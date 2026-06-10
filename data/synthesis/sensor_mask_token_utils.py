"""
Sensor Mask Token Utilities

Encodes binary segmentation mask videos into PaliGemma-style tokens:
  - 4 loc tokens  (bounding box, int32 in [0, 1023])
  - 16 seg tokens (VQ-VAE codebook indices, int32 in [0, 127])

For each mask video the output is a <mask_stem>_tokens.npz file saved in the
same directory as the source mask video, with keys:
  - loc_tokens : [N, 4]  int32
  - seg_tokens : [N, 16] int32
  - bboxes     : [N, 4]  float32  (y1, x1, y2, x2 in [0, 1])
  - valid      : [N]     bool     (False for empty-mask frames)

Supports all 5 datasets:
  VITRA_Robot_Data, InternA1_Robot_Data, MolmoAct_Data, Agibot_Data, Muse

Usage:
  Single process:  python sensor_mask_token_utils.py
  Distributed:     set IF_DIST=True, then torchrun --nproc_per_node=N sensor_mask_token_utils.py
  Test mode:       set TEST_MODE=True and TEST_VIDEO_PATH, then run directly
"""

import functools
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import av
import cv2
import numpy as np
import PIL.Image

# ---------------------------------------------------------------------------
# Add big_vision to the import path so we can use the encoder / decoder
# ---------------------------------------------------------------------------
_BIG_VISION_ROOT = Path(__file__).parent / "big_vision"
sys.path.insert(0, str(_BIG_VISION_ROOT))

# Hide all GPUs from TensorFlow before it initializes — must be set before import.
# This script runs CPU-only TF ops (small 64x64 conv); no GPU needed.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

# Belt-and-suspenders: also restrict via TF API
tf.config.set_visible_devices([], 'GPU')

# Encoder (TF-based)
from big_vision.pp.proj.paligemma.segmentation import encode_to_codebook_indices

# Decoder (JAX/Flax-based) – only imported when needed to avoid mandatory JAX dep
# We import lazily inside get_reconstruct_masks_cached() below.

# ---------------------------------------------------------------------------
# Force unbuffered output
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------
SKIP_EXISTING = False
IF_DIST      = True   # Set True to enable torchrun-style distributed processing
TEST_MODE    = False   # Set True to run test mode instead of full pipeline
TEST_VIDEO_PATH  = "<LOCAL_PATH>"  # Path to a mask video for test mode
TEST_N_FRAMES    = 10  # Number of frames to test
TEST_OUTPUT_DIR  = "<LOCAL_PATH>"

BATCH_SIZE = 64  # Frames per TF encoder call

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
_CHECKPOINT_URL   = 'gs://big_vision/paligemma/vae-oid.npz'
_CHECKPOINT_CACHE = _BIG_VISION_ROOT / "vae-oid.npz"

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------
_DATA_ROOT = Path("<DATA_ROOT>")
# _DATA_ROOT = Path("<LOCAL_PATH>")


DATASET_CONFIGS = {
    "VITRA_Robot_Data": {
        "type": "vitra",
        "root": _DATA_ROOT / "VITRA_Robot_Data",
        "json_file": "all_instructions_0910.json",
        "video_name": "video.mp4",
        "mask_name": "video_seg.mp4",
    },
    "InternA1_Robot_Data": {
        "type": "interna1",
        "root": _DATA_ROOT / "InternA1_Robot_Data/real/genie1",
    },
    "MolmoAct_Data": {
        "type": "molmoact",
        "root": _DATA_ROOT / "MolmoAct_Data",
        "subdirs": ["molmoact_dataset_household", "molmoact_dataset_tabletop"],
    },
    "Agibot_Data": {
        "type": "agibot",
        "root": _DATA_ROOT / "Agibot_Data_Copy/observations_extract",
        "video_name": "videos/head_color.mp4",
        "mask_name": "head_mask.mp4",
    },
    "Muse_Robot_Dataset": {
        "type": "muse",
        "root": _DATA_ROOT / "Muse_Robot_Dataset",
    },
}


# ===========================================================================
# Phase 1 — Checkpoint loading
# ===========================================================================

def load_vae_checkpoint() -> dict:
    """Load the PaliGemma VAE-OID checkpoint, caching it locally."""
    if _CHECKPOINT_CACHE.exists():
        print(f"Loading cached checkpoint from {_CHECKPOINT_CACHE}")
        return dict(np.load(str(_CHECKPOINT_CACHE)))

    print(f"Downloading checkpoint from {_CHECKPOINT_URL} ...")
    _CHECKPOINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    from tensorflow.io import gfile
    with gfile.GFile(_CHECKPOINT_URL, 'rb') as f:
        data = f.read()
    with open(_CHECKPOINT_CACHE, 'wb') as f:
        f.write(data)
    print(f"Saved checkpoint to {_CHECKPOINT_CACHE}")
    return dict(np.load(_CHECKPOINT_CACHE))


@functools.cache
def get_reconstruct_masks_cached():
    """Return a JIT-compiled JAX reconstruction function (cached after first call).

    Imports JAX lazily so the script can be used without JAX installed when
    test mode is disabled.
    """
    from big_vision.evaluators.proj.paligemma.transfers.segmentation import (
        get_reconstruct_masks,
    )
    # Pass the local cache path so get_reconstruct_masks doesn't re-fetch from GCS.
    return get_reconstruct_masks(str(_CHECKPOINT_CACHE))


# ===========================================================================
# Phase 2 — Core encoding
# ===========================================================================

def compute_bbox(mask_gray: np.ndarray):
    """Compute the tight bounding box of the non-zero region in mask_gray.

    Args:
        mask_gray: 2-D uint8/bool array, H x W.

    Returns:
        (y1, x1, y2, x2) as float32 in [0, 1], normalized by image size.
        Returns (0.0, 0.0, 1.0, 1.0) if the mask is empty.
    """
    h, w = mask_gray.shape
    rows = np.any(mask_gray > 0, axis=1)
    cols = np.any(mask_gray > 0, axis=0)
    if not rows.any():
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array(
        [rmin / h, cmin / w, (rmax + 1) / h, (cmax + 1) / w],
        dtype=np.float32,
    )


def _prepare_crop(mask_gray: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop mask to bbox and resize to 64x64, normalized to [0, 1].

    Args:
        mask_gray: H x W uint8 mask (0 = background, >0 = object).
        bbox: (y1, x1, y2, x2) float32 in [0, 1].

    Returns:
        np.ndarray of shape [64, 64, 1], float32 in [0, 1].
    """
    h, w = mask_gray.shape
    y1 = int(round(bbox[0] * h))
    x1 = int(round(bbox[1] * w))
    y2 = int(round(bbox[2] * h))
    x2 = int(round(bbox[3] * w))

    # Clamp to valid pixel range
    y1, x1 = max(y1, 0), max(x1, 0)
    y2, x2 = min(y2, h), min(x2, w)

    if y2 <= y1 or x2 <= x1:
        crop = np.zeros((64, 64), dtype=np.float32)
    else:
        crop = mask_gray[y1:y2, x1:x2].astype(np.float32)
        crop = crop / 255.0 if crop.max() > 1.0 else crop.astype(np.float32)
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)

    return crop[:, :, np.newaxis]  # [64, 64, 1]


def encode_mask_video(mask_video_path: Path, checkpoint: dict, batch_size: int = BATCH_SIZE) -> dict:
    """Encode every frame of a mask video into loc + seg tokens.

    Args:
        mask_video_path: Path to the binary mask video.
        checkpoint: VAE checkpoint dict (from load_vae_checkpoint()).
        batch_size: Number of frames to encode in one TF call.

    Returns:
        dict with keys:
          loc_tokens : [N, 4]  int32   bbox tokens in [0, 1023]
          seg_tokens : [N, 16] int32   VQ-VAE indices in [0, 127]
          bboxes     : [N, 4]  float32 normalized bounding boxes
          valid      : [N]     bool    True if the frame had a non-empty mask
    """
    # --- decode all frames ---
    gray_frames = []
    container = av.open(str(mask_video_path))
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format='rgb24')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
    container.close()

    if not gray_frames:
        return {
            'loc_tokens': np.zeros((0, 4),  dtype=np.int32),
            'seg_tokens': np.zeros((0, 16), dtype=np.int32),
            'bboxes':     np.zeros((0, 4),  dtype=np.float32),
            'valid':      np.zeros((0,),    dtype=bool),
        }

    n = len(gray_frames)
    all_loc  = np.zeros((n, 4),  dtype=np.int32)
    all_seg  = np.zeros((n, 16), dtype=np.int32)
    all_bbox = np.zeros((n, 4),  dtype=np.float32)
    all_valid = np.zeros((n,),   dtype=bool)

    # --- compute bboxes and crops for all frames, then batch-encode ---
    crops = []
    bboxes = []
    valids = []
    for gray in gray_frames:
        bbox  = compute_bbox(gray)
        valid = bool(gray.max() > 0)
        crop  = _prepare_crop(gray, bbox)
        crops.append(crop)
        bboxes.append(bbox)
        valids.append(valid)

    bboxes_arr = np.stack(bboxes)   # [N, 4]
    crops_arr  = np.stack(crops)    # [N, 64, 64, 1]

    # batch-encode via TF encoder
    seg_tokens_all = []
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = tf.constant(crops_arr[start:end], dtype=tf.float32)
        indices = encode_to_codebook_indices(checkpoint, batch)  # [B, 16]
        seg_tokens_all.append(indices.numpy().astype(np.int32))

    seg_tokens_arr = np.concatenate(seg_tokens_all, axis=0)  # [N, 16]

    # loc tokens: round bbox * 1023, clip to [0, 1023]
    loc_tokens_arr = np.round(bboxes_arr * 1023).clip(0, 1023).astype(np.int32)

    all_loc   = loc_tokens_arr
    all_seg   = seg_tokens_arr
    all_bbox  = bboxes_arr
    all_valid = np.array(valids, dtype=bool)

    return {
        'loc_tokens': all_loc,
        'seg_tokens': all_seg,
        'bboxes':     all_bbox,
        'valid':      all_valid,
    }


# ===========================================================================
# Phase 3 — Reconstruction (test mode only)
# ===========================================================================

def reconstruct_frames(seg_tokens: np.ndarray, bboxes: np.ndarray, orig_hw: tuple) -> list:
    """Reconstruct full-resolution binary masks from seg tokens.

    Args:
        seg_tokens: [N, 16] int32 in [0, 127].
        bboxes:     [N, 4]  float32 (y1, x1, y2, x2) in [0, 1].
        orig_hw:    (H, W) of the original mask video.

    Returns:
        List of N HxW bool arrays (full-resolution reconstructed masks).
        Also returns the 64x64 decoded float arrays for visualization.
    """
    import jax

    reconstruct_fn = get_reconstruct_masks_cached()
    h, w = orig_hw

    full_masks   = []
    mask64_list  = []

    for i in range(len(seg_tokens)):
        indices = seg_tokens[i:i+1].astype(np.int32)  # [1, 16]
        tocpu   = lambda x: jax.device_put(x, jax.local_devices(backend='cpu')[0])
        m64 = jax.device_get(reconstruct_fn(tocpu(indices)))[0, ..., 0]  # [64, 64] float [-1,1]
        mask64_list.append(m64)

        bbox = bboxes[i]
        y1 = int(round(bbox[0] * h))
        x1 = int(round(bbox[1] * w))
        y2 = int(round(bbox[2] * h))
        x2 = int(round(bbox[3] * w))
        y1, x1 = max(y1, 0), max(x1, 0)
        y2, x2 = min(y2, h), min(x2, w)

        full = np.zeros((h, w), dtype=bool)
        if y2 > y1 and x2 > x1:
            # m64 is float in [-1, 1]; convert to uint8 for PIL resize
            m64_u8 = ((m64 + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            resized = np.asarray(
                PIL.Image.fromarray(m64_u8).resize(
                    [x2 - x1, y2 - y1], resample=PIL.Image.BILINEAR
                )
            )
            # threshold: decoder output > 0 in [-1,1] space ↔ > 127 in uint8
            full[y1:y2, x1:x2] = resized > 127
        full_masks.append(full)

    return full_masks, mask64_list


# ===========================================================================
# Phase 4 — Dataset mask finders
# ===========================================================================

def find_masks_vitra(config: dict) -> list:
    """Yields (mask_path, token_out_path) pairs for VITRA_Robot_Data."""
    json_file = config['root'] / config['json_file']
    if not json_file.exists():
        print(f"  JSON file not found: {json_file}")
        return []

    with open(json_file, 'r') as f:
        data = json.load(f)

    pairs = []
    seen = set()
    for item in data:
        video_dir = config['root'] / item['relative_path']
        if str(video_dir) in seen:
            continue
        mask_path = video_dir / config['mask_name']
        if mask_path.exists():
            seen.add(str(video_dir))
            token_out = mask_path.parent / f"{mask_path.stem}_tokens.npz"
            pairs.append((mask_path, token_out))
    return pairs


def find_masks_agibot(config: dict) -> list:
    """Yields (mask_path, token_out_path) pairs for Agibot_Data."""
    root = config['root']
    pairs = []
    if not root.exists():
        print(f"  Root not found: {root}")
        return []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for episode_dir in sorted(task_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            mask_path = episode_dir / config['mask_name']
            if mask_path.exists():
                token_out = mask_path.parent / f"{mask_path.stem}_tokens.npz"
                pairs.append((mask_path, token_out))
    return pairs


def find_masks_interna1(config: dict) -> list:
    """Yields (mask_path, token_out_path) pairs for InternA1_Robot_Data."""
    root = config['root']
    pairs = []
    if not root.exists():
        print(f"  Root not found: {root}")
        return []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        set_dir = task_dir / "set_0"
        if not set_dir.exists():
            continue
        videos_dir = set_dir / "videos"
        if not videos_dir.exists():
            continue
        for chunk_dir in sorted(videos_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            mask_dir = chunk_dir / "observation.images.mask"
            if not mask_dir.exists():
                continue
            for mask_file in sorted(mask_dir.glob("*.mp4")):
                token_out = mask_file.parent / f"{mask_file.stem}_tokens.npz"
                pairs.append((mask_file, token_out))
    return pairs


def find_masks_molmoact(config: dict) -> list:
    """Yields (mask_path, token_out_path) pairs for MolmoAct_Data."""
    pairs = []
    for subdir_name in config['subdirs']:
        subdir = config['root'] / subdir_name
        if not subdir.exists():
            continue
        videos_dir = subdir / "train" / "videos"
        if not videos_dir.exists():
            continue
        for chunk_dir in sorted(videos_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            mask_dir = chunk_dir / "observations.mask"
            if not mask_dir.exists():
                continue
            for mask_file in sorted(mask_dir.glob("*.mp4")):
                token_out = mask_file.parent / f"{mask_file.stem}_tokens.npz"
                pairs.append((mask_file, token_out))
    return pairs


def find_masks_muse(config: dict) -> list:
    """Yields (mask_path, token_out_path) pairs for Muse_Robot_Dataset."""
    root = config['root']
    pairs = []
    if not root.exists():
        print(f"  Root not found: {root}")
        return []
    for data_type_dir in sorted(root.iterdir()):
        if not data_type_dir.is_dir():
            continue
        for episode_dir in sorted(data_type_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            mask_path = episode_dir / "mask" / "video.mp4"
            if mask_path.exists():
                token_out = mask_path.parent / "video_tokens.npz"
                pairs.append((mask_path, token_out))
    return pairs


def find_masks_for_dataset(dataset_name: str, config: dict) -> list:
    """Dispatch to the correct finder for the given dataset."""
    dtype = config['type']
    if dtype == 'vitra':
        return find_masks_vitra(config)
    elif dtype == 'agibot':
        return find_masks_agibot(config)
    elif dtype == 'interna1':
        return find_masks_interna1(config)
    elif dtype == 'molmoact':
        return find_masks_molmoact(config)
    elif dtype == 'muse':
        return find_masks_muse(config)
    else:
        print(f"  Unknown dataset type '{dtype}' for {dataset_name}")
        return []


# ===========================================================================
# Phase 5 — Processing
# ===========================================================================

def init_distributed():
    """Read distributed rank/world-size from env vars set by torchrun (or manually).

    No process group is needed because ranks never communicate — work is split
    purely by reading RANK / WORLD_SIZE from the environment.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    print(f"Process {rank}/{world_size} (local_rank={local_rank})")
    return local_rank, world_size, rank


def split_by_rank(items: list, world_size: int, rank: int) -> list:
    """Round-robin split of items across ranks."""
    return [item for i, item in enumerate(items) if i % world_size == rank]


def process_mask_video(
    mask_path: Path,
    token_out_path: Path,
    checkpoint: dict,
    batch_size: int = BATCH_SIZE,
) -> bool:
    """Encode one mask video and save the tokens npz. Returns True on success."""
    if SKIP_EXISTING and token_out_path.exists():
        return True

    try:
        result = encode_mask_video(mask_path, checkpoint, batch_size)
        np.savez(
            str(token_out_path),
            loc_tokens=result['loc_tokens'],
            seg_tokens=result['seg_tokens'],
            bboxes=result['bboxes'],
            valid=result['valid'],
        )
        return True
    except Exception as e:
        print(f"  Error encoding {mask_path}: {e}")
        traceback.print_exc()
        return False


def process_dataset(
    dataset_name: str,
    config: dict,
    rank: int,
    world_size: int,
    checkpoint: dict,
):
    """Process all mask videos in one dataset with optional distributed split."""
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")

    # All ranks build the same full sorted list
    all_pairs = sorted(find_masks_for_dataset(dataset_name, config))

    if not all_pairs:
        if rank == 0:
            print(f"  No mask videos found for {dataset_name}")
        return

    if rank == 0:
        print(f"  Found {len(all_pairs)} mask video(s)")

    # Filter already-done pairs
    if SKIP_EXISTING:
        pending = [(m, t) for m, t in all_pairs if not t.exists()]
        skipped = len(all_pairs) - len(pending)
        if skipped and rank == 0:
            print(f"  Skipping {skipped} already tokenized video(s)")
    else:
        pending = all_pairs

    if not pending:
        if rank == 0:
            print(f"  All videos already tokenized for {dataset_name}")
        return

    # Distribute across ranks
    my_pairs = split_by_rank(pending, world_size, rank)
    print(f"  Rank {rank}: assigned {len(my_pairs)} video(s)")

    ok = fail = 0
    t0 = time.time()
    for idx, (mask_path, token_out_path) in enumerate(my_pairs, 1):
        print(f"  Rank {rank}: [{idx}/{len(my_pairs)}] {mask_path}")
        success = process_mask_video(mask_path, token_out_path, checkpoint)
        if success:
            ok += 1
        else:
            fail += 1
        gc.collect()

    elapsed = time.time() - t0
    print(
        f"  Rank {rank}: {dataset_name} done — "
        f"{ok} ok, {fail} failed, {elapsed:.1f}s total"
    )


# ===========================================================================
# Phase 6 — Test mode
# ===========================================================================

def run_test_mode(
    video_path: str,
    n_frames: int,
    output_dir: str,
    checkpoint: dict,
):
    """Encode the first n_frames of a mask video, reconstruct, and save PNGs.

    Saves per frame to output_dir/:
      frame_NNNN_raw.png           - original mask (grayscale, 0/255)
      frame_NNNN_crop64.png        - 64x64 crop fed to encoder (0-255)
      frame_NNNN_recon64.png       - decoded 64x64 mask (thresholded 0/255)
      frame_NNNN_recon_full.png    - full-resolution reconstruction
      frame_NNNN_strip.png         - horizontal concat of the 4 above
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = Path(video_path)
    print(f"Test mode: {mask_path}  (first {n_frames} frames)")
    print(f"Output dir: {out_dir}")

    # --- decode frames ---
    gray_frames = []
    container = av.open(str(mask_path))
    for frame in container.decode(video=0):
        if len(gray_frames) >= n_frames:
            break
        rgb  = frame.to_ndarray(format='rgb24')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
    container.close()

    if not gray_frames:
        print("No frames decoded — check video path.")
        return

    orig_hw = gray_frames[0].shape[:2]   # (H, W)
    print(f"Decoded {len(gray_frames)} frames, size {orig_hw}")

    # --- encode ---
    bboxes = []
    crops  = []
    valids = []
    for gray in gray_frames:
        bbox  = compute_bbox(gray)
        valid = bool(gray.max() > 0)
        crop  = _prepare_crop(gray, bbox)
        bboxes.append(bbox)
        crops.append(crop)
        valids.append(valid)

    bboxes_arr = np.stack(bboxes)
    crops_arr  = np.stack(crops)

    batch = tf.constant(crops_arr, dtype=tf.float32)
    seg_tokens = encode_to_codebook_indices(checkpoint, batch).numpy().astype(np.int32)  # [N, 16]
    loc_tokens = np.round(bboxes_arr * 1023).clip(0, 1023).astype(np.int32)

    print(f"Encoded {len(seg_tokens)} frames. "
          f"seg_tokens range: [{seg_tokens.min()}, {seg_tokens.max()}], "
          f"valid frames: {sum(valids)}/{len(valids)}")

    # --- reconstruct ---
    full_masks, mask64_list = reconstruct_frames(seg_tokens, bboxes_arr, orig_hw)

    # --- save images ---
    strip_height = 256  # all strip panels resized to this height

    for i, (gray, crop, m64, full_mask) in enumerate(
        zip(gray_frames, crops, mask64_list, full_masks)
    ):
        prefix = out_dir / f"frame_{i:04d}"

        # raw mask: grayscale to 3-channel for consistent RGB PNG
        raw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(prefix) + "_raw.png", raw_img)

        # 64x64 crop fed to encoder (float [0,1] → uint8 [0,255])
        crop_u8  = (crops[i][:, :, 0] * 255).clip(0, 255).astype(np.uint8)
        crop_rgb = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(prefix) + "_crop64.png", crop_rgb)

        # decoded 64x64 float in [-1,1] → uint8
        recon64_u8  = ((m64 + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        recon64_rgb = cv2.cvtColor(recon64_u8, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(prefix) + "_recon64.png", recon64_rgb)

        # full-resolution reconstruction
        full_u8  = full_mask.astype(np.uint8) * 255
        full_rgb = cv2.cvtColor(full_u8, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(prefix) + "_recon_full.png", full_rgb)

        # strip: resize all 4 panels to strip_height, concat horizontally
        def _to_strip_panel(img_bgr):
            h_img = img_bgr.shape[0]
            scale = strip_height / h_img
            w_new = max(1, int(img_bgr.shape[1] * scale))
            return cv2.resize(img_bgr, (w_new, strip_height), interpolation=cv2.INTER_NEAREST)

        panels = [
            _to_strip_panel(raw_img),
            _to_strip_panel(crop_rgb),
            _to_strip_panel(recon64_rgb),
            _to_strip_panel(full_rgb),
        ]
        strip = np.hstack(panels)
        cv2.imwrite(str(prefix) + "_strip.png", strip)

        print(f"  frame {i:04d}: valid={valids[i]}, "
              f"bbox={bboxes[i].round(3).tolist()}, "
              f"seg_tokens={seg_tokens[i].tolist()}")

    print(f"\nTest mode complete. Images saved to {out_dir}")


# ===========================================================================
# Phase 7 — Entry point
# ===========================================================================

def main():
    """Process all datasets, with optional distributed support."""
    os.environ.setdefault('NCCL_TIMEOUT', '600')
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')

    if IF_DIST:
        local_rank, world_size, rank = init_distributed()
    else:
        local_rank, world_size, rank = 0, 1, 0

    if rank == 0:
        print("=" * 80)
        print("Sensor Mask Token Encoding Pipeline")
        print("=" * 80)
        print(f"IF_DIST={IF_DIST}  world_size={world_size}  SKIP_EXISTING={SKIP_EXISTING}")
        print(f"BATCH_SIZE={BATCH_SIZE}")
        print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
        print("=" * 80)

    checkpoint = load_vae_checkpoint()

    t_start = time.time()
    for dataset_name, config in DATASET_CONFIGS.items():
        try:
            process_dataset(dataset_name, config, rank, world_size, checkpoint)
        except Exception as e:
            print(f"Rank {rank}: Error in {dataset_name}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\nRank {rank}: Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # No process group to destroy — ranks communicate via filesystem only.


if __name__ == "__main__":
    if TEST_MODE:
        if not TEST_VIDEO_PATH:
            raise ValueError("Set TEST_VIDEO_PATH before running in TEST_MODE.")
        checkpoint = load_vae_checkpoint()
        run_test_mode(TEST_VIDEO_PATH, TEST_N_FRAMES, TEST_OUTPUT_DIR, checkpoint)
    else:
        main()
