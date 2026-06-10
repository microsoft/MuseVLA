#!/usr/bin/env python
"""
VITRA Re-Synthesis Script (Phase 2)

Re-generates all training-required files for VITRA episodes:
  1. video_seg.mp4      — SAM3 segmentation mask (multi-prompt + single-object selection)
  2. thermal/*.mp4      — 9 thermal sensor overlay videos (3 labels × 3 backgrounds)
  3. radar/*.mp4        — 6 radar sensor overlay videos (2 labels × 3 backgrounds)
  4. audio/*.mp4        — 6 audio sensor overlay videos (2 labels × 3 backgrounds)

Does NOT regenerate: metadata, depth, action/state files, video_seg_tokens.npz.

Usage:
    # Local test on audit samples:
    python data_synthesis_vitra_reseg.py --local-test

    # Distributed on cluster (8 GPUs):
    torchrun --nproc_per_node=8 data_synthesis_vitra_reseg.py
"""

import json
import torch
import cv2
import numpy as np
import av
from pathlib import Path
import sys
import os
import time
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress ALL tqdm progress bars (SAM3 uses its own tqdm internally)
import tqdm
import tqdm.auto
import tqdm.std

class _SilentTqdm:
    """Drop-in tqdm replacement that produces no output."""
    def __init__(self, iterable=None, *args, **kwargs):
        self._it = iterable
    def __iter__(self):
        return iter(self._it) if self._it is not None else iter([])
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, n=1): pass
    def close(self): pass
    def set_description(self, *a, **kw): pass
    def set_postfix(self, *a, **kw): pass
    @staticmethod
    def write(s, **kw): pass

# Patch all tqdm entry points
tqdm.tqdm = _SilentTqdm
tqdm.auto.tqdm = _SilentTqdm
tqdm.std.tqdm = _SilentTqdm
os.environ['TQDM_DISABLE'] = '1'

# Add current dir to path for sensor_mask_utils import
sys.path.insert(0, str(Path(__file__).parent))
from sensor_mask_utils import traverse_single_video, set_verbose
from clone_augmentation import generate_clone_augmentation
# NOTE: sensor_mask_token_utils imports TF which sets CUDA_VISIBLE_DEVICES=''
# We must import it LAZILY after SAM3 (PyTorch/CUDA) is initialized.
# See _load_seg_token_encoder() below.


def _detect_video_codec():
    """Test if NVENC GPU encoder is available. Returns codec name and options."""
    import av as _av
    import numpy as _np
    # Try NVENC with a tiny test encode
    try:
        c = _av.open('/tmp/_codec_test.mp4', 'w')
        s = c.add_stream('h264_nvenc', rate=1)
        s.width, s.height, s.pix_fmt = 64, 64, 'yuv420p'
        s.options = {'preset': 'p1', 'rc': 'vbr', 'cq': '28'}
        f = _av.VideoFrame.from_ndarray(_np.zeros((64, 64, 3), dtype=_np.uint8), format='rgb24')
        for p in s.encode(f): c.mux(p)
        for p in s.encode(): c.mux(p)
        c.close()
        os.remove('/tmp/_codec_test.mp4')
        return 'h264_nvenc', {'preset': 'p1', 'rc': 'vbr', 'cq': '28'}
    except Exception:
        try:
            os.remove('/tmp/_codec_test.mp4')
        except Exception:
            pass
    return 'libx264', {'preset': 'ultrafast', 'crf': '23'}

# Detect once at import time
VIDEO_CODEC, VIDEO_CODEC_OPTS = _detect_video_codec()
print(f"Video codec: {VIDEO_CODEC} (options: {VIDEO_CODEC_OPTS})")
sys.stdout.flush()

# ─── VITRA multi-prompt lookup ───
# Prompts ordered by SPECIFICITY: most distinctive first to avoid wrong-object selection.
# Rule: color-specific ("the red item") before generic ("the tomato") before noun-only ("tomato").
# Based on Phase 1 empirical testing on 100 episodes.
VITRA_OBJECT_TO_PROMPTS: dict[str, list[str]] = {
    # Fruits — color-specific first (scenes often have multiple fruits)
    "orange mango":     ["the orange item", "the mango", "fruit", "orange fruit"],
    "red tomato":       ["the red item", "red tomato", "the tomato"],
    "red apple":        ["the red item", "apple", "the apple", "red fruit"],
    "green apple":      ["the green apple", "the green object", "apple", "green fruit"],
    "yellow pear":      ["the pear", "pear", "yellow fruit"],
    "yellow lemon":     ["the lemon", "the yellow item", "lemon"],
    "red pomegranate":  ["red fruit", "pomegranate", "the pomegranate"],
    "purple grape":     ["the purple item", "the grapes", "grapes", "purple fruit"],
    "purple grapes":    ["the purple item", "the grapes", "grapes", "purple fruit"],
    "pink pomegranate": ["red fruit", "pomegranate", "the pomegranate"],

    # Cups / containers — color-specific helps distinguish from other cups
    "yellow cup":       ["yellow cup", "the cup"],
    "green mug":        ["the mug", "mug", "green cup"],
    "green cup":        ["green cup", "the cup"],
    "white measuring cup": ["the measuring cup", "measuring cup", "pitcher"],
    "clear measuring cup": ["the measuring cup", "measuring cup"],
    "transparent cup with candy": ["the cup", "cup", "the container"],
    "transparent cup":  ["the cup", "cup"],
    "black cup":        ["the cup", "cup"],

    # Pot / lid / kitchen
    "clear pot lid":        ["lid", "pot lid"],
    "silver pot lid":       ["lid", "pot lid"],
    "transparent pot lid":  ["lid", "pot lid"],
    "white lid":            ["the lid", "lid"],
    "black pot":            ["the pot", "pot", "saucepan"],
    "black soup pot":       ["the pot", "soup pot", "pot"],
    "clear bottle":         ["the bottle", "bottle"],
    "black bottle":         ["the bottle", "bottle"],
    "purple bottle":        ["the purple item", "bottle", "the bottle"],
    "pink bottle":          ["the bottle", "bottle"],
    "black plastic bottle": ["the bottle", "bottle"],
    "black can":            ["the can", "can"],
    "green whisk":          ["whisk", "the whisk", "utensil"],
    "silver whisk":         ["whisk", "the whisk"],
    "white cup lid":        ["white lid", "white circle", "cup lid", "the lid"],
    "blue cup lid":         ["cup lid", "the lid"],
    "orange cup lid":       ["cup lid", "the lid"],
    "red cup lid":          ["cup lid", "the lid"],

    # Toys / small objects — color-specific first (scenes have multiple small items)
    "yellow duck":          ["yellow duck", "the yellow item", "toy duck", "rubber duck"],
    "white plush toy":      ["the white toy", "stuffed animal", "plush toy", "the plush"],
    "blue cube":            ["the blue item", "the cube", "cube", "block"],
    "blue wooden cube":     ["the blue item", "blue cube", "cube", "block"],
    "brown cube":           ["the cube", "the brown item", "cube", "block"],
    "wooden cube":          ["the cube", "cube", "block"],
    "brown wooden cube":    ["the cube", "the brown item", "cube"],
    "pink wooden cube":     ["the cube", "cube", "block"],
    "turquoise cube":       ["the cube", "cube", "block"],

    # Tools / household — "the X" pattern most reliable
    "brown broom":              ["the broom", "broom", "long stick", "cleaning tool"],
    "pink broom":               ["the broom", "broom", "long stick", "cleaning tool"],
    "green broom":              ["the broom", "broom", "long stick", "cleaning tool"],
    "tan broom":                ["the broom", "broom", "long stick", "cleaning tool"],
    "beige broom":              ["the broom", "broom", "long stick", "cleaning tool"],
    "black blackboard eraser":  ["the eraser", "eraser", "blackboard eraser"],
    "black chalkboard eraser":  ["the eraser", "eraser", "chalkboard eraser"],
    "black eraser":             ["the eraser", "eraser"],
    "orange blackboard eraser": ["the eraser", "eraser"],
    "yellow blackboard eraser": ["the eraser", "eraser"],
    "yellow tape":              ["the tape", "tape", "the yellow item"],
    "yellow tape roll":         ["the tape", "tape"],
    "brown cylinder":           ["the cylinder", "the brown item", "cylinder"],
    "wooden cylinder":          ["the cylinder", "cylinder", "wooden piece"],
    "brown wooden cylinder":    ["the cylinder", "cylinder"],
    "beige wooden cylinder":    ["the cylinder", "cylinder"],
    "tan wooden cylinder":      ["the cylinder", "cylinder"],
    "tan cylinder":             ["the cylinder", "cylinder"],
    "yellow cylinder":          ["the cylinder", "cylinder"],
    "pink cylinder":            ["the cylinder", "cylinder"],
    "green potted plant":       ["the green plant", "plant", "potted plant"],
    "pink sponge bottle brush": ["the pink item", "pink brush", "sponge brush"],
    "blue sponge bottle brush": ["the blue item", "blue brush", "sponge brush"],
    "pink bottle brush":        ["the pink item", "brush"],
    "white paper box":          ["the white box", "box", "paper box"],
    "brown paper box":          ["box", "paper box"],
    "black toolbox":            ["the toolbox", "toolbox", "black box"],
    "black toolbox lid":        ["the toolbox", "toolbox"],
    "black power screwdriver":  ["the screwdriver", "screwdriver", "power tool"],
    "gray power screwdriver":   ["the screwdriver", "screwdriver"],
    "green power screwdriver":  ["the screwdriver", "screwdriver"],
    "black and red power screwdriver": ["the screwdriver", "screwdriver"],
    "silver scoop":             ["the scoop", "the silver item", "scoop"],
    "pink gum jar":             ["the small jar", "pink jar", "the pink item", "jar"],
    "green gum jar":            ["the small jar", "the green item", "jar"],
    "brown wooden mortar":      ["the mortar", "mortar", "wooden bowl"],
    "brown mortar":             ["the mortar", "mortar"],
    "black flexible tripod":    ["the tripod", "tripod"],
    "pink flexible tripod":     ["the tripod", "tripod"],
    "pink tripod":              ["the tripod", "tripod"],
    "red tripod":               ["the tripod", "tripod"],
    "black and red tripod":     ["the tripod", "tripod"],
    "pink and black flexible tripod": ["the tripod", "tripod"],
    "orange popcorn box":       ["the orange item", "box", "the box"],
    "yellow popcorn box":       ["the box", "box"],
    "black water bottle":       ["the bottle", "water bottle"],
    "blue drink carton":        ["the bottle", "bottle"],
    "black pen container":      ["the container", "container"],
    "yellow rectangular box":   ["the box", "box"],
    "white book":               ["the book", "book"],
    "yellow book":              ["the book", "book"],
    "silver rectangular block": ["the silver block", "the silver item", "the block", "block"],
    "wooden rectangular block": ["the block", "block"],
    "silver metal block":       ["the silver block", "the silver item", "the block", "block"],
    "beige rectangular block":  ["the block", "block"],
    "white and brown plush toy": ["the white toy", "stuffed animal", "plush toy"],
    "clear cup":                ["the cup", "cup"],

    # ── New-format keys ("the X") matching updated metadata ──
    # These map to diverse prompts including color-specific and alternative nouns
    "the mango":        ["the orange item", "the mango", "fruit", "mango"],
    "the tomato":       ["the red item", "red tomato", "the tomato", "tomato"],
    "the apple":        ["the red item", "apple", "the apple", "fruit"],
    "the pear":         ["the pear", "pear", "yellow fruit"],
    "the lemon":        ["the lemon", "the yellow item", "lemon"],
    "the pomegranate":  ["red fruit", "pomegranate", "the pomegranate"],
    "the grapes":       ["the purple item", "the grapes", "grapes", "purple fruit"],
    "the cup":          ["the cup", "cup", "glass"],
    "the mug":          ["the mug", "mug", "cup"],
    "the measuring cup": ["the measuring cup", "measuring cup", "pitcher"],
    "the cup lid":      ["white lid", "white circle", "cup lid", "the lid", "cap"],
    "the pot":          ["the pot", "pot", "saucepan"],
    "the pot lid":      ["pot lid", "lid", "the lid"],
    "the lid":          ["the lid", "lid", "cap"],
    "the bottle":       ["the bottle", "bottle", "flask"],
    "the can":          ["the can", "can", "tin"],
    "the whisk":        ["whisk", "the whisk", "utensil"],
    "the jar":          ["the small jar", "pink jar", "the green item", "jar"],
    "the container":    ["the container", "container", "holder"],
    "the duck":         ["yellow duck", "the yellow item", "toy duck", "rubber duck"],
    "the plush toy":    ["the white toy", "stuffed animal", "plush toy", "the plush"],
    "the cube":         ["the cube", "cube", "block", "the block"],
    "the block":        ["the silver block", "the silver item", "the block", "block", "the cube"],
    "the cylinder":     ["the cylinder", "cylinder", "wooden piece"],
    "the broom":        ["the broom", "broom", "long stick", "cleaning tool"],
    "the brush":        ["the pink item", "brush", "sponge brush", "bottle brush"],
    "the eraser":       ["the eraser", "eraser", "blackboard eraser"],
    "the tape":         ["the tape", "tape", "the yellow item", "tape roll"],
    "the plant":        ["the green plant", "plant", "potted plant"],
    "the box":          ["the box", "box", "paper box"],
    "the toolbox":      ["the toolbox", "toolbox", "black box"],
    "the toolbox lid":  ["toolbox lid", "the toolbox", "toolbox"],
    "the screwdriver":  ["the screwdriver", "screwdriver", "power tool", "drill"],
    "the scoop":        ["the scoop", "the silver item", "scoop", "ladle"],
    "the mortar":       ["the mortar", "mortar", "wooden bowl", "pestle"],
    "the tripod":       ["the tripod", "tripod", "camera stand"],
    "the book":         ["the book", "book", "notebook"],
}

# ─── Lazy seg-token encoder (avoids TF killing CUDA at import time) ───
_seg_token_encoder = None
_seg_token_checkpoint = None

def _load_seg_token_encoder():
    """Lazily import sensor_mask_token_utils (which imports TF and hides GPUs).
    Must be called AFTER all PyTorch/CUDA initialization is complete."""
    global _seg_token_encoder, _seg_token_checkpoint
    if _seg_token_encoder is not None:
        return _seg_token_encoder, _seg_token_checkpoint
    # Save and restore CUDA_VISIBLE_DEVICES so TF doesn't break PyTorch
    saved_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    from sensor_mask_token_utils import encode_mask_video, load_vae_checkpoint
    # Restore CUDA visibility for PyTorch
    if saved_cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = saved_cuda
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    _seg_token_encoder = encode_mask_video
    _seg_token_checkpoint = load_vae_checkpoint()
    return _seg_token_encoder, _seg_token_checkpoint


# ─── Configuration ───
MIN_MASK_AREA = 200
MAX_MASK_AREA = 200_000
CHUNK_SIZE = 200
SKIP_EXISTING = True
PREDICTOR_RESTART_INTERVAL = 30

# Minimum file size in bytes to consider a video valid (not corrupted/truncated)
MIN_VIDEO_SIZE = 1024  # 1KB — any valid H.264 video is larger than this


NUM_CLONE_AUGMENTATIONS = 2  # Number of clone placements per episode


def _build_expected_files():
    """Build list of all expected output files for a complete episode."""
    modes = {
        'thermal': ['cold', 'room_temperature', 'hot'],
        'radar': ['hidden', 'absense'],
        'audio': ['ringing', 'silence'],
    }
    bgs = ['dark', 'color', 'grey']
    files = ['video_seg.mp4', 'video_seg_tokens.npz']
    for aug_i in range(NUM_CLONE_AUGMENTATIONS):
        files.append(f'video_aug{aug_i}.mp4')
    for mode, labels in modes.items():
        for label in labels:
            for bg in bgs:
                files.append(f'{mode}/video_{label}_{bg}.mp4')
                for aug_i in range(NUM_CLONE_AUGMENTATIONS):
                    files.append(f'{mode}/video_{label}_{bg}_aug{aug_i}.mp4')
    return files

EXPECTED_FILES = _build_expected_files()


def _build_old_format_files():
    """Build list of expected files using old _aug.mp4 suffix (from previous jobs)."""
    modes = {
        'thermal': ['cold', 'room_temperature', 'hot'],
        'radar': ['hidden', 'absense'],
        'audio': ['ringing', 'silence'],
    }
    bgs = ['dark', 'color', 'grey']
    files = ['video_seg.mp4', 'video_seg_tokens.npz', 'video_aug.mp4']
    for mode, labels in modes.items():
        for label in labels:
            for bg in bgs:
                files.append(f'{mode}/video_{label}_{bg}.mp4')
                files.append(f'{mode}/video_{label}_{bg}_aug.mp4')
    return files

OLD_FORMAT_FILES = _build_old_format_files()


def is_episode_complete(episode_dir: Path) -> bool:
    """Check if an episode has all expected output files and none are corrupted.
    
    Accepts EITHER the new format (_aug0/_aug1) OR the old format (_aug).
    Returns True only if ALL files in one format exist and are above minimum size.
    Any missing or truncated file → returns False (needs reprocessing).
    
    Uses batch listdir (4 calls: root + thermal/ + radar/ + audio/) instead of
    per-file stat (~67-112 calls), critical for blob mount performance.
    """
    # Batch-read all file sizes in 4 listdir calls
    try:
        root_files = {e.name: e.stat().st_size for e in episode_dir.iterdir() if e.is_file()}
    except (FileNotFoundError, OSError):
        return False
    sub_files = {}
    for sub in ('thermal', 'radar', 'audio'):
        try:
            for e in (episode_dir / sub).iterdir():
                if e.is_file():
                    sub_files[f'{sub}/{e.name}'] = e.stat().st_size
        except (FileNotFoundError, OSError):
            pass
    all_files = {**root_files, **sub_files}
    
    def _all_valid(file_list):
        for rel_path in file_list:
            size = all_files.get(rel_path)
            if size is None or size < MIN_VIDEO_SIZE:
                return False
        return True

    # Check new format first (_aug0, _aug1), then old format (_aug)
    return _all_valid(EXPECTED_FILES) or _all_valid(OLD_FORMAT_FILES)


def get_prompts_for_object(primary_object: str) -> list[str]:
    """Look up prompts for a primary object name, with auto-fallback."""
    key = primary_object.lower().strip()
    prompts = VITRA_OBJECT_TO_PROMPTS.get(key)
    if prompts:
        return prompts
    # Auto-generate: strip color prefix, use noun
    words = key.split()
    if len(words) > 1:
        noun = ' '.join(words[1:])
        return [f"the {noun}", noun, key]
    return [key, f"the {key}"]


def select_best_mask_frame0(masks_list, height, width):
    """
    From SAM3 detections on frame 0, select the single best mask.
    
    Returns (best_mask_idx, best_score, best_area) or (-1, 0, 0) if none valid.
    best_mask_idx is the index into the masks array.
    """
    if not masks_list or len(masks_list) == 0:
        return -1, 0.0, 0

    best_idx = -1
    best_score = 0.0
    best_area = 0

    for i in range(len(masks_list)):
        m = masks_list[i]
        while m.ndim > 2:
            m = m[0]
        area = int(m.astype(bool).sum())
        # Use implicit score = 1.0 for video predictor (no per-mask scores in propagation)
        # For frame 0 from add_prompt, we use area as quality proxy
        if MIN_MASK_AREA <= area <= MAX_MASK_AREA:
            if area > best_area:
                best_idx = i
                best_area = area
    
    return best_idx, 1.0 if best_idx >= 0 else 0.0, best_area


# Max frames per SAM3 session before chunking (prevents OOM on 40GB GPUs)
# 40GB A100: ~19.5 MB/frame → max ~1836 frames. Use 1500 with safety margin.
MAX_FRAMES_PER_SESSION = 1500


def _run_sam3_prompt_on_video(video_path, prompt, video_predictor, height, width, gpu_id=-1):
    """Run SAM3 with a single prompt on a video, with automatic chunking for long videos.
    
    Returns (all_masks, frame0_good, n_empty, n_large_filtered, prompts_msg) or raises on error.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if frame_count <= MAX_FRAMES_PER_SESSION:
        # Short video: process in one session
        return _run_sam3_single_session(video_path, prompt, video_predictor, height, width, 0, frame_count)
    
    # Long video: process in chunks
    all_masks = {}
    frame0_good = False
    total_empty = 0
    total_large = 0
    
    for chunk_start in range(0, frame_count, MAX_FRAMES_PER_SESSION):
        chunk_end = min(chunk_start + MAX_FRAMES_PER_SESSION, frame_count)
        
        # Create temp video for this chunk
        cap = cv2.VideoCapture(str(video_path))
        enc_w = width + (width % 2)
        enc_h = height + (height % 2)
        
        tmp_path = str(video_path) + f".chunk_{chunk_start}_{gpu_id}.mp4"
        container = av.open(tmp_path, 'w')
        stream = container.add_stream('libx264', rate=max(1, int(fps)))
        stream.width = enc_w
        stream.height = enc_h
        stream.pix_fmt = 'yuv420p'
        stream.options = {'preset': 'ultrafast', 'crf': '23'}
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        for _ in range(chunk_end - chunk_start):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[0] != enc_h or frame.shape[1] != enc_w:
                frame = cv2.resize(frame, (enc_w, enc_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vf = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        cap.release()
        
        try:
            chunk_masks, chunk_f0_good, chunk_empty, chunk_large = _run_sam3_single_session(
                tmp_path, prompt, video_predictor, height, width, chunk_start, chunk_end - chunk_start
            )
            # Merge chunk results
            all_masks.update(chunk_masks)
            total_empty += chunk_empty
            total_large += chunk_large
            if chunk_start == 0:
                frame0_good = chunk_f0_good
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_masks, frame0_good, total_empty, total_large


def _run_sam3_single_session(video_path, prompt, video_predictor, height, width, frame_offset, expected_frames):
    """Run SAM3 on a single video (or chunk) with one prompt. Returns (masks, frame0_good, n_empty, n_large)."""
    print(f"      [session] start_session: {Path(video_path).name} offset={frame_offset} expected={expected_frames}", flush=True)
    response = video_predictor.handle_request(
        request=dict(type="start_session", resource_path=str(video_path))
    )
    session_id = response["session_id"]
    
    print(f"      [session] add_prompt: '{prompt}'", flush=True)
    response = video_predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
    )
    
    all_masks = {}
    frame0_good = False
    n_empty = 0
    n_large = 0
    propagate_start = time.time()
    last_progress = time.time()
    
    try:
        print(f"      [session] propagating ...", end='', flush=True)
        for response in video_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id, propagation_direction="both")
        ):
            frame_idx = response["frame_index"]
            masks = response["outputs"].get("out_binary_masks", [])
            
            if len(masks) > 0:
                valid_masks = []
                for m in masks:
                    while m.ndim > 2:
                        m = m[0]
                    area = int(m.astype(bool).sum())
                    if MIN_MASK_AREA <= area <= MAX_MASK_AREA:
                        valid_masks.append(m)
                    elif area > MAX_MASK_AREA:
                        n_large += 1
                
                if valid_masks:
                    best = max(valid_masks, key=lambda m: m.astype(bool).sum())
                    all_masks[frame_offset + frame_idx] = best.astype(bool)
                else:
                    all_masks[frame_offset + frame_idx] = np.zeros((height, width), dtype=bool)
                    n_empty += 1
            else:
                all_masks[frame_offset + frame_idx] = np.zeros((height, width), dtype=bool)
                n_empty += 1
            
            if frame_idx == 0:
                frame0_good = all_masks[frame_offset].any()
                if not frame0_good:
                    print(f" frame0=MISS, aborting propagation", flush=True)
                    break
                else:
                    print(f" frame0=HIT", end='', flush=True)
            
            # Progress every 10 seconds
            now = time.time()
            if now - last_progress >= 10.0:
                elapsed = now - propagate_start
                fps_rate = len(all_masks) / elapsed if elapsed > 0 else 0
                print(f" [{len(all_masks)}/{expected_frames}f {fps_rate:.1f}fps]", end='', flush=True)
                last_progress = now
    finally:
        propagate_time = time.time() - propagate_start
        print(f" done ({len(all_masks)} frames, {propagate_time:.1f}s)", flush=True)
        try:
            video_predictor.handle_request(dict(type="close_session", session_id=session_id))
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    return all_masks, frame0_good, n_empty, n_large


def generate_mask_video_multi_prompt(
    video_path: Path,
    output_path: Path,
    primary_object: str,
    video_predictor,
    gpu_id: int = -1,
) -> dict:
    """
    Generate video_seg.mp4 using multi-prompt strategy.
    
    Strategy:
      1. Look up prompt candidates for primary_object
      2. For each prompt, start SAM3 video session, add text prompt on frame 0
      3. Check frame 0 detection: if a valid mask exists (area in bounds), propagate
      4. During propagation, apply per-frame area filtering
      5. Write single best mask per frame to output video
    
    Returns dict with stats.
    """
    prompts = get_prompts_for_object(primary_object)
    
    # Get video info (retry on blob mount timeout)
    frame_count = 0
    video_open_msgs = []
    for _attempt in range(3):
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '4096'
        print(f"    [video open] attempt {_attempt+1}/3: {video_path.name} ...", end='', flush=True)
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 60s open timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)   # 60s read timeout
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dt = time.time() - t0
        if frame_count > 0:
            print(f" OK ({frame_count} frames, {width}x{height}, {fps:.1f}fps, {dt:.1f}s)", flush=True)
            video_open_msgs.append(f"attempt {_attempt+1}: OK ({frame_count}f, {dt:.1f}s)")
            break
        print(f" TIMEOUT (w={width}, h={height}, fps={fps}, {dt:.1f}s)", flush=True)
        video_open_msgs.append(f"attempt {_attempt+1}: TIMEOUT ({dt:.1f}s)")
        time.sleep(3)  # brief wait before retry (blob mount may be slow)
    
    if frame_count == 0:
        err_detail = ' | '.join(video_open_msgs)
        return {"status": "no_frames", "prompt": "", "n_frames": 0,
                "time": 0, "empty_pct": 100,
                "prompts_tried": [f"(video unreadable after 3 attempts: {err_detail})"]}
    
    enc_w = width + (width % 2)
    enc_h = height + (height % 2)
    
    start_time = time.time()
    all_masks = {}
    winning_prompt = ""
    
    # Try each prompt until one gives good detection on frame 0
    prompts_tried = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"    [SAM3] prompt {prompt_idx+1}/{len(prompts)}: '{prompt}' on {frame_count} frames ({width}x{height}) ...", flush=True)
        prompt_start = time.time()
        try:
            masks, frame0_good, n_empty, n_large_filtered = _run_sam3_prompt_on_video(
                video_path, prompt, video_predictor, height, width, gpu_id
            )
            prompt_time = time.time() - prompt_start
            
            if frame0_good:
                all_masks = masks
                winning_prompt = prompt
                empty_pct = 100 * n_empty / len(all_masks) if all_masks else 100
                print(f"    [SAM3] prompt '{prompt}' -> HIT ({len(all_masks)} frames, {n_empty} empty, {prompt_time:.1f}s)", flush=True)
                prompts_tried.append(f"'{prompt}' -> OK ({len(all_masks)} frames, {n_empty} empty, {n_large_filtered} large-filtered, {prompt_time:.1f}s)")
                break
            else:
                print(f"    [SAM3] prompt '{prompt}' -> MISS frame0 ({prompt_time:.1f}s)", flush=True)
                prompts_tried.append(f"'{prompt}' -> FAIL frame0 ({prompt_time:.1f}s)")
                all_masks.clear()
                
        except Exception as e:
            prompt_time = time.time() - prompt_start
            prompts_tried.append(f"'{prompt}' -> ERROR: {str(e)[:100]} ({prompt_time:.1f}s)")
            print(f"    Prompt '{prompt}' failed: {str(e)[:150]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_masks.clear()
            continue
    
    processing_time = time.time() - start_time
    
    # If no prompt worked, write all-zero mask video
    if not winning_prompt:
        all_masks = {i: np.zeros((height, width), dtype=bool) for i in range(frame_count)}
        empty_pct = 100.0
    
    # Write mask video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_container = av.open(str(output_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=max(1, int(fps)))
    output_stream.width = enc_w
    output_stream.height = enc_h
    output_stream.pix_fmt = 'yuv420p'
    
    for frame_idx in sorted(all_masks.keys()):
        mask = all_masks[frame_idx].astype(np.uint8) * 255
        if mask.shape[0] != enc_h or mask.shape[1] != enc_w:
            mask = cv2.resize(mask, (enc_w, enc_h), interpolation=cv2.INTER_NEAREST)
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        frame = av.VideoFrame.from_ndarray(mask_rgb, format='rgb24')
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
    
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()
    
    return {
        "status": "ok" if winning_prompt else "no_detection",
        "prompt": winning_prompt,
        "n_frames": len(all_masks),
        "empty_pct": empty_pct if winning_prompt else 100.0,
        "time": processing_time,
        "prompts_tried": prompts_tried,
    }


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} on GPU {local_rank} (no dist process group needed)")
    return local_rank, world_size, rank


def split_data_by_video(data: list, world_size: int, rank: int) -> list:
    """Split data across GPUs, keeping same relative_path together."""
    path_groups = {}
    for item in data:
        rel_path = item['relative_path']
        if rel_path not in path_groups:
            path_groups[rel_path] = []
        path_groups[rel_path].append(item)
    groups = list(path_groups.values())
    my_groups = []
    for i, group in enumerate(groups):
        if i % world_size == rank:
            my_groups.extend(group)
    return my_groups


def main():
    parser = argparse.ArgumentParser(description='VITRA Re-Segmentation')
    parser.add_argument('--local-test', action='store_true',
                        help='Run on local audit samples instead of full dataset')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Override metadata JSON file path (for testing subsets)')
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip episodes where all 45 output files exist and are valid (default: on)')
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing')
    parser.add_argument('--force-reprocess', type=str, default=None,
                        help='JSON file mapping relative_paths to reasons. Deletes video_seg.mp4 '
                             'for listed episodes to force re-segmentation even with --skip-existing')
    args = parser.parse_args()
    
    local_rank, world_size, rank = init_distributed()
    
    def print0(msg):
        if rank == 0:
            print(msg)
    
    # Determine data root and metadata
    if args.local_test:
        # Local test mode: use audit samples
        data_root = Path(__file__).parent / "vitra_audit_local"
        metadata_source = Path(args.metadata) if args.metadata else Path(__file__).parent / "vitra_audit_samples.json"
        print0(f"LOCAL TEST mode: {metadata_source}")
        with open(metadata_source, 'r') as f:
            data = json.load(f)
    else:
        # Cluster mode: use VITRA_Robot_Data_ReSeg
        data_root = Path(args.data_root) if args.data_root else Path("<DATA_ROOT>")
        metadata_source = data_root / "all_metadata_dist.json"
        if not metadata_source.exists():
            # Fallback to non-ReSeg
            data_root = Path("<DATA_ROOT>")
            metadata_source = data_root / "all_metadata_dist.json"
        print0(f"CLUSTER mode: {data_root}")
        with open(metadata_source, 'r') as f:
            data = json.load(f)
    
    if args.max_episodes:
        data = data[:args.max_episodes]
    
    # Deduplicate by relative_path: multiple annotations (episodes) can share the
    # same video directory. Synthesis is per-video-directory (generates mask + overlays
    # once), so we process each unique directory exactly once.
    seen_paths = set()
    unique_data = []
    for item in data:
        rp = item['relative_path']
        if rp not in seen_paths:
            seen_paths.add(rp)
            unique_data.append(item)
    
    n_total_annotations = len(data)
    n_total_videos = len(unique_data)
    print0(f"Metadata: {n_total_annotations} annotations across {n_total_videos} unique video directories")
    
    # Load frame counts for load-balanced GPU assignment
    frame_counts_path = data_root / "vitra_frame_counts.json"
    frame_counts = {}
    if frame_counts_path.exists():
        with open(frame_counts_path, 'r') as f:
            frame_counts = json.load(f)
        print0(f"Loaded frame counts for {len(frame_counts)} videos")
    
    # Force-reprocess: delete video_seg.mp4 for listed episodes so they fail is_episode_complete()
    # Force-reprocess: only rank 0 deletes files; other ranks wait for completion marker
    # Marker on shared storage (data_root) so all nodes can see it
    _force_reprocess_marker = data_root / ".force_reprocess_done"
    if args.force_reprocess:
        if rank == 0:
            _force_reprocess_marker.unlink(missing_ok=True)
            fp = Path(args.force_reprocess)
            if not fp.exists():
                fp = Path(__file__).parent / args.force_reprocess
            if fp.exists():
                with open(fp) as f:
                    force_list = json.load(f)
                n_deleted = 0
                for rp, reason in force_list.items():
                    seg_path = data_root / rp / "video_seg.mp4"
                    if seg_path.exists():
                        seg_path.unlink(missing_ok=True)
                        n_deleted += 1
                print0(f"force-reprocess: deleted video_seg.mp4 for {n_deleted}/{len(force_list)} episodes")
            else:
                print0(f"WARNING: --force-reprocess file not found: {args.force_reprocess}")
            _force_reprocess_marker.touch()
        else:
            import time as _time
            for _ in range(360):  # wait up to 30 min
                if _force_reprocess_marker.exists():
                    break
                _time.sleep(5)
            else:
                print(f"Rank {rank}: WARNING — timed out waiting for rank 0 force-reprocess")
    
    # Sort by frame count descending for better load balancing
    # (longest videos first → round-robin spreads them evenly across GPUs)
    def _get_frame_count(item):
        key = str(Path(item['relative_path']) / "video.mp4")
        if key in frame_counts:
            return frame_counts[key]
        # Fallback: use end-start if available
        s, e = item.get('start', 0), item.get('end', 300)
        return e - s if e and s else 300
    
    # Filter out already-complete videos BEFORE splitting across GPUs
    # This ensures even distribution of remaining work (not just total videos)
    if args.skip_existing:
        incomplete_data = [item for item in unique_data
                           if not is_episode_complete(data_root / item['relative_path'])]
        n_complete = n_total_videos - len(incomplete_data)
        total_frames_all = sum(_get_frame_count(item) for item in unique_data)
        total_frames_remaining = sum(_get_frame_count(item) for item in incomplete_data)
        print0(f"skip_existing: {n_complete}/{n_total_videos} videos already complete (skipped)")
        print0(f"  Remaining: {len(incomplete_data)} videos, ~{total_frames_remaining} frames "
               f"(of ~{total_frames_all} total)")
        unique_data = incomplete_data
    
    unique_data.sort(key=_get_frame_count, reverse=True)
    
    # Split across GPUs (round-robin on sorted list → balanced by frame count)
    my_data = split_data_by_video(unique_data, world_size, rank)
    my_total_frames = sum(_get_frame_count(item) for item in my_data)
    print(f"Rank {rank}: assigned {len(my_data)} videos, ~{my_total_frames} frames", flush=True)
    
    # Print distribution summary from rank 0
    if rank == 0 and world_size > 1:
        print(f"\n  GPU work distribution ({world_size} GPUs, {len(unique_data)} videos):")
        for r in range(world_size):
            r_data = split_data_by_video(unique_data, world_size, r)
            r_frames = sum(_get_frame_count(item) for item in r_data)
            print(f"    Rank {r:>2}: {len(r_data):>4} videos, ~{r_frames:>7} frames")
        sys.stdout.flush()
    
    # Initialize SAM3 video predictor
    # Only local_rank 0 on each node downloads from HuggingFace first;
    # all other ranks wait, then load from the cached checkpoint.
    print(f"Rank {rank}: Loading SAM3 video predictor...")
    sys.path.insert(0, str(Path(__file__).parent / "sam3"))
    from sam3.model_builder import build_sam3_video_predictor
    
    # Check for local checkpoint
    ckpt_path = Path(__file__).resolve().parents[2] / "assets" / "sam3.pt"
    
    if ckpt_path.exists():
        # Local checkpoint exists — all ranks can load directly
        video_predictor = build_sam3_video_predictor(checkpoint_path=str(ckpt_path))
    else:
        # Need to download from HuggingFace — serialize via filesystem lock
        # Only local_rank 0 on each node downloads; others wait for marker file
        marker = Path("/tmp/.sam3_download_done")
        if local_rank == 0:
            video_predictor = build_sam3_video_predictor()
            marker.touch()
        else:
            import time as _time
            while not marker.exists():
                _time.sleep(5)
            video_predictor = build_sam3_video_predictor()
    print(f"Rank {rank}: SAM3 loaded")
    
    # Initialize VQ-VAE for seg tokens AFTER SAM3/CUDA init (TF import hides GPUs)
    print(f"Rank {rank}: Loading VQ-VAE checkpoint for seg tokens...")
    encode_mask_fn, vae_checkpoint = _load_seg_token_encoder()
    print(f"Rank {rank}: VQ-VAE checkpoint loaded")
    
    # NOTE: vitra_invalid_videos.json is managed by analyze_dataset_vitra.py
    # Do NOT auto-delete it here — training depends on it for filtering bad episodes
    
    # Process
    print0("\n" + "=" * 80)
    print0("Starting re-synthesis (mask + sensor overlays + seg tokens + clone aug)...")
    print0(f"  skip_existing={args.skip_existing}, expected files per episode: {len(EXPECTED_FILES)}")
    print0("=" * 80)
    
    start_time = time.time()
    stats = {"ok": 0, "no_detection": 0, "skipped": 0, "error": 0, "no_video": 0}
    items_since_restart = 0
    processing_times = []  # Track per-episode times for ETA
    
    # Background thread pool for CPU-bound post-processing (render + tokens)
    # This overlaps with next episode's GPU-bound SAM3 segmentation
    bg_executor = ThreadPoolExecutor(max_workers=1)
    bg_future = None
    
    for idx, item in enumerate(my_data, 1):
        rp = item['relative_path']
        primary_object = item.get('primary_object', '')
        video_path = data_root / rp / "video.mp4"
        output_path = data_root / rp / "video_seg.mp4"
        
        if not video_path.exists():
            print(f"[{idx}/{len(my_data)}] SKIP (no video): {rp}")
            stats["no_video"] += 1
            continue
        
        # If partially processed, clean up corrupted files to start fresh
        episode_dir = data_root / rp
        for rel_f in EXPECTED_FILES:
            fpath = episode_dir / rel_f
            if fpath.exists() and fpath.stat().st_size < MIN_VIDEO_SIZE:
                fpath.unlink()  # Remove truncated/corrupted file
        
        # Restart predictor periodically to prevent memory leaks
        items_since_restart += 1
        if items_since_restart >= PREDICTOR_RESTART_INTERVAL:
            del video_predictor
            torch.cuda.empty_cache()
            if ckpt_path.exists():
                video_predictor = build_sam3_video_predictor(checkpoint_path=str(ckpt_path))
            else:
                video_predictor = build_sam3_video_predictor()
            items_since_restart = 0
            print(f"Rank {rank}: SAM3 predictor restarted")
        
        try:
            result = generate_mask_video_multi_prompt(
                video_path=video_path,
                output_path=output_path,
                primary_object=primary_object,
                video_predictor=video_predictor,
                gpu_id=rank,
            )
            
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            
            prompt_str = result.get("prompt", "")
            empty_pct = result.get("empty_pct", 100)
            proc_time = result.get("time", 0)
            n_frames = result.get("n_frames", 0)
            
            prompts_tried_str = ' | '.join(result.get('prompts_tried', []))
            
            # GPU memory info
            gpu_mem = ""
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                gpu_mem = f"GPU:{mem_used:.0f}/{mem_reserved:.0f}MB"
            
            marker = "OK" if status == "ok" else ("VNOF" if status == "no_frames" else "MISS")
            
            # ETA calculation (include no_frames in total for accurate ETA)
            elapsed_so_far = time.time() - start_time
            total_attempted = stats.get('ok', 0) + stats.get('no_detection', 0) + stats.get('no_frames', 0)
            total_processed = stats.get('ok', 0) + stats.get('no_detection', 0)
            if total_attempted > 0 and elapsed_so_far > 0:
                rate = total_attempted / elapsed_so_far * 60
                remaining = len(my_data) - idx
                eta_min = remaining / rate if rate > 0 else 0
                eta_str = f"ETA:{eta_min:.0f}min"
            else:
                eta_str = ""
            success_pct = 100 * stats.get('ok', 0) / max(total_processed, 1)
            
            print(f"[{idx:>4}/{len(my_data)}] {marker:>4} {primary_object:<30} "
                  f"prompt='{prompt_str}' frames={n_frames} empty={empty_pct:.0f}% "
                  f"time={proc_time:.1f}s {gpu_mem} "
                  f"[OK:{stats['ok']}/{total_processed}={success_pct:.0f}% {eta_str}]  {rp}")
            print(f"         prompts: {prompts_tried_str}")
            sys.stdout.flush()
            
            # Step 2+3: Render sensor overlays + seg tokens in background thread
            # These are CPU-bound and can overlap with next episode's SAM3 (GPU-bound)
            if status == "ok":
                def _postprocess(vp, mp, ep_dir, encode_fn, ckpt, rp_label):
                    """Render sensor overlays, encode seg tokens, generate clone augmentation."""
                    import traceback as _tb
                    import psutil
                    msgs = []
                    pp_start = time.time()
                    
                    # Memory baseline
                    proc = psutil.Process()
                    mem_start = proc.memory_info().rss / 1024**2
                    
                    try:
                        # Render sensor overlays (21 videos, parallel CPU encoding)
                        render_start = time.time()
                        try:
                            rr = traverse_single_video(str(vp), str(mp), ep_dir)
                            ok = sum(r['successful'] for r in rr.values())
                            fail = sum(r['failed'] for r in rr.values())
                            rt = time.time() - render_start
                            mem_after_render = proc.memory_info().rss / 1024**2
                            msgs.append(f"         render: {ok} sensor videos OK, {fail} failed ({rt:.1f}s) mem={mem_after_render:.0f}MB(+{mem_after_render-mem_start:.0f})")
                        except Exception as e:
                            rt = time.time() - render_start
                            msgs.append(f"         render: FAILED ({type(e).__name__}: {e}) ({rt:.1f}s)")
                        
                        # Seg tokens
                        token_start = time.time()
                        try:
                            token_out = Path(mp).parent / "video_seg_tokens.npz"
                            tr = encode_fn(Path(mp), ckpt)
                            np.savez_compressed(str(token_out),
                                loc_tokens=tr['loc_tokens'], seg_tokens=tr['seg_tokens'],
                                bboxes=tr['bboxes'], valid=tr['valid'])
                            nv = int(tr['valid'].sum())
                            nt = len(tr['valid'])
                            tt = time.time() - token_start
                            msgs.append(f"         tokens: {nv}/{nt} valid frames ({tt:.1f}s)")
                        except Exception as e:
                            tt = time.time() - token_start
                            msgs.append(f"         tokens: FAILED ({type(e).__name__}: {e}) ({tt:.1f}s)")
                        
                        # Clone augmentations (2 placements in parallel, each with internal thread pool)
                        clone_start = time.time()
                        try:
                            from concurrent.futures import ThreadPoolExecutor as _ClonePool
                            clone_futures = {}
                            with _ClonePool(max_workers=NUM_CLONE_AUGMENTATIONS) as cpool:
                                for aug_i in range(NUM_CLONE_AUGMENTATIONS):
                                    clone_futures[aug_i] = cpool.submit(
                                        generate_clone_augmentation, ep_dir, str(mp), aug_index=aug_i)
                                for aug_i in range(NUM_CLONE_AUGMENTATIONS):
                                    cr = clone_futures[aug_i].result()
                                    msgs.append(f"         clone[{aug_i}]: {cr['status']}, {cr['n_aug_videos']} videos, offset={cr.get('clone_offset','')}")
                            ct = time.time() - clone_start
                            mem_after_clone = proc.memory_info().rss / 1024**2
                            msgs.append(f"         clone total: {ct:.1f}s mem={mem_after_clone:.0f}MB(+{mem_after_clone-mem_start:.0f})")
                        except Exception as e:
                            ct = time.time() - clone_start
                            msgs.append(f"         clone: FAILED ({type(e).__name__}: {e}) ({ct:.1f}s)")
                    except Exception as outer_e:
                        msgs.append(f"         POSTPROCESS CRASH: {type(outer_e).__name__}: {outer_e}")
                        msgs.append(f"         {_tb.format_exc()}")
                    pp_total = time.time() - pp_start
                    mem_end = proc.memory_info().rss / 1024**2
                    msgs.append(f"         postprocess total: {pp_total:.1f}s (render={rt:.1f}s tokens={tt:.1f}s clone={ct:.1f}s) mem={mem_end:.0f}MB")
                    return msgs
                
                # Wait for any previous background job to finish
                if bg_future is not None:
                    wait_start = time.time()
                    if not bg_future.done():
                        print(f"         [waiting for previous postprocess to finish...]", flush=True)
                    try:
                        bg_msgs = bg_future.result(timeout=1800)
                        wait_time = time.time() - wait_start
                        if wait_time > 1.0:
                            print(f"         [waited {wait_time:.1f}s for previous postprocess]")
                        for m in bg_msgs:
                            print(m)
                        sys.stdout.flush()
                    except Exception as e:
                        import traceback
                        print(f"         bg job error: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        sys.stdout.flush()
                
                # Submit this episode's postprocessing to background
                bg_future = bg_executor.submit(
                    _postprocess, video_path, output_path,
                    str(data_root / rp), encode_mask_fn, vae_checkpoint, rp
                )
            
            processing_times.append(proc_time)
            
        except Exception as e:
            import traceback
            print(f"[{idx:>4}/{len(my_data)}] ERR  {rp}: {e}")
            traceback.print_exc()
            stats["error"] += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Drain last background job
    if bg_future is not None:
        try:
            bg_msgs = bg_future.result(timeout=1800)
            for m in bg_msgs:
                print(m)
        except Exception as e:
            import traceback
            print(f"  Last bg job error: {type(e).__name__}: {e}")
            traceback.print_exc()
    bg_executor.shutdown(wait=True)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\nRank {rank} complete in {elapsed/60:.1f}min")
    print(f"  Stats: {dict(stats)}")
    print(f"  OK: {stats.get('ok',0)}, No detection: {stats.get('no_detection',0)}, "
          f"No frames (blob timeout): {stats.get('no_frames',0)}, "
          f"Skipped: {stats.get('skipped',0)}, Error: {stats.get('error',0)}, No video: {stats.get('no_video',0)}")
    
    total_processed = stats.get('ok', 0) + stats.get('no_detection', 0)
    if total_processed > 0:
        success_rate = 100 * stats.get('ok', 0) / total_processed
        print(f"  Success rate: {stats.get('ok', 0)}/{total_processed} ({success_rate:.1f}%)")
        if processing_times:
            avg_t = sum(processing_times) / len(processing_times)
            print(f"  Avg time/episode: {avg_t:.1f}s")
            print(f"  Total wall time: {elapsed/60:.1f}min")
            print(f"  Throughput: {total_processed / elapsed * 60:.1f} episodes/min")
    
    if world_size > 1 and rank == 0:
        print("\nAll ranks complete (no barrier needed — independent processing).")


if __name__ == "__main__":
    main()
