#!/usr/bin/env python
"""
MolmoAct Re-Segmentation Script

Re-generates segmentation masks and sensor overlays for FILTERED MolmoAct episodes
using multi-prompt SAM3 strategy. Writes to MolmoAct_Data_ReSeg/.

Outputs per episode:
  1. observations.mask/episode_NNNNNN.mp4          — SAM3 mask
  2. observations.mask/episode_NNNNNN_tokens.npz   — PaliGemma seg tokens
  3. observations.sensor/{thermal,radar,audio}/episode_NNNNNN_{label}_{bg}.mp4  — sensor overlays
  4. observations.sensor/episode_NNNNNN/video_aug{0,1}.mp4  — clone augmented RGB
  5. observations.sensor/episode_NNNNNN/{mode}/video_{label}_{bg}_aug{0,1}.mp4  — clone sensor

Usage:
    # Local test on filtered samples:
    python data_synthesis_molmoact_reseg.py --local-test

    # Distributed on cluster (8 GPUs):
    torchrun --nproc_per_node=8 data_synthesis_molmoact_reseg.py
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
from concurrent.futures import ThreadPoolExecutor, Future

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress ALL tqdm progress bars
import tqdm, tqdm.auto, tqdm.std

class _SilentTqdm:
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

tqdm.tqdm = _SilentTqdm
tqdm.auto.tqdm = _SilentTqdm
tqdm.std.tqdm = _SilentTqdm
os.environ['TQDM_DISABLE'] = '1'

sys.path.insert(0, str(Path(__file__).parent))
from sensor_mask_utils import traverse_single_video, set_verbose
from clone_augmentation import generate_clone_augmentation


def _detect_video_codec():
    """Test if NVENC GPU encoder is available."""
    try:
        c = av.open('/tmp/_codec_test_m.mp4', 'w')
        s = c.add_stream('h264_nvenc', rate=1)
        s.width, s.height, s.pix_fmt = 64, 64, 'yuv420p'
        s.options = {'preset': 'p1', 'rc': 'vbr', 'cq': '28'}
        f = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format='rgb24')
        for p in s.encode(f): c.mux(p)
        for p in s.encode(): c.mux(p)
        c.close()
        os.remove('/tmp/_codec_test_m.mp4')
        return 'h264_nvenc', {'preset': 'p1', 'rc': 'vbr', 'cq': '28'}
    except Exception:
        try: os.remove('/tmp/_codec_test_m.mp4')
        except: pass
    return 'libx264', {'preset': 'ultrafast', 'crf': '23'}

VIDEO_CODEC, VIDEO_CODEC_OPTS = _detect_video_codec()
print(f"Video codec: {VIDEO_CODEC} (options: {VIDEO_CODEC_OPTS})")
sys.stdout.flush()


# ─── MolmoAct multi-prompt lookup ───────────────────────────────────────────
MOLMOACT_OBJECT_TO_PROMPTS: dict[str, list[str]] = {
    # ── Tabletop objects ──
    "brown box":                    ["the brown item", "brown box", "the box", "box"],
    "blue mug":                     ["the blue item", "blue mug", "the mug", "mug"],
    "gray towel":                   ["the gray item", "gray towel", "the towel", "towel"],
    "orange shorts":                ["the orange item", "orange shorts", "shorts", "clothing"],
    "black bottle":                 ["the bottle", "bottle", "black bottle"],
    "white and green water bottle": ["the green item", "water bottle", "the bottle", "bottle"],
    # ── Household — drinks/bottles ──
    "white bottle":                 ["the white bottle", "white bottle", "the bottle", "bottle"],
    "clear plastic bottle":         ["the bottle", "plastic bottle", "bottle"],
    # ── Household — clothes/fabric ──
    "black clothes":                ["the clothes", "clothing", "black fabric", "garment"],
    "gray cloth":                   ["the cloth", "gray cloth", "the gray item", "rag"],
    # ── Household — basket items ──
    "black gamepad":                ["the gamepad", "gamepad", "controller", "the black item"],
    "white stuffed toy":            ["the white toy", "stuffed animal", "plush toy", "the plush"],
    "red toy car":                  ["the red item", "red toy car", "toy car", "the car"],
    "gray toy cat":                 ["the gray item", "gray toy cat", "toy cat", "stuffed cat"],
    "white cup":                    ["the white cup", "white cup", "the cup", "cup"],
    "white mug":                    ["the white mug", "white mug", "the mug", "mug"],
    # ── Household — fruits ──
    "yellow mango":                 ["the orange item", "the mango", "mango", "fruit"],
    "green apple":                  ["the green apple", "the green item", "apple", "green fruit"],
    "red apple":                    ["the red item", "red apple", "the apple", "apple"],
    "green pear":                   ["the green item", "green pear", "the pear", "pear"],
    "yellow pear":                  ["the pear", "pear", "the yellow item", "yellow fruit"],
    "green lemon":                  ["the green item", "green lemon", "the lemon", "lemon"],
    "yellow lemon":                 ["the yellow item", "the lemon", "lemon", "yellow fruit"],
    "orange orange":                ["the orange item", "the orange", "orange", "orange fruit"],
    "purple plum":                  ["the purple item", "purple fruit", "the plum", "plum", "the fruit"],
}


def get_prompts_for_object(primary_object: str) -> list[str]:
    key = primary_object.lower().strip()
    prompts = MOLMOACT_OBJECT_TO_PROMPTS.get(key)
    if prompts:
        return prompts
    words = key.split()
    if len(words) > 1:
        noun = words[-1]
        return [f"the {noun}", noun, f"the {key}"]
    return [f"the {key}", key]


# ─── Configuration ──────────────────────────────────────────────────────────

MIN_MASK_AREA = 200
MAX_MASK_AREA = 200_000
PREDICTOR_RESTART_INTERVAL = 30
MIN_VIDEO_SIZE = 1024
NUM_CLONE_AUGMENTATIONS = 2
MAX_FRAMES_PER_SESSION = 1500

# Only generate color and grey backgrounds (skip dark to save ~33% rendering time)
BACKGROUND_MODES = ['color', 'grey']


def episode_name(idx: int) -> str:
    return f"episode_{idx:06d}"


def get_source_rgb_path(source_root: Path, split: str, chunk: str, ep_idx: int) -> Path:
    return source_root / f"molmoact_dataset_{split}" / "train" / "videos" / chunk / "observations.rgb" / f"{episode_name(ep_idx)}.mp4"


def get_output_paths(output_root: Path, split: str, chunk: str, ep_idx: int) -> dict:
    base = output_root / f"molmoact_dataset_{split}" / "train" / "videos" / chunk
    ep = episode_name(ep_idx)
    return {
        "rgb_link": base / "observations.rgb" / f"{ep}.mp4",
        "mask": base / "observations.mask" / f"{ep}.mp4",
        "tokens": base / "observations.mask" / f"{ep}_tokens.npz",
        "sensor_dir": base / "observations.sensor",
        "clone_dir": base / "observations.sensor" / ep,
    }


def _build_expected_files(ep_name: str):
    modes = {
        'thermal': ['cold', 'room_temperature', 'hot'],
        'radar': ['hidden', 'absense'],
        'audio': ['ringing', 'silence'],
    }
    bgs = BACKGROUND_MODES
    files = [
        f"observations.mask/{ep_name}.mp4",
        f"observations.mask/{ep_name}_tokens.npz",
    ]
    for mode, labels in modes.items():
        for label in labels:
            for bg in bgs:
                files.append(f"observations.sensor/{mode}/{ep_name}_{label}_{bg}.mp4")
    for aug_i in range(NUM_CLONE_AUGMENTATIONS):
        files.append(f"observations.sensor/{ep_name}/video_aug{aug_i}.mp4")
        for mode, labels in modes.items():
            for label in labels:
                for bg in bgs:
                    files.append(f"observations.sensor/{ep_name}/{mode}/video_{label}_{bg}_aug{aug_i}.mp4")
    return files


def _listdir_sizes(dirpath: Path) -> dict[str, int]:
    """Batch-read all file names+sizes in a directory (single readdir syscall)."""
    try:
        return {e.name: e.stat().st_size for e in dirpath.iterdir() if e.is_file()}
    except (FileNotFoundError, OSError):
        return {}


def is_episode_complete(chunk_base: Path, ep_idx: int) -> bool:
    """Fast completeness check using a single clone sentinel stat instead of 46 stats."""
    ep = episode_name(ep_idx)
    # Quick check: if clone aug0 exists, episode is almost certainly complete
    clone_path = chunk_base / "observations.sensor" / ep / "video_aug0.mp4"
    try:
        return clone_path.exists() and clone_path.stat().st_size >= MIN_VIDEO_SIZE
    except OSError:
        return False


def filter_incomplete_episodes(episodes: list, output_root: Path) -> tuple[list, int]:
    """Fast batch filter: group by chunk, read shared dirs once, spot-check clone sentinel.
    Returns (incomplete_episodes, n_complete)."""
    from collections import defaultdict
    
    chunk_groups = defaultdict(list)
    for ep in episodes:
        chunk_groups[(ep["split"], ep["chunk"])].append(ep)
    
    incomplete = []
    n_complete = 0
    
    for (split, chunk), chunk_eps in chunk_groups.items():
        chunk_base = output_root / f"molmoact_dataset_{split}" / "train" / "videos" / chunk
        
        # Read shared mask dir once to quickly check which episodes have masks
        mask_dir = _listdir_sizes(chunk_base / "observations.mask")
        
        for ep_data in chunk_eps:
            ep_idx = ep_data["episode_index"]
            ep = episode_name(ep_idx)
            
            # Fast check: mask + tokens + clone sentinel
            has_mask = mask_dir.get(f"{ep}.mp4", 0) >= MIN_VIDEO_SIZE
            has_tokens = mask_dir.get(f"{ep}_tokens.npz", 0) >= MIN_VIDEO_SIZE
            
            if has_mask and has_tokens:
                # Spot-check clone aug0 (1 stat call)
                clone_path = chunk_base / "observations.sensor" / ep / "video_aug0.mp4"
                try:
                    clone_ok = clone_path.exists() and clone_path.stat().st_size >= MIN_VIDEO_SIZE
                except OSError:
                    clone_ok = False
                
                if clone_ok:
                    n_complete += 1
                    continue
            
            incomplete.append(ep_data)
        
        print(f"  Checked {split}/{chunk}: {len(chunk_eps)} episodes, "
              f"{len(chunk_eps) - len([e for e in incomplete if e['split']==split and e['chunk']==chunk])} complete",
              flush=True)
    
    return incomplete, n_complete


# ─── SAM3 mask generation ──────────────────────────────────────────────────

def select_best_mask_frame0(masks_list, height, width):
    best_idx, best_area = -1, 0
    for i in range(len(masks_list)):
        m = masks_list[i]
        while m.ndim > 2:
            m = m[0]
        area = int(m.astype(bool).sum())
        if MIN_MASK_AREA <= area <= MAX_MASK_AREA:
            if area > best_area:
                best_idx = i
                best_area = area
    return best_idx, 1.0 if best_idx >= 0 else 0.0, best_area


def _run_sam3_single_session(video_path, prompt, video_predictor, height, width, frame_offset=0, expected_frames=None):
    response = video_predictor.handle_request(dict(
        type="start_session", resource_path=str(video_path)))
    session_id = response["session_id"]

    video_predictor.handle_request(dict(
        type="add_prompt", session_id=session_id, frame_index=0, text=prompt))

    all_masks = {}
    frame0_good = False
    n_empty = 0
    n_large = 0

    propagate_start = time.time()
    last_progress = propagate_start
    try:
        for response in video_predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=session_id, propagation_direction="both")):
            frame_idx = response["frame_index"]
            masks = response["outputs"]["out_binary_masks"]

            if masks.ndim == 4:
                masks = masks.squeeze(0)
            best_i, _, best_area = select_best_mask_frame0(masks, height, width)
            if best_i >= 0:
                m = masks[best_i]
                while m.ndim > 2:
                    m = m[0]
                mask_bool = m.astype(bool)
                area = int(mask_bool.sum())
                if area > MAX_MASK_AREA:
                    mask_bool = np.zeros((height, width), dtype=bool)
                    n_large += 1
            else:
                mask_bool = np.zeros((height, width), dtype=bool)
                n_empty += 1

            all_masks[frame_idx + frame_offset] = mask_bool

            if frame_idx == 0:
                frame0_good = all_masks[frame_offset].any()
                if not frame0_good:
                    print(f" frame0=MISS", end='', flush=True)
                    break
                else:
                    print(f" frame0=HIT", end='', flush=True)

            now = time.time()
            if now - last_progress >= 10.0:
                elapsed = now - propagate_start
                fps_rate = len(all_masks) / elapsed if elapsed > 0 else 0
                ef = expected_frames or "?"
                print(f" [{len(all_masks)}/{ef}f {fps_rate:.1f}fps]", end='', flush=True)
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


def _run_sam3_prompt_on_video(video_path, prompt, video_predictor, height, width, gpu_id=-1):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frame_count <= MAX_FRAMES_PER_SESSION:
        return _run_sam3_single_session(video_path, prompt, video_predictor, height, width, 0, frame_count)

    all_masks = {}
    frame0_good = False
    total_empty, total_large = 0, 0

    for chunk_start in range(0, frame_count, MAX_FRAMES_PER_SESSION):
        chunk_end = min(chunk_start + MAX_FRAMES_PER_SESSION, frame_count)
        enc_w = width + (width % 2)
        enc_h = height + (height % 2)

        tmp_path = str(video_path) + f".chunk_{chunk_start}_{gpu_id}.mp4"
        container = av.open(tmp_path, 'w')
        stream = container.add_stream('libx264', rate=max(1, int(fps)))
        stream.width, stream.height, stream.pix_fmt = enc_w, enc_h, 'yuv420p'
        stream.options = {'preset': 'ultrafast', 'crf': '23'}

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
        for _ in range(chunk_end - chunk_start):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[0] != enc_h or frame.shape[1] != enc_w:
                frame = cv2.resize(frame, (enc_w, enc_h))
            vf = av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format='rgb24')
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        cap.release()

        try:
            chunk_masks, chunk_f0, chunk_empty, chunk_large = _run_sam3_single_session(
                tmp_path, prompt, video_predictor, height, width, chunk_start, chunk_end - chunk_start)
            if chunk_start == 0:
                frame0_good = chunk_f0
            all_masks.update(chunk_masks)
            total_empty += chunk_empty
            total_large += chunk_large
        finally:
            try: os.remove(tmp_path)
            except: pass

        if chunk_start == 0 and not frame0_good:
            break

    return all_masks, frame0_good, total_empty, total_large


def generate_mask_video(
    video_path: Path,
    output_path: Path,
    primary_object: str,
    video_predictor,
    gpu_id: int = -1,
) -> dict:
    prompts = get_prompts_for_object(primary_object)

    frame_count = 0
    for _attempt in range(3):
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '4096'
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count > 0:
            break
        time.sleep(3)

    if frame_count == 0:
        return {"status": "no_frames", "prompt": "", "n_frames": 0,
                "time": 0, "empty_pct": 100, "prompts_tried": ["(video unreadable)"]}

    enc_w = width + (width % 2)
    enc_h = height + (height % 2)

    start_time = time.time()
    all_masks = {}
    winning_prompt = ""
    prompts_tried = []
    empty_pct = 100.0

    for prompt_idx, prompt in enumerate(prompts):
        print(f"    [SAM3] prompt {prompt_idx+1}/{len(prompts)}: '{prompt}' on {frame_count} frames ...", flush=True)
        prompt_start = time.time()
        try:
            masks, frame0_good, n_empty, n_large = _run_sam3_prompt_on_video(
                video_path, prompt, video_predictor, height, width, gpu_id)
            prompt_time = time.time() - prompt_start

            if frame0_good:
                all_masks = masks
                winning_prompt = prompt
                empty_pct = 100 * n_empty / len(all_masks) if all_masks else 100
                prompts_tried.append(f"'{prompt}' -> OK ({len(all_masks)} frames, {n_empty} empty, {prompt_time:.1f}s)")
                break
            else:
                prompts_tried.append(f"'{prompt}' -> FAIL frame0 ({prompt_time:.1f}s)")
                all_masks.clear()
        except Exception as e:
            prompt_time = time.time() - prompt_start
            prompts_tried.append(f"'{prompt}' -> ERROR: {str(e)[:100]} ({prompt_time:.1f}s)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_masks.clear()

    processing_time = time.time() - start_time

    if not winning_prompt:
        all_masks = {i: np.zeros((height, width), dtype=bool) for i in range(frame_count)}
        empty_pct = 100.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), 'w')
    stream = container.add_stream('libx264', rate=max(1, int(fps)))
    stream.width, stream.height, stream.pix_fmt = enc_w, enc_h, 'yuv420p'

    for frame_idx in sorted(all_masks.keys()):
        mask = all_masks[frame_idx].astype(np.uint8) * 255
        if mask.shape[0] != enc_h or mask.shape[1] != enc_w:
            mask = cv2.resize(mask, (enc_w, enc_h), interpolation=cv2.INTER_NEAREST)
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        frame = av.VideoFrame.from_ndarray(mask_rgb, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    return {
        "status": "ok" if winning_prompt else "no_detection",
        "prompt": winning_prompt,
        "n_frames": len(all_masks),
        "empty_pct": empty_pct if winning_prompt else 100.0,
        "time": processing_time,
        "prompts_tried": prompts_tried,
    }


# ─── Distributed helpers ───────────────────────────────────────────────────

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} on GPU {local_rank}")
    return local_rank, world_size, rank


# ─── VQ-VAE seg token encoder (lazy TF import) ─────────────────────────────

def _load_seg_token_encoder():
    saved_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    from sensor_mask_token_utils import encode_mask_video, load_vae_checkpoint
    if saved_cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = saved_cuda
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    return encode_mask_video, load_vae_checkpoint()


# ─── Clone augmentation adapter for MolmoAct ───────────────────────────────

def generate_clone_aug_molmoact(rgb_path: Path, mask_path: Path, clone_dir: Path, aug_index: int = 0, background_modes: list[str] | None = None) -> dict:
    clone_dir.mkdir(parents=True, exist_ok=True)

    # Atomic symlink to avoid race between parallel clone workers
    link_rgb = clone_dir / "video.mp4"
    tmp_link = clone_dir / f".video_tmp_{aug_index}.mp4"
    try:
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(rgb_path.resolve())
        os.replace(str(tmp_link), str(link_rgb))
    except (FileExistsError, OSError):
        tmp_link.unlink(missing_ok=True)

    try:
        result = generate_clone_augmentation(str(clone_dir), str(mask_path), aug_index=aug_index,
                                              background_modes=background_modes)
    finally:
        if link_rgb.is_symlink():
            link_rgb.unlink()

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MolmoAct Re-Segmentation')
    parser.add_argument('--local-test', action='store_true')
    parser.add_argument('--source-root', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--filtered-episodes', type=str, default=None)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing')
    args = parser.parse_args()

    local_rank, world_size, rank = init_distributed()

    def print0(msg):
        if rank == 0:
            print(msg)

    script_dir = Path(__file__).resolve().parent
    if args.local_test:
        source_root = Path(args.source_root) if args.source_root else script_dir.parent.parent / "mount" / "data" / "MolmoAct_Data"
        output_root = Path(args.output_root) if args.output_root else script_dir.parent.parent / "mount" / "data" / "MolmoAct_Data_ReSeg"
        filtered_path = args.filtered_episodes or str(script_dir / "molmoact_filtered_episodes.json")
    else:
        source_root = Path(args.source_root or "<DATA_ROOT>")
        output_root = Path(args.output_root or "<DATA_ROOT>")
        filtered_path = args.filtered_episodes or str(script_dir / "molmoact_filtered_episodes.json")

    print0(f"Source: {source_root}")
    print0(f"Output: {output_root}")
    print0(f"Filtered episodes: {filtered_path}")

    with open(filtered_path) as f:
        filtered = json.load(f)

    episodes = []
    for split in ["tabletop", "household"]:
        split_data = filtered.get(split, {})
        for ep in split_data.get("episodes", []):
            episodes.append({
                "split": split,
                "episode_index": ep["episode_index"],
                "chunk": ep["chunk"],
                "primary_object": ep["primary_object"],
                "task": ep["task"],
                "length": ep.get("length", 200),
            })

    print0(f"Total filtered episodes: {len(episodes)}")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    if args.skip_existing:
        print0(f"Scanning for incomplete episodes...")
        sys.stdout.flush()
        episodes, n_complete = filter_incomplete_episodes(episodes, output_root)
        print0(f"Skip existing: {n_complete}/{n_complete + len(episodes)} already complete, {len(episodes)} remaining")
        sys.stdout.flush()

    episodes.sort(key=lambda x: -x.get("length", 200))

    my_episodes = [ep for i, ep in enumerate(episodes) if i % world_size == rank]
    print(f"Rank {rank}: assigned {len(my_episodes)} episodes", flush=True)

    # Initialize SAM3
    print(f"Rank {rank}: Loading SAM3 video predictor...", flush=True)
    sys.path.insert(0, str(script_dir / "sam3"))
    from sam3.model_builder import build_sam3_video_predictor

    ckpt_path = script_dir.parent.parent / "assets" / "sam3.pt"
    if ckpt_path.exists():
        video_predictor = build_sam3_video_predictor(checkpoint_path=str(ckpt_path))
    else:
        marker = Path("/tmp/.sam3_download_done")
        if local_rank == 0:
            video_predictor = build_sam3_video_predictor()
            marker.touch()
        else:
            while not marker.exists():
                time.sleep(5)
            video_predictor = build_sam3_video_predictor()
    print(f"Rank {rank}: SAM3 loaded", flush=True)

    # Load VQ-VAE
    print(f"Rank {rank}: Loading VQ-VAE...", flush=True)
    encode_mask_fn, vae_checkpoint = _load_seg_token_encoder()
    print(f"Rank {rank}: VQ-VAE loaded", flush=True)

    set_verbose(False)

    # Copy metadata to output dir (rank 0 only)
    if rank == 0:
        for split in ["tabletop", "household"]:
            src_meta = source_root / f"molmoact_dataset_{split}" / "train" / "meta"
            dst_meta = output_root / f"molmoact_dataset_{split}" / "train" / "meta"
            if src_meta.exists():
                dst_meta.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in src_meta.iterdir():
                    dst_f = dst_meta / f.name
                    if not dst_f.exists():
                        shutil.copy2(str(f), str(dst_f))

    print0(f"\n{'='*80}")
    print0(f"Starting MolmoAct re-segmentation ({len(my_episodes)} episodes on rank {rank})")
    print0(f"{'='*80}")

    start_time = time.time()
    stats = {"ok": 0, "no_detection": 0, "no_frames": 0, "error": 0}
    items_since_restart = 0

    # For local test / small runs, do postprocessing inline for immediate feedback
    inline_postprocess = args.local_test or len(my_episodes) <= 10
    bg_executor = ThreadPoolExecutor(max_workers=1) if not inline_postprocess else None
    bg_future = None
    # Track ALL background futures for sync barrier at the end
    all_bg_futures: list[Future] = []

    for idx, ep in enumerate(my_episodes, 1):
        split = ep["split"]
        ep_idx = ep["episode_index"]
        chunk = ep["chunk"]
        primary_object = ep["primary_object"]
        ep_name = episode_name(ep_idx)

        rgb_path = get_source_rgb_path(source_root, split, chunk, ep_idx)
        paths = get_output_paths(output_root, split, chunk, ep_idx)

        if not rgb_path.exists():
            print(f"[{idx}/{len(my_episodes)}] SKIP (no RGB): {rgb_path}")
            stats["error"] += 1
            continue

        # Symlink RGB to output dir
        paths["rgb_link"].parent.mkdir(parents=True, exist_ok=True)
        if not paths["rgb_link"].exists():
            try:
                paths["rgb_link"].symlink_to(rgb_path.resolve())
            except FileExistsError:
                pass

        # Restart predictor periodically
        items_since_restart += 1
        if items_since_restart >= PREDICTOR_RESTART_INTERVAL:
            del video_predictor
            torch.cuda.empty_cache()
            if ckpt_path.exists():
                video_predictor = build_sam3_video_predictor(checkpoint_path=str(ckpt_path))
            else:
                video_predictor = build_sam3_video_predictor()
            items_since_restart = 0

        try:
            # Step 1: Generate mask
            result = generate_mask_video(
                video_path=rgb_path,
                output_path=paths["mask"],
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

            total_done = sum(v for k, v in stats.items() if k in ('ok', 'no_detection'))
            elapsed = time.time() - start_time
            rate = total_done / elapsed * 60 if elapsed > 0 and total_done > 0 else 0
            remaining = len(my_episodes) - idx
            eta = remaining / rate if rate > 0 else 0
            success_pct = 100 * stats['ok'] / max(total_done, 1)

            marker = "OK" if status == "ok" else "MISS"
            print(f"[{idx:>4}/{len(my_episodes)}] {marker:>4} {primary_object:<30} "
                  f"prompt='{prompt_str}' frames={n_frames} empty={empty_pct:.0f}% "
                  f"time={proc_time:.1f}s "
                  f"[OK:{stats['ok']}/{total_done}={success_pct:.0f}% ETA:{eta:.0f}min] "
                  f"{split}/{chunk}/{ep_name}")
            print(f"         prompts: {prompts_tried_str}")
            sys.stdout.flush()

            # Step 2+3: Postprocessing (sensor overlays + tokens + clone aug)
            if status == "ok":
                def _postprocess(rgb_p, mask_p, paths_dict, encode_fn, ckpt, ep_info):
                    import traceback as _tb
                    msgs = []
                    pp_start = time.time()

                    # Render sensor overlays
                    try:
                        sensor_dir = paths_dict["sensor_dir"]
                        rr = traverse_single_video(str(rgb_p), str(mask_p), str(sensor_dir),
                                                   background_modes=BACKGROUND_MODES)
                        ok = sum(r['successful'] for r in rr.values())
                        fail = sum(r['failed'] for r in rr.values())
                        msgs.append(f"         render: {ok} OK, {fail} failed ({time.time()-pp_start:.1f}s)")
                    except Exception as e:
                        msgs.append(f"         render: FAILED ({e})")

                    # Seg tokens
                    token_start = time.time()
                    try:
                        tr = encode_fn(Path(mask_p), ckpt)
                        token_out = paths_dict["tokens"]
                        token_out.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(str(token_out),
                            loc_tokens=tr['loc_tokens'], seg_tokens=tr['seg_tokens'],
                            bboxes=tr['bboxes'], valid=tr['valid'])
                        nv = int(tr['valid'].sum())
                        msgs.append(f"         tokens: {nv}/{len(tr['valid'])} valid ({time.time()-token_start:.1f}s)")
                    except Exception as e:
                        msgs.append(f"         tokens: FAILED ({e})")

                    # Clone augmentations
                    clone_start = time.time()
                    try:
                        clone_dir = paths_dict["clone_dir"]
                        from concurrent.futures import ThreadPoolExecutor as _CP
                        with _CP(max_workers=NUM_CLONE_AUGMENTATIONS) as cpool:
                            futs = {i: cpool.submit(generate_clone_aug_molmoact,
                                                     Path(rgb_p), Path(mask_p), clone_dir, aug_index=i,
                                                     background_modes=BACKGROUND_MODES)
                                    for i in range(NUM_CLONE_AUGMENTATIONS)}
                            for i in range(NUM_CLONE_AUGMENTATIONS):
                                cr = futs[i].result()
                                msgs.append(f"         clone[{i}]: {cr['status']}, {cr.get('n_aug_videos',0)} videos")
                        msgs.append(f"         clone total: {time.time()-clone_start:.1f}s")
                    except Exception as e:
                        msgs.append(f"         clone: FAILED ({e})")

                    msgs.append(f"         postprocess total: {time.time()-pp_start:.1f}s")
                    return msgs

                # Print previous bg job result if ready
                if bg_future is not None and bg_future.done():
                    try:
                        for m in bg_future.result(timeout=0):
                            print(m)
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"         bg error: {e}")

                if inline_postprocess:
                    print(f"         [postprocessing inline...]", flush=True)
                    pp_msgs = _postprocess(str(rgb_path), str(paths["mask"]),
                                           paths, encode_mask_fn, vae_checkpoint, ep)
                    for m in pp_msgs:
                        print(m, flush=True)
                else:
                    # Wait for previous bg job before submitting new one (pool has 1 worker)
                    if bg_future is not None and not bg_future.done():
                        print(f"         [waiting for previous postprocess...]", flush=True)
                        try:
                            for m in bg_future.result(timeout=1800):
                                print(m)
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"         bg error: {e}")

                    bg_future = bg_executor.submit(
                        _postprocess, str(rgb_path), str(paths["mask"]),
                        paths, encode_mask_fn, vae_checkpoint, ep)
                    all_bg_futures.append(bg_future)

        except Exception as e:
            import traceback
            print(f"[{idx:>4}/{len(my_episodes)}] ERR  {split}/{chunk}/{ep_name}: {e}")
            traceback.print_exc()
            stats["error"] = stats.get("error", 0) + 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ─── Sync barrier: wait for ALL background postprocessing to complete ───
    if all_bg_futures:
        n_pending = sum(1 for f in all_bg_futures if not f.done())
        if n_pending > 0:
            print(f"\n[Rank {rank}] Waiting for {n_pending}/{len(all_bg_futures)} background postprocess jobs to finish...", flush=True)
        barrier_start = time.time()
        for i, fut in enumerate(all_bg_futures):
            try:
                msgs = fut.result(timeout=1800)
                for m in msgs:
                    print(m)
            except Exception as e:
                print(f"  bg job {i} error: {e}")
        barrier_time = time.time() - barrier_start
        print(f"[Rank {rank}] All {len(all_bg_futures)} background jobs complete ({barrier_time:.1f}s)", flush=True)
    if bg_executor is not None:
        bg_executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    total_done = stats.get('ok', 0) + stats.get('no_detection', 0)
    print(f"\nRank {rank} complete in {elapsed/60:.1f}min")
    print(f"  Stats: {dict(stats)}")
    if total_done > 0:
        print(f"  Success: {stats['ok']}/{total_done} ({100*stats['ok']/total_done:.1f}%)")
        print(f"  Throughput: {total_done/elapsed*60:.1f} eps/min")


if __name__ == "__main__":
    main()
