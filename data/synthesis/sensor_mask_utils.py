"""
Sensor Mask Utilities for Thermal, Radar, and Audio Overlay Generation

This module contains all the logic for generating different sensor visualizations
(thermal, radar, audio) by applying colored overlays to masked regions of videos.

Key Features:
- Thermal overlay: Solid color with Gaussian blur (INFERNO colormap)
- Radar overlay: Tiled square pattern (VIRIDIS colormap)
- Audio overlay: Smooth pink pattern (COOL colormap)
- Alpha blending for semi-transparent effects
"""

import cv2
import numpy as np
import av
from pathlib import Path
from typing import Literal, Optional
import time
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Force unbuffered output for immediate flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

FIX_MODE = False

# Configuration
VERBOSE = False
STD = 7  # Gaussian noise std for sensor intensity (±14 at 2σ on [0,255] scale)
SKIP_EXISTING = True  # Skip already processed videos
TEST_MODE = False  # Process only first 2 episodes per dataset

# Dataset configurations
# DATA_ROOT = Path("<DATA_ROOT>")
# DATA_ROOT = Path("<LOCAL_PATH>")
DATA_ROOT = Path("<LOCAL_PATH>")
DATASET_CONFIGS = {
    "VITRA_Robot_Data": {
        "root": DATA_ROOT / "VITRA_Robot_Data",
        "json_file": "all_instructions_0910.json",
        "video_name": "video.mp4",
        "mask_name": "video_seg.mp4",
    },

    "InternA1_Robot_Data": {
        "root": DATA_ROOT / "InternA1_Robot_Data/real/genie1",
    },
    "MolmoAct_Data": {
        "root": DATA_ROOT / "MolmoAct_Data",
        "subdirs": ["molmoact_dataset_household", "molmoact_dataset_tabletop"],
        "video_path": "train/videos/{chunk}/observations.rgb/{episode}.mp4",
        "mask_path": "train/videos/{chunk}/observations.mask/{episode}.mp4",
    },
    "Agibot_Data": {
        "root": DATA_ROOT / "Agibot_Data_Copy/observations_extract",
        "video_name": "videos/head_color.mp4",
        "mask_name": "head_mask.mp4",
    },

}

def vprint(*args, **kwargs):
    """Verbose print helper; prints only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)

def set_verbose(verbose: bool):
    """Set the verbosity level."""
    global VERBOSE
    VERBOSE = verbose


def _get_encode_workers():
    """Number of parallel video encoding threads per GPU process."""
    total_cpus = os.cpu_count() or 8
    local_gpus = int(os.environ.get('LOCAL_WORLD_SIZE',
                     os.environ.get('WORLD_SIZE', '1')))
    cpus_per_process = max(1, total_cpus // max(1, local_gpus))
    return max(1, min(cpus_per_process - 2, 10))


def init_distributed():
    """Initialize distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        # Set device before dist.init_process_group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        
        print(f"Initialized process {rank}/{world_size} on GPU {local_rank}")
    
    return local_rank, world_size, rank


def split_videos_by_rank(all_pairs: list, world_size: int, rank: int) -> list:
    """Split video pairs across GPUs using round-robin distribution."""
    my_pairs = []
    for i, pair in enumerate(all_pairs):
        if i % world_size == rank:
            my_pairs.append(pair)
    return my_pairs


def create_video_from_frames(frames: list, output_path: Path, fps: float) -> bool:
    """
    Create a video from a list of frames using PyAV with CPU libx264 encoder.
    
    Args:
        frames: List of video frames (BGR format)
        output_path: Path to save the output video
        fps: Frames per second
        
    Returns:
        True if successful, False otherwise
    """
    if not frames:
        vprint(f"No frames to create video")
        return False

    height, width = frames[0].shape[:2]
    
    # Ensure width and height are divisible by 2 for yuv420p
    output_width = width if width % 2 == 0 else width + 1
    output_height = height if height % 2 == 0 else height + 1
    
    try:
        output_container = av.open(str(output_path), 'w')
        output_stream = output_container.add_stream('libx264', rate=int(fps))
        output_stream.options = {'preset': 'ultrafast', 'crf': '23'}
        output_stream.width = output_width
        output_stream.height = output_height
        output_stream.pix_fmt = 'yuv420p'
        
        # Encode all frames
        for frame_bgr in frames:
            # Pad if needed
            if output_height != height or output_width != width:
                pad_h = output_height - height
                pad_w = output_width - width
                frame_bgr = np.pad(frame_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            
            # Convert BGR to RGB for av.VideoFrame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Create VideoFrame and encode
            video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            for packet in output_stream.encode(video_frame):
                output_container.mux(packet)
        
        # Flush remaining packets
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        output_container.close()
        vprint(f"Successfully created video: {output_path}")
        return True
        
    except Exception as e:
        try:
            output_container.close()
        except:
            pass
        print(f"Error creating video {output_path}: {e}")
        return False


def traverse_single_video(
    video_path: str,
    mask_video_path: str,
    output_path: str,
    background_modes: list[str] | None = None,
) -> dict:
    """
    Generate all sensor overlay videos (thermal, radar, audio) for a single video.
    
    Processes all modes in a single pass to maximize efficiency by decoding
    frames and computing backgrounds only once.
    
    Args:
        video_path: Path to the original video
        mask_video_path: Path to the mask video
        output_path: Output directory path
        
    Returns:
        Dictionary with generation results for all modes:
        {
            'thermal': {'successful': int, 'failed': int, 'generated_files': list[str]},
            'radar': {'successful': int, 'failed': int, 'generated_files': list[str]},
            'audio': {'successful': int, 'failed': int, 'generated_files': list[str]}
        }
        
    Directory structure:
        output_path/
            thermal/
                <video_name>_<semantic_label>_<background_mode>.mp4
            radar/
                <video_name>_<semantic_label>_<background_mode>.mp4
            audio/
                <video_name>_<semantic_label>_<background_mode>.mp4
    """
    try:
        # Define semantic labels for each mode
        semantic_labels_map = {
            'thermal': ['cold', 'room_temperature', 'hot'],
            'radar': ['hidden', 'absense'],
            'audio': ['ringing', 'silence']
        }
        
        # Define colormaps for each mode
        colormap_map = {
            'thermal': cv2.COLORMAP_INFERNO,
            'radar': cv2.COLORMAP_VIRIDIS,
            'audio': cv2.COLORMAP_COOL
        }
        
        # Define background modes
        if background_modes is None:
            background_modes = ['dark', 'color', 'grey']
        
        # Get original video name (without extension)
        video_name = Path(video_path).stem
        
        # Track results for all modes
        results = {}
        for mode in ['thermal', 'radar', 'audio']:
            results[mode] = {'successful': 0, 'failed': 0, 'generated_files': []}
        
        # Calculate total combinations across all modes
        total_combinations = sum(len(labels) for labels in semantic_labels_map.values()) * len(background_modes)
        vprint(f"Generating {total_combinations} variants total (21 files)...")
        
        # Open videos once and decode all frames
        vprint(f"Opening and decoding videos...")
        original_container = av.open(str(video_path))
        mask_container = av.open(str(mask_video_path))
        
        original_stream = original_container.streams.video[0]
        fps = float(original_stream.average_rate)
        
        # Decode all frames at once
        original_frames_bgr = []
        mask_frames_gray = []
        
        try:
            original_frames = original_container.decode(video=0)
            mask_frames = mask_container.decode(video=0)
            
            for frame, mask_frame in zip(original_frames, mask_frames):
                # Convert original frame to BGR
                frame_rgb = frame.to_ndarray(format='rgb24')
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                original_frames_bgr.append(frame_bgr)
                
                # Convert mask to grayscale
                mask_rgb = mask_frame.to_ndarray(format='rgb24')
                if len(mask_rgb.shape) == 3:
                    mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
                else:
                    mask = mask_rgb
                mask_frames_gray.append(mask)
        
        except Exception as e:
            vprint(f"Error decoding frames: {e}")
            original_container.close()
            mask_container.close()
            return {'successful': 0, 'failed': 0, 'generated_files': []}
        
        original_container.close()
        mask_container.close()
        
        if not original_frames_bgr or not mask_frames_gray:
            vprint(f"No frames decoded")
            return {'successful': 0, 'failed': 0, 'generated_files': []}
        
        vprint(f"Decoded {len(original_frames_bgr)} frames")
        
        # Pre-compute background bases once (independent of semantic label and mode)
        dark_frames = [np.zeros_like(frame, dtype=np.uint8) for frame in original_frames_bgr] if 'dark' in background_modes else []
        grey_frames = [cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) 
                       for frame in original_frames_bgr] if 'grey' in background_modes else []
        
        # Pre-compute 3D masks once (reused for all combinations)
        mask_3d_frames = []
        for mask in mask_frames_gray:
            mask_bool = mask > 0
            if np.any(mask_bool):
                mask_3d = np.stack([mask_bool, mask_bool, mask_bool], axis=2)
            else:
                mask_3d = None
            mask_3d_frames.append(mask_3d)
        
        # Thread pool for parallel video encoding (libx264 releases GIL)
        encode_pool = ThreadPoolExecutor(max_workers=_get_encode_workers())
        encode_tasks = []  # (mode, filename, filepath, future)
        
        # Process all modes together
        for mode in ['thermal', 'radar', 'audio']:
            # Create output directory for this mode
            output_dir = Path(output_path) / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get semantic labels and colormap for this mode
            semantic_labels = semantic_labels_map[mode]
            colormap = colormap_map[mode]
            
            vprint(f"\nProcessing {mode} mode ({len(semantic_labels) * len(background_modes)} variants)...")
            
            # Process each semantic label for this mode
            for semantic_label in semantic_labels:
                # Determine intensity value based on semantic label
                # Use values away from extremes so random variation
                # covers a natural-looking range. With STD=15, 2σ gives ±30.
                if mode == 'thermal':
                    if semantic_label == 'cold':
                        base_intensity = 50  # INFERNO: dark purple (not black, visible on grey bg)
                    elif semantic_label == 'room_temperature':
                        base_intensity = 128
                    elif semantic_label == 'hot':
                        base_intensity = 210  # INFERNO: warm orange (~RGB 252,176,20), matching real thermal
                elif mode == 'radar':
                    if semantic_label == 'hidden':
                        base_intensity = 240
                    elif semantic_label == 'absense':
                        base_intensity = 20
                elif mode == 'audio':
                    if semantic_label == 'ringing':
                        base_intensity = 240
                    elif semantic_label == 'silence':
                        base_intensity = 20
                
                # Wider std for thermal hot to cover orange-to-yellow range
                noise_std = 10 if (mode == 'thermal' and semantic_label == 'hot') else STD
                
                # Sample intensity value with Gaussian variation (once per semantic label)
                sampled = np.random.normal(loc=base_intensity, scale=noise_std)
                low_2s = base_intensity - 2 * noise_std
                high_2s = base_intensity + 2 * noise_std
                low = max(0, low_2s)
                high = min(255, high_2s)
                intensity_value = int(np.clip(sampled, low, high))
                
                # Pre-compute colored overlay ONCE (same color for all frames)
                # applyColorMap on a single pixel, then broadcast
                single_color = cv2.applyColorMap(
                    np.array([[intensity_value]], dtype=np.uint8), colormap
                )[0, 0]  # [3] BGR
                
                # Generate backgrounds in single pass over frames
                bg_frames = {bg: [] for bg in background_modes}
                bg_source_map = {'dark': dark_frames, 'grey': grey_frames, 'color': original_frames_bgr}
                
                for idx, mask_3d in enumerate(mask_3d_frames):
                    mask_2d = mask_3d[:, :, 0] if mask_3d is not None else None
                    
                    for bg_mode in background_modes:
                        result_frame = bg_source_map[bg_mode][idx].copy()
                        if mask_2d is not None:
                            result_frame[mask_2d] = single_color
                        bg_frames[bg_mode].append(result_frame)
                
                # Submit video writes to thread pool
                for background_mode in background_modes:
                    output_filename = f"{video_name}_{semantic_label}_{background_mode}.mp4"
                    output_file_path = output_dir / output_filename
                    encode_tasks.append((mode, output_filename, str(output_file_path),
                        encode_pool.submit(create_video_from_frames,
                                           bg_frames[background_mode], output_file_path, fps)))
        
        # Collect results from all parallel encodes
        for mode_name, filename, filepath, future in encode_tasks:
            try:
                success = future.result()
                if success:
                    results[mode_name]['successful'] += 1
                    results[mode_name]['generated_files'].append(filepath)
                    vprint(f"    ✓ Generated: {filename}")
                else:
                    results[mode_name]['failed'] += 1
                    vprint(f"    ✗ Failed: {filename}")
            except Exception as e:
                vprint(f"    ✗ Failed: {filename} - {e}")
                results[mode_name]['failed'] += 1
        encode_pool.shutdown(wait=False)
        
        # Print summary for all modes
        vprint(f"\nTraversal completed:")
        for mode in ['thermal', 'radar', 'audio']:
            mode_total = len(semantic_labels_map[mode]) * len(background_modes)
            vprint(f"  {mode.upper()}: {results[mode]['successful']}/{mode_total} successful")
        
        return results
        
    except Exception as e:
        vprint(f"Error in traverse_single_video: {e}")
        return {
            'thermal': {'successful': 0, 'failed': 0, 'generated_files': []},
            'radar': {'successful': 0, 'failed': 0, 'generated_files': []},
            'audio': {'successful': 0, 'failed': 0, 'generated_files': []}
        }


def traverse_single_video_test():
    """
    Test function to generate sensor overlays for a specific video.
    
    Tests all three sensor modes (thermal, radar, audio) on a sample video
    from the VITRA Robot Data dataset.
    """
    video_path = "<LOCAL_PATH>"
    mask_video_path = "<LOCAL_PATH>"
    output_path = "<LOCAL_PATH>"
    
    modes = ['thermal', 'radar', 'audio']
    start_time = time.time()
    
    vprint("=" * 60)
    vprint("Starting traverse_single_video_test")
    vprint("=" * 60)
    vprint(f"Video path: {video_path}")
    vprint(f"Mask path: {mask_video_path}")
    vprint(f"Output path: {output_path}")
    vprint()
    
    all_results = {}
    
    for mode in modes:
        vprint(f"\n{'='*60}")
        vprint(f"Processing mode: {mode.upper()}")
        vprint(f"{'='*60}")
        
        result = traverse_single_video(
            video_path=video_path,
            mask_video_path=mask_video_path,
            output_path=output_path,
            mode=mode
        )
        
        all_results[mode] = result
    
    # Print final summary
    vprint("\n" + "=" * 60)
    vprint("FINAL SUMMARY")
    vprint("=" * 60)
    total_successful = sum(r['successful'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    for mode, result in all_results.items():
        vprint(f"{mode.upper()}: {result['successful']} successful, {result['failed']} failed")
    
    vprint(f"\nTotal: {total_successful} successful, {total_failed} failed")
    vprint("=" * 60)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    return all_results


def find_video_pairs_vitra(config: dict) -> list:
    """Find all video/mask pairs for VITRA dataset."""
    json_file = config['root'] / config['json_file']
    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        return []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pairs = []
    seen_dirs = set()  # Track unique video directories
    
    for item in data:
        video_dir = config['root'] / item['relative_path']
        
        # Skip if we've already processed this video directory
        if str(video_dir) in seen_dirs:
            continue
        
        video_path = video_dir / config['video_name']
        mask_path = video_dir / config['mask_name']
        
        if video_path.exists() and mask_path.exists():
            seen_dirs.add(str(video_dir))
            pairs.append({
                'video_path': str(video_path),
                'mask_path': str(mask_path),
                'output_dir': str(video_dir),
                'name': item['relative_path']
            })
    
    return pairs


def find_video_pairs_agibot(config: dict) -> list:
    """Find all video/mask pairs for AgiBotWorld dataset."""
    root = config['root']
    pairs = []
    
    if not root.exists():
        print(f"Root directory not found: {root}")
        return []
    
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for episode_dir in sorted(task_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            video_path = episode_dir / config['video_name']
            mask_path = episode_dir / config['mask_name']
            
            if video_path.exists() and mask_path.exists():
                pairs.append({
                    'video_path': str(video_path),
                    'mask_path': str(mask_path),
                    'output_dir': str(episode_dir),
                    'name': f"{task_dir.name}/{episode_dir.name}"
                })
    
    return pairs


def find_video_pairs_interna1(config: dict) -> list:
    """Find all video/mask pairs for InternA1 dataset."""
    root = config['root']
    pairs = []
    
    if not root.exists():
        print(f"Root directory not found: {root}")
        return []
    
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        
        # Navigate to set_0/videos
        set_dir = task_dir / "set_0"
        if not set_dir.exists():
            continue
        
        videos_dir = set_dir / "videos"
        if not videos_dir.exists():
            continue
        
        # Find all chunks
        for chunk_dir in sorted(videos_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            
            rgb_dir = chunk_dir / "observation.images.head"
            mask_dir = chunk_dir / "observation.images.mask"
            
            if not rgb_dir.exists() or not mask_dir.exists():
                continue
            
            # Find all video/mask pairs in the chunk
            for video_file in sorted(rgb_dir.glob("*_h264.mp4")):
                episode_name = video_file.stem.replace('_h264', '')  # Remove _h264 suffix
                mask_file = mask_dir / f"{episode_name}.mp4"
                
                if mask_file.exists():
                    output_dir = chunk_dir / "observation.images.sensor"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    pairs.append({
                        'video_path': str(video_file),
                        'mask_path': str(mask_file),
                        'output_dir': str(output_dir),
                        'name': f"{task_dir.name}/set_0/{chunk_dir.name}/{episode_name}"
                    })
    
    return pairs


def find_video_pairs_molmoact(config: dict) -> list:
    """Find all video/mask pairs for MolmoAct dataset."""
    pairs = []
    
    for subdir_name in config['subdirs']:
        subdir = config['root'] / subdir_name
        if not subdir.exists():
            continue
        
        train_dir = subdir / "train"
        if not train_dir.exists():
            continue
        
        # Find all chunks
        videos_dir = train_dir / "videos"
        if not videos_dir.exists():
            continue
        
        for chunk_dir in sorted(videos_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            
            rgb_dir = chunk_dir / "observations.rgb"
            mask_dir = chunk_dir / "observations.mask"
            
            if not rgb_dir.exists() or not mask_dir.exists():
                continue
            
            for video_file in sorted(rgb_dir.glob("*.mp4")):
                episode_name = video_file.stem
                mask_file = mask_dir / f"{episode_name}.mp4"
                
                if mask_file.exists():
                    output_dir = chunk_dir / "observations.sensor"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    pairs.append({
                        'video_path': str(video_file),
                        'mask_path': str(mask_file),
                        'output_dir': str(output_dir),
                        'name': f"{subdir_name}/{chunk_dir.name}/{episode_name}"
                    })
    
    return pairs


def process_single_video_pair(args):
    """Worker function to process a single video/mask pair."""
    pair, modes = args
    
    video_path = pair['video_path']
    mask_path = pair['mask_path']
    output_dir = pair['output_dir']
    name = pair['name']
    
    start_time = time.time()
    
    try:
        # Process all modes in a single pass
        mode_results = traverse_single_video(
            video_path=video_path,
            mask_video_path=mask_path,
            output_path=output_dir
        )
        
        # Add processing time to each mode result
        total_time = time.time() - start_time
        results = {'name': name, 'modes': mode_results, 'processing_time': total_time}
        
        # Calculate per-mode timing (approximate, since they're processed together)
        for mode in mode_results:
            mode_results[mode]['processing_time'] = total_time / 3
        
    except Exception as e:
        # Return error structure for all modes
        results = {
            'name': name,
            'modes': {
                'thermal': {'successful': 0, 'failed': 0, 'generated_files': [], 'error': str(e), 'processing_time': 0},
                'radar': {'successful': 0, 'failed': 0, 'generated_files': [], 'error': str(e), 'processing_time': 0},
                'audio': {'successful': 0, 'failed': 0, 'generated_files': [], 'error': str(e), 'processing_time': 0}
            },
            'processing_time': time.time() - start_time
        }
    
    return results


def process_dataset(dataset_name: str, config: dict, modes: list, rank: int, world_size: int):
    """Process all videos in a dataset with distributed processing."""
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
    
    # Find all video/mask pairs (all ranks do this to get same list)
    if dataset_name == "VITRA_Robot_Data":
        pairs = find_video_pairs_vitra(config)
    elif dataset_name == "Agibot_Data":
        pairs = find_video_pairs_agibot(config)
    elif dataset_name == "InternA1_Robot_Data":
        pairs = find_video_pairs_interna1(config)
    elif dataset_name == "MolmoAct_Data":
        pairs = find_video_pairs_molmoact(config)
    else:
        if rank == 0:
            print(f"Unknown dataset: {dataset_name}")
        return
    
    if not pairs:
        if rank == 0:
            print(f"No video/mask pairs found for {dataset_name}")
        return
    
    # Filter out already processed videos if SKIP_EXISTING is True
    if SKIP_EXISTING:
        # Define expected filenames for each mode
        semantic_labels_map = {
            'thermal': ['cold', 'room_temperature', 'hot'],
            'radar': ['hidden', 'absense'],
            'audio': ['ringing', 'silence']
        }
        background_modes = ['dark', 'color', 'grey']
        
        filtered_pairs = []
        for pair in pairs:
            output_dir = Path(pair['output_dir'])
            video_name = Path(pair['video_path']).stem
            
            # Check if all expected files exist with exact names
            all_exist = True
            for mode in modes:
                mode_dir = output_dir / mode
                if not mode_dir.exists():
                    all_exist = False
                    break
                
                # Check each expected filename
                for semantic_label in semantic_labels_map[mode]:
                    for background_mode in background_modes:
                        expected_filename = f"{video_name}_{semantic_label}_{background_mode}.mp4"
                        expected_file = mode_dir / expected_filename
                        if not expected_file.exists():
                            all_exist = False
                            break
                    if not all_exist:
                        break
                if not all_exist:
                    break
            
            if not all_exist:
                filtered_pairs.append(pair)
        
        skipped = len(pairs) - len(filtered_pairs)
        if skipped > 0 and rank == 0:
            print(f"Skipping {skipped} already processed videos")
        pairs = filtered_pairs
    
    if not pairs:
        if rank == 0:
            print(f"All videos already processed for {dataset_name}")
        return
    
    # Limit to first 2 if TEST_MODE
    if TEST_MODE:
        pairs = pairs[:2]
        if rank == 0:
            print(f"TEST MODE: Processing only first {len(pairs)} videos")
    
    if rank == 0:
        print(f"Found {len(pairs)} video/mask pairs to process")
        print(f"Modes: {', '.join(modes)}")
        print(f"World size: {world_size} processes\n")
    
    # Split pairs across ranks using round-robin
    my_pairs = split_videos_by_rank(pairs, world_size, rank)
    print(f"Rank {rank}: Assigned {len(my_pairs)} videos to process")
    
    # Process assigned pairs sequentially on each rank
    all_results = []
    total_time = 0
    
    for pair_idx, pair in enumerate(my_pairs, 1):
        try:
            print(f"\nRank {rank}: [{pair_idx}/{len(my_pairs)}] Processing: {pair['name']}")
            result = process_single_video_pair((pair, modes))
            all_results.append(result)
            total_time += result['processing_time']
            
            # Print completion with timing
            mode_summary = ", ".join([f"{mode}: {result['modes'][mode]['successful']} ok ({result['modes'][mode].get('processing_time', 0):.1f}s)" 
                                      for mode in modes if mode in result['modes']])
            print(f"Rank {rank}: {result['name']} - {result['processing_time']:.1f}s total - {mode_summary}")
        except Exception as e:
            print(f"Rank {rank}: Error processing {pair['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary for this rank
    print(f"\nRank {rank}: {'='*80}")
    print(f"Rank {rank}: Summary for {dataset_name}:")
    print(f"Rank {rank}: {'='*80}")
    
    for mode in modes:
        total_successful = sum(r['modes'].get(mode, {}).get('successful', 0) for r in all_results)
        total_failed = sum(r['modes'].get(mode, {}).get('failed', 0) for r in all_results)
        mode_time = sum(r['modes'].get(mode, {}).get('processing_time', 0) for r in all_results)
        print(f"Rank {rank}:   {mode.upper()}: {total_successful} successful, {total_failed} failed, {mode_time:.2f}s total")
    
    avg_time = total_time / len(all_results) if all_results else 0
    print(f"Rank {rank}:\n  Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Rank {rank}:   Average time per video: {avg_time:.2f}s")
    print(f"Rank {rank}: {'='*80}\n")


def _verify_sensor_video(local_path: Path) -> dict:
    """
    Verify a single sensor video by reading every frame.
    Mirrors the _load_single_sensor_frame logic from robot_dataset_vlm_vla.py.

    Returns a dict with keys: path, status, frame_count, readable_frames,
    and optionally failed_frames.
    """
    if not local_path.exists():
        print(f"  [MISSING] File does not exist: {local_path}")
        return {'path': str(local_path), 'status': 'missing', 'frame_count': 0, 'readable_frames': 0}

    cap = cv2.VideoCapture(str(local_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        cap.release()
        print(f"  [BROKEN] Video has 0 frames: {local_path}")
        return {'path': str(local_path), 'status': 'broken_zero_frames', 'frame_count': 0, 'readable_frames': 0}

    readable_frames = 0
    failed_frames = []
    for frame_idx in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            readable_frames += 1
        else:
            failed_frames.append(frame_idx)
    cap.release()

    if readable_frames == frame_count:
        print(f"  [OK] All {frame_count} frames readable")
        return {'path': str(local_path), 'status': 'ok', 'frame_count': frame_count, 'readable_frames': readable_frames}
    else:
        failed_preview = failed_frames[:10]
        suffix = '...' if len(failed_frames) > 10 else ''
        print(
            f"  [PARTIAL] {readable_frames}/{frame_count} frames readable. "
            f"Failed frames: {failed_preview}{suffix}"
        )
        return {
            'path': str(local_path),
            'status': 'partial',
            'frame_count': frame_count,
            'readable_frames': readable_frames,
            'failed_frames': failed_frames,
        }


def _resolve_video_mask_paths(episode_dir: Path) -> tuple:
    """
    Resolve the video and mask paths for a given episode directory by matching
    it against DATASET_CONFIGS (with the local mount prefix substituted).

    Supports:
      - VITRA_Robot_Data : video.mp4 / video_seg.mp4  (directly in episode dir)
      - Agibot_Data       : videos/head_color.mp4 / head_mask.mp4
      - any other config  : uses video_name / mask_name from config

    Falls back to VITRA defaults if no config root matches.

    Returns:
        (video_path, mask_path) as Path objects
    """
    OLD_PREFIX = "<DATA_ROOT>"
    NEW_PREFIX = "<LOCAL_PATH>"

    for config in DATASET_CONFIGS.values():
        config_root_str = str(config['root'])
        # Convert the config root to the local mount path for comparison
        if config_root_str.startswith(OLD_PREFIX):
            local_root = Path(NEW_PREFIX + config_root_str[len(OLD_PREFIX):])
        else:
            local_root = Path(config_root_str)

        try:
            episode_dir.relative_to(local_root)
            # Matched — use this config's video/mask names
            return episode_dir / config['video_name'], episode_dir / config['mask_name']
        except ValueError:
            continue

    # Fallback to VITRA defaults
    return episode_dir / "video.mp4", episode_dir / "video_seg.mp4"


def traverse_fix_video():
    """
    Regenerate all sensor overlay videos for episodes that contain broken/missing
    sensor videos, then verify every regenerated file frame-by-frame.

    For each path in `target_video_paths`:
      1. Convert <DATA_ROOT> prefix to the local mount path.
      2. Resolve the episode directory (parent of the sensor sub-folder, e.g. thermal/).
      3. Locate the original video and mask using _resolve_video_mask_paths(), which
         matches the episode dir against DATASET_CONFIGS to pick the correct filenames
         (e.g. video.mp4/video_seg.mp4 for VITRA, videos/head_color.mp4/head_mask.mp4
         for Agibot).
      4. Call traverse_single_video() to regenerate ALL sensor overlays (thermal,
         radar, audio) for that episode — identical to traverse_single_video_test().
      5. Verify every regenerated video frame-by-frame using _verify_sensor_video().

    Add the broken/missing sensor video paths to `target_video_paths` below.
    Paths may use either the <DATA_ROOT> or the local mount prefix.
    """
    # -----------------------------------------------------------------
    # List of broken/missing sensor video paths that need to be fixed.
    # The episode directory is inferred as the parent of the sensor sub-folder.
    # Example layout:
    #   .../20250827_Result/2025_08_27_09_54_02_000/thermal/video_room_temperature_grey.mp4
    #                           ^--- episode dir ----^  ^sensor^  ^-----filename-----^
    # -----------------------------------------------------------------
    target_video_paths = [
        "<DATA_ROOT>",
        "<DATA_ROOT>",
        "<DATA_ROOT>",
        "<DATA_ROOT>",
        # Add more paths here as needed
    ]

    OLD_PREFIX = "<DATA_ROOT>"
    NEW_PREFIX = "<LOCAL_PATH>"

    print("=" * 80)
    print("traverse_fix_video: Regenerating and verifying sensor videos")
    print("=" * 80)

    # Deduplicate episode directories so we regenerate each episode only once
    episode_dirs_seen = set()
    episode_dir_list = []
    for orig_path in target_video_paths:
        if orig_path.startswith(OLD_PREFIX):
            local_path = NEW_PREFIX + orig_path[len(OLD_PREFIX):]
        else:
            local_path = orig_path
        # Sensor video lives in  <episode_dir>/<sensor_type>/<filename>.mp4
        # so episode dir = grandparent of the file
        episode_dir = Path(local_path).parent.parent
        if str(episode_dir) not in episode_dirs_seen:
            episode_dirs_seen.add(str(episode_dir))
            episode_dir_list.append(episode_dir)

    total_episodes = len(episode_dir_list)
    regen_ok = 0
    regen_fail = 0
    all_verify_results = []

    for ep_idx, episode_dir in enumerate(episode_dir_list, 1):
        print(f"\n{'='*80}")
        print(f"[{ep_idx}/{total_episodes}] Episode: {episode_dir}")
        print(f"{'='*80}")

        video_path, mask_path = _resolve_video_mask_paths(episode_dir)
        print(f"  video : {video_path}")
        print(f"  mask  : {mask_path}")

        if not video_path.exists():
            print(f"  [SKIP] Original video not found: {video_path}")
            regen_fail += 1
            continue
        if not mask_path.exists():
            print(f"  [SKIP] Mask video not found: {mask_path}")
            regen_fail += 1
            continue

        # --- Regenerate all sensor overlays (same as traverse_single_video_test) ---
        print(f"  Regenerating sensor overlays...")
        start_time = time.time()
        mode_results = traverse_single_video(
            video_path=str(video_path),
            mask_video_path=str(mask_path),
            output_path=str(episode_dir),
        )
        elapsed = time.time() - start_time

        total_ok = sum(r['successful'] for r in mode_results.values())
        total_fail = sum(r['failed'] for r in mode_results.values())
        print(f"  Regeneration done in {elapsed:.1f}s — {total_ok} succeeded, {total_fail} failed")
        for mode, r in mode_results.items():
            print(f"    {mode.upper()}: {r['successful']} ok, {r['failed']} failed")

        if total_fail > 0:
            regen_fail += 1
        else:
            regen_ok += 1

        # --- Verify every generated file frame-by-frame ---
        print(f"\n  Verifying regenerated files...")
        for mode, r in mode_results.items():
            for gen_path in r['generated_files']:
                local_gen = Path(gen_path)
                print(f"    Verifying: {local_gen.name}")
                verify_result = _verify_sensor_video(local_gen)
                all_verify_results.append(verify_result)

    # --- Final summary ---
    verify_ok = sum(1 for r in all_verify_results if r['status'] == 'ok')
    verify_fail = len(all_verify_results) - verify_ok

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Episodes regenerated successfully : {regen_ok}/{total_episodes}")
    print(f"  Episodes with regeneration errors : {regen_fail}/{total_episodes}")
    print(f"  Videos verified OK                : {verify_ok}/{len(all_verify_results)}")
    print(f"  Videos with verification failures : {verify_fail}/{len(all_verify_results)}")
    print("=" * 80)

    return {
        'regen_ok': regen_ok,
        'regen_fail': regen_fail,
        'verify_results': all_verify_results,
    }


def main():
    """Main function to process all datasets with distributed processing."""
    # Set NCCL timeout to 10 minutes for very long processing times
    os.environ['NCCL_TIMEOUT'] = '600'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    
    # Initialize distributed environment
    local_rank, world_size, rank = init_distributed()
    
    start_time = time.time()
    
    # Sensor modes to generate
    modes = ['thermal', 'radar', 'audio']
    
    if rank == 0:
        print("="*80)
        print("Sensor Mask Generation Pipeline (Distributed)")
        print("="*80)
        print(f"Skip existing: {SKIP_EXISTING}")
        print(f"Test mode: {TEST_MODE}")
        print(f"World size: {world_size}")
        print(f"Modes: {', '.join(modes)}")
        print(f"Datasets: {', '.join(DATASET_CONFIGS.keys())}")
        print("="*80)
    
    # Process each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        try:
            process_dataset(dataset_name, config, modes, rank, world_size)
        except Exception as e:
            print(f"Rank {rank}: Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print total time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nRank {rank}: {'='*80}")
    print(f"Rank {rank}: Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Rank {rank}: {'='*80}")
    
    # Cleanup distributed
    if world_size > 1 and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed")
        except Exception as e:
            print(f"Rank {rank}: Error destroying process group: {e}")


if __name__ == "__main__":
    if FIX_MODE:
        traverse_fix_video()
    else:
        main()
