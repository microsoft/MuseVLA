#!/usr/bin/env python
"""
AgiBotWorld Dataset Processing and Synthesis Script (Distributed)

Processes AgiBotWorld robot data to generate:
1. Segmentation mask videos from RGB videos
2. Depth videos from depth PNG sequences
3. Task annotations with object lists and rephrased instructions

Usage:
    torchrun --nproc_per_node=8 data_synthesis_agibot_dist.py --stage 2
"""

import json
import torch
import torch.distributed as dist
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os
import time
import re
import logging
import av
import argparse
import traceback

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress LLM interface logging
logging.getLogger('llm.custom_logger').setLevel(logging.WARNING)
logging.getLogger('llm_interface').setLevel(logging.WARNING)
logging.getLogger('llm').setLevel(logging.WARNING)

# Disable ANSI color codes
os.environ['ANSI_COLORS_DISABLED'] = '1'
os.environ['NO_COLOR'] = '1'

# Completely disable tqdm by replacing it with a no-op class
class NoOpTqdm:
    def __init__(self, *args, **kwargs):
        self.iterable = args[0] if args else None
        self.disable = True
        self.n = 0
        self.total = kwargs.get('total', 0)
        
    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
        return iter([])
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass
    
    def update(self, n=1):
        pass
    
    def close(self):
        pass
    
    def set_description(self, desc):
        pass
    
    def set_postfix(self, **kwargs):
        pass
    
    @staticmethod
    def write(s, **kwargs):
        pass

# Replace tqdm module
import tqdm as tqdm_module
tqdm_module.tqdm = NoOpTqdm
tqdm_module.std.tqdm = NoOpTqdm
from tqdm import tqdm

# Wrapper to strip ANSI escape codes from output
class StripAnsiStream:
    def __init__(self, stream):
        self.stream = stream
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    
    def write(self, text):
        return self.stream.write(self.ansi_escape.sub('', text))
    
    def flush(self):
        return self.stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.stream, name)

# Apply wrapper to stdout and stderr
sys.stdout = StripAnsiStream(sys.stdout)
sys.stderr = StripAnsiStream(sys.stderr)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from vlm_utils import object_annotation, task_rephrase, set_verbose

# Configuration
DATA_ROOT = Path("<DATA_ROOT>")
TASK_INFO_DIR = DATA_ROOT / "task_info"
OBSERVATIONS_DIR = DATA_ROOT / "observations_extract"
VERBOSE = False
MODEL_TYPE = 'sam3'  # 'sam2' or 'sam3'
SKIP_EXISTING = False  # Set to True to skip already processed files
TEST_MODE = False  # Set to True to process first 16 episodes only

# Depth configuration (same as MolmoAct)
DEPTH_MIN_CLIP = 0.3  # meters = 300mm
DEPTH_MAX_CLIP = 1.5  # meters = 1500mm


def load_depth_images_from_folder(depth_folder: Path) -> list:
    """
    Load depth images from folder containing head_depth_XXXXXX.png files.
    
    Returns:
        List of numpy arrays with depth values in millimeters (uint16)
    """
    depth_files = sorted(depth_folder.glob("head_depth_*.png"))
    depth_images = []
    
    for depth_file in depth_files:
        # Load 16-bit PNG
        depth_img = Image.open(depth_file)
        depth_array = np.array(depth_img, dtype=np.uint16)
        depth_images.append(depth_array)
    
    return depth_images


def create_depth_video_from_images(depth_images: list, output_video_path: Path, 
                                   output_npy_path: Path, fps: float = 30.0) -> tuple:
    """
    Create depth video from depth PNG images (16-bit, in millimeters).
    Applies same clipping and colormap as MolmoAct processing.
    
    Returns:
        (processing_time, fps_processing)
    """
    start_time = time.time()
    
    if len(depth_images) == 0:
        print(f"    ⚠ No depth images found")
        return 0, 0
    
    # Convert depth from millimeters to meters
    depth_maps_meters = np.array(depth_images, dtype=np.float32) / 1000.0
    
    # Save depth maps to file (in meters)
    np.save(output_npy_path, depth_maps_meters)
    
    # Clip and normalize depth
    depth_clipped = np.clip(depth_maps_meters, DEPTH_MIN_CLIP, DEPTH_MAX_CLIP)
    depth_normalized = ((depth_clipped - DEPTH_MIN_CLIP) / (DEPTH_MAX_CLIP - DEPTH_MIN_CLIP) * 255).astype(np.uint8)
    
    height, width = depth_images[0].shape[:2]
    
    # Ensure width and height are divisible by 2
    output_width = width if width % 2 == 0 else width + 1
    output_height = height if height % 2 == 0 else height + 1
    
    # Save depth video
    output_container = av.open(str(output_video_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=int(fps))
    output_stream.width = output_width
    output_stream.height = output_height
    output_stream.pix_fmt = 'yuv420p'
    
    for i in range(len(depth_normalized)):
        depth_frame = depth_normalized[i]
        
        # Pad if needed
        if output_height != height or output_width != width:
            pad_h = output_height - height
            pad_w = output_width - width
            depth_frame = np.pad(depth_frame, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # Apply colormap (JET)
        depth_colormap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        frame = av.VideoFrame.from_ndarray(depth_colormap_rgb, format='rgb24')
        
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
    
    # Flush
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    output_container.close()
    
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processing = len(depth_images) / processing_time if processing_time > 0 else 0
    
    return processing_time, fps_processing


def generate_mask_video_from_rgb_video(rgb_video_path: Path, annotation_result: dict, 
                                       output_path: Path, video_predictor, max_frames_per_chunk: int = 300) -> tuple:
    """
    Generate segmentation mask video from RGB video using SAM3.
    Uses pre-computed annotation_result from extra_sensor.jsonl.
    Processes video in chunks if it's too long to avoid OOM.
    
    Args:
        max_frames_per_chunk: Maximum frames to process in one chunk (default: 300)
    
    Returns:
        (processing_time, fps_processing)
    """
    # Use first 4 objects if more are detected
    all_objects_str = annotation_result.get("all_objects", "")
    if all_objects_str:
        # Split by period and take first 4
        objects = [obj.strip() for obj in all_objects_str.split('.') if obj.strip()]
        text_prompt = '. '.join(objects[:4])
    else:
        text_prompt = ""
    
    if not text_prompt:
        print(f"    ⚠ No object detected, skipping mask generation")
        return 0, 0
    
    # Get video properties
    container = av.open(str(rgb_video_path))
    video_stream = container.streams.video[0]
    width = video_stream.width
    height = video_stream.height
    fps_video = float(video_stream.average_rate)
    total_frames = video_stream.frames
    container.close()
    
    # Determine if we need to process in chunks
    need_chunking = total_frames > max_frames_per_chunk
    
    if need_chunking:
        print(f"    Video has {total_frames} frames, processing in chunks of {max_frames_per_chunk}")
    
    # Start timing for segmentation
    start_time = time.time()
    
    # Collect all masks
    all_masks = {}
    
    if need_chunking:
        # Process video in chunks
        num_chunks = (total_frames + max_frames_per_chunk - 1) // max_frames_per_chunk
        
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * max_frames_per_chunk
            end_frame = min((chunk_idx + 1) * max_frames_per_chunk, total_frames)
            
            print(f"    Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame})")
            
            # Create temporary chunk video
            temp_chunk_path = output_path.parent / f"temp_chunk_{chunk_idx}_{output_path.stem}.mp4"
            
            # Extract chunk from video
            container = av.open(str(rgb_video_path))
            output_container = av.open(str(temp_chunk_path), 'w')
            output_stream = output_container.add_stream('libx264', rate=int(fps_video))
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = 'yuv420p'
            
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame <= i < end_frame:
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                if i >= end_frame:
                    break
            
            # Flush
            for packet in output_stream.encode():
                output_container.mux(packet)
            
            container.close()
            output_container.close()
            
            # Process chunk with SAM3
            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(temp_chunk_path),
                )
            )
            session_id = response["session_id"]
            
            # Add text prompt on first frame
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text_prompt,
                )
            )
            
            # Propagate through chunk
            for response in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="both",
                )
            ):
                frame_idx = response["frame_index"]
                outputs = response["outputs"]
                # Adjust frame index to global position
                global_frame_idx = start_frame + frame_idx
                all_masks[global_frame_idx] = outputs["out_binary_masks"]
            
            # Close session
            video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            
            # Clean up temp chunk
            if temp_chunk_path.exists():
                temp_chunk_path.unlink()
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    else:
        # Process entire video at once (original behavior)
        # Start session
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(rgb_video_path),
            )
        )
        session_id = response["session_id"]
        
        # Add text prompt on first frame
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt,
            )
        )
        
        # Propagate through video and collect masks
        for response in video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="both",
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            all_masks[frame_idx] = outputs["out_binary_masks"]
        
        # Close session
        video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
    
    # Clear GPU cache after propagation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    num_frames = len(all_masks)
    fps_processing = num_frames / processing_time if processing_time > 0 else 0
    
    # Ensure width and height are divisible by 2
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    
    # Save mask video
    output_container = av.open(str(output_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=int(fps_video))
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = 'yuv420p'
    
    for frame_idx in sorted(all_masks.keys()):
        masks = all_masks[frame_idx]
        
        # Combine all masks for this frame
        if len(masks) > 0:
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        else:
            combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Resize if needed
        if combined_mask.shape[0] != height or combined_mask.shape[1] != width:
            combined_mask = cv2.resize(combined_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Convert grayscale to RGB
        mask_rgb = np.stack([combined_mask, combined_mask, combined_mask], axis=-1)
        
        # Create VideoFrame
        frame = av.VideoFrame.from_ndarray(mask_rgb, format='rgb24')
        
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
    
    # Flush
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    output_container.close()
    
    return processing_time, fps_processing


def process_episode(task_id: str, episode_id: str, annotation_result: dict, 
                   rephrased_instruction: str, video_predictor, 
                   episode_idx: int, total_episodes: int, rank: int):
    """
    Process a single episode (generate mask and depth videos).
    """
    print(f"\nRank {rank}: [{episode_idx}/{total_episodes}] Processing: Task {task_id}, Episode {episode_id}")
    
    # Episode directory
    episode_dir = OBSERVATIONS_DIR / task_id / episode_id
    
    if not episode_dir.exists():
        print(f"  ⚠ Episode directory not found: {episode_dir}")
        return
    
    # Input paths
    rgb_video_path = episode_dir / "videos" / "head_color.mp4"
    depth_folder = episode_dir / "depth"
    
    # Output paths
    mask_video_path = episode_dir / "head_mask.mp4"
    depth_video_path = episode_dir / "head_depth.mp4"
    depth_npy_path = episode_dir / "head_depth_maps.npy"
    
    # Check if files exist
    if not rgb_video_path.exists():
        print(f"  ⚠ RGB video not found: {rgb_video_path}")
        return
    
    if not depth_folder.exists():
        print(f"  ⚠ Depth folder not found: {depth_folder}")
        return
    
    # Check if already processed
    if SKIP_EXISTING and mask_video_path.exists() and depth_video_path.exists():
        print(f"  ⏭ Already processed, skipping")
        return
    
    # Get video info
    container = av.open(str(rgb_video_path))
    video_stream = container.streams.video[0]
    num_frames = video_stream.frames
    fps = float(video_stream.average_rate)
    container.close()
    
    print(f"  Frames: {num_frames}, FPS: {fps:.2f}")
    
    # 1. Generate mask video from RGB video
    print(f"  Generating segmentation mask...")
    mask_time, mask_fps = generate_mask_video_from_rgb_video(
        rgb_video_path, annotation_result, mask_video_path, video_predictor
    )
    print(f"  ✓ Mask video: {mask_time:.2f}s, {mask_fps:.2f} FPS")
    
    # Clear GPU cache after mask generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Generate depth video from depth PNGs
    print(f"  Processing depth images...")
    depth_images = load_depth_images_from_folder(depth_folder)
    print(f"    Loaded {len(depth_images)} depth images")
    
    depth_time, depth_fps = create_depth_video_from_images(
        depth_images, depth_video_path, depth_npy_path, fps
    )
    
    # Free depth images from memory
    del depth_images
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  ✓ Depth video: {depth_time:.2f}s, {depth_fps:.2f} FPS")
    print(f"  ✓ Saved depth maps: {depth_npy_path.name}")
    
    # Summary
    total_time = mask_time + depth_time
    print(f"  📊 Total processing time: {total_time:.2f}s")


def save_task_metadata(task_id: str, episode_id: str, annotation_result: dict, rephrase_result: dict,
                      task_name: str, init_scene_text: str, metadata_file: Path):
    """
    Append episode metadata to extra_sensor.jsonl file.
    """
    metadata = {
        "task_id": task_id,
        "episode_id": episode_id,
        "task_name": task_name,
        "init_scene_text": init_scene_text,
        "original_instruction": rephrase_result.get('original_instruction', ''),
        "rephrased_instruction": rephrase_result.get('rephrased_instruction', ''),
        "primary_object": annotation_result.get('object', ''),
        "related_objects": annotation_result.get('related_objects', []),
        "all_objects": annotation_result.get('all_objects', '')
    }
    
    # Append to JSONL file
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')


def load_task_metadata(metadata_file: Path) -> dict:
    """
    Load episode metadata from extra_sensor.jsonl file.
    Returns dict mapping (task_id, episode_id) -> metadata
    """
    metadata_dict = {}
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    task_id = data['task_id']
                    episode_id = data['episode_id']
                    metadata_dict[(task_id, episode_id)] = data
    
    return metadata_dict


def split_episodes_by_rank(all_episodes: list, world_size: int, rank: int) -> list:
    """
    Split episodes across GPUs using round-robin distribution.
    
    Args:
        all_episodes: List of episode info dicts
        world_size: Total number of GPUs
        rank: Current GPU rank
    
    Returns:
        List of episodes assigned to this GPU
    """
    # Distribute episodes across GPUs using round-robin
    my_episodes = []
    for i, episode_info in enumerate(all_episodes):
        if i % world_size == rank:
            my_episodes.append(episode_info)
    
    return my_episodes


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


def main():
    """Main processing function with distributed support."""
    # Set NCCL timeout to 10 minutes for very long processing times
    os.environ['NCCL_TIMEOUT'] = '600'  # 10 minutes in seconds
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AgiBotWorld Data Synthesis with Distributed Processing')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Initialize distributed environment
    local_rank, world_size, rank = init_distributed()
    
    # Helper function to print only on rank 0
    def print_rank0(msg):
        if rank == 0:
            print(msg)
    
    print_rank0("=" * 80)
    print_rank0("AgiBotWorld Dataset Processing and Synthesis (Distributed)")
    print_rank0("=" * 80)
    
    # Set verbosity
    set_verbose(VERBOSE)
    
    # STAGE 1: Set to True to run task annotation only (no models needed)
    # STAGE 2: Set to False to run episode processing (needs models)
    STAGE_1_ANNOTATION_ONLY = False
    
    video_predictor = None
    
    if not STAGE_1_ANNOTATION_ONLY:
        # Initialize SAM3 video predictor once per GPU
        print(f"Rank {rank}: Initializing SAM3 video predictor...")
        from sam3.model_builder import build_sam3_video_predictor
        video_predictor = build_sam3_video_predictor()
        print(f"Rank {rank}: ✓ SAM3 video predictor loaded")
    
    # Process dataset
    print_rank0("\n" + "=" * 80)
    print_rank0("Starting processing...")
    print_rank0("=" * 80)
    
    start_time = time.time()
    
    # Get all task files
    task_files = sorted(TASK_INFO_DIR.glob("task_*.json"))
    print_rank0(f"\nFound {len(task_files)} task files")
    
    metadata_file = TASK_INFO_DIR / "extra_sensor.jsonl"
    
    if STAGE_1_ANNOTATION_ONLY:
        # Only rank 0 processes Phase 1
        if rank == 0:
            # ========================================================================
            # PHASE 1: Process task rephrasing and episode annotations (Rank 0 only)
            # ========================================================================
            print(f"\n{'='*80}")
            print(f"Phase 1: Processing task rephrasing and episode annotations (Rank 0)")
            print(f"{'='*80}")
            
            # Clear existing metadata file
            if metadata_file.exists():
                metadata_file.unlink()
            
            tasks_processed = 0
            
            for task_file in task_files:
                try:
                    # Extract task_id from filename
                    task_id = task_file.stem.replace('task_', '')
                    
                    print(f"\n{'-'*80}")
                    print(f"Task {task_id}")
                    print(f"{'-'*80}")
                    
                    # Load task file
                    with open(task_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    if len(task_data) == 0:
                        print(f"  ⚠ Empty task file, skipping")
                        continue
                    
                    # Get first episode
                    first_episode = task_data[0]
                    episode_id = str(first_episode['episode_id'])
                    task_name = first_episode['task_name']
                    init_scene_text = first_episode['init_scene_text']
                    
                    print(f"  Task name: {task_name}")
                    print(f"  Init scene: {init_scene_text}")
                    
                    # Process all episodes for this task
                    print(f"  Processing {len(task_data)} episodes for task rephrasing and annotation...")
                    episodes_annotated = 0
                    
                    for episode_data in task_data:
                        episode_id = str(episode_data['episode_id'])
                        episode_dir = OBSERVATIONS_DIR / task_id / episode_id
                        rgb_video_path = episode_dir / "videos" / "head_color.mp4"
                        
                        if not rgb_video_path.exists():
                            print(f"    ⚠ Episode {episode_id}: RGB video not found, skipping")
                            continue
                        
                        print(f"    Episode {episode_id}: Rephrasing instruction...")
                        rephrase_start = time.time()
                        rephrase_result = task_rephrase(task_name, init_scene_text, str(rgb_video_path))
                        rephrase_time = time.time() - rephrase_start
                        
                        print(f"      ✓ Rephrased ({rephrase_time:.2f}s): {rephrase_result['rephrased_instruction']}")
                        
                        print(f"    Episode {episode_id}: Getting object annotation...")
                        annotation_start = time.time()
                        annotation_result = object_annotation(str(rgb_video_path), rephrase_result['rephrased_instruction'])
                        annotation_time = time.time() - annotation_start
                        
                        print(f"      ✓ Annotation ({annotation_time:.2f}s) - Primary: {annotation_result.get('object', '')}")
                        
                        # Save episode metadata
                        save_task_metadata(
                            task_id, episode_id, annotation_result, rephrase_result,
                            task_name, init_scene_text, metadata_file
                        )
                        
                        episodes_annotated += 1
                    
                    print(f"  ✓ Processed {episodes_annotated}/{len(task_data)} episodes")
                    tasks_processed += 1
                    
                except KeyboardInterrupt:
                    print("\n\n⚠ Processing interrupted by user")
                    break
                except Exception as e:
                    print(f"\n⚠ Error processing task {task_id}: {e}")
                    traceback.print_exc()
                    continue
            
            print(f"\n✓ Completed task rephrasing and episode annotation for {tasks_processed} tasks")
            
            print(f"\n{'='*80}")
            print(f"STAGE 1 COMPLETE: All annotations saved to {metadata_file.name}")
            print(f"To run Stage 2 (video processing):")
            print(f"  torchrun --nproc_per_node=8 data_synthesis_agibot_dist.py --stage 2")
            print(f"{'='*80}")
        else:
            print(f"Rank {rank}: Waiting (only rank 0 processes Phase 1)")
        
    else:
        # ========================================================================
        # PHASE 2: Process episode videos (mask and depth generation) - All ranks
        # ========================================================================
        print_rank0(f"\n{'='*80}")
        print_rank0(f"Phase 2: Processing episode videos (mask and depth generation)")
        print_rank0(f"{'='*80}")
        
        # Load episode metadata from Stage 1 (all ranks load)
        if not metadata_file.exists():
            print_rank0(f"  ⚠ Metadata file not found: {metadata_file}")
            print_rank0(f"  Please run Stage 1 first to generate annotations")
            return
        
        print_rank0(f"  Loading episode metadata from {metadata_file.name}...")
        saved_metadata = load_task_metadata(metadata_file)
        print_rank0(f"  ✓ Loaded metadata for {len(saved_metadata)} episodes")
        
        # Build episode list from saved metadata (all ranks build same list)
        print_rank0(f"  Building episode list from metadata...")
        all_episodes_to_process = []
        
        for (task_id, episode_id), episode_meta in saved_metadata.items():
            # Check if episode directory exists
            episode_dir = OBSERVATIONS_DIR / task_id / episode_id
            if episode_dir.exists():
                all_episodes_to_process.append({
                    'task_id': task_id,
                    'episode_id': episode_id,
                    'rephrased_instruction': episode_meta['rephrased_instruction'],
                    'annotation_result': {
                        'object': episode_meta['primary_object'],
                        'related_objects': episode_meta['related_objects'],
                        'all_objects': episode_meta['all_objects']
                    }
                })
        
        if TEST_MODE:
            all_episodes_to_process = all_episodes_to_process[:16]
            print_rank0(f"  TEST MODE: Processing first 16 episodes")
        
        print_rank0(f"  Found {len(all_episodes_to_process)} episodes to process")
        
        # Split episodes across GPUs using round-robin
        my_episodes = split_episodes_by_rank(all_episodes_to_process, world_size, rank)
        print(f"Rank {rank}: Assigned {len(my_episodes)} episodes to process")
        
        # Process each episode assigned to this rank
        total_episodes_processed = 0
        for ep_idx, episode_info in enumerate(my_episodes, 1):
            try:
                # Process episode (generate mask and depth videos)
                process_episode(
                    episode_info['task_id'],
                    episode_info['episode_id'],
                    episode_info['annotation_result'],
                    episode_info['rephrased_instruction'],
                    video_predictor,
                    ep_idx,
                    len(my_episodes),
                    rank
                )
                total_episodes_processed += 1
            except KeyboardInterrupt:
                print(f"\n\nRank {rank}: ⚠ Processing interrupted by user")
                raise
            except Exception as e:
                print(f"\nRank {rank}: ⚠ Error processing episode: {e}")
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        
        print(f"\nRank {rank}: " + "=" * 80)
        print(f"Rank {rank}: Processing completed!")
        print(f"Rank {rank}: Total episodes processed: {total_episodes_processed}")
        print(f"Rank {rank}: Total time: {total_time / 60:.2f} minutes")
        print(f"Rank {rank}: " + "=" * 80)
    
    # Cleanup distributed
    if world_size > 1 and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed")
        except Exception as e:
            print(f"Rank {rank}: Error destroying process group (may already be destroyed): {e}")


if __name__ == "__main__":
    main()
