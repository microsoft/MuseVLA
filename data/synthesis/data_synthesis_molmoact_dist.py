#!/usr/bin/env python
"""
MolmoAct Dataset Processing and Synthesis Script (Distributed)

Processes MolmoAct robot data to generate:
1. RGB videos from first_view images in videos/<chunk>/observations.rgb
2. Segmentation mask videos in videos/<chunk>/observations.mask
3. Depth videos in videos/<chunk>/observations.depth
4. Modified parquet files without image columns in data_new/<chunk>/
5. Task annotations in meta/extra_sensor.json with object lists and rephrased instructions

Usage:
    torchrun --nproc_per_node=8 data_synthesis_molmoact_dist.py
"""

import json
import torch
import torch.distributed as dist
import cv2
import numpy as np
import pandas as pd
import av
from pathlib import Path
import sys
import os
import time
import re
import logging
import argparse

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress LLM interface logging
logging.getLogger('llm.custom_logger').setLevel(logging.CRITICAL)
logging.getLogger('llm_interface').setLevel(logging.CRITICAL)
logging.getLogger('llm').setLevel(logging.CRITICAL)

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
from vlm_utils import object_annotation, modify_task_with_sensor, set_verbose
from depth_anything_3.api import DepthAnything3

# Configuration
DATA_ROOTS = [
    Path("<DATA_ROOT>"),
    Path("<DATA_ROOT>"),
]
VERBOSE = False
MODEL_TYPE = 'sam3'  # 'sam2' or 'sam3'
SKIP_EXISTING = True  # Set to True to skip already processed files
TEST_MODE = False  # Set to True to process first 2 tasks with first 2 episodes each

# Depth configuration
DEPTH_MODEL_TYPE = 'nested-giant-large'
DEPTH_MIN_CLIP = 0.3  # meters
DEPTH_MAX_CLIP = 1.5  # meters

# Model mapping for depth
DEPTH_MODEL_MAP = {
    'giant': 'depth-anything/DA3-GIANT',
    'large': 'depth-anything/DA3-LARGE',
    'base': 'depth-anything/DA3-BASE',
    'small': 'depth-anything/DA3-SMALL',
    'nested-giant-large': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'
}


def create_video_from_images(images: list, output_path: Path, fps: float = 15.0):
    """
    Create video from list of RGB images.
    
    Args:
        images: List of RGB numpy arrays (H, W, 3)
        output_path: Output video path
        fps: Frames per second
    """
    if len(images) == 0:
        print(f"    ⚠ No images to create video")
        return
    
    height, width = images[0].shape[:2]
    
    # Ensure width and height are divisible by 2
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    
    # Create video container
    output_container = av.open(str(output_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=int(fps))
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = 'yuv420p'
    output_stream.options = {'crf': '23', 'preset': 'fast'}
    
    for img in images:
        # Resize if needed
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Create VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        
        for packet in output_stream.encode(frame):
            output_container.mux(packet)
    
    # Flush
    for packet in output_stream.encode():
        output_container.mux(packet)
    
    output_container.close()


def generate_mask_video_from_images(images: list, annotation_result: dict, output_path: Path, 
                                    fps: float, video_predictor, temp_video_path: Path) -> tuple:
    """
    Generate segmentation mask video from images using SAM3.
    Uses pre-computed annotation_result from extra_sensor.json.
    
    Returns:
        (processing_time, fps_processing)
    """
    # First create a temporary video from images for SAM3
    create_video_from_images(images, temp_video_path, fps)
    
    # Use first 2 objects if more are detected
    all_objects_str = annotation_result.get("all_objects", "")
    if all_objects_str:
        # Split by period and take first 2
        objects = [obj.strip() for obj in all_objects_str.split('.') if obj.strip()]
        text_prompt = '. '.join(objects[:2])
    else:
        text_prompt = ""
    
    if not text_prompt:
        print(f"    ⚠ No object detected, skipping mask generation")
        return 0, 0
    
    # Start timing for segmentation
    start_time = time.time()
    
    # Start session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(temp_video_path),
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
    all_masks = {}
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
    
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processing = len(images) / processing_time if processing_time > 0 else 0
    
    height, width = images[0].shape[:2]
    
    # Ensure width and height are divisible by 2
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    
    # Save mask video
    output_container = av.open(str(output_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=int(fps))
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


def generate_depth_video_from_images(images: list, output_path: Path, fps: float, 
                                     depth_model) -> tuple:
    """
    Generate depth video from images using DepthAnything3.
    
    Returns:
        (processing_time, fps_processing)
    """
    start_time = time.time()
    
    # Process frames in batches of 100 to avoid OOM
    BATCH_SIZE = 100
    all_depth_maps = []
    
    for i in range(0, len(images), BATCH_SIZE):
        batch_frames = images[i:i + BATCH_SIZE]
        prediction = depth_model.inference(batch_frames)
        all_depth_maps.append(prediction.depth)
        
        # Clear CUDA cache after each batch to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all depth maps
    depth_maps = np.concatenate(all_depth_maps, axis=0)
    
    # Save depth maps to file
    depth_maps_path = output_path.parent / f"{output_path.stem}_depth_maps.npy"
    np.save(depth_maps_path, depth_maps)
    
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processing = len(images) / processing_time if processing_time > 0 else 0
    
    # Clip and normalize depth
    depth_clipped = np.clip(depth_maps, DEPTH_MIN_CLIP, DEPTH_MAX_CLIP)
    depth_normalized = ((depth_clipped - DEPTH_MIN_CLIP) / (DEPTH_MAX_CLIP - DEPTH_MIN_CLIP) * 255).astype(np.uint8)
    
    height, width = images[0].shape[:2]
    
    # Ensure width and height are divisible by 2
    output_width = width if width % 2 == 0 else width + 1
    output_height = height if height % 2 == 0 else height + 1
    
    # Save depth video
    output_container = av.open(str(output_path), 'w')
    output_stream = output_container.add_stream('libx264', rate=int(fps))
    output_stream.width = output_width
    output_stream.height = output_height
    output_stream.pix_fmt = 'yuv420p'
    
    for i in range(len(depth_normalized)):
        depth_frame = depth_normalized[i]
        
        # Resize to original image size
        depth_frame_resized = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Pad if needed
        if output_height != height or output_width != width:
            pad_h = output_height - height
            pad_w = output_width - width
            depth_frame_resized = np.pad(depth_frame_resized, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # Apply colormap (JET)
        depth_colormap = cv2.applyColorMap(depth_frame_resized, cv2.COLORMAP_JET)
        
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
    
    return processing_time, fps_processing


def process_episode(episode_path: Path, task_index: int, rephrased_instruction: str,
                   annotation_result: dict, chunk_name: str, train_dir: Path,
                   depth_model, video_predictor, episode_idx: int, total_episodes: int, rank: int):
    """
    Process a single episode parquet file.
    """
    episode_name = episode_path.stem  # e.g., episode_000000
    
    print(f"\nRank {rank}: [{episode_idx}/{total_episodes}] Processing: {episode_name} (Task {task_index})")
    
    # Create output directories
    videos_dir = train_dir / "videos" / chunk_name
    data_new_dir = train_dir / "data_new" / chunk_name
    
    rgb_dir = videos_dir / "observations.rgb"
    depth_dir = videos_dir / "observations.depth"
    mask_dir = videos_dir / "observations.mask"
    
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    data_new_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    rgb_video_path = rgb_dir / f"{episode_name}.mp4"
    depth_video_path = depth_dir / f"{episode_name}.mp4"
    mask_video_path = mask_dir / f"{episode_name}.mp4"
    new_parquet_path = data_new_dir / f"{episode_name}.parquet"
    
    # Check if already processed
    if SKIP_EXISTING and rgb_video_path.exists() and depth_video_path.exists() and mask_video_path.exists() and new_parquet_path.exists():
        print(f"  ⏭ Already processed, skipping")
        return
    
    # Load parquet file
    df = pd.read_parquet(episode_path)
    num_frames = len(df)
    print(f"  Frames: {num_frames}")
    
    # Extract first_view images
    first_view_images = []
    for idx, row in df.iterrows():
        img_dict = row['first_view']
        # Image is stored as bytes in the parquet
        img_bytes = img_dict['bytes']
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        first_view_images.append(img_rgb)
    
    fps = 15.0  # Default FPS for MolmoAct
    
    # 1. Create RGB video from first_view images
    print(f"  Creating RGB video...")
    rgb_start = time.time()
    create_video_from_images(first_view_images, rgb_video_path, fps)
    rgb_time = time.time() - rgb_start
    print(f"  ✓ RGB video: {rgb_time:.2f}s")
    
    # 2. Generate mask video using pre-computed annotation from metadata
    print(f"  Generating segmentation mask...")
    temp_video_path = rgb_video_path.parent / f"temp_{episode_name}_rank{rank}.mp4"
    mask_time, mask_fps = generate_mask_video_from_images(
        first_view_images, annotation_result, mask_video_path, fps, 
        video_predictor, temp_video_path
    )
    # Clean up temp video
    if temp_video_path.exists():
        temp_video_path.unlink()
    print(f"  ✓ Mask video: {mask_time:.2f}s, {mask_fps:.2f} FPS")
    
    # Clear GPU cache before depth generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Generate depth video
    print(f"  Generating depth video...")
    depth_time, depth_fps = generate_depth_video_from_images(
        first_view_images, depth_video_path, fps, depth_model
    )
    
    # Free frames from memory
    del first_view_images
    print(f"  ✓ Depth video: {depth_time:.2f}s, {depth_fps:.2f} FPS")
    
    # 4. Save modified parquet without image columns
    print(f"  Saving modified parquet...")
    df_new = df.drop(columns=['first_view', 'second_view', 'wrist_image'], errors='ignore')
    df_new.to_parquet(new_parquet_path, index=False)
    print(f"  ✓ Saved parquet without image columns")
    
    # Summary
    total_time = rgb_time + mask_time + depth_time
    print(f"  📊 Total processing time: {total_time:.2f}s")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_task_metadata(task_index: int, annotation_result: dict, rephrased_instruction: str, 
                      metadata_file: Path):
    """
    Save task metadata to extra_sensor.json file.
    """
    # Load existing metadata if file exists
    metadata_dict = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
        except:
            print(f"  ⚠ Could not load existing metadata, creating new file")
            metadata_dict = {}
    
    # Add or update metadata for this task
    metadata_dict[f"task_{task_index}"] = {
        "task_index": task_index,
        "primary_object": annotation_result.get('object', ''),
        "related_objects": annotation_result.get('related_objects', []),
        "all_objects": annotation_result.get('all_objects', ''),
        "rephrased_instruction": rephrased_instruction
    }
    
    # Save updated metadata dict to file
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


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
    parser = argparse.ArgumentParser(description='MolmoAct Data Synthesis with Distributed Processing')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2], 
                       help='Stage: 1=annotation only, 2=episode processing (default: 2)')
    args = parser.parse_args()
    
    # Initialize distributed environment
    local_rank, world_size, rank = init_distributed()
    
    # Helper function to print only on rank 0
    def print_rank0(msg):
        if rank == 0:
            print(msg)
    
    # Set verbosity
    set_verbose(VERBOSE)
    
    STAGE_1_ANNOTATION_ONLY = (args.stage == 1)
    
    depth_model = None
    video_predictor = None
    
    if not STAGE_1_ANNOTATION_ONLY:
        # Initialize depth model once per GPU
        print(f"Rank {rank}: Initializing Depth Anything 3 model ({DEPTH_MODEL_TYPE})...")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        depth_model = DepthAnything3.from_pretrained(DEPTH_MODEL_MAP[DEPTH_MODEL_TYPE])
        depth_model = depth_model.to(device=device)
        print(f"Rank {rank}: ✓ Depth model loaded on {device}")
        
        # Initialize SAM3 video predictor once per GPU
        print(f"Rank {rank}: Initializing SAM3 video predictor...")
        from sam3.model_builder import build_sam3_video_predictor
        video_predictor = build_sam3_video_predictor()
        print(f"Rank {rank}: ✓ SAM3 video predictor loaded")
    
    # Process each dataset
    print_rank0("\n" + "=" * 80)
    print_rank0("Starting processing...")
    print_rank0("=" * 80)
    
    start_time = time.time()
    total_episodes_processed = 0
    
    for data_root in DATA_ROOTS:
        if not data_root.exists():
            print_rank0(f"\n⚠ Dataset not found: {data_root}")
            continue
        
        print_rank0(f"\n{'='*80}")
        print_rank0(f"Processing Dataset: {data_root.name}")
        print_rank0(f"{'='*80}")
        
        train_dir = data_root / "train"
        meta_dir = train_dir / "meta"
        data_dir = train_dir / "data"
        
        # Load tasks.jsonl (all ranks load the same data)
        tasks_file = meta_dir / "tasks.jsonl"
        if not tasks_file.exists():
            print_rank0(f"  ⚠ tasks.jsonl not found, skipping dataset")
            continue
        
        tasks = {}
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                task_data = json.loads(line.strip())
                tasks[task_data['task_index']] = task_data['task']
        
        print_rank0(f"  Loaded {len(tasks)} tasks")
        
        # Create reverse mapping: task_description -> task_index
        task_desc_to_idx = {desc: idx for idx, desc in tasks.items()}
        
        # Load episodes.jsonl to find first episode for each task
        episodes_file = meta_dir / "episodes.jsonl"
        if not episodes_file.exists():
            print_rank0(f"  ⚠ episodes.jsonl not found, skipping dataset")
            continue
        
        print_rank0(f"  Processing episodes.jsonl to find first episode for each task...")
        task_first_episodes = {}  # task_index -> first episode relative path
        
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                episode_idx = episode_data['episode_index']
                task_list = episode_data['tasks']
                
                # Get task description (first task in list)
                if task_list:
                    task_desc = task_list[0]
                    # Find task_index from description
                    task_idx = task_desc_to_idx.get(task_desc)
                    
                    if task_idx is not None:
                        # Keep only the first episode for each task
                        if task_idx not in task_first_episodes:
                            # Build episode path: chunk-XXX/episode_YYYYYY.parquet
                            chunk_num = episode_idx // 1000
                            chunk_name = f"chunk-{chunk_num:03d}"
                            episode_name = f"episode_{episode_idx:06d}.parquet"
                            episode_path = Path(chunk_name) / episode_name
                            task_first_episodes[task_idx] = episode_path
        
        print_rank0(f"  Found first episodes for {len(task_first_episodes)} tasks")
        
        # Get task indices to process
        task_indices = sorted(tasks.keys())
        
        metadata_file = meta_dir / "extra_sensor.json"

        if STAGE_1_ANNOTATION_ONLY:
            # Only rank 0 processes Phase 1
            if rank == 0:
                # ========================================================================
                # PHASE 1: Process task annotations and metadata (Rank 0 only)
                # ========================================================================
                print(f"\n{'='*80}")
                print(f"Phase 1: Processing task annotations and metadata (Rank 0)")
                print(f"{'='*80}")
                
                tasks_processed = 0
                
                for task_idx in task_indices:
                    try:
                        print(f"\n{'-'*80}")
                        print(f"Task {task_idx}: {tasks[task_idx]}")
                        print(f"{'-'*80}")
                        
                        original_instruction = tasks[task_idx]
                        
                        # Rephrase instruction
                        print(f"  Rephrasing instruction...")
                        rephrase_start = time.time()
                        rephrased_instructions = modify_task_with_sensor([original_instruction])
                        rephrased_instruction = rephrased_instructions[0] if rephrased_instructions else original_instruction
                        rephrase_time = time.time() - rephrase_start
                        print(f"  ✓ Rephrased ({rephrase_time:.2f}s): {rephrased_instruction}")
                        
                        # Get object annotation from first episode of this task
                        first_episode_path = data_dir / task_first_episodes.get(task_idx, Path(""))
                        if not first_episode_path.exists():
                            print(f"  ⚠ First episode not found, skipping task")
                            continue
                        
                        print(f"  Getting object annotation from first episode...")
                        # Load first episode to create temp video for annotation
                        df_first = pd.read_parquet(first_episode_path)
                        first_images = []
                        for idx, row in df_first.iterrows():
                            img_dict = row['first_view']
                            img_bytes = img_dict['bytes']
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            first_images.append(img_rgb)
                        
                        # Create temp video for annotation
                        temp_annotation_video = train_dir / f"temp_task_{task_idx}_annotation.mp4"
                        create_video_from_images(first_images, temp_annotation_video, fps=15.0)
                        
                        annotation_start = time.time()
                        annotation_result = object_annotation(str(temp_annotation_video), rephrased_instruction)
                        annotation_time = time.time() - annotation_start
                        
                        # Clean up temp video
                        if temp_annotation_video.exists():
                            temp_annotation_video.unlink()
                        
                        print(f"  ✓ Object annotation ({annotation_time:.2f}s)")
                        print(f"    Primary: {annotation_result.get('object', '')}")
                        print(f"    All: {annotation_result.get('all_objects', '')}")
                        
                        # Save task metadata
                        save_task_metadata(task_idx, annotation_result, rephrased_instruction, metadata_file)
                        print(f"  ✓ Saved task metadata")
                        
                        tasks_processed += 1
                        
                    except KeyboardInterrupt:
                        print("\n\n⚠ Processing interrupted by user")
                        break
                    except Exception as e:
                        print(f"\n⚠ Error processing task {task_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"\n✓ Completed task annotation for {tasks_processed} tasks")
                print(f"\n{'='*80}")
                print(f"STAGE 1 COMPLETE: Task annotations saved to {metadata_file.name}")
                print(f"To run Stage 2 (episode processing):")
                print(f"  torchrun --nproc_per_node=8 data_synthesis_molmoact_dist.py --stage 2")
                print(f"{'='*80}")
            else:
                print(f"Rank {rank}: Waiting (only rank 0 processes Phase 1)")
            
            continue
        
        # ========================================================================
        # PHASE 2: Process all episodes (All ranks)
        # ========================================================================
        print_rank0(f"\n{'='*80}")
        print_rank0(f"Phase 2: Processing all episodes")
        print_rank0(f"{'='*80}")
        
        # Load saved task metadata from extra_sensor.json (all ranks load)
        if not metadata_file.exists():
            print_rank0(f"  ⚠ Metadata file not found, skipping episode processing")
            continue
        
        print_rank0(f"  Loading task metadata from {metadata_file.name}...")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            saved_metadata = json.load(f)
        
        print_rank0(f"  ✓ Loaded metadata for {len(saved_metadata)} tasks")
        
        # Build episode list with task metadata using episodes.jsonl (all ranks build same list)
        print_rank0(f"  Building episode list from episodes.jsonl...")
        all_episodes_to_process = []
        
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                episode_idx = episode_data['episode_index']
                task_list = episode_data['tasks']
                
                if not task_list:
                    continue
                
                # Get task description and map to task_idx
                task_desc = task_list[0]
                task_idx = task_desc_to_idx.get(task_desc)
                
                if task_idx is None:
                    continue
                
                # Check if we have metadata for this task
                task_key = f"task_{task_idx}"
                if task_key not in saved_metadata:
                    continue
                
                task_meta = saved_metadata[task_key]
                rephrased_instruction = task_meta['rephrased_instruction']
                annotation_result = {
                    'object': task_meta['primary_object'],
                    'related_objects': task_meta['related_objects'],
                    'all_objects': task_meta['all_objects']
                }
                
                # Build episode path from episode_index
                chunk_num = episode_idx // 1000
                chunk_name = f"chunk-{chunk_num:03d}"
                episode_name = f"episode_{episode_idx:06d}.parquet"
                episode_path = data_dir / chunk_name / episode_name
                
                if episode_path.exists():
                    all_episodes_to_process.append({
                        'episode_path': episode_path,
                        'chunk_name': chunk_name,
                        'task_idx': task_idx,
                        'rephrased_instruction': rephrased_instruction,
                        'annotation_result': annotation_result
                    })
        
        if TEST_MODE:
            all_episodes_to_process = all_episodes_to_process[:16]
            print_rank0(f"  TEST MODE: Processing first 16 episodes")
        
        print_rank0(f"  Found {len(all_episodes_to_process)} total episodes to process")
        
        # Split episodes across GPUs using round-robin
        my_episodes = split_episodes_by_rank(all_episodes_to_process, world_size, rank)
        print(f"Rank {rank}: Assigned {len(my_episodes)} episodes to process")
        
        # Process each episode assigned to this rank
        for ep_idx, episode_info in enumerate(my_episodes, 1):
            try:
                process_episode(
                    episode_info['episode_path'], 
                    episode_info['task_idx'], 
                    episode_info['rephrased_instruction'],
                    episode_info['annotation_result'], 
                    episode_info['chunk_name'], 
                    train_dir,
                    depth_model, 
                    video_predictor, 
                    ep_idx, 
                    len(my_episodes),
                    rank
                )
                total_episodes_processed += 1
            except KeyboardInterrupt:
                print(f"\n\nRank {rank}: ⚠ Processing interrupted by user")
                break
            except Exception as e:
                print(f"\nRank {rank}: ⚠ Error processing episode {episode_info['episode_path'].name}: {e}")
                import traceback
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
