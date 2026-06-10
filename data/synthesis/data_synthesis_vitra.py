#!/usr/bin/env python
"""
VITRA Robot Data Synthesis Script

Processes VITRA robot data to generate:
1. Segmentation mask videos (video_seg.mp4)
2. Depth videos (video_depth.mp4)
3. Sensor annotations (sensor.txt) with object lists and rephrased instructions

Usage:
    python data_synthesis_vitra.py
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
import re
import logging

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
from vlm_utils import object_annotation, modify_task_with_sensor, set_verbose
from depth_anything_3.api import DepthAnything3

# Configuration
DATA_ROOT = Path("<LOCAL_PATH>")
JSON_FILE = DATA_ROOT / "all_instructions_0910.json"
VERBOSE = False
MODEL_TYPE = 'sam3'  # 'sam2' or 'sam3'
SKIP_EXISTING = True  # Set to False to overwrite existing files
TEST_MODE = False  # Set to True to process only first 5 items

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


def extract_frames_from_video(video_path: Path) -> tuple:
    """
    Extract all frames from video.
    
    Returns:
        (frames_rgb, fps, width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    return frames, fps, width, height


def generate_mask_video(frames_rgb: list, video_path: Path, instruction: str, output_path: Path, 
                       fps: float, width: int, height: int, video_predictor) -> tuple:
    """
    Generate segmentation mask video using SAM3.
    
    Returns:
        (annotation_result, annotation_time, processing_time, fps_processing)
    """
    # Get object annotation from VLM
    annotation_start = time.time()
    annotation_result = object_annotation(str(video_path), instruction)
    annotation_time = time.time() - annotation_start
    
    # Use first 3 objects if more are detected
    all_objects_str = annotation_result["all_objects"]
    if all_objects_str:
        # Split by period and take first 3
        objects = [obj.strip() for obj in all_objects_str.split('.') if obj.strip()]
        text_prompt = '. '.join(objects[:2])
    else:
        text_prompt = ""
    
    if not text_prompt:
        print(f"Warning: No object detected for instruction: {instruction}")
        return annotation_result, annotation_time, 0, 0
    
    # Start timing for segmentation
    start_time = time.time()
    
    # Start session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_path),
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
    fps_processing = len(frames_rgb) / processing_time if processing_time > 0 else 0
    
    # Ensure width and height are divisible by 2 for libx264
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
        
        # Combine all masks for this frame (if multiple objects detected)
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
    
    return annotation_result, annotation_time, processing_time, fps_processing


def generate_depth_video(frames_rgb: list, output_path: Path, fps: float, 
                        width: int, height: int, depth_model) -> tuple:
    """
    Generate depth video using DepthAnything3.
    
    Returns:
        (processing_time, fps_processing)
    """
    start_time = time.time()
    
    # Process frames in batches of 200
    BATCH_SIZE = 200
    all_depth_maps = []
    
    for i in range(0, len(frames_rgb), BATCH_SIZE):
        batch_frames = frames_rgb[i:i + BATCH_SIZE]
        prediction = depth_model.inference(batch_frames)
        all_depth_maps.append(prediction.depth)
    
    # Concatenate all depth maps
    depth_maps = np.concatenate(all_depth_maps, axis=0)
    
    # Save depth maps to file
    depth_maps_path = output_path.parent / "depth_maps.npy"
    np.save(depth_maps_path, depth_maps)
    
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processing = len(frames_rgb) / processing_time if processing_time > 0 else 0
    
    # Clip and normalize depth
    depth_clipped = np.clip(depth_maps, DEPTH_MIN_CLIP, DEPTH_MAX_CLIP)
    depth_normalized = ((depth_clipped - DEPTH_MIN_CLIP) / (DEPTH_MAX_CLIP - DEPTH_MIN_CLIP) * 255).astype(np.uint8)
    
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
        
        # Resize to original video size
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


def save_metadata_json(annotation_result: dict, item: dict, metadata_file: Path):
    """
    Save metadata to all_metadata.json file immediately after processing.
    Creates file if it doesn't exist, otherwise loads and updates it.
    Returns the time taken for rephrasing.
    """
    # Load existing metadata if file exists
    metadata_list = []
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
        except:
            print(f"  ⚠ Could not load existing metadata, creating new file")
            metadata_list = []
    
    # Get rephrased instruction with timing
    rephrase_start = time.time()
    rephrased_instructions = modify_task_with_sensor([item['instruction']])
    rephrased = rephrased_instructions[0] if rephrased_instructions else item['instruction']
    rephrase_time = time.time() - rephrase_start
    
    # Check if this item already exists in the metadata list
    updated = False
    for existing in metadata_list:
        if (existing.get('instruction') == item['instruction'] and
            existing.get('start') == item['start'] and
            existing.get('end') == item['end'] and
            existing.get('relative_path') == item['relative_path']):
            # Item already exists, update it
            existing['primary_object'] = annotation_result.get('object', '')
            existing['related_objects'] = annotation_result.get('related_objects', [])
            existing['all_objects'] = annotation_result.get('all_objects', '')
            existing['rephrased_instruction'] = rephrased
            updated = True
            break
    
    # If not found, add new entry
    if not updated:
        metadata = {
            "instruction": item['instruction'],
            "start": item['start'],
            "end": item['end'],
            "relative_path": item['relative_path'],
            "primary_object": annotation_result.get('object', ''),
            "related_objects": annotation_result.get('related_objects', []),
            "all_objects": annotation_result.get('all_objects', ''),
            "rephrased_instruction": rephrased
        }
        metadata_list.append(metadata)
    
    # Save updated metadata list to file
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    
    return rephrase_time


def process_item(item: dict, depth_model, video_predictor, metadata_file: Path, item_idx: int, total_items: int):
    """
    Process a single item from the JSON file.
    """
    instruction = item['instruction']
    start_frame = item['start']
    end_frame = item['end']
    relative_path = item['relative_path']
    
    # Construct paths
    video_dir = DATA_ROOT / relative_path
    video_path = video_dir / "video.mp4"
    
    if not video_path.exists():
        print(f"[{item_idx}/{total_items}] ⚠ Video not found: {video_path}")
        return
    
    # Output paths
    output_seg = video_dir / "video_seg.mp4"
    output_depth = video_dir / "video_depth.mp4"
    
    # Check if already in metadata file with same start/end
    videos_already_generated = False
    existing_annotation = None
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            for existing in metadata_list:
                if (existing.get('relative_path') == relative_path and
                    existing.get('start') == start_frame and
                    existing.get('end') == end_frame):
                    print(f"[{item_idx}/{total_items}] ⏭ Already in metadata: {relative_path} (frames {start_frame}-{end_frame})")
                    return
                elif existing.get('relative_path') == relative_path:
                    # Same video but different start/end - don't regenerate videos
                    # Reuse annotation from existing entry
                    videos_already_generated = True
                    existing_annotation = {
                        'object': existing.get('primary_object', ''),
                        'related_objects': existing.get('related_objects', []),
                        'all_objects': existing.get('all_objects', '')
                    }
        except:
            pass  # File exists but couldn't be read, continue processing
    
    # Check if already processed (files exist)
    if SKIP_EXISTING and output_seg.exists() and output_depth.exists():
        videos_already_generated = True
    
    print(f"\n[{item_idx}/{total_items}] Processing: {relative_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Frames: {start_frame} to {end_frame}")
    
    if videos_already_generated:
        print(f"  ⏭ Videos already exist for this path, only generating metadata")
    
    # Extract frames only if needed for video generation
    if not videos_already_generated:
        frames_rgb, fps, width, height = extract_frames_from_video(video_path)
        print(f"  Extracted {len(frames_rgb)} frames at {fps:.2f} FPS ({width}x{height})")
        
        if len(frames_rgb) == 0:
            print(f"  ⚠ No frames extracted, skipping")
            return
    
    # Generate mask video only if not already generated
    if not videos_already_generated:
        print(f"  Generating segmentation mask...")
        annotation_result, annotation_time, mask_time, mask_fps = generate_mask_video(
            frames_rgb, video_path, instruction, output_seg, fps, width, height, video_predictor
        )
        print(f"  ✓ Annotation: {annotation_time:.2f}s")
        print(f"  ✓ Mask video: {mask_time:.2f}s, {mask_fps:.2f} FPS")
    else:
        # Reuse annotation from existing entry
        print(f"  ⏭ Reusing annotation from existing entry")
        annotation_result = existing_annotation
        annotation_time, mask_time, mask_fps = 0, 0, 0
    
    # Generate depth video only if not already generated
    if not videos_already_generated:
        print(f"  Generating depth video...")
        depth_time, depth_fps = generate_depth_video(
            frames_rgb, output_depth, fps, width, height, depth_model
        )
        print(f"  ✓ Depth video: {depth_time:.2f}s, {depth_fps:.2f} FPS")
    else:
        depth_time, depth_fps = 0, 0
    
    # Save metadata to file immediately
    rephrase_time = save_metadata_json(annotation_result, item, metadata_file)
    print(f"  ✓ Metadata saved to file")
    
    # Add rephrase time to annotation time for total
    total_annotation_time = annotation_time + rephrase_time
    
    # Summary
    total_time = total_annotation_time + mask_time + depth_time
    if total_time > 0:
        print(f"  📊 Total processing time: {total_time:.2f}s (annotation+rephrase: {total_annotation_time:.2f}s)")
        if mask_time + depth_time > 0:
            print(f"  📊 Average FPS: {len(frames_rgb) / (mask_time + depth_time):.2f}")


def main():
    """Main processing function."""
    print("=" * 80)
    print("VITRA Robot Data Synthesis")
    print("=" * 80)
    
    # Set verbosity
    set_verbose(VERBOSE)
    
    # Load JSON file
    if not JSON_FILE.exists():
        print(f"Error: JSON file not found: {JSON_FILE}")
        return
    
    print(f"Loading: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if TEST_MODE:
        data = data[:3]
        print(f"TEST MODE: Processing only first 3 items")
    
    print(f"Found {len(data)} items to process")
    
    # Initialize depth model once
    print(f"\nInitializing Depth Anything 3 model ({DEPTH_MODEL_TYPE})...")
    device = torch.device("cuda")
    depth_model = DepthAnything3.from_pretrained(DEPTH_MODEL_MAP[DEPTH_MODEL_TYPE])
    depth_model = depth_model.to(device=device)
    print("✓ Depth model loaded")
    
    # Initialize SAM3 video predictor once
    print(f"\nInitializing SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor
    video_predictor = build_sam3_video_predictor()
    print("✓ SAM3 video predictor loaded")
    
    # Define metadata file path
    metadata_file = DATA_ROOT / "all_metadata.json"
    
    # Check if metadata file exists and report count
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            print(f"\n✓ Found existing metadata file with {len(existing_metadata)} entries")
        except:
            print(f"\n⚠ Metadata file exists but could not be read")
    
    # Process each item
    print("\n" + "=" * 80)
    print("Starting processing...")
    print("=" * 80)
    
    start_time = time.time()
    
    for idx, item in enumerate(data, 1):
        try:
            process_item(item, depth_model, video_predictor, metadata_file, idx, len(data))
        except KeyboardInterrupt:
            print("\n\n⚠ Processing interrupted by user")
            break
        except Exception as e:
            print(f"\n⚠ Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Report final metadata count
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                final_metadata = json.load(f)
            print(f"\n✓ Final metadata file contains {len(final_metadata)} entries")
        except:
            print(f"\n⚠ Could not read final metadata file")
    
    print("\n" + "=" * 80)
    print("Processing completed!")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
