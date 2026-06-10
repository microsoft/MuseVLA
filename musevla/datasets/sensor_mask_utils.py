"""
Sensor Mask Processing Utility

This script processes robot sensor data to generate masks using SAM3 and create
sensor-masked images for acoustic, radar, and thermal modalities.

Directory structure:
    data_root/
    ├── acoustic_data/
    │   └── episode_folders/
    │       ├── images/           # RGB images
    │       ├── acoustic/images/  # Acoustic sensor images
    │       ├── radar/images/     # Radar sensor images
    │       └── thermal/          # Thermal sensor images
    │           ├── left/images/
    │           └── right/images/
    └── thermal_data/
        └── ... (similar structure)
"""

import os
import sys
import cv2
import numpy as np
import av
from pathlib import Path
from PIL import Image
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import json
import logging

# Disable ANSI color codes and progress bars
os.environ['TQDM_DISABLE'] = '1'
os.environ['NO_COLOR'] = '1'
os.environ['TERM'] = 'dumb'
os.environ['ANSI_COLORS_DISABLED'] = '1'
os.environ['PY_COLORS'] = '0'

# Disable tqdm monitoring thread to prevent crashes
import tqdm
tqdm.tqdm.monitor_interval = 0
try:
    tqdm.tqdm._instances.clear()
except:
    pass

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Configure logging to remove ANSI formatting
logging.basicConfig(format='%(message)s', level=logging.INFO)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(message)s'))

# Add vitra/datasets to path for acoustic_overlay_optimized
sys.path.insert(0, str(Path(__file__).parent))
from acoustic_overlay_optimized import overlay_acoustic_on_image_optimized_bgr
from radar_overlay_optimized import overlay_radar_on_image_optimized_bgr
from thermal_overlay_optimized import overlay_thermal_on_image_optimized_bgr

# Add calibration directory to path
calibration_path = Path(__file__).parent.parent.parent / "calibration"
sys.path.insert(0, str(calibration_path))

from src.thermal import ThermalMaskGenerator
from src.radar import RadarMaskGenerator
from src.acoustic import AcousticMaskGenerator

# Import SAM3 video predictor
from sam3.model_builder import build_sam3_video_predictor


# Prompt configuration for different data types
PROMPT_CONFIG = {
    'acoustic_data': 'towels.',
    'thermal_data': 'bottles.',
    'radar_data': 'boxes.'
}

SKIP_EXISTING = False

def process_episode_folder(
    episode_path: Path,
    data_type: str,
    video_predictor,
    episode_idx: int = None,
    total_episodes: int = None
):
    """
    Process a single episode folder to generate masks and sensor-masked images.
    
    Args:
        episode_path: Path to episode folder
        data_type: Type of data (e.g., 'acoustic_data', 'thermal_data')
        video_predictor: SAM3 video predictor instance
        episode_idx: Current episode index (1-based)
        total_episodes: Total number of episodes
    """
    if episode_idx is not None and total_episodes is not None:
        print(f"\nProcessing episode {episode_idx}/{total_episodes}: {episode_path.name}")
    else:
        print(f"\nProcessing episode: {episode_path.name}")
    episode_start_time = time.perf_counter()
    
    # Check for label file and determine prompt
    label_folder = episode_path.parent.parent / f"{data_type.replace('_data', '_labels')}"
    label_file = label_folder / f"{episode_path.name}.json"
    validity_file = label_folder / f"{episode_path.name}_validity.json"
    
    if label_file.exists():
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            
            # Check if episode is abolished
            if label_data.get('problem', {}).get('abolished', False):
                print(f"  ⚠ Episode marked as abolished, skipping")
                # Save validity file
                label_folder.mkdir(parents=True, exist_ok=True)
                with open(validity_file, 'w') as f:
                    json.dump({
                        'abolished': True,
                        'reason': 'Already abolished in dataset label file'
                    }, f, indent=2)
                return
            
            # Determine prompt based on first instruction
            annotations = label_data.get('annotations', [])
            if annotations and len(annotations) > 0 and 'acoustic' in data_type:
                first_instruction = annotations[0].get('instruction', '').lower()
                if 'towel' in first_instruction:
                    prompt = 'towels.'
                else:
                    prompt = 'clothes.'
                print(f"  Prompt from label: '{prompt}' (based on: '{annotations[0].get('instruction', '')}')")
            else:
                # Fallback to default prompt
                prompt = PROMPT_CONFIG.get(data_type, 'object.')
                print(f"  Using default prompt: '{prompt}'")
        except Exception as e:
            print(f"  Warning: Could not read label file: {e}")
            prompt = PROMPT_CONFIG.get(data_type, 'object.')
    else:
        print(f"Warning: label file not found")
        # Get prompt for this data type from config
        prompt = PROMPT_CONFIG.get(data_type, 'object.')
        return
    
    # Define paths
    images_dir = episode_path / "images"
    mask_dir = episode_path / "mask"
    
    if not images_dir.exists():
        print(f"  Warning: images directory not found, skipping: {images_dir}")
        return
    
    # Create mask directory
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all RGB image paths
    rgb_image_paths = sorted(images_dir.glob("*.jpg"))
    if not rgb_image_paths:
        print(f"  Warning: no images found in {images_dir}")
        return
    
    print(f"  Found {len(rgb_image_paths)} RGB images")
    
    # Preload all RGB images in batches
    print(f"  Preloading all RGB images (batch size: 20)...")
    start_preload = time.perf_counter()
    rgb_images = []
    batch_size = 20
    for batch_start in range(0, len(rgb_image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(rgb_image_paths))
        batch_paths = rgb_image_paths[batch_start:batch_end]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            batch_images = list(executor.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), batch_paths))
        rgb_images.extend(batch_images)
    preload_time = time.perf_counter() - start_preload
    print(f"    Loaded {len(rgb_images)} images in {preload_time:.2f}s ({preload_time/len(rgb_images)*1000:.1f}ms/image)")
    
    # Pre-compute background bases for different background modes
    background_modes = ['dark', 'color', 'grey']
    # background_modes = ['color']
    
    # Check if masks need to be generated
    mask_video_path = mask_dir / "video.mp4"
    # all_masks_exist = SKIP_EXISTING and mask_video_path.exists()
    all_masks_exist = False
    all_masks = {}  # Store masks in memory
    
    if not all_masks_exist:
        # Generate masks for all frames using video tracking
        print(f"  Generating masks with prompt: '{prompt}' (using video tracking)")
        
        # Define video path in episode folder
        video_path = episode_path / "video.mp4"
        
        # Create video from RGB images if it doesn't exist
        if not (video_path.exists() and SKIP_EXISTING):
            print(f"    Creating video.mp4 from {len(rgb_images)} images...")
            
            # Get dimensions from first preloaded image
            height, width = rgb_images[0].shape[:2]
            
            # Ensure dimensions are even for video encoding
            if width % 2 != 0:
                width += 1
            if height % 2 != 0:
                height += 1
            
            # Create video from RGB images
            output_container = av.open(str(video_path), 'w')
            output_stream = output_container.add_stream(
                'libx264',
                rate=30,
                options={'preset': 'veryfast', 'crf': '23', 'tune': 'zerolatency'}
            )
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = 'yuv420p'
            
            for img in rgb_images:
                # Resize if needed (should be rare)
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height))
                
                frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
            
            # Flush
            for packet in output_stream.encode():
                output_container.mux(packet)
            output_container.close()
        else:
            print(f"    video.mp4 already exists, skipping creation")
        
        # Run SAM3 on the video
        print(f"    Running SAM3 on video...")
        sam3_start_time = time.perf_counter()
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
                text=prompt,
            )
        )
        
        # Get dimensions from first RGB image
        height, width = rgb_images[0].shape[:2]
        
        # Propagate through video and collect masks (keep in memory)
        num_instances = 0  # Track number of instances in first frame
        processed_frames = 0
        
        try:
            for response in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="both",
                )
            ):
                frame_idx = response["frame_index"]
                outputs = response["outputs"]
                masks = outputs.get("out_binary_masks", [])
                
                # Count instances from first frame
                if frame_idx == 0:
                    num_instances = len(masks)
                
                # Combine all masks for this frame
                if len(masks) > 0:
                    combined_mask = np.any(masks, axis=0).astype(bool)
                else:
                    combined_mask = np.zeros((height, width), dtype=bool)
                
                all_masks[frame_idx] = combined_mask
        
        finally:
            # Always close session even if error occurs
            try:
                video_predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
            except:
                pass  # Ignore errors during cleanup
            
            # Force CUDA sync and cleanup after SAM3 processing
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        sam3_time = time.perf_counter() - sam3_start_time
        print(f"    SAM3 detected {num_instances} instance(s) in {sam3_time:.2f}s ({sam3_time/len(rgb_images)*1000:.1f}ms/frame)")
        
        # Check validity and save validity file
        if num_instances < 2:
            print(f"  ⚠ Warning: number of segmented instances is {num_instances}, expected at least 2")
            label_folder.mkdir(parents=True, exist_ok=True)
            with open(validity_file, 'w') as f:
                json.dump({
                    'abolished': True,
                    'reason': f'SAM3 detected only {num_instances} instance(s), expected at least 2'
                }, f, indent=2)
            print(f"  ⚠ Episode abolished due to insufficient instances, skipping remaining processing")
        else:
            # Episode is valid
            label_folder.mkdir(parents=True, exist_ok=True)
            with open(validity_file, 'w') as f:
                json.dump({
                    'abolished': False,
                    'reason': f'Valid episode with {num_instances} instances detected'
                }, f, indent=2)
        
        # Save all masks as video
        print(f"    Saving {len(all_masks)} masks as video...")
        
        # Ensure dimensions are even
        mask_height = height
        mask_width = width
        if mask_width % 2 != 0:
            mask_width += 1
        if mask_height % 2 != 0:
            mask_height += 1
        
        # Create video writer for masks
        output_container = av.open(str(mask_video_path), 'w')
        output_stream = output_container.add_stream(
            'libx264',
            rate=30,
            options={'preset': 'veryfast', 'crf': '23', 'tune': 'zerolatency'}
        )
        output_stream.width = mask_width
        output_stream.height = mask_height
        output_stream.pix_fmt = 'yuv420p'
        
        for idx in sorted(all_masks.keys()):
            # Convert bool mask to uint8 (0 or 255)
            mask_uint8 = (all_masks[idx].astype(np.uint8) * 255)
            
            # Resize if needed
            if mask_uint8.shape[0] != mask_height or mask_uint8.shape[1] != mask_width:
                mask_uint8 = cv2.resize(mask_uint8, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert grayscale to RGB for video
            mask_rgb = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
            
            # Write frame to video
            frame = av.VideoFrame.from_ndarray(mask_rgb, format='rgb24')
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        
        # Flush
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        print(f"    ✓ Saved mask video (kept {len(all_masks)} masks in memory)")
    else:
        # Load masks from existing video
        print(f"  Loading existing masks from video...")
        mask_cap = cv2.VideoCapture(str(mask_video_path))
        frame_idx = 0
        while True:
            ret, mask_frame = mask_cap.read()
            if not ret:
                break
            mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            all_masks[frame_idx] = (mask > 0).astype(bool)
            frame_idx += 1
        mask_cap.release()
        print(f"    Loaded {len(all_masks)} masks into memory")
    
    # Load homography matrices from auto-detected calibration directory
    calibration_dir = Path(__file__).parent.parent.parent / "calibration"
    acoustic_H = np.load(calibration_dir / "homography/acoustic/homography_rgb_to_acoustic.npy")
    radar_H = np.load(calibration_dir / "homography/radar/homography_rgb_to_radar_old.npy")
    thermal_left_H = np.load(calibration_dir / "homography/thermal/left/homography_rgb_to_ir.npy")
    thermal_right_H = np.load(calibration_dir / "homography/thermal/right/homography_rgb_to_ir.npy")
    
    # Initialize generators
    acoustic_generator = AcousticMaskGenerator(rgb_to_acoustic_transform=acoustic_H)
    radar_generator = RadarMaskGenerator(rgb_to_radar_transform=radar_H)
    thermal_generator = ThermalMaskGenerator(
        rgb_to_left_thermal_transform=thermal_left_H,
        rgb_to_right_thermal_transform=thermal_right_H
    )
    
    # Process acoustic sensor images
    acoustic_images_dir = episode_path / "acoustic" / "images"
    if acoustic_images_dir.exists():
        acoustic_video_path = acoustic_images_dir / "video.mp4"
        acoustic_masked_dir = episode_path / "acoustic" / "images_masked"
        acoustic_masked_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for each background mode
        acoustic_bg_paths = {
            'dark': acoustic_masked_dir / "video_dark.mp4",
            'color': acoustic_masked_dir / "video.mp4",
            'grey': acoustic_masked_dir / "video_grey.mp4"
        }
        
        need_raw = not (SKIP_EXISTING and acoustic_video_path.exists())
        backgrounds_needed = [bg for bg in background_modes 
                             if not (SKIP_EXISTING and acoustic_bg_paths[bg].exists())]
        
        if need_raw or backgrounds_needed:
            if need_raw and backgrounds_needed:
                print(f"  Creating acoustic raw and {len(backgrounds_needed)} background variant videos")
            elif need_raw:
                print("  Creating acoustic sensor video")
            else:
                print(f"  Creating {len(backgrounds_needed)} acoustic background variant videos")
            
            start_time = time.perf_counter()
            
            # Get dimensions
            height, width = rgb_images[0].shape[:2]
            if width % 2 != 0:
                width += 1
            if height % 2 != 0:
                height += 1
            
            # Preload all acoustic images at once
            acoustic_image_paths = []
            for rgb_image_path in rgb_image_paths:
                image_name = rgb_image_path.stem
                acoustic_image_path = acoustic_images_dir / f"{image_name}.jpg"
                if acoustic_image_path.exists():
                    acoustic_image_paths.append(acoustic_image_path)
            
            print(f"    Preloading {len(acoustic_image_paths)} acoustic images (batch size: 20)...")
            start_load = time.perf_counter()
            acoustic_images = []
            batch_size = 20
            for batch_start in range(0, len(acoustic_image_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(acoustic_image_paths))
                batch_paths = acoustic_image_paths[batch_start:batch_end]
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    batch_images = list(executor.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), batch_paths))
                acoustic_images.extend(batch_images)
            load_time = time.perf_counter() - start_load
            print(f"      Loaded in {load_time:.2f}s ({load_time/len(acoustic_images)*1000:.1f}ms/image)")
            
            # Create video writers
            raw_container = None
            raw_stream = None
            bg_containers = {}
            bg_streams = {}
            
            if need_raw:
                raw_container = av.open(str(acoustic_video_path), 'w')
                raw_stream = raw_container.add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                raw_stream.width = width
                raw_stream.height = height
                raw_stream.pix_fmt = 'yuv420p'
            
            for bg in backgrounds_needed:
                bg_containers[bg] = av.open(str(acoustic_bg_paths[bg]), 'w')
                bg_streams[bg] = bg_containers[bg].add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                bg_streams[bg].width = width
                bg_streams[bg].height = height
                bg_streams[bg].pix_fmt = 'yuv420p'
            
            # Process all frames
            start_encode = time.perf_counter()
            for idx in range(len(acoustic_images)):
                rgb_bgr = rgb_images[idx]
                acoustic_bgr = acoustic_images[idx]
                
                # Generate raw video
                if need_raw:
                    img = acoustic_bgr
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    for packet in raw_stream.encode(frame):
                        raw_container.mux(packet)
                
                # Generate background variant videos
                if backgrounds_needed and idx in all_masks:
                    mask = (all_masks[idx].astype(np.uint8) * 255)
                    
                    # Generate all 3 variants at once (internally creates dark/grey backgrounds)
                    dark_result, color_result, grey_result = overlay_acoustic_on_image_optimized_bgr(
                        acoustic_generator=acoustic_generator,
                        bgr_image=rgb_bgr,
                        mask=mask,
                        acoustic_bgr_image=acoustic_bgr,
                        alpha=1.0
                    )
                    
                    # Write each variant to its container
                    if 'dark' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(dark_result, format='bgr24')
                        for packet in bg_streams['dark'].encode(frame):
                            bg_containers['dark'].mux(packet)
                    
                    if 'color' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(color_result, format='bgr24')
                        for packet in bg_streams['color'].encode(frame):
                            bg_containers['color'].mux(packet)
                    
                    if 'grey' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(grey_result, format='bgr24')
                        for packet in bg_streams['grey'].encode(frame):
                            bg_containers['grey'].mux(packet)
            encode_time = time.perf_counter() - start_encode
            print(f"      Encoded in {encode_time:.2f}s ({encode_time/len(acoustic_images)*1000:.1f}ms/frame)")
            
            # Flush and close
            if need_raw:
                for packet in raw_stream.encode():
                    raw_container.mux(packet)
                raw_container.close()
            
            for bg in backgrounds_needed:
                for packet in bg_streams[bg].encode():
                    bg_containers[bg].mux(packet)
                bg_containers[bg].close()
            
            total_time = time.perf_counter() - start_time
            if need_raw and backgrounds_needed:
                print(f"    ✓ Saved acoustic raw + {len(backgrounds_needed)} background variants ({len(acoustic_images)} frames in {total_time:.2f}s, {total_time/len(acoustic_images)*1000:.1f}ms/frame)")
            elif need_raw:
                print(f"    ✓ Saved acoustic sensor video ({len(acoustic_images)} frames in {total_time:.2f}s, {total_time/len(acoustic_images)*1000:.1f}ms/frame)")
            else:
                print(f"    ✓ Saved {len(backgrounds_needed)} acoustic background variants ({len(acoustic_images)} frames in {total_time:.2f}s, {total_time/len(acoustic_images)*1000:.1f}ms/frame)")
        else:
            print("  Acoustic videos already exist, skipping")
    
    # Process radar sensor images
    radar_images_dir = episode_path / "radar" / "images"
    if radar_images_dir.exists():
        radar_video_path = radar_images_dir / "video.mp4"
        radar_masked_dir = episode_path / "radar" / "images_masked"
        radar_masked_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for each background mode
        radar_bg_paths = {
            'dark': radar_masked_dir / "video_dark.mp4",
            'color': radar_masked_dir / "video.mp4",
            'grey': radar_masked_dir / "video_grey.mp4"
        }
        
        need_raw = not (SKIP_EXISTING and radar_video_path.exists())
        backgrounds_needed = [bg for bg in background_modes 
                             if not (SKIP_EXISTING and radar_bg_paths[bg].exists())]
        
        if need_raw or backgrounds_needed:
            if need_raw and backgrounds_needed:
                print(f"  Creating radar raw and {len(backgrounds_needed)} background variant videos")
            elif need_raw:
                print("  Creating radar sensor video")
            else:
                print(f"  Creating {len(backgrounds_needed)} radar background variant videos")
            
            start_time = time.perf_counter()
            
            # Get dimensions
            height, width = rgb_images[0].shape[:2]
            if width % 2 != 0:
                width += 1
            if height % 2 != 0:
                height += 1
            
            # Preload all radar images at once
            radar_image_paths = []
            for rgb_image_path in rgb_image_paths:
                image_name = rgb_image_path.stem
                radar_image_path = radar_images_dir / f"{image_name}.jpg"
                if radar_image_path.exists():
                    radar_image_paths.append(radar_image_path)
            
            print(f"    Preloading {len(radar_image_paths)} radar images (batch size: 20)...")
            start_load = time.perf_counter()
            radar_images = []
            batch_size = 20
            for batch_start in range(0, len(radar_image_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(radar_image_paths))
                batch_paths = radar_image_paths[batch_start:batch_end]
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    batch_images = list(executor.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), batch_paths))
                radar_images.extend(batch_images)
            load_time = time.perf_counter() - start_load
            print(f"      Loaded in {load_time:.2f}s ({load_time/len(radar_images)*1000:.1f}ms/image)")
            
            # Create video writers
            raw_container = None
            raw_stream = None
            bg_containers = {}
            bg_streams = {}
            
            if need_raw:
                raw_container = av.open(str(radar_video_path), 'w')
                raw_stream = raw_container.add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                raw_stream.width = width
                raw_stream.height = height
                raw_stream.pix_fmt = 'yuv420p'
            
            for bg in backgrounds_needed:
                bg_containers[bg] = av.open(str(radar_bg_paths[bg]), 'w')
                bg_streams[bg] = bg_containers[bg].add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                bg_streams[bg].width = width
                bg_streams[bg].height = height
                bg_streams[bg].pix_fmt = 'yuv420p'
            
            # Process all frames
            start_encode = time.perf_counter()
            for idx in range(len(radar_images)):
                rgb_bgr = rgb_images[idx]
                radar_bgr = radar_images[idx]
                
                # Generate raw video
                if need_raw:
                    img = radar_bgr
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    for packet in raw_stream.encode(frame):
                        raw_container.mux(packet)
                
                # Generate background variant videos
                if backgrounds_needed and idx in all_masks:
                    mask = (all_masks[idx].astype(np.uint8) * 255)
                    
                    # Generate all 3 variants at once (internally creates dark/grey backgrounds)
                    dark_result, color_result, grey_result = overlay_radar_on_image_optimized_bgr(
                        radar_generator=radar_generator,
                        bgr_image=rgb_bgr,
                        mask=mask,
                        radar_bgr_image=radar_bgr,
                        alpha=1.0
                    )
                    
                    # Write each variant to its container
                    if 'dark' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(dark_result, format='bgr24')
                        for packet in bg_streams['dark'].encode(frame):
                            bg_containers['dark'].mux(packet)
                    
                    if 'color' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(color_result, format='bgr24')
                        for packet in bg_streams['color'].encode(frame):
                            bg_containers['color'].mux(packet)
                    
                    if 'grey' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(grey_result, format='bgr24')
                        for packet in bg_streams['grey'].encode(frame):
                            bg_containers['grey'].mux(packet)
            encode_time = time.perf_counter() - start_encode
            print(f"      Encoded in {encode_time:.2f}s ({encode_time/len(radar_images)*1000:.1f}ms/frame)")
            
            # Flush and close
            if need_raw:
                for packet in raw_stream.encode():
                    raw_container.mux(packet)
                raw_container.close()
            
            for bg in backgrounds_needed:
                for packet in bg_streams[bg].encode():
                    bg_containers[bg].mux(packet)
                bg_containers[bg].close()
            
            total_time = time.perf_counter() - start_time
            frames_count = len(radar_images)
            if need_raw and backgrounds_needed:
                print(f"    ✓ Saved radar raw + {len(backgrounds_needed)} background variants ({frames_count} frames in {total_time:.2f}s, {total_time/frames_count*1000:.1f}ms/frame)")
            elif need_raw:
                print(f"    ✓ Saved radar sensor video ({frames_count} frames in {total_time:.2f}s, {total_time/frames_count*1000:.1f}ms/frame)")
            else:
                print(f"    ✓ Saved {len(backgrounds_needed)} radar background variants ({frames_count} frames in {total_time:.2f}s, {total_time/frames_count*1000:.1f}ms/frame)")
        else:
            print("  Radar videos already exist, skipping")
    
    # Process thermal sensor images (raw, masked, and combined with shared loading)
    thermal_left_dir = episode_path / "thermal" / "left" / "images"
    thermal_right_dir = episode_path / "thermal" / "right" / "images"
    if thermal_left_dir.exists() and thermal_right_dir.exists():
        thermal_left_video_path = thermal_left_dir / "video.mp4"
        thermal_right_video_path = thermal_right_dir / "video.mp4"
        thermal_masked_dir = episode_path / "thermal" / "images_masked"
        thermal_masked_dir.mkdir(parents=True, exist_ok=True)
        thermal_combined_dir = episode_path / "thermal" / "images"
        thermal_combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for each background mode (only for masked)
        thermal_bg_paths = {
            'dark': thermal_masked_dir / "video_dark.mp4",
            'color': thermal_masked_dir / "video.mp4",
            'grey': thermal_masked_dir / "video_grey.mp4"
        }
        
        thermal_combined_video_path = thermal_combined_dir / "video.mp4"
        
        need_raw_left = not (SKIP_EXISTING and thermal_left_video_path.exists())
        need_raw_right = not (SKIP_EXISTING and thermal_right_video_path.exists())
        need_combined = not (SKIP_EXISTING and thermal_combined_video_path.exists())
        backgrounds_needed = [bg for bg in background_modes 
                             if not (SKIP_EXISTING and thermal_bg_paths[bg].exists())]
        
        if need_raw_left or need_raw_right or need_combined or backgrounds_needed:
            msg_parts = []
            if need_raw_left:
                msg_parts.append("left raw")
            if need_raw_right:
                msg_parts.append("right raw")
            if need_combined:
                msg_parts.append("combined")
            if backgrounds_needed:
                msg_parts.append(f'{len(backgrounds_needed)} masked bg')
            print(f"  Creating thermal {', '.join(msg_parts)} videos (shared loading)")
            
            start_time = time.perf_counter()
            
            # Get dimensions
            height, width = rgb_images[0].shape[:2]
            if width % 2 != 0:
                width += 1
            if height % 2 != 0:
                height += 1
            
            # Preload all thermal images at once
            thermal_left_paths = []
            thermal_right_paths = []
            for rgb_image_path in rgb_image_paths:
                image_name = rgb_image_path.stem
                thermal_left_path = thermal_left_dir / f"{image_name}.jpg"
                thermal_right_path = thermal_right_dir / f"{image_name}.jpg"
                if thermal_left_path.exists() and thermal_right_path.exists():
                    thermal_left_paths.append(thermal_left_path)
                    thermal_right_paths.append(thermal_right_path)
            
            print(f"    Preloading {len(thermal_left_paths)} thermal left images (batch size: 20)...")
            start_load_left = time.perf_counter()
            thermal_left_images = []
            batch_size = 20
            for batch_start in range(0, len(thermal_left_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(thermal_left_paths))
                batch_paths = thermal_left_paths[batch_start:batch_end]
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    batch_images = list(executor.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), batch_paths))
                thermal_left_images.extend(batch_images)
            load_left_time = time.perf_counter() - start_load_left
            print(f"      Loaded in {load_left_time:.2f}s ({load_left_time/len(thermal_left_images)*1000:.1f}ms/image)")
            
            print(f"    Preloading {len(thermal_right_paths)} thermal right images (batch size: 20)...")
            start_load_right = time.perf_counter()
            thermal_right_images = []
            batch_size = 20
            for batch_start in range(0, len(thermal_right_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(thermal_right_paths))
                batch_paths = thermal_right_paths[batch_start:batch_end]
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    batch_images = list(executor.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), batch_paths))
                thermal_right_images.extend(batch_images)
            load_right_time = time.perf_counter() - start_load_right
            print(f"      Loaded in {load_right_time:.2f}s ({load_right_time/len(thermal_right_images)*1000:.1f}ms/image)")
            
            # Create video writers
            raw_left_container = None
            raw_left_stream = None
            raw_right_container = None
            raw_right_stream = None
            combined_container = None
            combined_stream = None
            bg_containers = {}
            bg_streams = {}
            
            if need_raw_left:
                raw_left_container = av.open(str(thermal_left_video_path), 'w')
                raw_left_stream = raw_left_container.add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                raw_left_stream.width = width
                raw_left_stream.height = height
                raw_left_stream.pix_fmt = 'yuv420p'
            
            if need_raw_right:
                raw_right_container = av.open(str(thermal_right_video_path), 'w')
                raw_right_stream = raw_right_container.add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                raw_right_stream.width = width
                raw_right_stream.height = height
                raw_right_stream.pix_fmt = 'yuv420p'
            
            if need_combined:
                combined_container = av.open(str(thermal_combined_video_path), 'w')
                combined_stream = combined_container.add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                combined_stream.width = width
                combined_stream.height = height
                combined_stream.pix_fmt = 'yuv420p'
            
            for bg in backgrounds_needed:
                bg_containers[bg] = av.open(str(thermal_bg_paths[bg]), 'w')
                bg_streams[bg] = bg_containers[bg].add_stream(
                    'libx264', rate=30,
                    options={'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
                )
                bg_streams[bg].width = width
                bg_streams[bg].height = height
                bg_streams[bg].pix_fmt = 'yuv420p'
            
            # Pre-compute INFERNO default color outside loop
            inferno_color = cv2.applyColorMap(np.array([[110]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
            default_bgr = tuple(int(x) for x in inferno_color)
            
            # Process all frames
            start_encode = time.perf_counter()
            for idx in range(len(thermal_left_images)):
                rgb_bgr = rgb_images[idx]
                thermal_left_bgr = thermal_left_images[idx]
                thermal_right_bgr = thermal_right_images[idx]
                
                # Generate raw left video
                if need_raw_left:
                    img = thermal_left_bgr
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    for packet in raw_left_stream.encode(frame):
                        raw_left_container.mux(packet)
                
                # Generate raw right video
                if need_raw_right:
                    img = thermal_right_bgr
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                    for packet in raw_right_stream.encode(frame):
                        raw_right_container.mux(packet)
                
                # Optimize: if both masked and combined are needed, compute combined first and reuse
                if need_combined and backgrounds_needed:
                    # Compute combined version (full thermal overlay with no mask)
                    _, combined_result, _ = overlay_thermal_on_image_optimized_bgr(
                        thermal_generator=thermal_generator,
                        bgr_image=rgb_bgr,
                        mask=None,  # No mask for combined
                        thermal_left_bgr_image=thermal_left_bgr,
                        thermal_right_bgr_image=thermal_right_bgr,
                        alpha=1.0,
                        default_color=default_bgr
                    )
                    
                    # Save combined video
                    frame = av.VideoFrame.from_ndarray(combined_result, format='bgr24')
                    for packet in combined_stream.encode(frame):
                        combined_container.mux(packet)
                    
                    # For masked version, composite combined result with original RGB using SAM3 mask
                    if idx in all_masks:
                        mask_bool = all_masks[idx].astype(bool)
                        
                        # Create masked variants by compositing with different backgrounds
                        if 'dark' in backgrounds_needed:
                            dark_base = np.zeros_like(rgb_bgr, dtype=np.uint8)
                            dark_base[mask_bool] = combined_result[mask_bool]
                            frame = av.VideoFrame.from_ndarray(dark_base, format='bgr24')
                            for packet in bg_streams['dark'].encode(frame):
                                bg_containers['dark'].mux(packet)
                        
                        if 'color' in backgrounds_needed:
                            color_base = rgb_bgr.copy()
                            color_base[mask_bool] = combined_result[mask_bool]
                            frame = av.VideoFrame.from_ndarray(color_base, format='bgr24')
                            for packet in bg_streams['color'].encode(frame):
                                bg_containers['color'].mux(packet)
                        
                        if 'grey' in backgrounds_needed:
                            grey_base = cv2.cvtColor(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                            grey_base[mask_bool] = combined_result[mask_bool]
                            frame = av.VideoFrame.from_ndarray(grey_base, format='bgr24')
                            for packet in bg_streams['grey'].encode(frame):
                                bg_containers['grey'].mux(packet)
                
                elif need_combined:
                    # Only combined needed, no masked variants
                    _, combined_result, _ = overlay_thermal_on_image_optimized_bgr(
                        thermal_generator=thermal_generator,
                        bgr_image=rgb_bgr,
                        mask=None,  # No mask for combined
                        thermal_left_bgr_image=thermal_left_bgr,
                        thermal_right_bgr_image=thermal_right_bgr,
                        alpha=1.0,
                        default_color=default_bgr
                    )
                    frame = av.VideoFrame.from_ndarray(combined_result, format='bgr24')
                    for packet in combined_stream.encode(frame):
                        combined_container.mux(packet)
                
                elif backgrounds_needed and idx in all_masks:
                    # Only masked needed, no combined
                    mask = (all_masks[idx].astype(np.uint8) * 255)
                    
                    # Generate all 3 variants at once (internally creates dark/grey backgrounds)
                    dark_result, color_result, grey_result = overlay_thermal_on_image_optimized_bgr(
                        thermal_generator=thermal_generator,
                        bgr_image=rgb_bgr,
                        mask=mask,
                        thermal_left_bgr_image=thermal_left_bgr,
                        thermal_right_bgr_image=thermal_right_bgr,
                        alpha=1.0,
                        default_color=default_bgr
                    )
                    
                    # Write each variant to its container
                    if 'dark' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(dark_result, format='bgr24')
                        for packet in bg_streams['dark'].encode(frame):
                            bg_containers['dark'].mux(packet)
                    
                    if 'color' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(color_result, format='bgr24')
                        for packet in bg_streams['color'].encode(frame):
                            bg_containers['color'].mux(packet)
                    
                    if 'grey' in backgrounds_needed:
                        frame = av.VideoFrame.from_ndarray(grey_result, format='bgr24')
                        for packet in bg_streams['grey'].encode(frame):
                            bg_containers['grey'].mux(packet)
            encode_time = time.perf_counter() - start_encode
            print(f"      Encoded in {encode_time:.2f}s ({encode_time/len(thermal_left_images)*1000:.1f}ms/frame)")
            
            # Flush and close
            if need_raw_left:
                for packet in raw_left_stream.encode():
                    raw_left_container.mux(packet)
                raw_left_container.close()
            
            if need_raw_right:
                for packet in raw_right_stream.encode():
                    raw_right_container.mux(packet)
                raw_right_container.close()
            
            if need_combined:
                for packet in combined_stream.encode():
                    combined_container.mux(packet)
                combined_container.close()
            
            for bg in backgrounds_needed:
                for packet in bg_streams[bg].encode():
                    bg_containers[bg].mux(packet)
                bg_containers[bg].close()
            
            total_time = time.perf_counter() - start_time
            frames_count = len(thermal_left_images)
            variants = []
            if need_raw_left:
                variants.append('left raw')
            if need_raw_right:
                variants.append('right raw')
            if need_combined:
                variants.append('combined')
            if backgrounds_needed:
                variants.append(f'{len(backgrounds_needed)} masked bg')
            print(f"    ✓ Saved thermal {', '.join(variants)} ({frames_count} frames in {total_time:.2f}s, {total_time/frames_count*1000:.1f}ms/frame)")
        else:
            print("  Thermal videos already exist, skipping")
    
    # Cleanup at end of episode to prevent memory issues
    all_masks.clear()
    rgb_images.clear()
    import gc
    gc.collect()
    
    # Report total episode processing time
    total_episode_time = time.perf_counter() - episode_start_time
    print(f"\n  ✓ Episode complete: {total_episode_time:.2f}s total")


def process_data_folder(data_folder_path: Path, video_predictor):
    """
    Process all episodes in a data folder.
    
    Args:
        data_folder_path: Path to data folder (e.g., acoustic_data)
        video_predictor: SAM3 video predictor instance
    """
    data_type = data_folder_path.name
    print(f"\n{'='*80}")
    print(f"Processing data folder: {data_type}")
    print(f"{'='*80}")
    
    # Get all episode folders
    episode_folders = sorted([d for d in data_folder_path.iterdir() if d.is_dir()])
    # episode_folders = [data_folder_path.joinpath('2026_01_14_13_24_57_011')]
    
    if not episode_folders:
        print(f"Warning: no episode folders found in {data_folder_path}")
        return
    
    print(f"Found {len(episode_folders)} episode folders")
    
    # Process each episode with periodic predictor restart to prevent memory corruption
    for idx, episode_folder in enumerate(episode_folders, 1):
        # Restart predictor every 30 episodes to prevent memory corruption
        if idx > 1 and (idx - 1) % 30 == 0:
            print(f"\n{'='*80}")
            print(f"Restarting SAM3 predictor after {idx-1} episodes to prevent memory corruption...")
            print(f"{'='*80}")
            
            # Force aggressive cleanup
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Rebuild predictor
            del video_predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            from sam3.model_builder import build_sam3_video_predictor
            video_predictor = build_sam3_video_predictor()
            print("SAM3 predictor restarted successfully\n")
        
        try:
            process_episode_folder(episode_folder, data_type, video_predictor, idx, len(episode_folders))
            
            # Force cleanup between episodes to prevent memory corruption
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Error processing episode {episode_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup after error as well
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue


def main():
    # Configuration - modify these as needed
    data_root = Path('<PROJECT_PATH>')
    data_folders_to_process = ['acoustic_data', 'thermal_data', 'radar_data']  # None = process all *_data folders, or specify list like ['acoustic_data', 'thermal_data']
    
    # Setup paths
    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")
    
    print(f"Data root: {data_root}")
    
    # Initialize SAM3 video predictor
    print("\nInitializing SAM3 video predictor...")
    video_predictor = build_sam3_video_predictor()
    
    # Find data folders to process
    if data_folders_to_process:
        data_folders = [data_root / folder for folder in data_folders_to_process]
    else:
        # Process all folders ending with '_data'
        data_folders = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.endswith('_data')])
    
    if not data_folders:
        print("No data folders found to process")
        return
    
    print(f"\nData folders to process: {[f.name for f in data_folders]}")
    
    # Process each data folder
    for data_folder in data_folders:
        if not data_folder.exists():
            print(f"Warning: data folder does not exist: {data_folder}")
            continue
        
        process_data_folder(data_folder, video_predictor)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
