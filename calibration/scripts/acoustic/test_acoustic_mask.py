#!/usr/bin/env python3
"""
Test script for acoustic mask generation functionality.

Usage:
    python test_acoustic_mask.py \
        --rgb /path/to/rgb.jpg \
        --acoustic /path/to/acoustic.jpg \
        --masks /path/to/image_masks.npz \
        --homography /path/to/homography_rgb_to_acoustic.npy \
        --output /path/to/output.jpg \
        --alpha 1.0
"""

import sys
import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import time

# Add calibration root to path
calibration_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(calibration_root))

from src.acoustic.acoustic_mask_generator import AcousticMaskGenerator


def load_masks_from_npz(npz_path: str) -> np.ndarray:
    """
    Load masks from .npz file.

    Args:
        npz_path: Path to .npz file containing masks

    Returns:
        masks_np: Array of shape [N, H, W] where N is number of objects
    """
    data = np.load(npz_path)
    masks_np = data['masks']  # Shape: [N, 1, H, W]

    # Squeeze the channel dimension: [N, 1, H, W] -> [N, H, W]
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(axis=1)

    print(f"Loaded {len(masks_np)} masks from {npz_path}")
    print(f"Mask shape: {masks_np.shape}")

    return masks_np


def combine_masks(masks_np: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Combine multiple masks into a single binary mask.

    Args:
        masks_np: Array of shape [N, H, W] with mask probabilities
        threshold: Threshold for binarization

    Returns:
        Combined binary mask of shape [H, W], dtype=uint8
    """
    binary_masks = (masks_np > threshold).astype(np.uint8)
    combined = np.any(binary_masks, axis=0).astype(np.uint8) * 255

    print(f"Combined {len(masks_np)} masks into single mask")
    print(f"Combined mask coverage: {np.sum(combined > 0)} pixels")

    return combined


def load_homography(path: str) -> np.ndarray:
    """Load homography matrix from .npy or .json file."""
    if path.endswith('.npy'):
        return np.load(path)
    else:
        import json
        with open(path) as f:
            return np.array(json.load(f), dtype=np.float32)


def visualize_results(
    rgb_image: np.ndarray,
    masks: np.ndarray,
    result: np.ndarray,
    sensor_image: np.ndarray = None,
    sensor_name: str = "Acoustic",
    output_path: str = None
):
    """
    Create visualization with input images and result.

    Layout:
    +-------------------+-------------------+
    |   RGB Input       |   Sensor Result   |
    +-------------------+-------------------+
    |   Combined Mask   |   Mask Overlay    |
    +-------------------+-------------------+
    """
    # Create combined mask visualization
    combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8) * 255
    mask_vis = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # Create colored mask overlay on RGB
    mask_overlay = rgb_image.copy()
    for mask in masks:
        mask_2d = (mask > 0.5).astype(np.uint8)
        color = np.random.randint(0, 255, 3).tolist()
        mask_overlay[mask_2d > 0] = (
            0.5 * mask_overlay[mask_2d > 0] +
            0.5 * np.array(color, dtype=np.uint8)
        )

    # Resize all to same height
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, target_h))

    max_h = 720
    rgb_resized = resize_to_height(rgb_image, max_h)
    result_resized = resize_to_height(result, max_h)
    mask_resized = resize_to_height(mask_vis, max_h)
    overlay_resized = resize_to_height(mask_overlay, max_h)

    # Stack images
    top_row = np.hstack([rgb_resized, result_resized])
    bottom_row = np.hstack([mask_resized, overlay_resized])
    combined = np.vstack([top_row, bottom_row])

    # Add labels
    cv2.putText(combined, "RGB Input", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"{sensor_name} Overlay Result",
                (rgb_resized.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"Combined Mask ({len(masks)} objects)",
                (10, max_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Mask Overlay",
                (mask_resized.shape[1] + 10, max_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"\nVisualization saved to: {output_path}")
    else:
        cv2.imshow(f"{sensor_name} Mask Results", combined)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Generate acoustic-masked images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_acoustic_mask.py \\
    --rgb data/rgb.jpg \\
    --acoustic data/acoustic.jpg \\
    --masks data/rgb_masks.npz \\
    --homography calibration/homography_rgb_to_acoustic.npy \\
    --output output/result.jpg \\
    --alpha 1.0
        """
    )

    # Required arguments
    parser.add_argument("--rgb", type=str, required=True,
                       help="Path to RGB image")
    parser.add_argument("--acoustic", type=str, required=True,
                       help="Path to acoustic image")
    parser.add_argument("--masks", type=str, required=True,
                       help="Path to masks .npz file")
    parser.add_argument("--homography", type=str, required=True,
                       help="Path to RGB->acoustic homography matrix (.npy or .json)")

    # Optional arguments
    parser.add_argument("--output", type=str,
                       help="Output path for result image")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Blending factor (0=only RGB, 1=only acoustic)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Mask threshold for binarization")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization image")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Acoustic Mask Generator")
    print("="*60)

    # Load images
    print("\nLoading images...")
    rgb_image = cv2.imread(args.rgb)
    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image: {args.rgb}")
    print(f"  RGB image: {rgb_image.shape}")

    acoustic_image = cv2.imread(args.acoustic)
    if acoustic_image is None:
        raise ValueError(f"Failed to load acoustic image: {args.acoustic}")
    print(f"  Acoustic image: {acoustic_image.shape}")

    # Load masks
    print("\nLoading masks...")
    masks_np = load_masks_from_npz(args.masks)

    # Load homography matrix
    print("\nLoading homography matrix...")
    homography = load_homography(args.homography)
    print(f"  Homography: {homography.shape}")

    # Create generator
    print("\nInitializing acoustic mask generator...")
    generator = AcousticMaskGenerator(rgb_to_acoustic_transform=homography)

    # Combine all masks
    print("\nProcessing masks...")
    combined_mask = combine_masks(masks_np, threshold=args.threshold)

    # Generate acoustic overlay
    print("\nGenerating acoustic overlay...")
    start_time = time.time()
    result = generator.overlay_acoustic_on_image(
        rgb_image=rgb_image,
        rgb_mask=combined_mask,
        acoustic_image=acoustic_image,
        alpha=args.alpha
    )
    elapsed = time.time() - start_time
    print(f"  Generation completed in {elapsed:.3f}s")

    # Save result
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(calibration_root, "output", "acoustic_mask")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"result_{timestamp}.jpg")

    cv2.imwrite(args.output, result)
    print(f"\nResult saved to: {args.output}")

    # Generate visualization if requested
    if args.visualize:
        vis_path = args.output.replace('.jpg', '_visualization.jpg')
        visualize_results(
            rgb_image=rgb_image,
            masks=masks_np,
            result=result,
            sensor_image=acoustic_image,
            sensor_name="Acoustic",
            output_path=vis_path
        )

    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
