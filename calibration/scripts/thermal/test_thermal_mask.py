#!/usr/bin/env python3
"""
Test script for thermal mask generation functionality.

Usage:
    python test_thermal_mask.py \
        --rgb /path/to/rgb.jpg \
        --left-thermal /path/to/left_thermal.jpg \
        --right-thermal /path/to/right_thermal.jpg \
        --masks /path/to/image_masks.npz \
        --left-homography /path/to/left_homography.npy \
        --right-homography /path/to/right_homography.npy \
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

from src.thermal.thermal_mask_generator import ThermalMaskGenerator


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
    # Binarize each mask
    binary_masks = (masks_np > threshold).astype(np.uint8)

    # Combine all masks (any pixel that's in any mask)
    combined = np.any(binary_masks, axis=0).astype(np.uint8) * 255

    print(f"Combined {len(masks_np)} masks into single mask")
    print(f"Combined mask coverage: {np.sum(combined > 0)} pixels")

    return combined


def process_single_mask(
    mask: np.ndarray,
    rgb_image: np.ndarray,
    left_thermal: np.ndarray,
    right_thermal: np.ndarray,
    generator: ThermalMaskGenerator,
    alpha: float = 1.0,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Process a single mask and overlay thermal values.

    Args:
        mask: Single mask of shape [H, W] with probabilities
        rgb_image: RGB image
        left_thermal: Left thermal image
        right_thermal: Right thermal image
        generator: ThermalMaskGenerator instance
        alpha: Blending factor
        threshold: Threshold for mask binarization

    Returns:
        Result image with thermal overlay
    """
    # Binarize mask
    mask_binary = (mask > threshold).astype(np.uint8) * 255

    # Apply thermal overlay
    result = generator.overlay_thermal_on_image(
        rgb_image=rgb_image,
        rgb_mask=mask_binary,
        left_thermal_image=left_thermal,
        right_thermal_image=right_thermal,
        alpha=alpha
    )

    return result


def visualize_results(
    rgb_image: np.ndarray,
    masks: np.ndarray,
    result: np.ndarray,
    left_thermal: np.ndarray = None,
    right_thermal: np.ndarray = None,
    output_path: str = None
):
    """
    Create visualization with input images and result.

    Layout:
    +-------------------+-------------------+
    |   RGB Input       |   Thermal Result  |
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
    cv2.putText(combined, "Thermal Overlay Result",
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
        cv2.imshow("Thermal Mask Results", combined)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Generate thermal-masked images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with all masks combined
  python test_thermal_mask.py \\
    --rgb data/tmp/test_image/rgb.jpg \\
    --left-thermal data/tmp/test_image/left_thermal.jpg \\
    --right-thermal data/tmp/test_image/right_thermal.jpg \\
    --masks data/tmp/test_image/rgb_masks.npz \\
    --left-homography homography/left_H.npy \\
    --right-homography homography/right_H.npy \\
    --output output/result.jpg \\
    --alpha 1.0
        """
    )

    # Required arguments
    parser.add_argument("--rgb", type=str, required=True,
                       help="Path to RGB image")
    parser.add_argument("--left-thermal", type=str, required=True,
                       help="Path to left thermal image")
    parser.add_argument("--right-thermal", type=str, required=True,
                       help="Path to right thermal image")
    parser.add_argument("--masks", type=str, required=True,
                       help="Path to masks .npz file")
    parser.add_argument("--left-homography", type=str, required=True,
                       help="Path to left homography matrix (.npy or .json)")
    parser.add_argument("--right-homography", type=str, required=True,
                       help="Path to right homography matrix (.npy or .json)")

    # Optional arguments
    parser.add_argument("--output", type=str,
                       help="Output path for result image")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Blending factor (0=only RGB, 1=only thermal)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Mask threshold for binarization")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization image")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Thermal Mask Generator")
    print("="*60)

    # Load images
    print("\nLoading images...")
    rgb_image = cv2.imread(args.rgb)
    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image: {args.rgb}")
    print(f"✓ RGB image: {rgb_image.shape}")

    left_thermal = cv2.imread(args.left_thermal)
    if left_thermal is None:
        raise ValueError(f"Failed to load left thermal: {args.left_thermal}")
    print(f"✓ Left thermal: {left_thermal.shape}")

    right_thermal = cv2.imread(args.right_thermal)
    if right_thermal is None:
        raise ValueError(f"Failed to load right thermal: {args.right_thermal}")
    print(f"✓ Right thermal: {right_thermal.shape}")

    # Load masks
    print("\nLoading masks...")
    masks_np = load_masks_from_npz(args.masks)

    # Load homography matrices
    print("\nLoading homography matrices...")
    if args.left_homography.endswith('.npy'):
        left_H = np.load(args.left_homography)
    else:
        import json
        with open(args.left_homography) as f:
            left_H = np.array(json.load(f), dtype=np.float32)
    print(f"✓ Left homography: {left_H.shape}")

    if args.right_homography.endswith('.npy'):
        right_H = np.load(args.right_homography)
    else:
        import json
        with open(args.right_homography) as f:
            right_H = np.array(json.load(f), dtype=np.float32)
    print(f"✓ Right homography: {right_H.shape}")

    # Create generator
    print("\nInitializing thermal mask generator...")
    generator = ThermalMaskGenerator(
        rgb_to_left_thermal_transform=left_H,
        rgb_to_right_thermal_transform=right_H
    )

    # Combine all masks
    print("\nProcessing masks...")
    combined_mask = combine_masks(masks_np, threshold=args.threshold)

    # Generate thermal overlay
    print("\nGenerating thermal overlay...")
    start_time = time.time()
    result = generator.overlay_thermal_on_image(
        rgb_image=rgb_image,
        rgb_mask=combined_mask,
        left_thermal_image=left_thermal,
        right_thermal_image=right_thermal,
        alpha=args.alpha
    )
    elapsed = time.time() - start_time
    print(f"✓ Generation completed in {elapsed:.3f}s")

    # Save result
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(calibration_root, "output", "thermal_mask")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"result_{timestamp}.jpg")

    cv2.imwrite(args.output, result)
    print(f"\n✓ Result saved to: {args.output}")

    # Generate visualization if requested
    if args.visualize:
        vis_path = args.output.replace('.jpg', '_visualization.jpg')
        visualize_results(
            rgb_image=rgb_image,
            masks=masks_np,
            result=result,
            left_thermal=left_thermal,
            right_thermal=right_thermal,
            output_path=vis_path
        )

    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
