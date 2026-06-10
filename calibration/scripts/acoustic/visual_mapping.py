#!/usr/bin/env python3
"""
Visual Mapping Tool for Acoustic Calibration

Visualizes RGB -> acoustic coordinate mapping to verify calibration accuracy.

Usage:
    python visual_mapping.py <data_dir> [--no_connections]

Description:
    - <data_dir>: Directory containing images, annotations, and calibration_* subdirectory
    - Finds the latest calibration_* directory for homography matrices
    - Generates visualization for each image pair
    - Output saved to <data_dir>/visualization_<timestamp>/
"""

import cv2
import numpy as np
import yaml
import argparse
import os
import sys
import glob
import time
from pathlib import Path

# Add calibration root to path for src module imports
calibration_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(calibration_root))

from src.common.coordinate_transformer import CoordinateTransformer, load_transformer_from_dir


def find_latest_calibration_dir(data_dir):
    """
    Find the latest calibration_* directory in data_dir.

    Args:
        data_dir: Data directory path

    Returns:
        Path to latest calibration directory, or None if not found
    """
    calibration_dirs = glob.glob(os.path.join(data_dir, "calibration_*"))
    if not calibration_dirs:
        return None

    # Sort by modification time, return latest
    calibration_dirs.sort(key=os.path.getmtime, reverse=True)
    return calibration_dirs[0]


def load_multi_image_annotations(yaml_path):
    """
    Load annotations from multi-image YAML format.

    Returns:
        Dictionary mapping image basename to points list
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    annotations = {}
    images = data.get('images', [])

    for img_data in images:
        path = img_data.get('path', '')
        points = img_data.get('points', [])
        if path and points:
            annotations[path] = np.array(points, dtype=np.float32)

    return annotations


def draw_points(image, points, color=(0, 255, 0), radius=5, labels=None):
    """Draw points on image."""
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 2)

        if labels is not None:
            cv2.putText(image, str(labels[i]), (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def visualize_single_pair(
    rgb_image_path: str,
    acoustic_image_path: str,
    transformer: CoordinateTransformer,
    rgb_points: np.ndarray,
    acoustic_points: np.ndarray,
    show_connections: bool = True,
    output_path: str = None
):
    """
    Visualize RGB -> acoustic mapping for a single image pair.

    Args:
        rgb_image_path: Path to RGB image
        acoustic_image_path: Path to acoustic image
        transformer: CoordinateTransformer instance
        rgb_points: RGB annotation points (N, 2)
        acoustic_points: Acoustic annotation points (N, 2) - ground truth
        show_connections: Whether to draw connection lines
        output_path: Optional output path to save visualization
    """
    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    acoustic_image = cv2.imread(acoustic_image_path)

    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image: {rgb_image_path}")
    if acoustic_image is None:
        raise ValueError(f"Failed to load acoustic image: {acoustic_image_path}")

    # Transform RGB points to acoustic coordinates
    acoustic_points_pred = transformer.transform_src_to_dst(rgb_points.tolist())
    acoustic_points_pred = np.array(acoustic_points_pred, dtype=np.float32)

    # Create display copies
    rgb_display = rgb_image.copy()
    acoustic_display = acoustic_image.copy()

    # Draw RGB points
    draw_points(rgb_display, rgb_points, color=(0, 255, 0), radius=5)

    # Draw acoustic GT points (green) and predicted points (red)
    draw_points(acoustic_display, acoustic_points, color=(0, 255, 0), radius=5)  # GT
    draw_points(acoustic_display, acoustic_points_pred, color=(0, 0, 255), radius=4)  # Predicted

    # Resize images to match height
    max_height = max(rgb_display.shape[0], acoustic_display.shape[0])
    scale_rgb = max_height / rgb_display.shape[0]
    scale_acoustic = max_height / acoustic_display.shape[0]

    new_width_rgb = int(rgb_display.shape[1] * scale_rgb)
    new_width_acoustic = int(acoustic_display.shape[1] * scale_acoustic)

    rgb_resized = cv2.resize(rgb_display, (new_width_rgb, max_height))
    acoustic_resized = cv2.resize(acoustic_display, (new_width_acoustic, max_height))

    # Scale point coordinates
    rgb_points_scaled = rgb_points * scale_rgb
    acoustic_points_pred_scaled = acoustic_points_pred * scale_acoustic

    # Concatenate side by side
    combined = np.hstack([rgb_resized, acoustic_resized])

    # Draw connection lines if requested
    if show_connections and len(rgb_points_scaled) <= 200:
        for p_rgb, p_acoustic_pred in zip(rgb_points_scaled, acoustic_points_pred_scaled):
            x1, y1 = int(p_rgb[0]), int(p_rgb[1])
            x2, y2 = int(p_acoustic_pred[0] + new_width_rgb), int(p_acoustic_pred[1])
            cv2.line(combined, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Calculate projection error
    diffs = acoustic_points_pred - acoustic_points
    dists = np.linalg.norm(diffs, axis=1)
    mean_err = float(np.mean(dists))
    max_err = float(np.max(dists))

    # Add labels
    cv2.putText(combined, "RGB Image (source points)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Acoustic Image (GT: green, Pred: red)", (new_width_rgb + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f"Error: mean={mean_err:.2f}px, max={max_err:.2f}px", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"Saved visualization: {output_path}")

    return combined, mean_err, max_err


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RGB->acoustic homography mapping accuracy'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing images and annotation files')
    parser.add_argument('--no_connections', action='store_true',
                       help='Do not draw connection lines')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory does not exist: {args.data_dir}")
        return

    # Find required files
    rgb_yaml = os.path.join(args.data_dir, "rgb.yaml")
    acoustic_yaml = os.path.join(args.data_dir, "acoustic.yaml")

    if not os.path.exists(rgb_yaml):
        print(f"Error: RGB annotation file not found: {rgb_yaml}")
        return

    if not os.path.exists(acoustic_yaml):
        print(f"Error: Acoustic annotation file not found: {acoustic_yaml}")
        return

    # Find latest calibration directory
    homography_dir = find_latest_calibration_dir(args.data_dir)
    if not homography_dir:
        print(f"Error: No calibration_* directory found in {args.data_dir}")
        print("Please run homography_calculator.py first")
        return

    print(f"Using homography directory: {homography_dir}")

    # Generate output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"visualization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load transformer
        print(f"Loading homography matrices from: {homography_dir}")
        transformer = load_transformer_from_dir(homography_dir)
        print("Homography matrices loaded successfully")

        # Load annotations
        rgb_annotations = load_multi_image_annotations(rgb_yaml)
        acoustic_annotations = load_multi_image_annotations(acoustic_yaml)

        print(f"\nRGB annotations: {len(rgb_annotations)} images")
        print(f"Acoustic annotations: {len(acoustic_annotations)} images")

        # Match image pairs by index
        rgb_files = sorted(rgb_annotations.keys())
        acoustic_files = sorted(acoustic_annotations.keys())

        if len(rgb_files) != len(acoustic_files):
            print(f"Warning: Image count mismatch ({len(rgb_files)} RGB vs {len(acoustic_files)} acoustic)")
            n_pairs = min(len(rgb_files), len(acoustic_files))
        else:
            n_pairs = len(rgb_files)

        all_mean_errors = []
        all_max_errors = []

        for i in range(n_pairs):
            rgb_file = rgb_files[i]
            acoustic_file = acoustic_files[i]

            rgb_image_path = os.path.join(args.data_dir, rgb_file)
            acoustic_image_path = os.path.join(args.data_dir, acoustic_file)

            rgb_points = rgb_annotations[rgb_file]
            acoustic_points = acoustic_annotations[acoustic_file]

            if len(rgb_points) != len(acoustic_points):
                print(f"Warning: Point count mismatch for pair {i}: {len(rgb_points)} vs {len(acoustic_points)}")
                n_pts = min(len(rgb_points), len(acoustic_points))
                rgb_points = rgb_points[:n_pts]
                acoustic_points = acoustic_points[:n_pts]

            output_path = os.path.join(output_dir, f"mapping_{i:03d}.png")

            print(f"\nProcessing pair {i + 1}/{n_pairs}: {rgb_file} <-> {acoustic_file}")

            _, mean_err, max_err = visualize_single_pair(
                rgb_image_path=rgb_image_path,
                acoustic_image_path=acoustic_image_path,
                transformer=transformer,
                rgb_points=rgb_points,
                acoustic_points=acoustic_points,
                show_connections=not args.no_connections,
                output_path=output_path
            )

            all_mean_errors.append(mean_err)
            all_max_errors.append(max_err)
            print(f"  Projection error: mean={mean_err:.2f}px, max={max_err:.2f}px")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Processed {n_pairs} image pairs")
        print(f"Overall mean error: {np.mean(all_mean_errors):.2f} pixels")
        print(f"Overall max error: {np.max(all_max_errors):.2f} pixels")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
