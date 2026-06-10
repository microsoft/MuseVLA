#!/usr/bin/env python3
"""
Visual Mapping Tool for Radar Calibration

Visualizes RGB -> radar coordinate mapping to verify calibration accuracy.

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
    radar_image_path: str,
    transformer: CoordinateTransformer,
    rgb_points: np.ndarray,
    radar_points: np.ndarray,
    show_connections: bool = True,
    output_path: str = None
):
    """
    Visualize RGB -> radar mapping for a single image pair.

    Args:
        rgb_image_path: Path to RGB image
        radar_image_path: Path to radar image
        transformer: CoordinateTransformer instance
        rgb_points: RGB annotation points (N, 2)
        radar_points: Radar annotation points (N, 2) - ground truth
        show_connections: Whether to draw connection lines
        output_path: Optional output path to save visualization
    """
    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    radar_image = cv2.imread(radar_image_path)

    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image: {rgb_image_path}")
    if radar_image is None:
        raise ValueError(f"Failed to load radar image: {radar_image_path}")

    # Transform RGB points to radar coordinates
    radar_points_pred = transformer.transform_src_to_dst(rgb_points.tolist())
    radar_points_pred = np.array(radar_points_pred, dtype=np.float32)

    # Create display copies
    rgb_display = rgb_image.copy()
    radar_display = radar_image.copy()

    # Draw RGB points
    draw_points(rgb_display, rgb_points, color=(0, 255, 0), radius=5)

    # Draw radar GT points (green) and predicted points (red)
    draw_points(radar_display, radar_points, color=(0, 255, 0), radius=5)  # GT
    draw_points(radar_display, radar_points_pred, color=(0, 0, 255), radius=4)  # Predicted

    # Resize images to match height
    max_height = max(rgb_display.shape[0], radar_display.shape[0])
    scale_rgb = max_height / rgb_display.shape[0]
    scale_radar = max_height / radar_display.shape[0]

    new_width_rgb = int(rgb_display.shape[1] * scale_rgb)
    new_width_radar = int(radar_display.shape[1] * scale_radar)

    rgb_resized = cv2.resize(rgb_display, (new_width_rgb, max_height))
    radar_resized = cv2.resize(radar_display, (new_width_radar, max_height))

    # Scale point coordinates
    rgb_points_scaled = rgb_points * scale_rgb
    radar_points_pred_scaled = radar_points_pred * scale_radar

    # Concatenate side by side
    combined = np.hstack([rgb_resized, radar_resized])

    # Draw connection lines if requested
    if show_connections and len(rgb_points_scaled) <= 200:
        for p_rgb, p_radar_pred in zip(rgb_points_scaled, radar_points_pred_scaled):
            x1, y1 = int(p_rgb[0]), int(p_rgb[1])
            x2, y2 = int(p_radar_pred[0] + new_width_rgb), int(p_radar_pred[1])
            cv2.line(combined, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Calculate projection error
    diffs = radar_points_pred - radar_points
    dists = np.linalg.norm(diffs, axis=1)
    mean_err = float(np.mean(dists))
    max_err = float(np.max(dists))

    # Add labels
    cv2.putText(combined, "RGB Image (source points)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Radar Image (GT: green, Pred: red)", (new_width_rgb + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f"Error: mean={mean_err:.2f}px, max={max_err:.2f}px", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"Saved visualization: {output_path}")

    return combined, mean_err, max_err


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RGB->radar homography mapping accuracy'
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
    radar_yaml = os.path.join(args.data_dir, "radar.yaml")

    if not os.path.exists(rgb_yaml):
        print(f"Error: RGB annotation file not found: {rgb_yaml}")
        return

    if not os.path.exists(radar_yaml):
        print(f"Error: Radar annotation file not found: {radar_yaml}")
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
        radar_annotations = load_multi_image_annotations(radar_yaml)

        print(f"\nRGB annotations: {len(rgb_annotations)} images")
        print(f"Radar annotations: {len(radar_annotations)} images")

        # Match image pairs by index
        rgb_files = sorted(rgb_annotations.keys())
        radar_files = sorted(radar_annotations.keys())

        if len(rgb_files) != len(radar_files):
            print(f"Warning: Image count mismatch ({len(rgb_files)} RGB vs {len(radar_files)} radar)")
            n_pairs = min(len(rgb_files), len(radar_files))
        else:
            n_pairs = len(rgb_files)

        all_mean_errors = []
        all_max_errors = []

        for i in range(n_pairs):
            rgb_file = rgb_files[i]
            radar_file = radar_files[i]

            rgb_image_path = os.path.join(args.data_dir, rgb_file)
            radar_image_path = os.path.join(args.data_dir, radar_file)

            rgb_points = rgb_annotations[rgb_file]
            radar_points = radar_annotations[radar_file]

            if len(rgb_points) != len(radar_points):
                print(f"Warning: Point count mismatch for pair {i}: {len(rgb_points)} vs {len(radar_points)}")
                n_pts = min(len(rgb_points), len(radar_points))
                rgb_points = rgb_points[:n_pts]
                radar_points = radar_points[:n_pts]

            output_path = os.path.join(output_dir, f"mapping_{i:03d}.png")

            print(f"\nProcessing pair {i + 1}/{n_pairs}: {rgb_file} <-> {radar_file}")

            _, mean_err, max_err = visualize_single_pair(
                rgb_image_path=rgb_image_path,
                radar_image_path=radar_image_path,
                transformer=transformer,
                rgb_points=rgb_points,
                radar_points=radar_points,
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
