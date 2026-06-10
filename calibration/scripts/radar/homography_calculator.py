#!/usr/bin/env python3
"""
Homography Calculator for Radar Calibration

Calculates homography matrix between RGB and radar images using annotated points.

Usage:
    python homography_calculator.py <data_dir>

Description:
    - <data_dir>: Directory containing rgb.yaml and radar.yaml annotation files
    - Output saved to <data_dir>/calibration_<timestamp>/
"""

import cv2
import numpy as np
import yaml
import json
import argparse
import os
import time


def load_multi_image_annotations(yaml_path):
    """
    Load annotations from multi-image YAML format.

    Expected format:
        sensor_type: radar
        images:
          - path: 000_radar.png
            points: [[x1, y1], [x2, y2], ...]
          - path: 001_radar.png
            points: [[x1, y1], [x2, y2], ...]
        total_points: N

    Returns:
        List of all points concatenated from all images
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    all_points = []
    images = data.get('images', [])

    for img_data in images:
        points = img_data.get('points', [])
        all_points.extend(points)

    if len(all_points) < 4:
        raise ValueError(f"Insufficient annotation points (need at least 4, got {len(all_points)})")

    return np.array(all_points, dtype=np.float32)


def calculate_homography(src_points, dst_points, ransac_threshold=10.0):
    """
    Calculate homography matrix using RANSAC.

    Args:
        src_points: Source image points (N, 2)
        dst_points: Destination image points (N, 2)
        ransac_threshold: RANSAC reprojection threshold (default: 10.0, larger than thermal's 5.0)

    Returns:
        3x3 homography matrix
    """
    if len(src_points) != len(dst_points):
        raise ValueError("Source and destination point counts must match")

    if len(src_points) < 4:
        raise ValueError("At least 4 corresponding points required for homography calculation")

    # Use RANSAC for robust homography estimation
    homography, mask = cv2.findHomography(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    if homography is None:
        raise ValueError("Failed to compute homography. Check if annotation points correspond correctly.")

    inliers = np.sum(mask)
    print(f"Homography computed: {inliers}/{len(src_points)} inliers")

    return homography


def save_homography(homography, output_dir, prefix):
    """
    Save homography matrix in JSON and .npy formats.

    Args:
        homography: 3x3 homography matrix
        output_dir: Output directory
        prefix: Filename prefix (e.g., 'rgb_to_radar')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_dir, f'homography_{prefix}.json')
    homography_list = homography.tolist()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(homography_list, f, indent=2)
    print(f"Saved JSON format: {json_path}")

    # Save as .npy
    npy_path = os.path.join(output_dir, f'homography_{prefix}.npy')
    np.save(npy_path, homography)
    print(f"Saved NumPy format: {npy_path}")

    return json_path, npy_path


def main():
    parser = argparse.ArgumentParser(
        description='Calculate homography between RGB and radar images'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing rgb.yaml and radar.yaml annotation files')
    parser.add_argument('--ransac-threshold', type=float, default=10.0,
                       help='RANSAC reprojection threshold (default: 10.0)')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory does not exist: {args.data_dir}")
        return

    # Construct annotation file paths
    rgb_annotations = os.path.join(args.data_dir, "rgb.yaml")
    radar_annotations = os.path.join(args.data_dir, "radar.yaml")

    # Check input files exist
    if not os.path.exists(rgb_annotations):
        print(f"Error: RGB annotation file not found: {rgb_annotations}")
        return

    if not os.path.exists(radar_annotations):
        print(f"Error: Radar annotation file not found: {radar_annotations}")
        return

    # Generate timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"calibration_{timestamp}")

    try:
        # Load annotation points
        print(f"Loading RGB annotations: {rgb_annotations}")
        rgb_points = load_multi_image_annotations(rgb_annotations)
        print(f"RGB annotation points: {len(rgb_points)}")

        print(f"Loading radar annotations: {radar_annotations}")
        radar_points = load_multi_image_annotations(radar_annotations)
        print(f"Radar annotation points: {len(radar_points)}")

        if len(rgb_points) != len(radar_points):
            print(f"Warning: RGB and radar point counts differ ({len(rgb_points)} vs {len(radar_points)})")
            print("Using minimum point count")
            min_points = min(len(rgb_points), len(radar_points))
            rgb_points = rgb_points[:min_points]
            radar_points = radar_points[:min_points]

        # Calculate RGB->radar homography
        print(f"\nCalculating RGB -> radar homography (RANSAC threshold: {args.ransac_threshold})...")
        homography_rgb_to_radar = calculate_homography(
            rgb_points, radar_points, args.ransac_threshold
        )
        save_homography(homography_rgb_to_radar, output_dir, 'rgb_to_radar')

        # Calculate radar->RGB homography (inverse)
        print("\nCalculating radar -> RGB homography (inverse)...")
        homography_radar_to_rgb = np.linalg.inv(homography_rgb_to_radar)
        save_homography(homography_radar_to_rgb, output_dir, 'radar_to_rgb')

        # Verification: Calculate radar->RGB homography directly
        print("\nVerification: Computing radar -> RGB homography directly...")
        homography_radar_to_rgb_verify = calculate_homography(
            radar_points, rgb_points, args.ransac_threshold
        )
        save_homography(homography_radar_to_rgb_verify, output_dir, 'radar_to_rgb_verify')

        print("\nHomography calculation complete!")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
