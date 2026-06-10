#!/usr/bin/env python3
"""
Homography Calculator for Acoustic Calibration

Calculates homography matrix between RGB and acoustic images using annotated points.

Usage:
    python homography_calculator.py <data_dir>

Description:
    - <data_dir>: Directory containing rgb.yaml and acoustic.yaml annotation files
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
        sensor_type: acoustic
        images:
          - path: 000_acoustic.png
            points: [[x1, y1], [x2, y2], ...]
          - path: 001_acoustic.png
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
        prefix: Filename prefix (e.g., 'rgb_to_acoustic')
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
        description='Calculate homography between RGB and acoustic images'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing rgb.yaml and acoustic.yaml annotation files')
    parser.add_argument('--ransac-threshold', type=float, default=10.0,
                       help='RANSAC reprojection threshold (default: 10.0)')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory does not exist: {args.data_dir}")
        return

    # Construct annotation file paths
    rgb_annotations = os.path.join(args.data_dir, "rgb.yaml")
    acoustic_annotations = os.path.join(args.data_dir, "acoustic.yaml")

    # Check input files exist
    if not os.path.exists(rgb_annotations):
        print(f"Error: RGB annotation file not found: {rgb_annotations}")
        return

    if not os.path.exists(acoustic_annotations):
        print(f"Error: Acoustic annotation file not found: {acoustic_annotations}")
        return

    # Generate timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"calibration_{timestamp}")

    try:
        # Load annotation points
        print(f"Loading RGB annotations: {rgb_annotations}")
        rgb_points = load_multi_image_annotations(rgb_annotations)
        print(f"RGB annotation points: {len(rgb_points)}")

        print(f"Loading acoustic annotations: {acoustic_annotations}")
        acoustic_points = load_multi_image_annotations(acoustic_annotations)
        print(f"Acoustic annotation points: {len(acoustic_points)}")

        if len(rgb_points) != len(acoustic_points):
            print(f"Warning: RGB and acoustic point counts differ ({len(rgb_points)} vs {len(acoustic_points)})")
            print("Using minimum point count")
            min_points = min(len(rgb_points), len(acoustic_points))
            rgb_points = rgb_points[:min_points]
            acoustic_points = acoustic_points[:min_points]

        # Calculate RGB->acoustic homography
        print(f"\nCalculating RGB -> acoustic homography (RANSAC threshold: {args.ransac_threshold})...")
        homography_rgb_to_acoustic = calculate_homography(
            rgb_points, acoustic_points, args.ransac_threshold
        )
        save_homography(homography_rgb_to_acoustic, output_dir, 'rgb_to_acoustic')

        # Calculate acoustic->RGB homography (inverse)
        print("\nCalculating acoustic -> RGB homography (inverse)...")
        homography_acoustic_to_rgb = np.linalg.inv(homography_rgb_to_acoustic)
        save_homography(homography_acoustic_to_rgb, output_dir, 'acoustic_to_rgb')

        # Verification: Calculate acoustic->RGB homography directly
        print("\nVerification: Computing acoustic -> RGB homography directly...")
        homography_acoustic_to_rgb_verify = calculate_homography(
            acoustic_points, rgb_points, args.ransac_threshold
        )
        save_homography(homography_acoustic_to_rgb_verify, output_dir, 'acoustic_to_rgb_verify')

        print("\nHomography calculation complete!")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
