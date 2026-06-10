#!/usr/bin/env python3
"""
Coordinate Transformer - Wraps homography matrix for coordinate transformation.

Usage:
    from src.common import CoordinateTransformer

    transformer = CoordinateTransformer(
        homography_src_to_dst_path='path/to/homography.npy'
    )

    # Transform single point
    dst_point = transformer.transform_src_to_dst([100, 200])

    # Transform multiple points
    src_points = [[100, 200], [150, 250], [200, 300]]
    dst_points = transformer.transform_src_to_dst(src_points)
"""

import numpy as np
import json
import os


class CoordinateTransformer:
    """Coordinate transformer for image coordinate conversion using homography."""

    def __init__(self, homography_src_to_dst_path=None, homography_dst_to_src_path=None):
        """
        Initialize coordinate transformer.

        Args:
            homography_src_to_dst_path: Path to src->dst homography matrix (.npy or .json)
            homography_dst_to_src_path: Path to dst->src homography matrix (.npy or .json)

        Note: If only one path is provided, the inverse matrix is computed for the other direction.
        """
        self.homography_src_to_dst = None
        self.homography_dst_to_src = None

        if homography_src_to_dst_path:
            self.homography_src_to_dst = self._load_homography(homography_src_to_dst_path)

        if homography_dst_to_src_path:
            self.homography_dst_to_src = self._load_homography(homography_dst_to_src_path)
        elif self.homography_src_to_dst is not None:
            self.homography_dst_to_src = np.linalg.inv(self.homography_src_to_dst)

        if self.homography_src_to_dst is None and self.homography_dst_to_src is not None:
            self.homography_src_to_dst = np.linalg.inv(self.homography_dst_to_src)

        if self.homography_src_to_dst is None:
            raise ValueError("At least one homography matrix file path must be provided")

    def _load_homography(self, file_path):
        """Load homography matrix (.npy or .json format)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Homography matrix file not found: {file_path}")

        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported file format: {file_path}, use .npy or .json")

    def _transform_points(self, points, homography_matrix):
        """
        Transform point coordinates using homography matrix.

        Args:
            points: Point coordinates, single point [x, y] or batch [[x1, y1], [x2, y2], ...]
            homography_matrix: 3x3 homography matrix

        Returns:
            Transformed point coordinates in same format as input
        """
        points = np.array(points, dtype=np.float32)

        is_single_point = points.ndim == 1
        if is_single_point:
            points = points.reshape(1, -1)

        # Convert to homogeneous coordinates
        n_points = len(points)
        homogeneous_points = np.ones((n_points, 3), dtype=np.float32)
        homogeneous_points[:, :2] = points

        # Apply homography transformation
        transformed = (homography_matrix @ homogeneous_points.T).T

        # Normalize homogeneous coordinates
        transformed_points = transformed[:, :2] / transformed[:, 2:3]

        if is_single_point:
            return transformed_points[0].tolist()
        else:
            return transformed_points.tolist()

    def transform_src_to_dst(self, points):
        """
        Transform source image coordinates to destination image coordinates.

        Args:
            points: Source image point coordinates, single [x, y] or batch [[x1, y1], ...]

        Returns:
            Destination image point coordinates
        """
        if self.homography_src_to_dst is None:
            raise ValueError("src->dst homography matrix not loaded")

        return self._transform_points(points, self.homography_src_to_dst)

    def transform_dst_to_src(self, points):
        """
        Transform destination image coordinates to source image coordinates.

        Args:
            points: Destination image point coordinates, single [x, y] or batch [[x1, y1], ...]

        Returns:
            Source image point coordinates
        """
        if self.homography_dst_to_src is None:
            raise ValueError("dst->src homography matrix not loaded")

        return self._transform_points(points, self.homography_dst_to_src)

    # Aliases for thermal compatibility
    def transform_rgb_to_ir(self, points):
        """Alias for transform_src_to_dst (RGB -> IR/sensor)."""
        return self.transform_src_to_dst(points)

    def transform_ir_to_rgb(self, points):
        """Alias for transform_dst_to_src (IR/sensor -> RGB)."""
        return self.transform_dst_to_src(points)

    def get_homography_src_to_dst(self):
        """Get src->dst homography matrix copy."""
        return self.homography_src_to_dst.copy() if self.homography_src_to_dst is not None else None

    def get_homography_dst_to_src(self):
        """Get dst->src homography matrix copy."""
        return self.homography_dst_to_src.copy() if self.homography_dst_to_src is not None else None


def load_transformer_from_dir(homography_dir, src_name='rgb', dst_name='sensor'):
    """
    Load homography matrices from directory and create transformer.

    Args:
        homography_dir: Directory containing homography matrix files
        src_name: Source name in filename (default: 'rgb')
        dst_name: Destination name in filename (default: 'sensor')

    Returns:
        CoordinateTransformer instance
    """
    # Try common naming patterns
    patterns = [
        (f'homography_{src_name}_to_{dst_name}.npy', f'homography_{dst_name}_to_{src_name}.npy'),
        (f'homography_{src_name}_to_ir.npy', f'homography_ir_to_{src_name}.npy'),
        ('homography_rgb_to_ir.npy', 'homography_ir_to_rgb.npy'),
        ('homography_rgb_to_radar.npy', 'homography_radar_to_rgb.npy'),
        ('homography_rgb_to_acoustic.npy', 'homography_acoustic_to_rgb.npy'),
    ]

    for src_to_dst_name, dst_to_src_name in patterns:
        src_to_dst_npy = os.path.join(homography_dir, src_to_dst_name)
        src_to_dst_json = src_to_dst_npy.replace('.npy', '.json')

        if os.path.exists(src_to_dst_npy):
            src_to_dst_path = src_to_dst_npy
        elif os.path.exists(src_to_dst_json):
            src_to_dst_path = src_to_dst_json
        else:
            continue

        dst_to_src_npy = os.path.join(homography_dir, dst_to_src_name)
        dst_to_src_json = dst_to_src_npy.replace('.npy', '.json')

        dst_to_src_path = None
        if os.path.exists(dst_to_src_npy):
            dst_to_src_path = dst_to_src_npy
        elif os.path.exists(dst_to_src_json):
            dst_to_src_path = dst_to_src_json

        return CoordinateTransformer(
            homography_src_to_dst_path=src_to_dst_path,
            homography_dst_to_src_path=dst_to_src_path
        )

    raise FileNotFoundError(f"No homography matrix files found in directory: {homography_dir}")
