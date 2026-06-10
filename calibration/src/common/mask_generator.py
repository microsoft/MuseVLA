#!/usr/bin/env python3
"""
Sensor Mask Generator - Overlay sensor values on RGB images using binary masks.

This module provides functionality to generate sensor-masked images by:
1. Transforming RGB mask coordinates to sensor coordinates
2. Extracting sensor values from the sensor image
3. Overlaying sensor values on the RGB image

Usage:
    from src.common import SensorMaskGenerator

    generator = SensorMaskGenerator(rgb_to_sensor_transform=H)

    result = generator.overlay_sensor_on_image(
        rgb_image=rgb_img,
        rgb_mask=mask,
        sensor_image=sensor_img,
        alpha=1.0
    )
"""

import numpy as np
from typing import Optional, List, Union


class _MinimalTransformer:
    """
    Minimal coordinate transformer that accepts homography matrix directly.
    """

    def __init__(self, homography_matrix: np.ndarray):
        """
        Initialize transformer with homography matrix.

        Args:
            homography_matrix: 3x3 homography matrix for RGB->sensor transformation
        """
        self.H = np.array(homography_matrix, dtype=np.float32)
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")

    def transform(self, points: Union[List, np.ndarray]) -> List:
        """
        Transform RGB points to sensor coordinates using homography.

        Args:
            points: Single point [x, y] or batch [[x1, y1], [x2, y2], ...]

        Returns:
            Transformed points in same format as input
        """
        points = np.array(points, dtype=np.float32)
        is_single = points.ndim == 1

        if is_single:
            points = points.reshape(1, -1)

        # Convert to homogeneous coordinates [x, y, 1]
        n_points = len(points)
        homo = np.ones((n_points, 3), dtype=np.float32)
        homo[:, :2] = points

        # Apply homography transformation
        transformed = (self.H @ homo.T).T

        # Normalize by third coordinate
        result = transformed[:, :2] / transformed[:, 2:3]

        if is_single:
            return result[0].tolist()
        else:
            return result.tolist()


class SensorMaskGenerator:
    """
    Generate sensor-masked RGB images by overlaying sensor values on masked regions.

    This is a simplified version for single-sensor systems (radar, acoustic),
    unlike ThermalMaskGenerator which handles dual cameras.
    """

    def __init__(self, rgb_to_sensor_transform: np.ndarray):
        """
        Initialize the sensor mask generator.

        Args:
            rgb_to_sensor_transform: 3x3 homography matrix for RGB->sensor

        Raises:
            ValueError: If transformation matrix is not provided
        """
        if rgb_to_sensor_transform is None:
            raise ValueError("Transformation matrix must be provided")

        self.transformer = _MinimalTransformer(rgb_to_sensor_transform)

    def _is_valid_coord(self, x: float, y: float, image: np.ndarray) -> bool:
        """
        Check if coordinates are within valid image bounds.

        Args:
            x: X coordinate
            y: Y coordinate
            image: Image to check bounds against

        Returns:
            True if coordinates are valid (within bounds)
        """
        h, w = image.shape[:2]
        x_int = int(round(x))
        y_int = int(round(y))
        return 0 <= x_int < w and 0 <= y_int < h

    def _extract_pixel_value(self, image: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        """
        Extract pixel value from image at given coordinates.

        Args:
            image: Image (H, W, C) or (H, W)
            x: X coordinate (will be rounded to int)
            y: Y coordinate (will be rounded to int)

        Returns:
            Pixel value or None if out of bounds
        """
        if not self._is_valid_coord(x, y, image):
            return None

        x_int = int(round(x))
        y_int = int(round(y))

        return image[y_int, x_int]

    def overlay_sensor_on_image(
        self,
        rgb_image: np.ndarray,
        rgb_mask: np.ndarray,
        sensor_image: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Overlay sensor values on RGB image based on binary mask.

        Algorithm:
        1. For each pixel in rgb_mask:
           a. Transform RGB coordinates to sensor coordinates
           b. Extract sensor value (if coordinates are valid)
           c. Overlay on RGB image using alpha blending

        Args:
            rgb_image: RGB image (H, W, 3), dtype=uint8
            rgb_mask: Binary mask (H, W), dtype=uint8 or bool
            sensor_image: Sensor image (H_s, W_s, 3), dtype=uint8
            alpha: Blending factor (0=only RGB, 1=only sensor)

        Returns:
            Sensor-masked RGB image (H, W, 3), dtype=uint8

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"rgb_image must be (H, W, 3), got {rgb_image.shape}")

        if rgb_mask.ndim != 2:
            raise ValueError(f"rgb_mask must be 2D, got {rgb_mask.ndim}D")

        if rgb_image.shape[:2] != rgb_mask.shape:
            raise ValueError(
                f"rgb_image and rgb_mask dimensions must match: "
                f"{rgb_image.shape[:2]} vs {rgb_mask.shape}"
            )

        if sensor_image is None:
            raise ValueError("Sensor image must be provided")

        # Create output image (copy of RGB)
        result = rgb_image.copy()

        # Normalize mask to binary
        if rgb_mask.dtype == bool:
            mask_binary = rgb_mask.astype(np.uint8) * 255
        else:
            mask_binary = (rgb_mask > 0).astype(np.uint8) * 255

        # Get masked pixel coordinates
        mask_coords = np.argwhere(mask_binary > 0)  # Returns (N, 2) array of [y, x]

        if len(mask_coords) == 0:
            return result

        # Convert to (N, 2) array of [x, y] for transformation
        rgb_points = mask_coords[:, [1, 0]].astype(np.float32)

        # Transform to sensor coordinates (batch processing)
        sensor_coords = self.transformer.transform(rgb_points.tolist())
        sensor_coords = np.array(sensor_coords, dtype=np.float32)

        # Process each pixel
        for i, (rgb_y, rgb_x) in enumerate(mask_coords):
            sensor_x, sensor_y = sensor_coords[i]

            # Extract sensor value
            sensor_value = self._extract_pixel_value(sensor_image, sensor_x, sensor_y)

            if sensor_value is None:
                # Out of bounds, keep original RGB pixel
                continue

            # Blend sensor with RGB
            original_pixel = rgb_image[rgb_y, rgb_x].astype(np.float32)
            sensor_pixel = sensor_value.astype(np.float32)
            blended_pixel = (
                alpha * sensor_pixel + (1 - alpha) * original_pixel
            ).astype(np.uint8)

            result[rgb_y, rgb_x] = blended_pixel

        return result


def overlay_sensor_on_image(
    sensor_image: np.ndarray,
    rgb_image: np.ndarray,
    rgb_mask: np.ndarray,
    rgb_to_sensor_transform: np.ndarray,
    alpha: float = 1
) -> np.ndarray:
    """
    Standalone function to overlay sensor values on RGB image.

    This is a convenience wrapper around SensorMaskGenerator.

    Args:
        sensor_image: Sensor image (H_s, W_s, 3)
        rgb_image: RGB image (H, W, 3)
        rgb_mask: Binary mask (H, W)
        rgb_to_sensor_transform: 3x3 homography matrix
        alpha: Blending factor (0=only RGB, 1=only sensor)

    Returns:
        Sensor-masked RGB image (H, W, 3)
    """
    generator = SensorMaskGenerator(rgb_to_sensor_transform=rgb_to_sensor_transform)

    return generator.overlay_sensor_on_image(
        rgb_image=rgb_image,
        rgb_mask=rgb_mask,
        sensor_image=sensor_image,
        alpha=alpha
    )
