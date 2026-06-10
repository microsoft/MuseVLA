#!/usr/bin/env python3
"""
Thermal Mask Generator - Overlay thermal values on RGB images using binary masks

This module provides functionality to generate thermal-masked images by:
1. Transforming RGB mask coordinates to thermal coordinates
2. Extracting thermal values from left and right thermal images
3. Overlaying thermal values on the RGB image

Usage:
    from src.thermal import ThermalMaskGenerator

    generator = ThermalMaskGenerator(
        rgb_to_left_thermal_transform=left_transform,
        rgb_to_right_thermal_transform=right_transform
    )

    result = generator.overlay_thermal_on_image(
        left_thermal_image=left_ir,
        right_thermal_image=right_ir,
        rgb_image=rgb_img,
        rgb_mask=mask
    )
"""

import numpy as np
from typing import Optional, Tuple, List, Union


class _MinimalTransformer:
    """
    Minimal coordinate transformer that accepts homography matrix directly.

    This is an internal class to avoid dependency on file-based CoordinateTransformer.
    """

    def __init__(self, homography_matrix: np.ndarray):
        """
        Initialize transformer with homography matrix.

        Args:
            homography_matrix: 3x3 homography matrix for RGB->thermal transformation
        """
        self.H = np.array(homography_matrix, dtype=np.float32)
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")

    def transform_rgb_to_ir(self, points: Union[List, np.ndarray]) -> List:
        """
        Transform RGB points to thermal/IR coordinates using homography.

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
        transformed = (self.H @ homo.T).T  # Shape: (n_points, 3)

        # Normalize by third coordinate
        result = transformed[:, :2] / transformed[:, 2:3]

        # Return in same format as input
        if is_single:
            return result[0].tolist()
        else:
            return result.tolist()


class ThermalMaskGenerator:
    """
    Generate thermal-masked RGB images by overlaying thermal values on masked regions.

    This class handles coordinate transformation from RGB to thermal space and
    combines thermal values from left and right thermal cameras.
    """

    def __init__(
        self,
        rgb_to_left_thermal_transform: Optional[np.ndarray] = None,
        rgb_to_right_thermal_transform: Optional[np.ndarray] = None
    ):
        """
        Initialize the thermal mask generator.

        Args:
            rgb_to_left_thermal_transform: 3x3 homography matrix for RGB->left thermal
            rgb_to_right_thermal_transform: 3x3 homography matrix for RGB->right thermal

        Raises:
            ValueError: If neither transformation matrix is provided
        """
        # Initialize transformers
        if rgb_to_left_thermal_transform is not None:
            self.left_transformer = _MinimalTransformer(rgb_to_left_thermal_transform)
        else:
            self.left_transformer = None

        if rgb_to_right_thermal_transform is not None:
            self.right_transformer = _MinimalTransformer(rgb_to_right_thermal_transform)
        else:
            self.right_transformer = None

        if self.left_transformer is None and self.right_transformer is None:
            raise ValueError(
                "At least one transformation matrix must be provided "
                "(rgb_to_left_thermal_transform or rgb_to_right_thermal_transform)"
            )

    def _is_valid_thermal_coord(
        self,
        x: float,
        y: float,
        thermal_image: np.ndarray
    ) -> bool:
        """
        Check if thermal coordinates are within valid bounds.

        Args:
            x: X coordinate in thermal image
            y: Y coordinate in thermal image
            thermal_image: The thermal image to check bounds against

        Returns:
            True if coordinates are valid (within bounds)
        """
        h, w = thermal_image.shape[:2]
        x_int = int(round(x))
        y_int = int(round(y))
        return 0 <= x_int < w and 0 <= y_int < h

    def _extract_thermal_value(
        self,
        thermal_image: np.ndarray,
        x: float,
        y: float
    ) -> Optional[np.ndarray]:
        """
        Extract thermal value from thermal image at given coordinates.

        Args:
            thermal_image: Thermal image (H, W, C) or (H, W)
            x: X coordinate (will be rounded to int)
            y: Y coordinate (will be rounded to int)

        Returns:
            Thermal pixel value or None if out of bounds
        """
        # Check bounds
        if not self._is_valid_thermal_coord(x, y, thermal_image):
            return None

        # Round coordinates to integers
        x_int = int(round(x))
        y_int = int(round(y))

        return thermal_image[y_int, x_int]

    def overlay_thermal_on_image(
        self,
        rgb_image: np.ndarray,
        rgb_mask: np.ndarray,
        left_thermal_image: Optional[np.ndarray] = None,
        right_thermal_image: Optional[np.ndarray] = None,
        alpha: float = 1
    ) -> np.ndarray:
        """
        Overlay thermal values on RGB image based on binary mask.

        Algorithm:
        1. For each pixel in rgb_mask:
           a. Transform RGB coordinates to left/right thermal coordinates
           b. Extract thermal values (if coordinates are valid)
           c. Average thermal values if both valid, use single if only one valid
           d. Overlay on RGB image using alpha blending

        Args:
            rgb_image: RGB image (H, W, 3), dtype=uint8
            rgb_mask: Binary mask (H, W), dtype=uint8 or bool, values 0/255 or True/False
            left_thermal_image: Left thermal image (H_t, W_t, 3), dtype=uint8
            right_thermal_image: Right thermal image (H_t, W_t, 3), dtype=uint8
            alpha: Blending factor for thermal overlay (0=only RGB, 1=only thermal)

        Returns:
            Thermal-masked RGB image (H, W, 3), dtype=uint8

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

        if left_thermal_image is None and right_thermal_image is None:
            raise ValueError("At least one thermal image must be provided")

        if left_thermal_image is not None and self.left_transformer is None:
            raise ValueError("Left thermal image provided but no left transformer")

        if right_thermal_image is not None and self.right_transformer is None:
            raise ValueError("Right thermal image provided but no right transformer")

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
            return result  # No pixels to process

        # Convert to (N, 2) array of [x, y] for transformation
        rgb_points = mask_coords[:, [1, 0]].astype(np.float32)  # Swap to [x, y]

        # Transform to thermal coordinates (batch processing)
        left_thermal_coords = None
        right_thermal_coords = None

        if left_thermal_image is not None:
            left_thermal_coords = self.left_transformer.transform_rgb_to_ir(
                rgb_points.tolist()
            )
            left_thermal_coords = np.array(left_thermal_coords, dtype=np.float32)

        if right_thermal_image is not None:
            right_thermal_coords = self.right_transformer.transform_rgb_to_ir(
                rgb_points.tolist()
            )
            right_thermal_coords = np.array(right_thermal_coords, dtype=np.float32)

        # Process each pixel
        for i, (rgb_y, rgb_x) in enumerate(mask_coords):
            thermal_values = []

            # Extract left thermal value
            if left_thermal_image is not None:
                left_x, left_y = left_thermal_coords[i]
                left_value = self._extract_thermal_value(
                    left_thermal_image, left_x, left_y
                )
                if left_value is not None:
                    thermal_values.append(left_value)

            # Extract right thermal value
            if right_thermal_image is not None:
                right_x, right_y = right_thermal_coords[i]
                right_value = self._extract_thermal_value(
                    right_thermal_image, right_x, right_y
                )
                if right_value is not None:
                    thermal_values.append(right_value)

            # Combine thermal values
            if len(thermal_values) == 0:
                # No valid thermal values, keep original RGB pixel
                continue
            elif len(thermal_values) == 1:
                # Use single thermal value
                thermal_pixel = thermal_values[0]
            else:
                # Average both thermal values
                thermal_pixel = np.mean(thermal_values, axis=0).astype(np.uint8)

            # Blend thermal with RGB
            original_pixel = rgb_image[rgb_y, rgb_x].astype(np.float32)
            thermal_pixel = thermal_pixel.astype(np.float32)
            blended_pixel = (
                alpha * thermal_pixel + (1 - alpha) * original_pixel
            ).astype(np.uint8)

            result[rgb_y, rgb_x] = blended_pixel

        return result


# Standalone convenience function
def overlay_thermal_on_image(
    left_thermal_image: np.ndarray,
    right_thermal_image: np.ndarray,
    rgb_image: np.ndarray,
    rgb_mask: np.ndarray,
    rgb_to_left_thermal_transform: np.ndarray,
    rgb_to_right_thermal_transform: np.ndarray,
    alpha: float = 1
) -> np.ndarray:
    """
    Standalone function to overlay thermal values on RGB image.

    This is a convenience wrapper around ThermalMaskGenerator for simple use cases.

    Args:
        left_thermal_image: Left thermal image (H_t, W_t, 3)
        right_thermal_image: Right thermal image (H_t, W_t, 3)
        rgb_image: RGB image (H, W, 3)
        rgb_mask: Binary mask (H, W)
        rgb_to_left_thermal_transform: 3x3 homography matrix
        rgb_to_right_thermal_transform: 3x3 homography matrix
        alpha: Blending factor (0=only RGB, 1=only thermal)

    Returns:
        Thermal-masked RGB image (H, W, 3)
    """
    generator = ThermalMaskGenerator(
        rgb_to_left_thermal_transform=rgb_to_left_thermal_transform,
        rgb_to_right_thermal_transform=rgb_to_right_thermal_transform
    )

    return generator.overlay_thermal_on_image(
        rgb_image=rgb_image,
        rgb_mask=rgb_mask,
        left_thermal_image=left_thermal_image,
        right_thermal_image=right_thermal_image,
        alpha=alpha
    )
