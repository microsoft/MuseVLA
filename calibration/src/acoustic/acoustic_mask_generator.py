#!/usr/bin/env python3
"""
Acoustic Mask Generator - Overlay acoustic values on RGB images using binary masks.

This module extends SensorMaskGenerator with acoustic-specific default color
using cv2.COLORMAP_COOL for out-of-bounds pixels.
"""

import cv2
import numpy as np
from typing import Optional, List, Union


# Get COOL colormap minimum color (value 0)
_COOL_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_COOL)
COOL_MIN_COLOR = _COOL_LUT[0, 0]  # BGR color for value 0


class _MinimalTransformer:
    """Minimal coordinate transformer that accepts homography matrix directly."""

    def __init__(self, homography_matrix: np.ndarray):
        self.H = np.array(homography_matrix, dtype=np.float32)
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")

    def transform(self, points: Union[List, np.ndarray]) -> List:
        points = np.array(points, dtype=np.float32)
        is_single = points.ndim == 1

        if is_single:
            points = points.reshape(1, -1)

        n_points = len(points)
        homo = np.ones((n_points, 3), dtype=np.float32)
        homo[:, :2] = points

        transformed = (self.H @ homo.T).T
        result = transformed[:, :2] / transformed[:, 2:3]

        if is_single:
            return result[0].tolist()
        else:
            return result.tolist()


class AcousticMaskGenerator:
    """
    Generate acoustic-masked RGB images by overlaying acoustic values on masked regions.

    When RGB coordinates map outside the acoustic image bounds, fills with
    COOL colormap minimum color instead of keeping original RGB.
    """

    def __init__(
        self,
        rgb_to_acoustic_transform: np.ndarray,
        default_color: np.ndarray = None
    ):
        """
        Initialize the acoustic mask generator.

        Args:
            rgb_to_acoustic_transform: 3x3 homography matrix for RGB->acoustic
            default_color: Default BGR color for out-of-bounds pixels.
                          Defaults to COOL minimum color.
        """
        if rgb_to_acoustic_transform is None:
            raise ValueError("Transformation matrix must be provided")

        self.transformer = _MinimalTransformer(rgb_to_acoustic_transform)
        self.default_color = default_color if default_color is not None else COOL_MIN_COLOR

    def _is_valid_coord(self, x: float, y: float, image: np.ndarray) -> bool:
        h, w = image.shape[:2]
        x_int = int(round(x))
        y_int = int(round(y))
        return 0 <= x_int < w and 0 <= y_int < h

    def _extract_pixel_value(
        self,
        image: np.ndarray,
        x: float,
        y: float
    ) -> np.ndarray:
        """
        Extract pixel value from image, return default color if out of bounds.
        """
        if not self._is_valid_coord(x, y, image):
            return self.default_color

        x_int = int(round(x))
        y_int = int(round(y))
        return image[y_int, x_int]

    def overlay_acoustic_on_image(
        self,
        rgb_image: np.ndarray,
        rgb_mask: np.ndarray,
        acoustic_image: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Overlay acoustic values on RGB image based on binary mask.

        When coordinates are out of bounds, uses COOL minimum color
        instead of keeping the original RGB pixel.

        Args:
            rgb_image: RGB image (H, W, 3), dtype=uint8
            rgb_mask: Binary mask (H, W), dtype=uint8 or bool
            acoustic_image: Acoustic image (H_a, W_a, 3), dtype=uint8
            alpha: Blending factor (0=only RGB, 1=only acoustic)

        Returns:
            Acoustic-masked RGB image (H, W, 3), dtype=uint8
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

        if acoustic_image is None:
            raise ValueError("Acoustic image must be provided")

        result = rgb_image.copy()

        # Normalize mask to binary
        if rgb_mask.dtype == bool:
            mask_binary = rgb_mask.astype(np.uint8) * 255
        else:
            mask_binary = (rgb_mask > 0).astype(np.uint8) * 255

        mask_coords = np.argwhere(mask_binary > 0)

        if len(mask_coords) == 0:
            return result

        rgb_points = mask_coords[:, [1, 0]].astype(np.float32)
        acoustic_coords = self.transformer.transform(rgb_points.tolist())
        acoustic_coords = np.array(acoustic_coords, dtype=np.float32)

        for i, (rgb_y, rgb_x) in enumerate(mask_coords):
            acoustic_x, acoustic_y = acoustic_coords[i]

            # Extract acoustic value (returns default color if out of bounds)
            acoustic_value = self._extract_pixel_value(acoustic_image, acoustic_x, acoustic_y)

            # Blend acoustic with RGB
            original_pixel = rgb_image[rgb_y, rgb_x].astype(np.float32)
            acoustic_pixel = acoustic_value.astype(np.float32)
            blended_pixel = (
                alpha * acoustic_pixel + (1 - alpha) * original_pixel
            ).astype(np.uint8)

            result[rgb_y, rgb_x] = blended_pixel

        return result


def overlay_acoustic_on_image(
    acoustic_image: np.ndarray,
    rgb_image: np.ndarray,
    rgb_mask: np.ndarray,
    rgb_to_acoustic_transform: np.ndarray,
    alpha: float = 1.0,
    default_color: np.ndarray = None
) -> np.ndarray:
    """
    Standalone function to overlay acoustic values on RGB image.

    Args:
        acoustic_image: Acoustic image (H_a, W_a, 3)
        rgb_image: RGB image (H, W, 3)
        rgb_mask: Binary mask (H, W)
        rgb_to_acoustic_transform: 3x3 homography matrix
        alpha: Blending factor (0=only RGB, 1=only acoustic)
        default_color: Default BGR color for out-of-bounds pixels

    Returns:
        Acoustic-masked RGB image (H, W, 3)
    """
    generator = AcousticMaskGenerator(
        rgb_to_acoustic_transform=rgb_to_acoustic_transform,
        default_color=default_color
    )

    return generator.overlay_acoustic_on_image(
        rgb_image=rgb_image,
        rgb_mask=rgb_mask,
        acoustic_image=acoustic_image,
        alpha=alpha
    )
