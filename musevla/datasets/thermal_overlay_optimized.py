"""
Optimized thermal overlay using vectorized operations.
This is 10-20x faster than the pixel-by-pixel loop version.
"""

import cv2
import numpy as np
import time


def overlay_thermal_on_image_optimized_bgr(
    thermal_generator,
    bgr_image: np.ndarray,
    mask: np.ndarray,
    thermal_left_bgr_image: np.ndarray,
    thermal_right_bgr_image: np.ndarray,
    alpha: float = 1.0,
    default_color: tuple = None,
    profile: bool = False
) -> tuple:
    """
    BGR-native optimized thermal overlay returning 3 background variants.
    
    Args:
        thermal_generator: ThermalMaskGenerator instance (for transformers)
        bgr_image: BGR image (H, W, 3), dtype=uint8
        mask: Binary mask (H, W), dtype=uint8 or bool
        thermal_left_bgr_image: Left thermal BGR image (H_t, W_t, 3), dtype=uint8
        thermal_right_bgr_image: Right thermal BGR image (H_t, W_t, 3), dtype=uint8
        alpha: Blending factor (0=only RGB, 1=only thermal)
        default_color: Default BGR color tuple (B, G, R) for mask pixels without thermal data
        profile: If True, print detailed timing breakdown
    
    Returns:
        Tuple of (dark_result, color_result, grey_result), each (H, W, 3), dtype=uint8
    """
    return _process_single_thermal(
        thermal_generator, bgr_image, mask,
        thermal_left_bgr_image, thermal_right_bgr_image,
        alpha, default_color, profile
    )


def _process_single_thermal(
    thermal_generator,
    bgr_image: np.ndarray,
    mask: np.ndarray,
    left_thermal_bgr: np.ndarray,
    right_thermal_bgr: np.ndarray,
    alpha: float,
    default_color: tuple,
    profile: bool
) -> tuple:
    """Internal function to process thermal overlay returning 3 background variants."""
    t_start = time.perf_counter() if profile else None
    
    # Create all 3 background bases
    dark_base = np.zeros_like(bgr_image, dtype=np.uint8)
    color_base = bgr_image.copy()
    grey_base = cv2.cvtColor(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    # Handle None mask (for combined thermal - process full image)
    t0 = time.perf_counter() if profile else None
    if mask is None:
        # Create full mask for entire image
        h, w = bgr_image.shape[:2]
        mask_binary = np.ones((h, w), dtype=np.uint8) * 255
    elif mask.dtype == bool:
        mask_binary = mask.astype(np.uint8) * 255
    else:
        mask_binary = (mask > 0).astype(np.uint8) * 255
    
    mask_coords = np.argwhere(mask_binary > 0)
    
    if len(mask_coords) == 0:
        return dark_base, color_base, grey_base
    
    # Transform all coordinates at once
    points = mask_coords[:, [1, 0]].astype(np.float32)
    
    if profile:
        t1 = time.perf_counter()
        print(f"      [Thermal] Mask processing: {(t1-t0)*1000:.2f}ms ({len(mask_coords)} pixels)")
    
    # Transform to left thermal coordinates
    t2 = time.perf_counter() if profile else None
    left_thermal_coords = None
    if left_thermal_bgr is not None:
        # Use vectorized cv2.perspectiveTransform instead of looping
        points_homogeneous = points.reshape(-1, 1, 2)
        left_thermal_coords = cv2.perspectiveTransform(
            points_homogeneous, 
            thermal_generator.left_transformer.H
        ).reshape(-1, 2)
        left_thermal_coords_int = np.round(left_thermal_coords).astype(np.int32)
        
        # Create validity mask for left
        h_l, w_l = left_thermal_bgr.shape[:2]
        left_valid = (
            (left_thermal_coords_int[:, 0] >= 0) & (left_thermal_coords_int[:, 0] < w_l) &
            (left_thermal_coords_int[:, 1] >= 0) & (left_thermal_coords_int[:, 1] < h_l)
        )
    else:
        left_valid = np.zeros(len(mask_coords), dtype=bool)
    
    if profile:
        t3 = time.perf_counter()
        print(f"      [Thermal] Left transform: {(t3-t2)*1000:.2f}ms ({left_valid.sum()} valid)")
    
    # Transform to right thermal coordinates
    t4 = time.perf_counter() if profile else None
    right_thermal_coords = None
    if right_thermal_bgr is not None:
        # Use vectorized cv2.perspectiveTransform instead of looping
        points_homogeneous = points.reshape(-1, 1, 2)
        right_thermal_coords = cv2.perspectiveTransform(
            points_homogeneous,
            thermal_generator.right_transformer.H
        ).reshape(-1, 2)
        right_thermal_coords_int = np.round(right_thermal_coords).astype(np.int32)
        
        # Create validity mask for right
        h_r, w_r = right_thermal_bgr.shape[:2]
        right_valid = (
            (right_thermal_coords_int[:, 0] >= 0) & (right_thermal_coords_int[:, 0] < w_r) &
            (right_thermal_coords_int[:, 1] >= 0) & (right_thermal_coords_int[:, 1] < h_r)
        )
    else:
        right_valid = np.zeros(len(mask_coords), dtype=bool)
    
    t6 = time.perf_counter() if profile else None
    n_pixels = len(mask_coords)
    thermal_values = np.zeros((n_pixels, 3), dtype=np.float32)
    counts = np.zeros(n_pixels, dtype=np.int32)
    
    # Add left thermal values
    if left_thermal_bgr is not None and left_valid.any():
        valid_left_coords = left_thermal_coords_int[left_valid]
        thermal_values[left_valid] += left_thermal_bgr[
            valid_left_coords[:, 1],
            valid_left_coords[:, 0]
        ].astype(np.float32)
        counts[left_valid] += 1
    
    # Add right thermal values
    if right_thermal_bgr is not None and right_valid.any():
        valid_right_coords = right_thermal_coords_int[right_valid]
        thermal_values[right_valid] += right_thermal_bgr[
            valid_right_coords[:, 1],
            valid_right_coords[:, 0]
        ].astype(np.float32)
        counts[right_valid] += 1
    
    # Average thermal values where we have valid data
    valid_any = counts > 0
    
    if profile:
        t7 = time.perf_counter()
        print(f"      [Thermal] Extract values: {(t7-t6)*1000:.2f}ms ({valid_any.sum()} with data)")
    
    # Handle pixels with valid thermal data
    t8 = time.perf_counter() if profile else None
    if valid_any.any():
        thermal_values[valid_any] /= counts[valid_any, np.newaxis]
        thermal_values = thermal_values.astype(np.uint8)
        
        # Apply thermal values to all 3 backgrounds
        valid_coords = mask_coords[valid_any]
        valid_thermal = thermal_values[valid_any]
        
        if alpha == 1.0:
            # Fast path: no blending needed
            dark_base[valid_coords[:, 0], valid_coords[:, 1]] = valid_thermal
            color_base[valid_coords[:, 0], valid_coords[:, 1]] = valid_thermal
            grey_base[valid_coords[:, 0], valid_coords[:, 1]] = valid_thermal
        else:
            # Blend with each background
            thermal_float = valid_thermal.astype(np.float32)
            
            # Dark background
            dark_orig = dark_base[valid_coords[:, 0], valid_coords[:, 1]].astype(np.float32)
            dark_blended = (alpha * thermal_float + (1 - alpha) * dark_orig).astype(np.uint8)
            dark_base[valid_coords[:, 0], valid_coords[:, 1]] = dark_blended
            
            # Color background
            color_orig = color_base[valid_coords[:, 0], valid_coords[:, 1]].astype(np.float32)
            color_blended = (alpha * thermal_float + (1 - alpha) * color_orig).astype(np.uint8)
            color_base[valid_coords[:, 0], valid_coords[:, 1]] = color_blended
            
            # Grey background
            grey_orig = grey_base[valid_coords[:, 0], valid_coords[:, 1]].astype(np.float32)
            grey_blended = (alpha * thermal_float + (1 - alpha) * grey_orig).astype(np.uint8)
            grey_base[valid_coords[:, 0], valid_coords[:, 1]] = grey_blended
    
    # Handle pixels without thermal data using default color
    if default_color is not None:
        invalid_any = ~valid_any
        if invalid_any.any():
            invalid_coords = mask_coords[invalid_any]
            default_bgr = np.array(default_color, dtype=np.uint8)
            dark_base[invalid_coords[:, 0], invalid_coords[:, 1]] = default_bgr
            color_base[invalid_coords[:, 0], invalid_coords[:, 1]] = default_bgr
            grey_base[invalid_coords[:, 0], invalid_coords[:, 1]] = default_bgr
    
    if profile:
        t9 = time.perf_counter()
        print(f"      [Thermal] Blending: {(t9-t8)*1000:.2f}ms")
        print(f"      [Thermal] TOTAL: {(t9-t_start)*1000:.2f}ms")
    
    return dark_base, color_base, grey_base
