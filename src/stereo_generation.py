import cv2
import numpy as np
from typing import Tuple

def smooth_depth_map(depth_map: np.ndarray, smoothing: int = 5) -> np.ndarray:
    """
    Applies bilateral filtering to smooth depth while preserving edges.
    """
    if smoothing > 0:
        depth_8u = (depth_map * 255).astype(np.uint8)
        smoothed_8u = cv2.bilateralFilter(depth_8u, d=smoothing, sigmaColor=75, sigmaSpace=75)
        return smoothed_8u.astype(np.float32) / 255.0
    return depth_map

def generate_right_eye(
    left_img: np.ndarray,
    depth_map: np.ndarray,
    stereo_strength: float = 0.14,
    max_disparity: int = 70,
    smoothing: int = 3,
    inpaint_radius: int = 5,
    focal_plane: float = 0.0,
    fill_mode: str = "algorithmic",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a right-eye view by mapping right-image pixels back to the left image.
    Crucially, it detects and rejects "stretched" pixels on steep depth slopes
    (disocclusions), allowing these gaps to be cleanly filled by the true background.
    """
    h, w = left_img.shape[:2]
    # --- 1. Smooth depth & compute disparity ---
    depth_smoothed = smooth_depth_map(depth_map, smoothing)
    # Calculate disparity relative to the focal plane.
    # Positive disp = pops out (shifts left), Negative disp = pushes in (shifts right)
    disp = ((depth_smoothed - focal_plane) * stereo_strength * w).astype(np.float32)
    disp = np.clip(disp, -max_disparity, max_disparity)

    x_map_2d = np.full((h, w), -1, dtype=np.int32)

    # --- 2. Inverse map with stretch rejection (no filling yet) ---
    for y in range(h):
        d = disp[y]
        x_L = np.arange(w)
        # target x_R for each x_L
        x_R = np.round(x_L - d).astype(int)
        # Detect stretched pixels (where depth decreases sharply, i.e. foreground->background)
        # We reject pixels on slopes where disparity drops by > 0.3 px per pixel.
        dd = np.diff(d, append=d[-1])
        valid_L = dd >= -0.3
        # Painter's algorithm: sort by depth (so foreground overwrites background)
        order = np.argsort(d)

        for i in order:
            if not valid_L[i]:
                continue  # Reject stretched slope pixels
            pos = x_R[i]
            if 0 <= pos < w:
                x_map_2d[y, pos] = i

    # --- 3. Dilate gaps to uniformly eat outline pixels ---
    # The dark foreground outline bleeds 3-10px into the background depending
    # on the row. Instead of per-row offsets (inconsistent → streaks), we dilate
    # the gap mask horizontally. This converts ALL outline pixels to gaps on
    # every row uniformly. Then a simple nearest-right fill lands on guaranteed
    # clean background.
    gap_mask = (x_map_2d == -1).astype(np.uint8)
    kernel = np.ones((1, 15), np.uint8)  # 7px dilation each side, horizontal only
    gap_dilated = cv2.dilate(gap_mask, kernel, iterations=1)
    x_map_2d[gap_dilated > 0] = -1

    final_gap_mask = (x_map_2d == -1)

    # --- 4. Fill all gaps from nearest right (background) neighbor if algorithmic ---
    if fill_mode == "algorithmic":
        for y in range(h):
            valid_idx = np.where(x_map_2d[y] != -1)[0]
            if len(valid_idx) == 0:
                x_map_2d[y, :] = 0
                continue
            hole_idx = np.where(x_map_2d[y] == -1)[0]
            if len(hole_idx) > 0:
                idx = np.searchsorted(valid_idx, hole_idx)
                idx = np.clip(idx, 0, len(valid_idx) - 1)
                x_map_2d[y, hole_idx] = x_map_2d[y, valid_idx[idx]]

    # --- 5. Fast 2D Color Mapping ---
    y_idx = np.arange(h)[:, None]
    
    # Use np.maximum to prevent index out of bounds during mapping for -1 values
    safe_map = np.maximum(x_map_2d, 0)
    right_img = left_img[y_idx, safe_map]

    if fill_mode == "ai":
        # Black out the unfilled gaps for AI mode (AI will inpaint these)
        right_img[final_gap_mask] = 0

    return right_img, final_gap_mask