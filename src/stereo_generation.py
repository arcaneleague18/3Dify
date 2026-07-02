import cv2
import numpy as np
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
) -> np.ndarray:
    """
    Generates a right-eye view by mapping right-image pixels back to the left image.
    Crucially, it detects and rejects "stretched" pixels on steep depth slopes
    (disocclusions), allowing these gaps to be cleanly filled by the true background.
    """
    h, w = left_img.shape[:2]
    # --- 1. Smooth depth & compute disparity ---
    depth_smoothed = smooth_depth_map(depth_map, smoothing)
    disp = (depth_smoothed * stereo_strength * w).astype(np.float32)
    disp = np.clip(disp, 0, max_disparity)

    x_map_2d = np.full((h, w), -1, dtype=np.int32)
    # --- 2. Inverse map with stretch rejection ---
    for y in range(h):
        d = disp[y]
        x_L = np.arange(w)
        # target x_R for each x_L
        x_R = np.round(x_L - d).astype(int)
        # Detect stretched pixels (where depth decreases sharply, i.e. foreground->background)
        # We reject pixels on slopes where disparity drops by > 0.5 px per pixel.
        dd = np.diff(d, append=d[-1])
        valid_L = dd >= -0.5
        # Painter's algorithm: sort by depth (so foreground overwrites background)
        order = np.argsort(d)
            
        for i in order:
            if not valid_L[i]:
                continue  # Reject stretched slope pixels
            pos = x_R[i]
            if 0 <= pos < w:
                x_map_2d[y, pos] = i
        # --- 3. Fill gaps with pure background ---
        # Right-to-left fill: disocclusions happen on the right of foreground objects.
        valid_idx = np.where(x_map_2d[y] != -1)[0]
        if len(valid_idx) == 0:
            x_map_2d[y, :] = 0
            continue
            
        hole_idx = np.where(x_map_2d[y] == -1)[0]
        if len(hole_idx) > 0:
            idx = np.searchsorted(valid_idx, hole_idx)
            # Offset by 5 pixels to safely sample pure background.
            # This skips the noisy foreground outline (which might have bled into the background
            # due to depth map misalignment) and prevents horizontal dark streaks.
            idx = np.clip(idx + 5, 0, len(valid_idx) - 1)
            x_map_2d[y, hole_idx] = x_map_2d[y, valid_idx[idx]]
    # --- 4. Fast 2D Color Mapping ---
    y_idx = np.arange(h)[:, None]
    right_img = left_img[y_idx, x_map_2d]
    return right_img