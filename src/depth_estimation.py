import numpy as np
from PIL import Image
import cv2
import torch
from transformers import pipeline
from config import DEPTH_MODEL_NAME
# Global variable to cache the pipeline
_depth_pipeline = None
def get_depth_pipeline():
    global _depth_pipeline
    if _depth_pipeline is None:
        # Determine device
        if torch.cuda.is_available():
            device = 0
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_type = "mps"
        else:
            device = -1
            device_type = "cpu"
        print(f"Loading depth estimation pipeline on device: {device_type}")
        _depth_pipeline = pipeline(
            "depth-estimation",
            model=DEPTH_MODEL_NAME,
            device=device,
        )
    return _depth_pipeline
def estimate_depth(image: np.ndarray) -> np.ndarray:
    """
    Estimates the depth of the given image using Depth Anything V2.
    Expects a BGR numpy array (OpenCV format).
    Returns a normalized [0, 1] float32 depth map where
    **1.0 = closest to camera, 0.0 = farthest from camera**.
    """
    h, w = image.shape[:2]

    # Convert BGR → RGB PIL for the HF pipeline
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    # Run inference
    pipe = get_depth_pipeline()
    result = pipe(pil_img)
    # ---- Use the raw high-precision tensor instead of the 8-bit PIL image ----
    raw_depth = result["predicted_depth"]           # torch.Tensor [1, H_model, W_model]
    if isinstance(raw_depth, torch.Tensor):
        depth_map = raw_depth.squeeze().cpu().numpy().astype(np.float32)
    else:
        depth_map = np.array(raw_depth, dtype=np.float32)
    # Resize to the original image resolution
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 0:
        depth_map = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_map = np.zeros_like(depth_map)
    # Depth Anything V2 outputs disparity-like values where
    # higher = closer.  Our convention is the same (1 = closest),
    # so no inversion is needed.
    print(f"Depth map stats — min: {depth_map.min():.4f}, max: {depth_map.max():.4f}, "
          f"mean: {depth_map.mean():.4f}")
    return depth_map
