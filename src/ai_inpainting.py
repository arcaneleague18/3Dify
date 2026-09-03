import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import AutoPipelineForInpainting

# Global variable to cache the pipeline
_inpainting_pipeline = None

def get_inpainting_pipeline():
    global _inpainting_pipeline
    if _inpainting_pipeline is None:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32 # mps float16 support can be spotty for some ops
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"Loading inpainting pipeline on device: {device} with dtype: {dtype}")
        _inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
        )
        
        # Offload to CPU if we have CUDA to save VRAM
        if device == "cuda":
            _inpainting_pipeline.enable_model_cpu_offload()
        else:
            _inpainting_pipeline.to(device)

    return _inpainting_pipeline


def inpaint_gaps(image: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    """
    Inpaints the given gaps in the image using Stable Diffusion.
    image: BGR numpy array
    gap_mask: boolean numpy array (True where inpainting is needed)
    Returns: inpainted BGR numpy array
    """
    if not np.any(gap_mask):
        return image

    h, w = image.shape[:2]

    # Convert BGR -> RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SD models expect dimensions that are multiples of 8.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    ph, pw = h + pad_h, w + pad_w
    
    padded_img = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
    padded_mask_uint8 = np.pad(gap_mask.astype(np.uint8) * 255, ((0, pad_h), (0, pad_w)), mode='constant')

    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray(padded_mask_uint8)

    pipe = get_inpainting_pipeline()
    
    # Prompt for seamless background completion
    prompt = "seamless background, high quality, sharp focus, continuous pattern"
    negative_prompt = "artifacts, blur, distortion, text, watermark"

    print(f"Running AI Inpainting at {pw}x{ph}...")
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_img,
        mask_image=pil_mask,
        height=ph,
        width=pw,
        num_inference_steps=20,
        strength=0.99,
    ).images[0]

    # Convert back to OpenCV format
    result_np = np.array(result_img)
    
    # Resize back if the pipeline returned a different size
    if result_np.shape[0] != ph or result_np.shape[1] != pw:
        result_np = cv2.resize(result_np, (pw, ph), interpolation=cv2.INTER_LINEAR)
    
    # Crop padding back to original size
    result_np = result_np[:h, :w]
        
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    
    # Paste the original unmasked pixels back (just in case the model modified them slightly)
    final_img = image.copy()
    final_img[gap_mask] = result_bgr[gap_mask]

    return final_img
