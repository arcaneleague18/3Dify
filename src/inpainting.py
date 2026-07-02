import cv2
import numpy as np
class BaseInpainter:
    def inpaint(self, img, mask, radius=3):
        raise NotImplementedError("Subclasses must implement inpaint method.")
class OpenCVInpainter(BaseInpainter):
    def __init__(self, method=cv2.INPAINT_TELEA):
        self.method = method
    def inpaint(self, img, mask, radius=3):
        """
        Uses OpenCV's built-in inpainting to fill holes.
        mask should be a single-channel 8-bit image where non-zero pixels indicate the area to be inpainted.
        """
        # Ensure mask is 8-bit single channel
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask = mask.astype(np.uint8)
        
        return cv2.inpaint(img, mask, inpaintRadius=radius, flags=self.method)
