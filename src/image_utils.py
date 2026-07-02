import cv2
import numpy as np
def create_sbs(left_img, right_img, half_width=False):
    """
    Creates a Side-by-Side (SBS) stereoscopic image.
    """
    h, w, c = left_img.shape
    if half_width:
        left_img = cv2.resize(left_img, (w // 2, h))
        right_img = cv2.resize(right_img, (w // 2, h))
    return np.hstack((left_img, right_img))
def create_over_under(left_img, right_img):
    """
    Creates an Over-Under (Top/Bottom) stereoscopic image.
    """
    return np.vstack((left_img, right_img))
def create_anaglyph(left_img, right_img):
    """
    Creates a Red/Cyan Anaglyph stereoscopic image.
    left_img is used for Red channel.
    right_img is used for Green and Blue channels.
    """
    anaglyph = np.zeros_like(left_img)
    # OpenCV uses BGR format
    anaglyph[:, :, 2] = left_img[:, :, 2]   # Red from Left
    anaglyph[:, :, 1] = right_img[:, :, 1]  # Green from Right
    anaglyph[:, :, 0] = right_img[:, :, 0]  # Blue from Right
    return anaglyph
def format_stereo_output(left_img, right_img, output_format):
    """
    Formats the left and right images based on the requested output format.
    """
    if "Separate" in output_format:
        return left_img, right_img
    elif "Half Side-by-side" in output_format:
        return create_sbs(left_img, right_img, half_width=True), None
    elif "Side-by-side" in output_format:
        return create_sbs(left_img, right_img, half_width=False), None
    elif "Over-under" in output_format:
        return create_over_under(left_img, right_img), None
    elif "Anaglyph" in output_format:
        return create_anaglyph(left_img, right_img), None
    else:
        return left_img, right_img                                