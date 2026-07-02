import os
# Model Configuration
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
# Stereo Generation Defaults
DEFAULT_STEREO_STRENGTH = 1.00
DEFAULT_MAX_DISPARITY = 70
DEFAULT_INPAINTING_RADIUS = 5
DEFAULT_DEPTH_SMOOTHING = 3
# Supported output formats
OUTPUT_FORMATS = [
    "Side-by-side (SBS)",
    "Half Side-by-side",
    "Over-under (Top/Bottom)",
    "Separate (Left/Right)",
    "Anaglyph (Red/Cyan)"
]
