# 3Dify: 2D to 3D Stereoscopic Image Converter

**3Dify** is a powerful Python application that converts standard 2D images into immersive 3D stereoscopic views. By leveraging state-of-the-art local AI depth estimation models and custom stereo-generation algorithms, it allows you to dynamically generate and customize 3D effects from any flat image.

## 🌟 Key Features

### 1. State-of-the-Art Depth Estimation
The project uses the `depth-anything/Depth-Anything-V2-Small-hf` model via the Hugging Face `transformers` pipeline. This provides highly accurate depth maps where foreground objects and background landscapes are distinctly separated, ensuring a clean 3D effect. The pipeline automatically utilizes GPU acceleration (`cuda` or `mps`) if available, falling back to CPU if not.

### 2. Advanced Stereo Generation Algorithm
Rather than simple pixel shifting, the core engine in `stereo_generation.py` uses a custom inverse mapping technique with occlusion handling:
- **Disocclusion Rejection**: It detects "stretched" pixels on steep depth slopes (where depth decreases sharply) and rejects them.
- **Background Hole Filling**: Gaps created by foreground objects shifting are intelligently filled. The algorithm scans horizontally to sample pure background pixels (offsetting to avoid bleeding from noisy foreground outlines), preventing ugly stretching artifacts.
- **Bilateral Smoothing**: Applies OpenCV bilateral filtering (`cv2.bilateralFilter`) to the depth map to smooth noise while preserving sharp object edges.

### 3. Fully Customizable Parameters
Through the interactive UI, you can tweak the 3D generation to perfection:
- **Stereo Strength**: Controls the intensity of the 3D effect (i.e., how far apart the left/right views are shifted based on the depth map).
- **Maximum Disparity**: The absolute maximum horizontal pixel shift applied to the closest foreground objects.
- **Depth Smoothing**: Size of the bilateral filter to reduce noise on depth boundaries.
- **Inpainting Radius**: Controls the heuristic for hole-filling pixel-shifted occluded areas.

### 4. Multiple Output Formats
Whether you are using a VR headset, a 3D TV, or just red/cyan glasses, this tool supports multiple formats:
- **Anaglyph (Red/Cyan)**
- **Side-by-side (SBS)**
- **Half Side-by-side**
- **Over-under (Top/Bottom)**
- **Separate (Left/Right views)**

### 5. Interactive Web Interface
Built with **Gradio**, the UI provides immediate visual feedback. You can view the final 3D output alongside intermediate steps like the raw depth map and the isolated Left/Right eye views.

---

## 🚀 Installation & Setup

This project requires **Python 3.10+**. 

### Step 1: Clone the Repository
```bash
git clone https://github.com/arcaneleague18/3Dify.git
cd 3d
```

### Step 2: Install Dependencies
It is highly recommended to use a virtual environment. The project is pre-configured for `uv` (a fast Python package installer and resolver), but standard `pip` works perfectly as well.

**Using `uv` (Recommended):**
```bash
uv sync
```

**Using standard `pip`:**
```bash
python -m venv .venv
# Activate virtual env:
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r src/requirements.txt
```

---

## 🎮 Usage

To launch the Gradio web interface, simply run:

```bash
python src/main.py
```
*(Depending on your entry point, you can also run `python src/ui.py` directly).*

1. Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your web browser.
2. **Upload** a standard 2D image.
3. **Adjust** the stereo parameters (Strength, Disparity, Smoothing).
4. **Select** your desired output format from the dropdown.
5. Click **"Generate 3D Image"** and view your results!

---

## 📁 Repository Structure

- `src/ui.py`: Gradio interface setup and event handling.
- `src/depth_estimation.py`: Inference code for generating depth maps using `transformers`.
- `src/stereo_generation.py`: Core algorithm for inverse pixel mapping and occlusion hole-filling.
- `src/image_utils.py`: Utilities for formatting the left and right eyes into Anaglyph, SBS, Over/Under, etc.
- `src/inpainting.py`: Additional functions for gap filling (if applicable).
- `src/config.py`: Global constants, model names, and default parameters.
- `src/main.py`: Entry point for the application.
