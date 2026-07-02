import gradio as gr
import cv2
import numpy as np
from config import (
    DEFAULT_STEREO_STRENGTH,
    DEFAULT_MAX_DISPARITY,
    DEFAULT_INPAINTING_RADIUS,
    DEFAULT_DEPTH_SMOOTHING,
    OUTPUT_FORMATS
)
from depth_estimation import estimate_depth
from stereo_generation import generate_right_eye
from image_utils import format_stereo_output
def create_ui():
    with gr.Blocks(title="2D to 3D Stereoscopic Image Converter") as app:
        gr.Markdown("# 2D to 3D Stereoscopic Image Converter")
        gr.Markdown("Convert any 2D image into a 3D stereoscopic view using local AI depth estimation.")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input 2D Image", type="numpy")
                
                with gr.Accordion("Stereo Parameters", open=True):
                    stereo_strength = gr.Slider(0.01, 1.00, value=DEFAULT_STEREO_STRENGTH, step=0.01, label="Stereo Strength", info="Intensity of the 3D effect (how far apart the left/right views are shifted).")
                    max_disparity = gr.Slider(10, 100, value=DEFAULT_MAX_DISPARITY, step=1, label="Maximum Disparity", info="Maximum horizontal shift (in pixels) for the closest foreground objects.")
                    focal_plane = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Focal Plane (Zero Parallax)", info="0.0 = Background stays still, foreground pops out. 1.0 = Foreground stays still, background pushes in.")
                    depth_intensity = gr.Slider(0.1, 3.0, value=2.0, step=0.1, label="Depth Intensity (Gamma)", info="Adjusts the non-linear contrast of the depth map. Lower values bring the background closer, higher values push mid-tones deeper.")
                    smoothing = gr.Slider(0, 15, value=DEFAULT_DEPTH_SMOOTHING, step=1, label="Depth Smoothing", info="Bilateral filter size to reduce noise and rough edges on depth boundaries.")
                    inpaint_radius = gr.Slider(1, 10, value=DEFAULT_INPAINTING_RADIUS, step=1, label="Inpainting Radius", info="Hole-filling size for pixel-shifting occluded areas.")
                    
                output_format = gr.Dropdown(choices=OUTPUT_FORMATS, value="Anaglyph (Red/Cyan)", label="Output Format")
                
                generate_btn = gr.Button("Generate 3D Image", variant="primary")
            
            with gr.Column():
                final_output = gr.Image(label="Final Stereoscopic Output")
                with gr.Accordion("Intermediate Steps", open=False):
                    depth_output = gr.Image(label="Estimated Depth Map")
                    left_output = gr.Image(label="Left Eye View")
                    right_output = gr.Image(label="Right Eye View")
        
        # State to cache depth map
        depth_state = gr.State(None)
        
        def process_image(img, s_strength, m_disp, f_plane, d_intensity, smooth, i_rad, out_fmt, current_depth):
            if img is None:
                return None, None, None, None, current_depth
            
            if current_depth is None:
                try:
                    # img is RGB from Gradio, OpenCV expects BGR internally.
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    depth = estimate_depth(img_bgr)
                    current_depth = depth
                except Exception as e:
                    raise gr.Error(f"Depth estimation failed: {str(e)}")
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Apply depth intensity (gamma) adjustment
            adjusted_depth = current_depth
            if d_intensity != 1.0:
                adjusted_depth = np.power(current_depth, d_intensity)

            # Generate right eye
            right_bgr = generate_right_eye(img_bgr, adjusted_depth, s_strength, m_disp, smooth, i_rad, f_plane)
            
            # Left eye is original
            left_bgr = img_bgr
            
            # Format output
            out_left, out_right = format_stereo_output(left_bgr, right_bgr, out_fmt)
            
            # If the format is a combined format, out_left contains the combined image and out_right is empty or redundant
            if "Separate" in out_fmt:
                final_bgr = out_left # Just show left in final if separate.
            else:
                final_bgr = out_left
                
            # Convert back to RGB for Gradio
            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB) if len(final_bgr.shape) == 3 else final_bgr
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
            
            # Depth for display (0-255 grayscale)
            depth_display = (adjusted_depth * 255).astype(np.uint8)
            
            return final_rgb, depth_display, left_rgb, right_rgb, current_depth

        # When image changes, reset depth state
        input_image.change(
            fn=lambda: None,
            inputs=None,
            outputs=depth_state
        )
        generate_btn.click(
            fn=process_image,
            inputs=[input_image, stereo_strength, max_disparity, focal_plane, depth_intensity, smoothing, inpaint_radius, output_format, depth_state],
            outputs=[final_output, depth_output, left_output, right_output, depth_state]
        )
    return app
def launch_ui():
    app = create_ui()
    app.launch(inbrowser=True)
if __name__ == "__main__":
    launch_ui()                    