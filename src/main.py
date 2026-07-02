import logging
from ui import launch_ui
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting 2D to 3D Stereoscopic Image Converter...")
    
    try:
        launch_ui()
    except Exception as e:
        logging.error(f"Application crashed: {e}")
if __name__ == "__main__":
    main()
