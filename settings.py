from pathlib import Path
import sys
 
FILE = Path(__file__).resolve()
 
ROOT = FILE.parent
 
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
 
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
 

SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'test': VIDEO_DIR / 'TEST1.mov',
    'video_0': VIDEO_DIR / 'video.MOV',
    'video_3': VIDEO_DIR / 'videoplayback.mp4',
    'video_4': VIDEO_DIR / 'plongeur.mp4',
 
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
 
DETECTION_MODEL_YOLOv9 = MODEL_DIR / 'bestV9KAGGLE.pt'
#DETECTION_MODEL_YOLOv9 = MODEL_DIR / 'bestNEW.pt'
DETECTION_MODEL_YOLOv8 = MODEL_DIR / 'best.pt'
DETECTION_MODEL_FASTER = MODEL_DIR / 'FRCNN-V0-5epochs.pth'
#SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
