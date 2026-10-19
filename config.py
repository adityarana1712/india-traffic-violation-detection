import numpy as np

# Video source: 0 for webcam, path to video file, or an RTSP stream URL
VIDEO_SOURCE = "traffic_clip.mp4"

# Model configuration
YOLO_MODEL = "yolov8n.pt"  # Automatically downloads YOLOv8 nano
CONFIDENCE_THRESHOLD = 0.5
TRACKED_CLASSES = [2, 3, 5, 7]  # COCO IDs: 2:car, 3:motorcycle, 5:bus, 7:truck

# Speed & Calibration Settings (India Urban Limits)
SPEED_LIMIT_KMH = 50
PIXEL_TO_METERS = 0.05  # Camera calibration factor

# Restricted Zone (Polygon coordinates for Red Light / Stop-Line zone)
RESTRICTED_ZONE = np.array(
    [[200, 400], [600, 400], [700, 600], [100, 600]], dtype=np.int32
)

# Storage paths for E-Challan records
SNAPSHOT_DIR = "india_challan_snapshots"
LOG_FILE = "india_e_challan_logs.csv"