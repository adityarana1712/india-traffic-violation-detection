import cv2
from config import VIDEO_SOURCE
from detector import IndiaTrafficMonitor


def main():
  cap = cv2.VideoCapture(VIDEO_SOURCE)

  if not cap.isOpened():
    print(f"[ERROR] Could not open video feed source: {VIDEO_SOURCE}")
    return

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  if fps <= 0:
    fps = 30  # Fallback frame rate

  monitor = IndiaTrafficMonitor()

  # Create a resizable window so you can scale it down with your mouse
  cv2.namedWindow("India Traffic Enforcement System", cv2.WINDOW_NORMAL)
  # Optional: Set an initial window size (Width, Height) if it's too huge
  cv2.resizeWindow("India Traffic Enforcement System", 960, 540)

  print(
      "[INFO] India Smart Traffic Monitoring System Initialized. Press 'q' to"
      " exit."
  )

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("[INFO] End of video stream.")
      break

    processed_frame = monitor.process_frame(frame, fps)

    # Display live monitoring interface
    cv2.imshow("India Traffic Enforcement System", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()