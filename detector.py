import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from config import (
    CONFIDENCE_THRESHOLD,
    PIXEL_TO_METERS,
    RESTRICTED_ZONE,
    SPEED_LIMIT_KMH,
    TRACKED_CLASSES,
    YOLO_MODEL,
)
from logger import IndiaEChallanLogger


class IndiaTrafficMonitor:

  def __init__(self):
    self.model = YOLO(YOLO_MODEL)
    self.logger = IndiaEChallanLogger()
    self.vehicle_positions = {}
    self.logged_violations = set()

    print(
        "[INFO] Initializing EasyOCR reader for Indian vehicle plates (HSRP)..."
    )
    self.ocr_reader = easyocr.Reader(["en"], gpu=False)

  def extract_plate_number(self, vehicle_crop):
    try:
      h, w, _ = vehicle_crop.shape
      plate_region = vehicle_crop[int(h * 0.5) : h, int(w * 0.1) : int(w * 0.9)]
      results = self.ocr_reader.readtext(plate_region)

      for bbox, text, prob in results:
        if prob > 0.35:
          cleaned = "".join(e for e in text if e.isalnum())
          if len(cleaned) >= 6:
            return cleaned
    except Exception:
      pass
    return "UNKNOWN"

  def calculate_speed(self, prev_pos, curr_pos, fps):
    pixel_distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    distance_meters = pixel_distance * PIXEL_TO_METERS
    speed_ms = distance_meters * fps
    return speed_ms * 3.6

  def process_frame(self, frame, fps):
    results = self.model.track(
        frame,
        persist=True,
        classes=TRACKED_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    cv2.polylines(
        frame, [RESTRICTED_ZONE], isClosed=True, color=(0, 0, 255), thickness=2
    )

    if results[0].boxes.id is not None:
      boxes = results[0].boxes.xyxy.cpu().numpy()
      track_ids = results[0].boxes.id.cpu().numpy()

      for box, track_id in zip(boxes, track_ids):
        track_id = int(track_id)
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = int((x1 + x2) / 2), int(y2)

        is_inside = (
            cv2.pointPolygonTest(
                RESTRICTED_ZONE, (center_x, center_y), False
            )
            >= 0
        )

        current_pos = (center_x, center_y)
        if track_id in self.vehicle_positions:
          prev_pos = self.vehicle_positions[track_id]["pos"]
          speed = self.calculate_speed(prev_pos, current_pos, fps)
        else:
          speed = 0.0

        self.vehicle_positions[track_id] = {"pos": current_pos}

        color = (0, 255, 0)
        status_text = f"ID: {track_id} | {speed:.1f} km/h"

        if is_inside:
          color = (0, 0, 255)
          status_text = f"VIOLATION: Red Light Jump! ID:{track_id}"
          violation_key = (track_id, "Red_Light_Jump")

          if violation_key not in self.logged_violations:
            vehicle_crop = frame[max(0, y1) : min(frame.shape[0], y2), max(
                0, x1
            ) : min(frame.shape[1], x2)]
            plate_num = self.extract_plate_number(vehicle_crop)

            self.logger.log_violation(
                frame,
                track_id,
                plate_num,
                "Red_Light_Jump",
                "Sec 184 MVA",
                1000,
            )
            self.logged_violations.add(violation_key)

        elif speed > SPEED_LIMIT_KMH:
          color = (0, 165, 255)
          status_text = f"SPEEDING: {speed:.1f} km/h"
          violation_key = (track_id, "Speeding")

          if violation_key not in self.logged_violations:
            vehicle_crop = frame[max(0, y1) : min(frame.shape[0], y2), max(
                0, x1
            ) : min(frame.shape[1], x2)]
            plate_num = self.extract_plate_number(vehicle_crop)

            self.logger.log_violation(
                frame,
                track_id,
                plate_num,
                "Speeding",
                "Sec 183 MVA",
                2000,
            )
            self.logged_violations.add(violation_key)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Increased fontScale from 0.5 to 0.8 and thickness from 2 to 2 for better legibility
        cv2.putText(
            frame,
            status_text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame