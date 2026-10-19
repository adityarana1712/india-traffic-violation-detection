import csv
import os
from datetime import datetime
import cv2
from config import LOG_FILE, SNAPSHOT_DIR


class IndiaEChallanLogger:

  def __init__(self):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    self._init_csv()

  def _init_csv(self):
    if not os.path.exists(LOG_FILE):
      # Added encoding='utf-8' here
      with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Vehicle_ID",
            "Plate_Number",
            "Violation_Type",
            "MVA_Section",
            "Fine_Amount_INR",
            "Snapshot_Path",
        ])

  def log_violation(
      self,
      frame,
      track_id,
      plate_number,
      violation_type,
      mva_section,
      fine_amt,
  ):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_filename = (
        f"{violation_type}_ID_{track_id}_{plate_number}_{timestamp}.jpg"
    )
    snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)

    # Save visual snapshot evidence
    cv2.imwrite(snapshot_path, frame)

    # Added encoding='utf-8' here as well
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow([
          timestamp,
          track_id,
          plate_number,
          violation_type,
          mva_section,
          f"₹{fine_amt}",
          snapshot_path,
      ])

    print(
        f"[E-CHALLAN GENERATED] Plate: {plate_number} | Type:"
        f" {violation_type} | Section: {mva_section} | Fine: ₹{fine_amt}"
    )