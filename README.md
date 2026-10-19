# India Traffic Violation & E-Challan Prototype 🇮🇳🚦

Hey! This is a computer vision project I put together to experiment with automated traffic monitoring on Indian roads. It tracks vehicles using YOLO, checks for common infractions like red-light jumps and speeding, tries to read license plates using OCR, and logs everything out as a mock e-challan with snapshot evidence.

## 🛠️ What's Inside
* **Python** (3.10 / 3.11)
* **Ultralytics YOLOv8** for real-time object tracking
* **OpenCV** for video handling and zone/speed logic
* **EasyOCR** for picking up Indian vehicle registration plates
* **Pandas & NumPy** for data handling and calculations

## 📌 Features
* Tracks cars, two-wheelers, buses, and trucks.
* Flags vehicles crossing a custom stop-line zone (Sec 184 MVA) or driving over the speed limit (Sec 183 MVA).
* Crops the vehicle region to grab license plate text.
* Saves a snapshot image and writes an entry to a CSV log with the violation details and fines.