# DrivingAssistant

# 🚗 Real-Time Traffic Sign and Lane Detection using YOLO and UFLD

This project performs real-time detection and tracking of traffic signs, lane lines, and traffic lights using **YOLO11n** and **Ultrafast Lane Detection v2 (UFLDv2)** with ONNX models. The system is designed to be used with dashcam video input and can annotate frames with recognized road signs and lane information.

---

## 📌 Features

* 🚦 **Traffic Light Detection**: Detects green and red traffic lights.
* 🛑 **Road Sign Recognition**: Recognizes various regulatory, warning, and informational signs.
* 🛣️ **Lane Detection**: Utilizes Ultrafast Lane Detection (UFLD) to identify road lanes.
* 🔁 **Multi-object Tracking**: Tracks objects across frames for better accuracy and state management.
* 📊 **Live Overlay**: Displays detected signs and traffic status in real-time.

---

## 🧰 Dependencies

Ensure you have the following Python packages installed:

```bash
pip install onnxruntime opencv-python numpy ultralytics
```
You also need models. Get CULANE_res18 from: [Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) [Googlew drive](https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view) and convert it to onnx

Also, install the custom lane detection module `TrafficLaneDetector` (provided in the project directory).

---



## ▶️ How to Run

1. **Select the appropriate language file** (`signs_en.json` or `signs_pl.json`).
2. **Set your video input** in `video_path`. Use a file path or a camera index like `0` for webcam.
3. **Run the script**:

```bash
python main.py
```

4. **Press `q`** during execution to quit.

---

## 🧠 Key Components Explained

### `display_multiple_sign_info(frame, sign_info_list)`

Draws a dynamic banner at the top of the frame showing currently active and detected traffic signs, with time-based persistence for display.

### `sign_display_counter`

A dictionary storing recognized signs with countdown timers for how long each should be displayed.

```python
sign_display_counter = {
  "A-7": {"time": 15, "type": "warning", "name": "Yield"},
  "Red Light": {"time": 3, "type": "", "name": "Stop"}
}
```

### YOLO Object Detection

The `road.onnx` model detects general objects like traffic lights and road signs, while `sign_best.onnx` specializes in recognizing specific traffic sign classes.

### Lane Detection

Uses the `UltrafastLaneDetectorV2` class from a custom `TrafficLaneDetector` package to detect lanes in real-time with an ONNX model (`culane_res18.onnx`).

---

## 🔄 Auto Reset

To prevent performance degradation over time, the main detection model (`YOLO("road.pt")`) and tracking history reset every 60 seconds.

---

## 🔍 Recognized Sign Classes

The system supports a wide range of signs defined in `signs_en.json` and `signs_pl.json`, including:

* A-7 (Yield)
* B-20 (Stop)
* D-1, D-2, D-3 (Priority roads)
* and more...



---

## 💡 Notes

* Make sure input signs in video are visible and large enough for the secondary sign classifier.
* Ensure minimum object size (`>35x35 px`) for sign classification to activate.
* For consistent FPS, you can adjust the inference frequency (`counter % 5`, etc.).
* Lane detection system inspiration: [Vehicle-CV-ADAS](https://github.com/jason-li-831202/Vehicle-CV-ADAS) which was using [Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

---



