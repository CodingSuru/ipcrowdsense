# IPCrowdSense 👥

IPCrowdSense is an AI-powered crowd monitoring system that uses computer vision to detect and count people in real-time via IP-based RTSP camera feeds. Powered by YOLOv11, it offers precise person detection, customizable boundary monitoring, and a web interface for live analytics and alerts. Ideal for security, crowd management, and analytics applications. 🚀
Features ✨

Real-time person detection and counting with YOLOv11 👀
Custom polygonal boundaries for crowd density monitoring 📍
Overcrowding alerts with configurable thresholds ⚠️
Web-based dashboard for live video and analytics 📊
RTSP camera integration for flexible deployment 📹
JSON export for detection and violation data 💾
Performance metrics (FPS, accuracy, system health) 📈
Simple tracking for consistent person counting 🕒

# Demo 🎥
Note: Replace this placeholder with a screenshot or GIF of your web interface showing boundary drawing and alerts.
Installation ⚙️
Prerequisites 🛠️

Python 3.8+ 🐍
OpenCV 🖼️
NumPy 🔢
PyTorch 🔥
Ultralytics YOLO 🤖
Flask 🌐
RTSP-compatible camera 📡
Web browser (Chrome/Firefox recommended) 🌍

# Steps 📋

Clone the Repository:
git clone https://github.com/your-username/ipcrowdsense.git 📂
cd ipcrowdsense


Set Up Virtual Environment:
python -m venv venv 🛡️
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install opencv-python numpy torch ultralytics flask 📦


Download YOLOv11 Model:

Download yolo11n.pt from Ultralytics YOLO and place it in the project root. 📥
Note: Do not commit yolo11n.pt to GitHub due to size and licensing.



# Usage 🚀

Start the Server:
python main.py ▶️


Access the web interface at http://localhost:8081.


Connect Camera:

In the web interface, click "Connect" and enter your RTSP URL (e.g., rtsp://user:pass@192.168.1.100:554/stream). 🔗


Draw Boundaries:

Click on the video feed to add boundary points. 📍
Close boundaries by clicking near the first point. 🔲
Add multiple boundaries as needed. ➕


Monitor & Analyze:

View real-time people counts and alerts. 🚨
Export analytics as JSON via the interface. 💾



Example RTSP URL
rtsp://admin:password123@192.168.1.100:554/h264

# Project Structure 🗂️
ipcrowdsense/
├── detection.py        # YOLO-based detection and tracking logic 🤖
├── main.py            # Flask server and camera handling 🌐
├── index.html         # Web interface for visualization 🖥️
├── video/             # Directory for recorded videos (auto-created) 📼
├── README.md          # Project documentation 📖
└── .gitignore         # Git ignore file 🚫

# Configuration ⚙️

RTSP URL: Set via the web interface. 🔗
Max People: Configure per boundary for alerts. 👥
Boundaries: Draw interactively on the video feed. ✏️
Detection Parameters (edit detection.py):
Confidence threshold: conf=0.5
IoU threshold: iou=0.45
Max detections: max_det=50



# Performance Tips 🏎️

Use yolo11n.pt for speed or yolo11s.pt for better accuracy. ⚖️
Reduce frame resolution in detection.py (e.g., 640px). 📏
Enable GPU for faster YOLO inference. 💻
Lower max_det for fewer detections. 🔢

# Troubleshooting 🐞

Camera Not Connecting:
Check RTSP URL and network. 🌐
Ensure port 554 is open. 🚫


Slow Performance:
Use a lighter YOLO model or lower resolution. ⚖️


Model Errors:
Verify yolo11n.pt exists and PyTorch/Ultralytics are installed. ✅



# Limitations ⚠️

Requires stable network for RTSP streaming. 🌐
Performance depends on hardware (CPU/GPU). 💻
YOLO accuracy varies with lighting/crowd density. 💡
Single-camera support (multi-camera planned). 📹

# Contributing 🤝

Fork the repo 🍴
Create a branch: git checkout -b feature/your-feature 🌿
Commit changes: git commit -m 'Add feature' ✅
Push: git push origin feature/your-feature 📤
Open a Pull Request 📬

See CONTRIBUTING.md for details (create one if needed).

# License 📜
Licensed under the MIT License. 🆓
Acknowledgments 🙌

Ultralytics for YOLOv11 🤖
OpenCV for image processing 🖼️
Flask for the web framework 🌐
PyTorch for AI backend 🔥


Start monitoring crowds with IPCrowdSense today! 🚀
