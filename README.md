# 🎯 Object Detection & Tracking

A computer vision application that detects and tracks objects in images and videos in real-time using YOLOv8.

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/sxrthxk10/CodeAlpha-ObjectDetection)

## 📌 Project Overview
This project was built as **Task 4** of the **CodeAlpha Artificial Intelligence Internship**. It uses a pre-trained YOLOv8 model to detect objects in uploaded images and videos, and applies object tracking to assign consistent IDs to objects across video frames.

## ✨ Features
- Real-time object detection using YOLOv8 (pre-trained model)
- Bounding boxes with object labels and confidence scores
- Object tracking with persistent IDs across video frames
- Supports both image and video uploads
- Clean, tab-based interface using Gradio

## 🛠️ Tech Stack
- **Python**
- **Ultralytics YOLOv8** – pre-trained object detection model
- **OpenCV** – video frame processing
- **Gradio** – web interface

## ⚙️ How It Works
1. User uploads an image or video
2. **Image mode:** YOLOv8 runs detection on the image, drawing bounding boxes with labels
3. **Video mode:** Each frame is processed using `model.track()`, which detects objects and assigns persistent tracking IDs
4. Annotated frames are written to an output video file
5. Result is displayed/returned to the user

## 📂 Files
- `app.py` – Main application code (detection, tracking logic, Gradio UI)
- `requirements.txt` – Python dependencies

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## 👤 Author
**Sarthak Tiwari**
CodeAlpha AI Internship — Task 4

## 🙏 Acknowledgment
Thanks to **CodeAlpha** for this internship opportunity.

#CodeAlpha #AIInternship #ComputerVision #YOLO #ObjectDetection #Python
