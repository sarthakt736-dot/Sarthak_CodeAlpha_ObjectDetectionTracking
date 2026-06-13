import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_image(image):
    results = model(image)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    return output_path

with gr.Blocks(title="🎯 Object Detection & Tracking — CodeAlpha") as app:
    gr.Markdown("# 🎯 Object Detection & Tracking")
    gr.Markdown("### CodeAlpha AI Internship | Sarthak Tiwari")
    gr.Markdown("Upload an image or video — objects will be detected and tracked using YOLOv8.")

    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            img_input = gr.Image(type="numpy", label="Upload Image")
            img_output = gr.Image(label="Detected Output")
        img_btn = gr.Button("🚀 Detect Objects", variant="primary")
        img_btn.click(fn=detect_image, inputs=img_input, outputs=img_output)

    with gr.Tab("🎥 Video Detection & Tracking"):
        with gr.Row():
            vid_input = gr.Video(label="Upload Video")
            vid_output = gr.Video(label="Tracked Output")
        vid_btn = gr.Button("🚀 Detect & Track", variant="primary")
        vid_btn.click(fn=detect_video, inputs=vid_input, outputs=vid_output)

app.launch()