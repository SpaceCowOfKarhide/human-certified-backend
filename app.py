import gradio as gr
import torch
from PIL import Image
import numpy as np

# --- LOAD YOUR MODEL HERE ---
# Replace this with your actual model loading
def load_model():
    # example placeholder
    model = None
    return model

model = load_model()

# --- IMAGE DETECTION ---
def detect_image(image):
    # TODO: replace with real inference
    ai_confidence = 0.82  # example
    human_confidence = 1 - ai_confidence

    return {
        "ai_generated_confidence": round(ai_confidence * 100, 1),
        "human_made_confidence": round(human_confidence * 100, 1),
        "note": "Confidence score, not percent of content"
    }

# --- VIDEO DETECTION ---
def detect_video(video):
    # TODO: real inference logic
    ai_confidence = 0.68
    human_confidence = 1 - ai_confidence

    return {
        "ai_generated_confidence": round(ai_confidence * 100, 1),
        "human_made_confidence": round(human_confidence * 100, 1),
        "note": "Confidence score based on detected artifacts"
    }

# --- UI ---
with gr.Blocks(title="Human Certified AI Detector") as demo:
    gr.Markdown("## 🧠 Human Certified AI Detector")

    with gr.Tab("Image"):
        img = gr.Image(type="pil")
        img_btn = gr.Button("Analyze Image")
        img_out = gr.JSON()
        img_btn.click(detect_image, img, img_out)

    with gr.Tab("Video"):
        vid = gr.Video()
        vid_btn = gr.Button("Analyze Video")
        vid_out = gr.JSON()
        vid_btn.click(detect_video, vid, vid_out)

demo.launch()
