from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import shutil
import os
import uuid
import cv2

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="Human Certified Backend", version="0.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Config
# ---------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_NAME = "umm-maybe/AI-image-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frame sampling settings
# Sample approximately 1 frame per second; fallback to every 10th frame if FPS is weird.
DEFAULT_FRAME_STEP = 10
MIN_SAMPLED_FRAMES = 3  # ensure at least a few frames are analyzed

# Mixed-class thresholds
AI_RATIO_HUMAN_MAX = 0.10   # <10% AI frames => Human-made
AI_RATIO_MIXED_MAX = 0.50   # 10%–50% => Mixed / AI-Assisted
# >50% => AI-generated

# ---------------------------
# Model (loaded once)
# ---------------------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------------------
# Health
# ---------------------------
@app.get("/")
def root():
    return {"message": "Human Certified Backend OK"}

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# ---------------------------
# Helpers
# ---------------------------
def _predict_image(pil_image: Image.Image):
    """Run model on a PIL image, return (label_str, conf_percent)."""
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)

    label = model.config.id2label[predicted_class.item()]
    conf_percent = round(confidence.item() * 100, 1)
    return label, conf_percent

def _label_to_verdict(label: str) -> str:
    """Map model label to verdict."""
    return "AI-generated" if label.lower() in {"artificial", "ai", "synthetic"} else "Human-made"


# ---------------------------
# IMAGE ENDPOINT
# ---------------------------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and predict
        image = Image.open(path).convert("RGB")
        label, conf_percent = _predict_image(image)
        verdict = _label_to_verdict(label)

        return {
            "filename": file.filename,
            "result": verdict,         # "Human-made" or "AI-generated"
            "label": label,            # raw model label
            "confidence": f"{conf_percent}%"  # confidence of the predicted class
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ---------------------------
# VIDEO ENDPOINT (3-tier)
# ---------------------------
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot open video"}, status_code=400)

        # Determine sampling step
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps and fps > 0:
            frame_step = max(1, int(round(fps)))  # ~1 sample per second
        else:
            frame_step = DEFAULT_FRAME_STEP       # fallback

        frame_results = []
        frame_index = 0

        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_step == 0:
                # BGR -> RGB -> PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                label, conf_percent = _predict_image(pil_img)
                verdict = _label_to_verdict(label)

                frame_results.append({
                    "frame": frame_index,
                    "label": label,
                    "result": verdict,          # "Human-made" or "AI-generated"
                    "confidence": conf_percent  # per-frame confidence
                })

            frame_index += 1

        cap.release()

        # Ensure we have a few samples (for super-short or weird videos)
        if len(frame_results) < MIN_SAMPLED_FRAMES and frame_index > 0:
            # Re-scan with denser sampling
            cap = cv2.VideoCapture(video_path)
            frame_results = []
            step = max(1, frame_index // max(1, MIN_SAMPLED_FRAMES))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    label, conf_percent = _predict_image(pil_img)
                    verdict = _label_to_verdict(label)
                    frame_results.append({
                        "frame": idx,
                        "label": label,
                        "result": verdict,
                        "confidence": conf_percent
                    })
                idx += 1
            cap.release()

        total = len(frame_results)
        if total == 0:
            return JSONResponse({"error": "No frames analyzed"}, status_code=400)

        ai_frames = sum(1 for r in frame_results if r["result"] == "AI-generated")
        human_frames = total - ai_frames
        ai_ratio = ai_frames / total

        # 3-tier classification
        if ai_ratio < AI_RATIO_HUMAN_MAX:
            final_result = "Human-made"
        elif ai_ratio < AI_RATIO_MIXED_MAX:
            final_result = "Mixed / AI-Assisted"
        else:
            final_result = "AI-generated"

        avg_conf = round(sum(r["confidence"] for r in frame_results) / total, 1)

        return {
            "final_result": final_result,
            "avg_confidence": f"{avg_conf}%",
            "total_frames_analyzed": total,
            "ai_frames": ai_frames,
            "human_frames": human_frames,
            "ai_ratio": round(ai_ratio * 100, 1),  # percent of frames that looked AI
            "frame_details": frame_results         # optional: remove in prod if too large
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
