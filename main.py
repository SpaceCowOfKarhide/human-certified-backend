from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, uuid, shutil

app = FastAPI()

# ---------------------------
# CORS (ALLOW EVERYTHING FOR NOW)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT for Railway + frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Health / Root
# ---------------------------
@app.get("/")
def root():
    return {"message": "Backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# IMAGE ENDPOINT (TEST VERSION)
# ---------------------------
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "ai_generated_confidence": 50,
        "human_made_confidence": 50
    }
