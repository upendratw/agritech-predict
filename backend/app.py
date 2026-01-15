# agritech-predict/backend/app.py

import io
import os
import base64
import traceback
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🔹 NEW: import advice router
from advice import router as advice_router

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
app = FastAPI(
    title="Agritech Predict Backend",
    description="AI inference + treatment advice service",
    version="1.0",
)

# ---------------------------------------------------------
# Middleware
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 🔹 REGISTER ADVICE ROUTER
# ---------------------------------------------------------
app.include_router(
    advice_router,
    prefix="",             # e.g. /treatment-advice
    tags=["Treatment Advice"]
)
app.include_router(advice_router)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

MODEL_PATH = CHECKPOINTS_DIR / "best.pt"
LABELS_PATH = CHECKPOINTS_DIR / "labels.txt"

# ---------------------------------------------------------
# Load Ultralytics
# ---------------------------------------------------------
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ULTRALYTICS_ERROR = str(e)

# ---------------------------------------------------------
# Globals
# ---------------------------------------------------------
MODEL = None
CLASS_NAMES: List[str] = []

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_labels() -> List[str]:
    if not LABELS_PATH.exists():
        raise RuntimeError(f"labels.txt not found at {LABELS_PATH}")

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise RuntimeError("labels.txt is empty")

    return labels


def load_model():
    global MODEL, CLASS_NAMES

    if MODEL is not None:
        return

    if YOLO is None:
        raise RuntimeError(f"Ultralytics not installed: {ULTRALYTICS_ERROR}")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    MODEL = YOLO(str(MODEL_PATH))
    CLASS_NAMES = load_labels()

    print("✅ Model loaded:", MODEL_PATH)
    print("✅ Labels:", CLASS_NAMES)

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str


class PredictResponse(BaseModel):
    detections: List[Detection]
    annotated_image_base64: str
    model_path: str

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "POST /predict | GET /treatment-advice"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    score_thresh: float = Query(0.25, ge=0.0, le=1.0),
):
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Read image
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Inference
    try:
        results = MODEL.predict(
            source=np.asarray(image),
            conf=score_thresh,
            imgsz=640,
            verbose=False,
        )
        r = results[0]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed:\n{traceback.format_exc()}",
        )

    detections: List[Detection] = []

    boxes = r.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cid = int(class_ids[i])
            label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)

            detections.append(
                Detection(
                    x1=float(xyxy[i][0]),
                    y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]),
                    y2=float(xyxy[i][3]),
                    score=float(scores[i]),
                    class_id=cid,
                    label=label,
                )
            )

    # Annotated image
    try:
        annotated = r.plot()
        annotated_pil = (
            annotated if isinstance(annotated, Image.Image)
            else Image.fromarray(annotated)
        )

        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate annotated image:\n{traceback.format_exc()}",
        )

    return {
        "detections": [d.dict() for d in detections],
        "annotated_image_base64": img_b64,
        "model_path": str(MODEL_PATH),
    }