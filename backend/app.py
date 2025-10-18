# backend/app.py
import os
import io
import time
import logging
from typing import Optional, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- internal model imports (adjust if your project layout differs) ---
from src.models.ssd import SSD

# optionally import decode helper if you have one. If not present, we'll handle gracefully.
try:
    from src.models.utils import decode_predictions
except Exception:
    decode_predictions = None

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# ---------------------------
# App & static files
# ---------------------------
app = FastAPI(title="Agritech Detector Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to a specific origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static annotated images
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------
# Utility: draw boxes (Pillow-safe)
# ---------------------------
def draw_boxes_on_pil(pil_img: Image.Image, boxes: List[List[float]], labels: List[str],
                      scores: Optional[List[float]] = None, labels_map: Optional[Dict[int, str]] = None,
                      score_fmt: str = "{:.2f}"):
    """
    Draw boxes, labels and (optional) scores onto a PIL image in-place.

    - pil_img: PIL.Image
    - boxes: list/iterable of [x1, y1, x2, y2] in pixel coords
    - labels: list of str label names (already mapped)
    - scores: optional list of floats same length as boxes
    - labels_map: not used here since labels are already strings
    """
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Try to load a truetype font if available, else default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label_txt = str(labels[i]) if labels is not None and i < len(labels) else ""

        if scores is not None and i < len(scores):
            score = scores[i]
            try:
                label_txt = f"{label_txt} {score_fmt.format(score)}"
            except Exception:
                label_txt = f"{label_txt} {score}"

        # box color and thickness
        color = (255, 0, 0, 200)  # semi-transparent red
        thickness = max(1, int(round(min(pil_img.size) / 200)))  # adaptive thickness

        # draw rectangle (thickness aware)
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Compute text size robustly across Pillow versions:
        try:
            bbox = draw.textbbox((0, 0), label_txt, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            try:
                bbox2 = font.getbbox(label_txt)
                text_w = bbox2[2] - bbox2[0]
                text_h = bbox2[3] - bbox2[1]
            except Exception:
                text_w, text_h = draw.textsize(label_txt, font=font)

        # background rectangle for text (slightly above the top-left of the box)
        pad = 3
        text_x0 = x1
        text_y0 = max(0, y1 - text_h - 2 * pad)
        text_x1 = x1 + text_w + 2 * pad
        text_y1 = text_y0 + text_h + 2 * pad

        # fill the background (semi-opaque)
        draw.rectangle([text_x0, text_y0, text_x1, text_y1], fill=(0, 0, 0, 160))

        # finally draw the text in white
        text_pos = (text_x0 + pad, text_y0 + pad)
        draw.text(text_pos, label_txt, fill=(255, 255, 255, 255), font=font)

# ---------------------------
# Checkpoint & model loader with flexible adaptation
# ---------------------------
MODEL: Optional[SSD] = None
DEVICE = torch.device("cpu")
LABELS_MAP: Optional[Dict[int, str]] = None  # load from file if provided

def infer_num_classes_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Try to infer number of classes (including background) from checkpoint by looking at
    pred_heads.*.cls_conv.weight shape (out_channels).
    Returns num_classes (including background) or None.
    """
    for k, v in state_dict.items():
        if k.endswith("cls_conv.weight"):
            out_ch = v.shape[0]
            # Try divisors to find plausible #classes
            for divisor in range(1, 65):
                if out_ch % divisor == 0:
                    num_cls = out_ch // divisor
                    if 2 <= num_cls <= 100:
                        return num_cls
            return out_ch
    return None

def _tile_or_truncate_tensor(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Tile or truncate src along dim=0 to match target_shape[0].
    For biases, will tile/truncate similarly for 1D tensors.
    """
    tgt0 = target_shape[0]
    src0 = src.shape[0]
    if src0 == tgt0:
        return src.clone()
    if src0 > tgt0:
        return src[:tgt0].clone()
    reps = int(np.ceil(tgt0 / src0))
    tiled = src.repeat(reps, *([1] * (src.ndim - 1)))
    return tiled[:tgt0].clone()

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device=torch.device("cpu")):
    """
    Flexible loader:
      - try strict load_state_dict
      - else try non-strict
      - try to adapt classification head shapes by tiling/truncation
    Returns info dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # ckpt might be a state_dict or dict with 'model_state' etc.
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt or "model" in ckpt):
        candidate = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        candidate = ckpt

    if not isinstance(candidate, dict):
        raise RuntimeError("Checkpoint format not recognized (expected state_dict or dict containing model_state).")

    state_dict_ckpt = candidate

    # try strict first
    try:
        model.load_state_dict(state_dict_ckpt, strict=True)
        logger.info("Loaded checkpoint with strict=True")
        return {"loaded": True, "mode": "strict", "adapted": 0}
    except Exception as e:
        logger.warning("Strict load failed, will try flexible loading. Error: %s", e)

    # try non-strict
    try:
        res = model.load_state_dict(state_dict_ckpt, strict=False)
        logger.info("Non-strict load result: %s", res)
    except Exception as e:
        logger.warning("Non-strict load threw exception: %s", e)
        res = None

    # attempt minimal adaptation on cls_conv.* (tile/truncate) where shapes mismatch
    model_state = model.state_dict()
    adapted = 0
    new_ckpt = dict(state_dict_ckpt)  # shallow copy

    for mkey, mparam in model_state.items():
        if "cls_conv.weight" in mkey or "cls_conv.bias" in mkey:
            if mkey in state_dict_ckpt:
                src = state_dict_ckpt[mkey]
                if src.shape != mparam.shape:
                    try:
                        new_val = _tile_or_truncate_tensor(src, mparam.shape)
                        new_ckpt[mkey] = new_val
                        adapted += 1
                        logger.info("Adapted %s from %s -> %s", mkey, tuple(src.shape), tuple(mparam.shape))
                    except Exception as ex:
                        logger.warning("Failed to adapt %s: %s", mkey, ex)

    # final attempt
    try:
        res2 = model.load_state_dict(new_ckpt, strict=False)
        logger.info("Flexible load result after attempted adaptations: %s", res2)
        return {"loaded": True, "mode": "flexible", "adapted": adapted, "load_result": str(res2)}
    except Exception as e:
        logger.error("Flexible load failed completely: %s", e)
        raise

# ---------------------------
# Startup: build model and load checkpoint (if provided)
# ---------------------------
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth"),
)
CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)

@app.on_event("startup")
def startup_event():
    global MODEL, DEVICE, LABELS_MAP
    # pick device
    device_str = os.environ.get("DEVICE", "cpu")
    try:
        DEVICE = torch.device(device_str)
    except Exception:
        DEVICE = torch.device("cpu")
    logger.info("Starting up. device=%s", DEVICE)

    # Build model with a safe default num_classes. We'll try to infer from checkpoint.
    guessed_num_classes = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt_raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
            if isinstance(ckpt_raw, dict) and "model_state" in ckpt_raw:
                ckpt_state = ckpt_raw["model_state"]
            elif isinstance(ckpt_raw, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_raw.values()):
                ckpt_state = ckpt_raw
            else:
                ckpt_state = ckpt_raw
            inferred = infer_num_classes_from_checkpoint(ckpt_state)
            if inferred is not None:
                guessed_num_classes = inferred
                logger.info("Inferred num_classes (including background) from checkpoint: %s", guessed_num_classes)
        except Exception as e:
            logger.warning("Could not inspect checkpoint for shape info: %s", e)

    # fallback
    if guessed_num_classes is None:
        guessed_num_classes = int(os.environ.get("DEFAULT_NUM_CLASSES", 2))  # including background

    # Build and load model
    MODEL = SSD(num_classes=guessed_num_classes)
    MODEL.to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            info = load_checkpoint_into_model(MODEL, CHECKPOINT_PATH, device=DEVICE)
            logger.info("Checkpoint load result: %s", info)
        except Exception as e:
            logger.exception("Failed to load checkpoint at startup: %s", e)
    else:
        logger.info("No checkpoint found at %s — starting with random weights.", CHECKPOINT_PATH)

    # optional labels.txt near the checkpoint or configured location
    labels_path = os.environ.get("LABELS_PATH")
    if not labels_path:
        ckpt_dir = os.path.dirname(CHECKPOINT_PATH)
        try_labels = os.path.join(ckpt_dir, "labels.txt")
        if os.path.exists(try_labels):
            labels_path = try_labels

    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf8") as fh:
                lines = [l.strip() for l in fh.readlines() if l.strip()]
            # assume class indices start at 1 (0 = background)
            LABELS_MAP = {i + 1: name for i, name in enumerate(lines)}
            logger.info("Loaded labels_map from %s: %s", labels_path, LABELS_MAP)
        except Exception as e:
            logger.warning("Failed to load labels file %s: %s", labels_path, e)
    else:
        logger.info("No labels file found or specified; LABELS_MAP will be None.")

# ---------------------------
# Infer endpoint
# ---------------------------
@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    image_size: int = Query(512, description="Resize shorter side to this while preserving aspect ratio"),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0),
    topk: int = Query(200, ge=1),
):
    """
    Run inference on a single uploaded image (multipart/form-data, field name 'image').
    """
    global MODEL, DEVICE, LABELS_MAP
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    # read image bytes
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {e}")

    orig_w, orig_h = pil.size

    # resize to square image_size x image_size (training used this)
    pil_resized = pil.resize((image_size, image_size), resample=Image.BILINEAR)

    # convert to tensor (C,H,W) normalized to [0,1]
    img_arr = np.array(pil_resized).astype("float32") / 255.0
    img_t = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(img_t)

    # Try decode_predictions helper if available
    boxes = []
    labels_out = []
    scores_out = []

    if decode_predictions is not None:
        try:
            # Expecting: boxes_rel, labels_out, scores_out
            _boxes_rel, _labels_out, _scores_out = decode_predictions(outputs, score_threshold=score_threshold, topk=topk)
            _boxes_rel = np.asarray(_boxes_rel, dtype=float)
            if _boxes_rel.size == 0:
                boxes = []
                labels_out = []
                scores_out = []
            else:
                # If normalized (max <= 1.0) scale to resized pixels
                if _boxes_rel.max() <= 1.0:
                    boxes = (_boxes_rel * np.array([image_size, image_size, image_size, image_size])).tolist()
                else:
                    boxes = _boxes_rel.tolist()
                labels_out = list(_labels_out) if _labels_out is not None else []
                scores_out = list(_scores_out) if _scores_out is not None else []
        except Exception as exc:
            logger.exception("decode_predictions failed: %s", exc)
            raw_path = os.path.join(RESULTS_DIR, f"out_{int(time.time())}.raw.pt")
            torch.save(outputs, raw_path)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "decode_predictions failed on model outputs. Raw outputs saved to server.",
                    "raw_path": raw_path,
                    "error": str(exc),
                },
            )
    else:
        # No decode helper available — save raw outputs for debugging
        raw_path = os.path.join(RESULTS_DIR, f"out_{int(time.time())}.raw.pt")
        torch.save(outputs, raw_path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "No decode_predictions available in src.models.utils. Raw model outputs saved on server for debugging.",
                "raw_path": raw_path,
            },
        )

    # If we have boxes, map them back to original image size
    boxes_scaled: List[List[float]] = []
    for b in boxes:
        x1, y1, x2, y2 = b
        sx = orig_w / image_size
        sy = orig_h / image_size
        boxes_scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])

    # Map class ids to readable names using LABELS_MAP (if available)
    labels_list: List[str] = []
    if LABELS_MAP:
        labels_list = [LABELS_MAP.get(int(l), str(int(l))) for l in labels_out]
    else:
        labels_list = [str(int(l)) for l in labels_out]

    # Draw boxes on original-sized PIL
    pil_out = pil.copy()
    draw_boxes_on_pil(pil_out, boxes_scaled, labels_list, scores=scores_out, labels_map=None)

    # Save annotated image to static results and return URL + counts
    out_fname = f"out_{int(time.time())}.png"
    out_path = os.path.join(RESULTS_DIR, out_fname)
    pil_out.save(out_path, format="PNG")

    # build counts
    counts: Dict[str, int] = {}
    for lbl in labels_list:
        counts[lbl] = counts.get(lbl, 0) + 1

    return {"image_url": f"/static/results/{out_fname}", "counts": counts, "raw_boxes_count": len(boxes)}

# ---------------------------
# Simple health check / info
# ---------------------------
@app.get("/info")
def info():
    global MODEL, DEVICE, LABELS_MAP
    return {
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "labels_map_provided": LABELS_MAP is not None,
        "static_results_url": "/static/results/",
    }

@app.get("/", response_class=PlainTextResponse)
def root():
    """
    Basic root endpoint so GET / returns 200.
    If the frontend build exists under backend/static/index.html, serve it.
    Otherwise return a small text health response.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return PlainTextResponse("Agritech Detector backend is running.", status_code=200)