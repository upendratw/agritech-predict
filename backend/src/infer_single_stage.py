#!/usr/bin/env python3
"""
infer_single_stage.py

Robust inference script for the single-stage detector in this repo.

Usage (example):
  python src/infer_single_stage.py \
    --weights checkpoints/best_model.pth \
    --image /Users/upendra/Desktop/input.jpg \
    --device mps \
    --image_size 512 \
    --score_thresh 0.3 \
    --iou_thresh 0.45 \
    --topk_pre 300 \
    --out /Users/upendra/Desktop/output_pred.png \
    --debug
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import math
import numpy as np
from typing import Tuple, Optional

# Make sure repo root is importable (so `src` modules load)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

# Try to import the project's model and utils; fallbacks provided if missing.
try:
    from src.single_stage_detector import SingleStageDetector  # user repo factory
except Exception:
    SingleStageDetector = None

try:
    from src.single_stage_utils import decode_predictions  # repo decoder
except Exception:
    decode_predictions = None


def load_model_factory(device: torch.device, verbose: bool = True):
    """
    Return a constructed model instance (uninitialized weights).
    Tries to find SingleStageDetector in repo; if not present raises.
    """
    if SingleStageDetector is None:
        raise RuntimeError(
            "Cannot find SingleStageDetector class in repo (src/single_stage_detector.py). "
            "Please ensure the file exists and defines SingleStageDetector."
        )
    model = SingleStageDetector()  # expect constructor without args; adjust if your factory needs args
    model.to(device)
    if verbose:
        print("Constructed model and moved to device:", device)
    return model


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device, verbose: bool = True):
    """
    Load checkpoint into model. Handles:
      - state_dict-only files
      - full dicts with 'model_state' or 'model_state_dict' keys.
    Ensures tensors in the loaded state_dict are moved to `device` before loading.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"weights not found: {ckpt_path}")

    # Load to CPU first (safe); we'll move to device explicitly.
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Determine state_dict
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "model_state_dict" in ckpt):
        # common checkpoint shape
        state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    elif isinstance(ckpt, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in ckpt.items()):
        state = ckpt  # looks like a state_dict already
    elif isinstance(ckpt, torch.nn.Module):
        # unlikely, but handle
        state = ckpt.state_dict()
    else:
        # fallback: maybe checkpoint is a state_dict
        state = ckpt

    if not isinstance(state, dict):
        raise RuntimeError("Unable to extract state_dict from checkpoint file")

    # Move tensors to target device and correct dtype (if device is mps convert to float)
    new_state = {}
    for k, v in state.items():
        try:
            if torch.is_tensor(v):
                # cast/convert
                v2 = v.to(device)
                new_state[k] = v2
            else:
                new_state[k] = v
        except Exception:
            # fallback: leave as-is (load_state_dict will error if incompatible)
            new_state[k] = v

    # Now load state
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if verbose:
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print("Missing keys when loading checkpoint (ignored):", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("Unexpected keys in checkpoint (ignored):", unexpected[:10], "..." if len(unexpected) > 10 else "")
    return model


def preprocess_image(image_path: str, image_size: int) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Load image, resize to (image_size, image_size) preserving aspect ratio by letterbox
    and return tensor shaped (1,3,H,W) on CPU (will be moved to device later),
    the original PIL image, and (orig_w, orig_h).
    """
    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size

    # letterbox resize to keep aspect ratio
    target = (image_size, image_size)
    ratio = min(target[0] / orig_w, target[1] / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)

    # create letterbox canvas
    canvas = Image.new("RGB", target, (114, 114, 114))
    paste_x = (target[0] - new_w) // 2
    paste_y = (target[1] - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Convert to tensor [0,1], C x H x W
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    # to CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

    # Return tensor (cpu), pil original, and details for mapping boxes back
    meta = {"orig_size": (orig_w, orig_h), "paste": (paste_x, paste_y), "scale": ratio, "canvas_size": target}
    return tensor, pil, meta


def _safe_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Return (w,h) for text using available PIL methods; avoids getsize/getbbox compatibility problems.
    """
    try:
        # Pillow >=8 has font.getbbox
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return int(w), int(h)
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            # final fallback
            return (len(text) * 6, 11)


def decode_and_postprocess(
    outputs,
    image_size: int,
    score_thresh: float = 0.3,
    topk_pre: int = 200,
    iou_thresh: float = 0.45,
    max_dets: int = 200,
    debug: bool = False,
):
    """
    Decode outputs using repository decoder if available, otherwise perform a simple decode:
      - expect outputs to be (locs, confs, anchors)
      - anchors may be normalized (0..1) or pixel coords (we detect this)
    Returns: boxes (N,4) in pixel coords (relative to the model input canvas), labels (N,), scores (N,)
    """
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
        locs, confs, anchors = outputs[:3]
    else:
        raise RuntimeError("Model did not return (locs, confs, anchors). Got: " + str(type(outputs)))

    # Move to cpu for numpy operations
    locs = locs.detach().cpu()
    confs = confs.detach().cpu()
    anchors = anchors.detach().cpu()

    # If decoder provided by repo, use it (it expects (locs, confs, anchors))
    if decode_predictions is not None:
        boxes, labels, scores = decode_predictions(
            (locs, confs, anchors),
            score_threshold=score_thresh,
            topk=topk_pre,
            iou_threshold=iou_thresh,
            max_detections=max_dets,
            image_size=image_size,
        )
        return boxes, labels, scores

    # --- fallback simple decoder (not as sophisticated as repo) ---
    # confs shape: (B,N,C) or (1,N,C)
    cls_logits = confs[0]  # (N,C)
    probs = F.softmax(cls_logits, dim=-1).numpy()  # (N,C)
    # treat index 0 as background
    fg_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]  # if only 1 class (maybe background+1); adapt
    # simple threshold
    keep = fg_scores > score_thresh
    if not keep.any():
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64), torch.zeros((0,))

    kept_scores = fg_scores[keep]
    kept_anchors = anchors[keep]
    kept_locs = locs[0][keep]  # (k,4)

    # If anchors are in xyxy form produced by model.generate_anchors earlier, detect the format.
    # Some models produce anchors as [x1,y1,x2,y2] or [cx,cy,w,h]. We'll try to detect.
    # Heuristic: if anchors values are <=1 assume normalized cx,cy,w,h in 0..1
    amax = float(kept_anchors.abs().max().item())
    anchors_are_normalized = amax <= 1.0001

    # Expect anchors from our model are either [x1,y1,x2,y2] in pixels OR [cx,cy,w,h] normalized or absolute.
    # Try to detect: if anchors have small values and first anchor looks like cx!=x1, treat as cx,cy,w,h
    # We'll support cx,cy,w,h normalized => need to decode using SSD-style encoding if locs are offsets.
    # However, if the repo uses direct corner predictions, this fallback may be inaccurate.
    # To remain robust, if anchors appear to be xyxy zeros (common bug), abort returning empty.
    if kept_anchors.numel() == 0 or torch.all(kept_anchors == 0):
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64), torch.zeros((0,))

    # Detect whether anchors look like cx,cy,w,h by checking if (x2 - x1) small or negative
    diffs = kept_anchors[:, 2] - kept_anchors[:, 0]
    if (diffs <= 0).all() or (amax <= 1.0 and kept_anchors.shape[1] == 4):
        # treat anchors as (cx,cy,w,h)
        acx = kept_anchors[:, 0]
        acy = kept_anchors[:, 1]
        aw = kept_anchors[:, 2]
        ah = kept_anchors[:, 3]
        # decode SSD-like offsets (assume locs encoded as dx,dy,dw,dh with variances (0.1,0.2))
        var_c, var_wh = 0.1, 0.2
        dx = kept_locs[:, 0]
        dy = kept_locs[:, 1]
        dw = kept_locs[:, 2]
        dh = kept_locs[:, 3]
        cx = acx + dx * aw * var_c
        cy = acy + dy * ah * var_c
        w = aw * torch.exp(dw * var_wh)
        h = ah * torch.exp(dh * var_wh)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        if anchors_are_normalized:
            boxes = boxes * float(image_size)
    else:
        # treat anchors as x1,y1,x2,y2 already
        # possibly locs are deltas; if locs are small we just return anchors
        boxes = kept_anchors.clone()
        # if anchors normalized scale to pixels
        if anchors_are_normalized:
            boxes = boxes * float(image_size)

    # NMS per-class – since we only support single foreground class in fallback, do a single nms
    scores_tensor = torch.from_numpy(kept_scores).float()
    keep_idx = nms(boxes, scores_tensor, iou_thresh)
    boxes = boxes[keep_idx]
    labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    scores = scores_tensor[keep_idx].cpu()
    return boxes.cpu(), labels.cpu(), scores.cpu()


def rescale_boxes_to_original(boxes, meta):
    """
    boxes: (N,4) in canvas coordinates (the letterboxed, image_size x image_size)
    meta: dict returned from preprocess_image with orig_size, paste, scale
    returns boxes in original image pixel coordinates
    """
    if boxes.numel() == 0:
        return boxes
    paste_x, paste_y = meta["paste"]
    scale = meta["scale"]
    # boxes were in canvas coords. Convert canvas -> resized image -> original
    # step1: subtract paste offsets
    boxes = boxes.clone().float()
    boxes[:, [0, 2]] -= paste_x
    boxes[:, [1, 3]] -= paste_y
    # step2: divide by scale to map back to original scale
    boxes = boxes / float(scale)
    # clamp to image
    ow, oh = meta["orig_size"]
    boxes[:, 0].clamp_(0, ow - 1)
    boxes[:, 2].clamp_(0, ow - 1)
    boxes[:, 1].clamp_(0, oh - 1)
    boxes[:, 3].clamp_(0, oh - 1)
    return boxes


def draw_boxes_on_image(pil_img: Image.Image, boxes, labels, scores, label_map=None, out_path: Optional[str] = None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(x) for x in boxes[i]]
        # ensure coordinates valid
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        # skip degenerate
        if not (x2 > x1 and y2 > y1):
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label_text = f"{int(labels[i])}" if labels is not None else ""
        if scores is not None:
            label_text = f"{label_text} {scores[i]:.2f}"
        # compute text size robustly
        w, h = _safe_text_size(draw, label_text, font)
        # draw background
        txt_x = x1
        txt_y = max(0, y1 - h - 4)
        draw.rectangle([txt_x, txt_y, txt_x + w + 4, txt_y + h + 2], fill="red")
        draw.text((txt_x + 2, txt_y), label_text, fill="white", font=font)
    if out_path:
        pil_img.save(out_path)
    return pil_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0, etc.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--iou_thresh", type=float, default=0.45)
    parser.add_argument("--topk_pre", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device_str = args.device
    try:
        device = torch.device(device_str)
    except Exception:
        device = torch.device("cpu")
    print("Device:", device)

    # Build model
    model = load_model_factory(device=device, verbose=True)

    # Load checkpoint into model (ensures weights moved to device)
    model = load_checkpoint_into_model(model, args.weights, device=device, verbose=True)
    model.eval()

    # Preprocess image
    tensor_cpu, pil_orig, meta = preprocess_image(args.image, args.image_size)
    if args.debug:
        print("Original image size:", meta["orig_size"], "canvas:", meta["canvas_size"], "paste:", meta["paste"], "scale:", meta["scale"])

    # Move tensor to same dtype/device as model parameters
    # Choose dtype by checking one parameter:
    model_param = next(model.parameters())
    target_dtype = model_param.dtype
    # Move input to device and dtype
    tensor = tensor_cpu.to(device=device, dtype=target_dtype)

    # Forward
    with torch.no_grad():
        outputs = model(tensor)
    if args.debug:
        print("Model forward done. Outputs type:", type(outputs))
        try:
            if isinstance(outputs, (list, tuple)):
                print("shapes:",
                      "locs", getattr(outputs[0], "shape", None),
                      "confs", getattr(outputs[1], "shape", None),
                      "anchors", getattr(outputs[2], "shape", None) if len(outputs) > 2 else None)
        except Exception:
            pass

    # Decode & postprocess
    boxes_canvas, labels, scores = decode_and_postprocess(
        outputs,
        image_size=args.image_size,
        score_thresh=args.score_thresh,
        topk_pre=args.topk_pre,
        iou_thresh=args.iou_thresh,
        max_dets=200,
        debug=args.debug,
    )

    # If no boxes, still save a copy and exit
    if boxes_canvas.numel() == 0 or boxes_canvas.shape[0] == 0:
        print("No detections above threshold.")
        if args.out:
            pil_orig.save(args.out)
            print("Saved image (no boxes) to:", args.out)
        return

    # Rescale boxes back to original image coordinates
    boxes_orig = rescale_boxes_to_original(boxes_canvas, meta)

    # Draw and save
    out_path = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_det.png"))
    pil_with = draw_boxes_on_image(pil_orig.copy(), boxes_orig, labels.numpy() if isinstance(labels, torch.Tensor) else labels, scores.numpy() if isinstance(scores, torch.Tensor) else scores, out_path=out_path)
    print("Saved:", out_path)
    if args.debug:
        print("Boxes saved:", boxes_orig.shape[0])

if __name__ == "__main__":
    main()