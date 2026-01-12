# debug_infer_verbose.py
"""
Debug helper: run one forward pass and print diagnostics + visualize anchors & final boxes.

Usage (from repo root):
python debug_infer_verbose.py --weights checkpoints/best_model.pth --image /path/to/input.jpg --device cpu --image_size 512 --score_thresh 0.3 --topk 200 --iou 0.45 --out /tmp/debug_out.png
"""
import argparse
import sys
import os
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

# Try to import your model & decode util in a tolerant way
# Adjust names if your repo uses different module paths.
try:
    # when running as "python debug_infer_verbose.py" from repo root
    from single_stage_detector import SingleStageDetector
except Exception:
    try:
        from src.single_stage_detector import SingleStageDetector
    except Exception:
        SingleStageDetector = None

# decode util: many repos provide a decode_predictions / utils.decode_predictions
try:
    from src.models.utils import decode_predictions as decode_preds
except Exception:
    try:
        from models.utils import decode_predictions as decode_preds
    except Exception:
        decode_preds = None

def load_checkpoint_to_model(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    # ck may be a state_dict or a full dict with key 'model_state' or 'model_state_dict'
    if isinstance(ck, dict) and any(k in ck for k in ("model_state", "model_state_dict", "state_dict")):
        # try common keys
        for k in ("model_state", "model_state_dict", "state_dict"):
            if k in ck:
                state = ck[k]
                break
        else:
            # take "model_state" fallback
            state = ck
    elif isinstance(ck, dict) and all(isinstance(k, str) for k in ck.keys()) and any(k.startswith("backbone") or k.startswith("pred_heads") for k in ck.keys()):
        # looks like a bare state dict
        state = ck
    else:
        # fallback: maybe full checkpoint under 'model' etc
        state = ck

    # load into CPU model
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint into model (non-strict).")
    except Exception as e:
        # try nested key 'model_state'
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
            print("Loaded nested 'model_state' into model (non-strict).")
        else:
            print("Warning: load_state_dict failed:", e)
    # Move model to device
    model.to(device)
    model.eval()
    return model

def draw_anchors_on_image(pil_img, anchors, topk=500, alpha=0.15):
    # anchors: (N,4) float in pixel coords
    draw = ImageDraw.Draw(pil_img)
    N = min(len(anchors), topk)
    for i in range(N):
        x1,y1,x2,y2 = anchors[i]
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0))
    return pil_img

def draw_boxes(pil_img, boxes, scores=None, labels=None, color=(0,255,0), width=2):
    draw = ImageDraw.Draw(pil_img)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = [int(float(v)) for v in b]
        draw.rectangle([x1,y1,x2,y2], outline=color, width=width)
        txt = ""
        if scores is not None:
            txt = f"{scores[i]:.2f}"
        if labels is not None:
            txt = f"{labels[i]} {txt}"
        if txt:
            text_w, text_h = font.getsize(txt)
            draw.rectangle([x1, y1-text_h-4, x1+text_w+4, y1], fill=(0,0,0))
            draw.text((x1+2, y1-text_h-2), txt, fill=(255,255,255), font=font)
    return pil_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", default="debug_out.png")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "mps" else "mps")
    print("Device:", device)

    if SingleStageDetector is None:
        print("ERROR: Could not find SingleStageDetector import. Edit this script to import your model class.")
        sys.exit(1)

    # construct model (use sensible defaults if your constructor needs other args)
    # Try to reflect what your repo's training used. If your model needs num_classes / anchors etc,
    # edit this line to match your model constructor.
    model = SingleStageDetector()  # <-- if your constructor needs args, change here

    # load checkpoint safely to CPU and then move model to device
    model = load_checkpoint_to_model(model, args.weights, device)

    # load image
    pil = Image.open(args.image).convert("RGB")
    orig_w, orig_h = pil.size
    print("Original image size (W,H):", orig_w, orig_h)

    # preprocess: resize to image_size (same as training) preserving aspect via letterbox?
    # Here we do a simple resize to square image_size (same as your training pipeline) — adjust if you used padding.
    pil_resized = pil.resize((args.image_size, args.image_size), Image.BILINEAR)
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(),])  # assume model expects [0,1] normalized tensors
    tensor = transform(pil_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)  # expecting either (locs, confs, anchors) OR a single output
    print("Model forward done. Outputs type:", type(outputs))

    # heuristics to parse outputs
    locs = confs = anchors = None
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        locs, confs, anchors = outputs
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        locs, confs = outputs
        print("Warning: anchors not returned by model. Anchors required for decode.")
    else:
        # single tensor? attempt to split? unknown format
        print("Unknown outputs format — printing repr")
        print(outputs)
        sys.exit(1)

    print("shapes: locs", getattr(locs, "shape", None), "confs", getattr(confs, "shape", None), "anchors", getattr(anchors, "shape", None))

    # confs shape expected (B,N,C) -> compute probs
    if confs is not None:
        confs_cpu = confs.detach().cpu()
        try:
            probs = F.softmax(confs_cpu[0], dim=-1)  # (N,C)
        except Exception as e:
            print("softmax failed:", e)
            probs = None
        if probs is not None:
            # assume foreground class index 1
            if probs.shape[-1] >= 2:
                fg = probs[:, 1].numpy()
            else:
                fg = probs.max(dim=-1)[0].numpy()
            print("FG probs stats: min,median,mean,max:",
                  np.nanmin(fg), np.nanmedian(fg), np.nanmean(fg), np.nanmax(fg))
            # top 20
            top_idx = np.argsort(-fg)[:20]
            print("Top 20 foreground probs:", list(np.round(fg[top_idx], 6)))
            # percentiles
            perc = {p: float(np.percentile(fg, p)) for p in (50,75,90,95,99)}
            print("percentiles:", perc)
        else:
            print("Could not compute probabilities (probs is None).")
    else:
        print("confs is None, cannot show probs")

    # anchors processing: anchors might be normalized (cx,cy,w,h) or xyxy. Try to print a few.
    anchors_cpu = anchors.detach().cpu()
    print("anchors example (first 5):", anchors_cpu[:5])

    # If anchors look like cx,cy,w,h in pixels / normalized, attempt to unify to xyxy in pixels:
    a = anchors_cpu.numpy()
    # heuristics:
    if a.shape[1] == 4:
        # are anchors in cx,cy,w,h normalized (0..1) ?
        max_val = float(np.abs(a).max())
        anchors_are_normalized = max_val <= 1.001
        print("anchors_are_normalized:", anchors_are_normalized, "max:", max_val)
        if anchors_are_normalized:
            # convert cx,cy,w,h -> x1,y1,x2,y2 in pixels
            cx = a[:,0] * args.image_size
            cy = a[:,1] * args.image_size
            w = a[:,2] * args.image_size
            h = a[:,3] * args.image_size
            x1 = cx - 0.5*w
            y1 = cy - 0.5*h
            x2 = cx + 0.5*w
            y2 = cy + 0.5*h
            anchors_xyxy = np.stack([x1,y1,x2,y2], axis=1)
        else:
            # anchors maybe already xyxy in pixel coords or cxcywh in pixels
            # if values look like centers and sizes (w/h small vs image_size), treat as cxcywh (pixel)
            if (a[:,2].mean() < args.image_size*0.9):
                # treat as cxcywh in pixels
                cx = a[:,0]
                cy = a[:,1]
                w = a[:,2]
                h = a[:,3]
                x1 = cx - 0.5*w
                y1 = cy - 0.5*h
                x2 = cx + 0.5*w
                y2 = cy + 0.5*h
                anchors_xyxy = np.stack([x1,y1,x2,y2], axis=1)
            else:
                # assume anchors are xyxy already
                anchors_xyxy = a.copy()
        print("anchors_xyxy sample:", anchors_xyxy[:5])
    else:
        print("Anchors shape not (N,4) — skip anchor visualization")
        anchors_xyxy = None

    # If probs exist, decode using decode_preds if available
    final_boxes = np.zeros((0,4))
    final_labels = np.zeros((0,), dtype=int)
    final_scores = np.zeros((0,))
    if decode_preds is not None:
        try:
            boxes_t, labels_t, scores_t = decode_preds((locs.cpu(), confs.cpu(), anchors.cpu()), score_threshold=args.score_thresh, topk=args.topk, iou_threshold=args.iou)
            final_boxes = boxes_t.numpy()
            final_labels = labels_t.numpy()
            final_scores = scores_t.numpy()
            print("Decoded final boxes:", final_boxes.shape[0])
        except Exception as e:
            print("decode_preds call failed:", e)
    else:
        print("No decode util available in repo (decode_preds is None). Cannot decode predictions.")

    # Visualize:
    # draw anchors (first 200) on resized image, and overlay final boxes (if any)
    out_img = pil_resized.copy()
    if anchors_xyxy is not None:
        # draw first 500 anchors faintly
        try:
            out_img = draw_anchors_on_image(out_img, anchors_xyxy, topk=500)
        except Exception as e:
            print("drawing anchors failed:", e)
    if final_boxes.shape[0] > 0:
        try:
            out_img = draw_boxes(out_img, final_boxes, scores=final_scores, labels=final_labels, color=(0,255,0), width=3)
        except Exception as e:
            print("drawing final boxes failed:", e)

    out_img.save(args.out)
    print("Saved debug image to", args.out)

if __name__ == "__main__":
    main()
