# src/infer.py
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np

from src.models.ssd import SSD
from src.models import loss as loss_module

# attempt to import decoding utilities from your project utils
try:
    from src.models import utils as box_utils
except Exception:
    box_utils = None


def infer_num_classes_from_state(state_dict: dict):
    head_indices = set()
    for k in state_dict.keys():
        if k.startswith("pred_heads."):
            parts = k.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    head_indices.add(idx)
                except Exception:
                    continue
    head_indices = sorted(head_indices)
    if not head_indices:
        return None

    inferred = []
    for i in head_indices:
        loc_key = f"pred_heads.{i}.loc_conv.weight"
        cls_key = f"pred_heads.{i}.cls_conv.weight"
        if loc_key not in state_dict or cls_key not in state_dict:
            return None
        loc_out = state_dict[loc_key].shape[0]  # anchors*4
        cls_out = state_dict[cls_key].shape[0]  # anchors * num_classes
        if loc_out % 4 != 0:
            return None
        anchors = loc_out // 4
        if anchors <= 0:
            return None
        if cls_out % anchors != 0:
            return None
        classes = cls_out // anchors
        inferred.append((i, anchors, classes))

    classes_set = set(c for (_, _, c) in inferred)
    if len(classes_set) == 1:
        return inferred[0][2]
    return None


def load_checkpoint(ckpt_path: str, device: torch.device, num_classes_arg: int = None):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        state_dict = ckpt_obj["model_state"]
    elif isinstance(ckpt_obj, dict) and any(k.startswith("pred_heads.") for k in ckpt_obj.keys()):
        state_dict = ckpt_obj
    elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state_dict = ckpt_obj["state_dict"]
    else:
        state_dict = ckpt_obj

    inferred = infer_num_classes_from_state(state_dict if isinstance(state_dict, dict) else {})
    if inferred is not None:
        num_classes = int(inferred)
        print(f"Inferred num_classes (including background) from checkpoint: {num_classes}")
    elif num_classes_arg:
        num_classes = int(num_classes_arg)
        print(f"Using provided --num_classes={num_classes}")
    else:
        num_classes = 2
        print("Could not infer num_classes and --num_classes not provided. Falling back to num_classes=2 (bg+1).")

    model = SSD(num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint (strict=True).")
    except Exception as e:
        print("Strict load failed:", e)
        print("Retrying with strict=False (will skip unmatched params).")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: str, image_size: int):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    tensor = transform(img)  # C,H,W  float [0,1]
    return img, tensor


def visualize_and_save(img_pil: Image.Image, boxes, labels, scores, labels_map, out_path: str, score_thresh=0.3):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, box in enumerate(boxes):
        score = float(scores[i])
        if score < score_thresh:
            continue
        x, y, w, h = box
        # box may be in xywh; if utilities produced xyxy, try to detect:
        if len(box) == 4:
            # if width looks like width (<=1e4) assume xywh; otherwise xyxy
            if w > 1.0 and h > 1.0 and x >= 0 and y >= 0:
                # treat as xywh
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                x1, y1, x2, y2 = x, y, w, h
        else:
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label_txt = labels_map.get(labels[i], str(labels[i])) if labels_map else str(labels[i])
        txt = f"{label_txt}:{score:.2f}"
        draw.text((x1 + 4, max(y1 - 10, 0)), txt, fill="white", font=font)
    img_pil.save(out_path)
    print("Saved visualization to:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_classes", type=int, default=None, help="(optional) include background")
    parser.add_argument("--score_thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--out", type=str, default="pred.png", help="Output visualization path")
    parser.add_argument("--labels", type=str, default=None, help="Optional labels.txt (one label per line) to map ids to names (excluding background order must match categories)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device, num_classes_arg=args.num_classes)

    orig_pil, tensor = preprocess_image(args.image, args.image_size)
    input_tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # outputs may be (loc_preds, cls_preds, anchors) or (loc, cls) or decoded depending on model implementation
    # If model returns decoded boxes directly, handle it
    boxes = None
    labels = None
    scores = None

    # try to decode with project utils if available
    if box_utils is not None:
        try:
            # common helper name might be decode_predictions / decode / decode_boxes
            if hasattr(box_utils, "decode_predictions"):
                boxes, labels, scores = box_utils.decode_predictions(outputs, score_threshold=args.score_thresh, topk=args.topk)
            elif hasattr(box_utils, "decode_from_raw"):
                boxes, labels, scores = box_utils.decode_from_raw(outputs, score_threshold=args.score_thresh, topk=args.topk)
            elif hasattr(box_utils, "decode_boxes") and hasattr(box_utils, "convert_scores_to_labels"):
                # some projects separate decode box & score logic
                locs, cls_logits, anchors = outputs if isinstance(outputs, (list, tuple)) else (outputs, None, None)
                boxes = box_utils.decode_boxes(anchors, locs)
                labels, scores = box_utils.convert_scores_to_labels(cls_logits, topk=args.topk, score_threshold=args.score_thresh)
            else:
                # fallback: try to interpret outputs as (loc, cls, anchors)
                if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                    locs, cls_logits = outputs[0], outputs[1]
                    anchors = outputs[2] if len(outputs) > 2 else None
                    if hasattr(box_utils, "decode_boxes"):
                        boxes = box_utils.decode_boxes(anchors, locs)
                    else:
                        boxes = None
                    # convert cls_logits -> scores/labels (softmax)
                    probs = torch.softmax(cls_logits, dim=-1)
                    probs = probs.cpu().numpy()
                    # naive selection: argmax per anchor
                    cls_ids = probs.argmax(axis=-1).squeeze(0)  # (n_anchors,)
                    cls_scores = probs.max(axis=-1).squeeze(0)
                    labels = cls_ids.tolist() if isinstance(cls_ids, np.ndarray) else [int(cls_ids)]
                    scores = cls_scores.tolist() if isinstance(cls_scores, np.ndarray) else [float(cls_scores)]
                else:
                    boxes = None
        except Exception as e:
            print("Decoding via box_utils failed:", e)
            boxes = None

    if boxes is None:
        # Save raw outputs to disk for inspection if decoding not available
        raw_out = {}
        if isinstance(outputs, (list, tuple)):
            for i, o in enumerate(outputs):
                try:
                    raw_out[f"out_{i}"] = o.cpu()
                except Exception:
                    raw_out[f"out_{i}"] = torch.tensor(o)
        else:
            raw_out["out_0"] = outputs.cpu() if torch.is_tensor(outputs) else outputs
        raw_path = Path(args.out).with_suffix(".raw.pt")
        torch.save(raw_out, str(raw_path))
        print("Could not decode boxes (no decode helper available). Raw model outputs saved to:", raw_path)
        print("If you have a decode function in src.models.utils, implement one named 'decode_predictions(outputs, score_threshold, topk)' that returns (boxes, labels, scores).")
        return

    # load labels map if provided
    labels_map = {}
    if args.labels:
        try:
            with open(args.labels, "r", encoding="utf-8") as fr:
                lines = [l.strip() for l in fr.readlines() if l.strip()]
                # assume categories in labels.txt are in order and ids start from 1 (if your categories in training were 1-indexed)
                # We'll map 1->lines[0], 2->lines[1], ...
                for idx, name in enumerate(lines, start=1):
                    labels_map[idx] = name
        except Exception:
            pass

    # ensure lists/np arrays
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # optionally filter by score threshold
    keep = scores >= args.score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    print("Predictions (kept):", len(boxes))
    for i, (b, lab, sc) in enumerate(zip(boxes.tolist(), labels.tolist(), scores.tolist())):
        lab_txt = labels_map.get(int(lab), int(lab))
        print(f"{i}: label={lab_txt}, score={sc:.3f}, box={b}")

    # visualize
    visualize_and_save(orig_pil, boxes.tolist(), labels.tolist(), scores.tolist(), labels_map, args.out, score_thresh=args.score_thresh)


if __name__ == "__main__":
    import argparse

    main()