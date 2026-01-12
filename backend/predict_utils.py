# backend/predict_utils.py
import os
import io
import base64
import numpy as np
from typing import Tuple, Dict, Any, List

# prefer ultralytics YOLO if available
try:
    from ultralytics import YOLO
    _HAVE_ULTRALYTICS = True
except Exception:
    _HAVE_ULTRALYTICS = False

from PIL import Image


def select_device_name(device_requested: str = "auto") -> str:
    """
    Return a device string usable by ultralytics / torch: 'mps' | 'cpu' | 'cuda:0'.
    """
    d = device_requested.lower() if device_requested else "auto"
    if d == "auto":
        # prefer mps on Apple silicon, else cuda, else cpu
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return d


def load_yolo_model(weights_path: str, device: str = "cpu"):
    """
    Loads ultralytics YOLO model object (YOLO class).
    Requires `ultralytics` Python package.
    """
    if not _HAVE_ULTRALYTICS:
        raise RuntimeError("ultralytics package not installed. Install with `pip install ultralytics`")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    # YOLO constructor accepts model path and returns a model object with .predict()
    model = YOLO(weights_path)
    # set device (model.predict accepts device param too)
    # We keep model as-is; pass device to predict call
    return model


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _image_to_base64_bytes(img: Image.Image, fmt: str = "PNG") -> str:
    out = io.BytesIO()
    img.save(out, format=fmt)
    b = out.getvalue()
    return base64.b64encode(b).decode("ascii")


def infer_from_bytes(model, image_bytes: bytes, imgsz: int = 640, conf: float = 0.25, device: str = "cpu") -> Dict[str, Any]:
    """
    Run inference on a single image provided as bytes using a loaded ultralytics YOLO model.

    Returns dict:
      - detections: list of {box: [x1,y1,x2,y2], conf: float, class_id: int, class_name: str}
      - annotated_image_b64: base64 PNG (if available)
    """
    if not _HAVE_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed")

    pil_img = _pil_from_bytes(image_bytes)

    # ultralytics expects numpy array (H,W,3) or path
    img_np = np.array(pil_img)

    # run predict
    # model.predict returns a Results object (or list of Results)
    # pass imgsz, conf, device to predict for consistent behavior
    results = model.predict(source=img_np, imgsz=imgsz, conf=conf, device=device, augment=False, verbose=False)

    if not results:
        return {"detections": [], "annotated_image_b64": None}

    res0 = results[0]  # first (and only) image

    detections: List[Dict[str, Any]] = []
    # res0.boxes exists (Boxes object) if any detections, else empty
    try:
        boxes_obj = getattr(res0, "boxes", None)
        if boxes_obj is None or len(boxes_obj) == 0:
            detections = []
        else:
            # boxes_obj.xyxy, boxes_obj.conf, boxes_obj.cls
            # Convert to numpy
            xyxy = boxes_obj.xyxy.cpu().numpy()  # shape (N,4)
            confs = boxes_obj.conf.cpu().numpy() if getattr(boxes_obj, "conf", None) is not None else [1.0] * len(xyxy)
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int) if getattr(boxes_obj, "cls", None) is not None else [0] * len(xyxy)

            # class names mapping available as model.names (dict)
            names_map = getattr(model, "names", None) or {}

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in box]
                confv = float(confs[i]) if i < len(confs) else 1.0
                cid = int(cls_ids[i]) if i < len(cls_ids) else 0
                name = names_map.get(cid, str(cid)) if isinstance(names_map, dict) else str(cid)
                detections.append({"box": [x1, y1, x2, y2], "conf": confv, "class_id": cid, "class_name": name})
    except Exception:
        # fallback: try accessing res0.boxes.xyxy etc. If fails, be permissive.
        detections = []

    # annotated image (plot)
    annotated_b64 = None
    try:
        # res0.plot() returns an image (numpy array HWC) with boxes drawn
        plotted = res0.plot()  # numpy array
        pil_plotted = Image.fromarray(plotted)
        annotated_b64 = _image_to_base64_bytes(pil_plotted, fmt="PNG")
    except Exception:
        # fallback: return original image as base64
        annotated_b64 = _image_to_base64_bytes(pil_img, fmt="PNG")

    return {"detections": detections, "annotated_image_b64": annotated_b64}