# coco_dataset.py
"""
Backward-compatible CocoDataset for agritech-detector.

Drop-in file path:
  agritech-detector/src/data/coco_dataset.py

Features:
- Accepts constructor args:
    annotations_json=..., OR annotations=...
    images_root=..., OR images=...
    image_size=..., OR target_size=...
  where image_size can be int, str like "512" or "512,512" or tuple/list (512,512).
- Letterbox (aspect-ratio preserved) resize to square S x S and maps boxes accordingly.
- Returns (img_tensor, target) where target = {"boxes": Tensor[N,4] (xyxy), "labels": Tensor[N]}.
- Robust file lookup (tries several fallbacks).
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional, List, Tuple, Callable, Any, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _parse_image_size(value, default=512) -> int:
    """
    Accepts int, str, tuple/list and returns single int.
    Examples:
      512 -> 512
      "512" -> 512
      "512,512" -> 512
      "512 512" -> 512
      (512,512) -> 512
      [512,512] -> 512
    """
    if value is None:
        return int(default)
    # already an int
    if isinstance(value, int):
        return value
    # float -> int
    if isinstance(value, float):
        return int(value)
    # tuple or list
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return int(default)
        first = value[0]
        return _parse_image_size(first, default=default)
    # string: extract first number
    if isinstance(value, str):
        # find integer substrings
        nums = re.findall(r"\d+", value)
        if len(nums) == 0:
            raise ValueError(f"Cannot parse image size from string: {value!r}")
        return int(nums[0])
    # torch/tensor-like?
    try:
        if hasattr(value, "item"):
            return int(value.item())
    except Exception:
        pass
    raise TypeError(f"Unsupported type for image_size: {type(value)}")


def letterbox_resize_and_map_boxes(
    pil_img: Image.Image,
    boxes_xyxy: np.ndarray,
    S: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, np.ndarray, float, int, int]:
    """
    Resize PIL image keeping aspect ratio and pad to S x S (centered).
    boxes_xyxy: numpy array shape (N,4) in (x1,y1,x2,y2) coordinates relative to original image.
    Returns: (new_pil_img, boxes_mapped (N,4), scale, pad_x, pad_y)
    """
    orig_w, orig_h = pil_img.size  # (W, H)
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Image has zero width or height")

    scale = min(S / orig_w, S / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new("RGB", (S, S), color=fill)
    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2
    new_img.paste(img_resized, (pad_x, pad_y))

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        boxes_scaled = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        x1 = boxes_xyxy[:, 0] * scale + pad_x
        y1 = boxes_xyxy[:, 1] * scale + pad_y
        x2 = boxes_xyxy[:, 2] * scale + pad_x
        y2 = boxes_xyxy[:, 3] * scale + pad_y
        boxes_scaled = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    return new_img, boxes_scaled, scale, pad_x, pad_y


class CocoDataset(Dataset):
    """
    COCO-style dataset.

    Constructor supports multiple kwarg names for compatibility:
      - annotations_json (preferred) or annotations
      - images_root (preferred) or images
      - image_size (preferred) or target_size

    Returns:
      img_tensor: torch.FloatTensor C x H x W (values 0..1)
      target: dict with keys "boxes" (FloatTensor Nx4 in xyxy coords relative to transformed image)
                       and "labels" (LongTensor N)
    """

    def __init__(
        self,
        annotations_json: Optional[str] = None,
        images_root: Optional[str] = None,
        transforms: Optional[Callable[[Image.Image], Any]] = None,
        image_size: Optional[Any] = None,
        # backward-compatible names
        annotations: Optional[str] = None,
        images: Optional[str] = None,
        target_size: Optional[Any] = None,
    ):
        # Resolve backward-compatible names
        self.annotations_json = annotations_json or annotations
        self.images_root = images_root or images
        # prefer explicit image_size, else target_size
        size_value = image_size if image_size is not None else target_size
        self.image_size = _parse_image_size(size_value, default=512)

        if not self.annotations_json:
            raise ValueError("annotations_json (or annotations) path must be provided")
        if not self.images_root:
            raise ValueError("images_root (or images) path must be provided")

        self.transforms = transforms
        self.records: List[Dict[str, Any]] = []
        self.cat_map: Dict[int, str] = {}

        self._load_json()

        if self.transforms is None:
            self.to_tensor = T.ToTensor()
        else:
            self.to_tensor = self.transforms

    def _load_json(self):
        if not os.path.isfile(self.annotations_json):
            raise FileNotFoundError(f"Annotations JSON not found: {self.annotations_json}")

        with open(self.annotations_json, "r") as f:
            data = json.load(f)

        # category id -> name
        for c in data.get("categories", []):
            self.cat_map[int(c["id"])] = c.get("name", str(c["id"]))

        images_index = {}
        for im in data.get("images", []):
            images_index[int(im["id"])] = {
                "file_name": im["file_name"],
                "width": im.get("width"),
                "height": im.get("height"),
            }

        ann_by_img = {}
        for ann in data.get("annotations", []):
            img_id = int(ann["image_id"])
            ann_by_img.setdefault(img_id, []).append(ann)

        for img_id, meta in images_index.items():
            file_name = meta["file_name"]
            width = meta.get("width")
            height = meta.get("height")

            anns = ann_by_img.get(img_id, [])
            boxes = []
            labels = []
            for a in anns:
                bbox = a.get("bbox", None)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                x1 = float(x)
                y1 = float(y)
                x2 = float(x + w)
                y2 = float(y + h)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(a.get("category_id", 0)))

            candidate = os.path.join(self.images_root, file_name)
            if not os.path.isfile(candidate):
                candidate_alt = os.path.join(self.images_root, os.path.basename(file_name))
                if os.path.isfile(candidate_alt):
                    candidate = candidate_alt
                else:
                    found = None
                    for root, _, files in os.walk(self.images_root):
                        if os.path.basename(file_name) in files:
                            found = os.path.join(root, os.path.basename(file_name))
                            break
                    if found:
                        candidate = found
                    else:
                        candidate = None

            self.records.append(
                {
                    "image_id": img_id,
                    "file_name": file_name,
                    "file_path": candidate,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        fp = rec["file_path"]
        if fp is None or not os.path.isfile(fp):
            raise FileNotFoundError(f"Image file not found: {fp} (record idx={idx}, file_name={rec['file_name']})")

        pil_img = Image.open(fp).convert("RGB")
        boxes_xyxy = np.array(rec["boxes"], dtype=np.float32) if len(rec["boxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
        labels_list = rec["labels"]

        img_resized_pil, boxes_mapped_np, scale, pad_x, pad_y = letterbox_resize_and_map_boxes(
            pil_img, boxes_xyxy, S=self.image_size
        )

        img_tensor = self.to_tensor(img_resized_pil)

        boxes_tensor = torch.from_numpy(boxes_mapped_np) if boxes_mapped_np.size else torch.zeros((0, 4), dtype=torch.float32)
        if labels_list:
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
        else:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    ds = CocoDataset(annotations_json=args.annotations, images_root=args.images_root, image_size=args.image_size)
    print(f"Constructed CocoDataset with {len(ds)} samples.")
    img, tgt = ds[args.index]
    print("Image tensor shape:", img.shape)
    print("Target keys:", list(tgt.keys()))
    print("Boxes shape:", tgt["boxes"].shape)