# src/data/transforms.py
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch

def letterbox_resize(image: Image.Image, target_size: Tuple[int, int] = (512, 512), color=(114, 114, 114)):
    """
    Resize PIL image to target_size using letterbox (keep aspect ratio + pad).
    Returns:
      new_img (PIL.Image),
      scale (float) -- scaling factor applied to original,
      pad (pad_left, pad_top) -- integer pixels
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size

    if orig_w == 0 or orig_h == 0:
        raise ValueError("Invalid image with zero width or height")

    # scale ratio (min of target/orig)
    scale = min(float(target_w) / orig_w, float(target_h) / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # resize image
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # compute padding (centered)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # create new image and paste (RGB)
    new_img = Image.new("RGB", (target_w, target_h), color)
    new_img.paste(resized, (pad_left, pad_top))

    return new_img, scale, (pad_left, pad_top)


def transform_bboxes_letterbox(boxes: Optional[np.ndarray], scale: float, pad: Tuple[int, int]):
    """
    Transform bounding boxes to letterbox-resized image coordinates.
    Args:
      boxes: (N,4) array-like in original image coords [x1,y1,x2,y2] or None/empty
      scale: scaling factor returned by letterbox_resize
      pad: (pad_left, pad_top)
    Returns:
      transformed_boxes: (N,4) numpy array (float32) clipped to image
    """
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pad_left, pad_top = pad
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left  # x coords
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top   # y coords
    return boxes.astype(np.float32)


def to_torch_boxes(a):
    """Convert numpy array to torch float tensor Nx4"""
    if a is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    arr = np.asarray(a, dtype=np.float32)
    if arr.size == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.from_numpy(arr)


def resize_image_and_boxes_pil(pil_img: Image.Image, boxes, target_size=(512, 512)):
    """
    Convenience: resize PIL image + transform boxes to new coords, and clip.
    Returns:
      new_img (PIL.Image),
      new_boxes (torch.FloatTensor Nx4)
    """
    new_img, scale, pad = letterbox_resize(pil_img, target_size=target_size)
    boxes_np = np.asarray(boxes, dtype=np.float32) if boxes is not None else np.zeros((0, 4), dtype=np.float32)
    new_boxes = transform_bboxes_letterbox(boxes_np, scale, pad)

    # clip to target
    tw, th = target_size
    if new_boxes.size:
        new_boxes[:, 0::2] = np.clip(new_boxes[:, 0::2], 0, tw - 1)
        new_boxes[:, 1::2] = np.clip(new_boxes[:, 1::2], 0, th - 1)

    return new_img, to_torch_boxes(new_boxes)