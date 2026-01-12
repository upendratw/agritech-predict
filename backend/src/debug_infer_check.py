# debug_infer_check.py
import torch, sys, argparse
from PIL import Image
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--image_size", type=int, default=512)
args = parser.parse_args()

device = torch.device(args.device)
print("Device:", device)

# Import model factory (adjust path if needed)
try:
    # If your repo layout requires it, run from repo root:
    # export PYTHONPATH=$(pwd)
    from single_stage_detector import SingleStageDetector  # or src.single_stage_detector if that is correct
except Exception as e:
    print("Failed to import SingleStageDetector:", e)
    raise

# construct model
model = SingleStageDetector(num_classes=2, image_size=args.image_size)  # adjust args if your constructor differs
model.to(device)

# load weights (handle both state_dict and full checkpoint)
ck = torch.load(args.weights, map_location=device)
if isinstance(ck, dict) and "model_state" in ck:
    sd = ck["model_state"]
else:
    sd = ck
# ensure state_dict tensors are same device dtype (torch will handle type but we move model and then load)
try:
    model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint into model (non-strict).")
except Exception as e:
    print("load_state_dict failed:", e)
    # try to load partially
    model.load_state_dict(sd, strict=False)
    print("Loaded partially into model (non-strict fallback).")

model.eval()

# Prepare input
pil = Image.open(args.image).convert("RGB")
# resize to image_size preserving aspect (this depends on how your dataset used transforms)
t = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
tensor = t(pil).unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)

with torch.no_grad():
    out = model(tensor)  # expected (locs, confs, anchors) or similar
    print("Model forward done. Outputs type:", type(out))
    if isinstance(out, (list, tuple)) and len(out) >= 3:
        locs, confs, anchors = out[0], out[1], out[2]
        print("shapes: locs", locs.shape, "confs", confs.shape, "anchors", anchors.shape)
        # anchors stats
        try:
            a = anchors.detach().cpu()
            print("anchors min/max/mean:", float(a.min()), float(a.max()), float(a.mean()))
            print("anchors sample (first 10):", a[:10].tolist())
        except Exception as e:
            print("anchors inspect failed:", e)

        # confidences -> probs for foreground (if confs are logits)
        import torch.nn.functional as F
        confs_cpu = confs.detach().cpu()
        if confs_cpu.ndim == 3:
            probs = F.softmax(confs_cpu[0], dim=-1)[:, 1]  # assuming class 1 is foreground
            print("FG probs stats: min,median,mean,max:", float(probs.min()), float(probs.median()), float(probs.mean()), float(probs.max()))
            print("Top-10 probs:", probs.topk(10).values.tolist())
        else:
            print("confs unexpected shape for softmax:", confs_cpu.shape)
    else:
        print("Unexpected model output. Please adapt debug script for your model's forward signature.")