# tools/test_load_strict.py
import torch, argparse
from importlib import import_module
import os
import sys

# make tools find src under backend/ or frontend/ automatically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_src_root = os.path.join(project_root, "backend")
frontend_src_root = os.path.join(project_root, "frontend")

# Insert possible package roots at front of sys.path so imports like `import src...` work.
for p in (backend_src_root, frontend_src_root):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--model_mod", default="src.models.ssd")
parser.add_argument("--model_cls", default="SSD")
parser.add_argument("--num_classes", type=int, default=None)
args = parser.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
state = ck.get("model_state", ck)
num_classes = args.num_classes or ck.get("num_classes", None) or 3

mod = import_module(args.model_mod)
ModelCls = getattr(mod, args.model_cls)
model = ModelCls(num_classes=num_classes)
try:
    model.load_state_dict(state, strict=True)
    print("Strict load succeeded.")
except Exception as e:
    print("Strict load FAILED:")
    print(e)