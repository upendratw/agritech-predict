# tools/compare_ckpt_model.py
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

ckpt = torch.load(args.ckpt, map_location="cpu")
state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

print("Checkpoint has", len(state), "keys. First 30 keys:")
for k in list(state.keys())[:30]:
    print("  ", k, tuple(state[k].shape) if hasattr(state[k],"shape") else None)

# import model and create instance
mod = import_module(args.model_mod)
ModelCls = getattr(mod, args.model_cls)

if args.num_classes is None:
    num_classes = ckpt.get("num_classes", None)
    if num_classes is None:
        print("Warning: checkpoint doesn't have num_classes metadata. Provide --num_classes if known.")
        num_classes = 3
else:
    num_classes = args.num_classes

print("\nInstantiating model with num_classes =", num_classes)
model = ModelCls(num_classes=num_classes)
mstate = model.state_dict()

print("\nModel has", len(mstate), "keys. First 30 keys:")
for k in list(mstate.keys())[:30]:
    print("  ", k, tuple(mstate[k].shape) if hasattr(mstate[k],"shape") else None)

m_only = set(mstate.keys()) - set(state.keys())
ckpt_only = set(state.keys()) - set(mstate.keys())
both = set(state.keys()) & set(mstate.keys())

print("\nKeys only in model:", len(m_only))
for k in sorted(list(m_only))[:40]:
    print("  model-only:", k, tuple(mstate[k].shape))

print("\nKeys only in checkpoint:", len(ckpt_only))
for k in sorted(list(ckpt_only))[:40]:
    print("  ckpt-only:", k, tuple(state[k].shape))

print("\nKeys in both but with shape mismatch:")
count = 0
for k in sorted(list(both)):
    s1 = tuple(state[k].shape) if hasattr(state[k],"shape") else None
    s2 = tuple(mstate[k].shape) if hasattr(mstate[k],"shape") else None
    if s1 != s2:
        print("  MISMATCH:", k, "ckpt:", s1, "model:", s2)
        count += 1
print("Total mismatches:", count)