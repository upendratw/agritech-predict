# tools/remap_and_load.py
import torch
from importlib import import_module
import argparse
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
parser.add_argument("--num_classes", type=int, default=3)
args = parser.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
state = ck.get("model_state", ck) if isinstance(ck, dict) else ck

mod = import_module(args.model_mod)
ModelCls = getattr(mod, args.model_cls)
model = ModelCls(num_classes=args.num_classes)
mstate = model.state_dict()

# Build mapping from suffix -> list of keys
def suffixes(k):
    parts = k.split('.')
    # try different suffix lengths
    out = []
    for L in range(1, min(6,len(parts))+1):
        out.append('.'.join(parts[-L:]))
    return out

ck_map = {}
for k in state.keys():
    for s in suffixes(k):
        ck_map.setdefault(s, []).append(k)

new_state = {}
used_ck = set()
for mk in mstate.keys():
    matched = None
    for s in suffixes(mk):
        candidates = ck_map.get(s, [])
        if len(candidates) == 1:
            matched = candidates[0]
            break
        elif len(candidates) > 1:
            # prefer exact shape match
            for c in candidates:
                if tuple(state[c].shape) == tuple(mstate[mk].shape):
                    matched = c
                    break
            if matched:
                break
    if matched:
        new_state[mk] = state[matched]
        used_ck.add(matched)
    else:
        print("NO MATCH for model key:", mk)

missing_from_ckpt = set(state.keys()) - used_ck
if missing_from_ckpt:
    print("Checkpoint keys not used:", len(missing_from_ckpt))
    print(list(missing_from_ckpt)[:30])

# Try load
try:
    model.load_state_dict(new_state, strict=True)
    print("Loaded after remapping strict=True succeeded.")
except Exception as e:
    print("Strict load failed after remap:", e)
    # try non-strict to see how many loaded
    res = model.load_state_dict(new_state, strict=False)
    print("Non-strict load result:", res)