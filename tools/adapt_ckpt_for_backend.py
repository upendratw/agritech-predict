# tools/adapt_ckpt_for_backend.py
import argparse
from pathlib import Path
import torch
import numpy as np

def safe_adapt(src, tgt_shape):
    src_shape = tuple(src.shape)
    if src_shape == tgt_shape:
        return src
    if len(src_shape) != len(tgt_shape):
        return None
    # If inner dims match except first dim => tile/truncate along dim 0
    if src_shape[1:] == tgt_shape[1:]:
        s0 = src_shape[0]
        t0 = tgt_shape[0]
        arr = src.cpu().numpy()
        if s0 == t0:
            return torch.from_numpy(arr).clone()
        if s0 > t0:
            return torch.from_numpy(arr[:t0].copy())
        reps = int((t0 + s0 - 1) // s0)
        arr2 = np.tile(arr, (reps,) + (1,) * (arr.ndim - 1))
        arr2 = arr2[:t0].copy()
        return torch.from_numpy(arr2)
    # If inner dims match except second dim => adapt on dim=1
    if src_shape[0] == tgt_shape[0] and src_shape[2:] == tgt_shape[2:]:
        s1 = src_shape[1]
        t1 = tgt_shape[1]
        arr = src.cpu().numpy()
        if s1 == t1:
            return torch.from_numpy(arr).clone()
        if s1 > t1:
            return torch.from_numpy(arr[:, :t1].copy())
        reps = int((t1 + s1 - 1) // s1)
        arr2 = np.tile(arr, (1, reps) + (1,) * (arr.ndim - 2))
        arr2 = arr2[:, :t1].copy()
        return torch.from_numpy(arr2)
    return None

def adapt_state_dict_for_model(ckpt_state, model_state):
    adapted = {}
    stats = {"adapted": [], "skipped": [], "mismatched": [], "used": []}
    for k_mod, v_mod in model_state.items():
        if k_mod in ckpt_state:
            v_ck = ckpt_state[k_mod]
            if tuple(v_ck.shape) == tuple(v_mod.shape):
                adapted[k_mod] = v_ck
                stats["used"].append(k_mod)
            else:
                adapted_tensor = safe_adapt(v_ck, tuple(v_mod.shape))
                if adapted_tensor is not None:
                    adapted[k_mod] = adapted_tensor
                    stats["adapted"].append((k_mod, tuple(v_ck.shape), tuple(v_mod.shape)))
                else:
                    stats["mismatched"].append((k_mod, tuple(v_ck.shape), tuple(v_mod.shape)))
                    adapted[k_mod] = v_mod  # keep model default
        else:
            stats["skipped"].append(k_mod)
            adapted[k_mod] = v_mod
    return adapted, stats

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_state_example", required=False, help="Path to an example model_state dict or None")
    p.add_argument("--out", default="adapted_ckpt.pth")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Accept both wrapper and bare dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt_state = ckpt["model_state"]
    else:
        # assume the ckpt variable is the state dict
        ckpt_state = ckpt if isinstance(ckpt, dict) else {}
    # If user provided a saved model_state to use as target shapes, load it.
    if args.model_state_example:
        model_state = torch.load(args.model_state_example, map_location="cpu")
    else:
        # Otherwise, we need you to run this script from your app and produce model.state_dict()
        print("No model_state_example provided. Please provide your current model.state_dict() saved to disk.")
        print("Example: in your app run: torch.save(model.state_dict(), 'model_state_example.pth') and pass that path.")
        raise SystemExit(1)

    adapted_state, stats = adapt_state_dict_for_model(ckpt_state, model_state)
    # Save adapted checkpoint wrapper similar to original
    out = {"model_state": adapted_state}
    # preserve other top-level keys if present
    if isinstance(ckpt, dict):
        for k in ckpt.keys():
            if k not in ("model_state",):
                out[k] = ckpt[k]
    torch.save(out, args.out)
    print("Wrote adapted checkpoint to", args.out)
    print("Stats:", stats)