#!/usr/bin/env python3
"""Read a YOLOX checkpoint and report what it actually contains.

    python tools/inspect_ckpt.py /path/to/best_ckpt.pth

Answers, from the weights themselves rather than from assumption:
  - how many classes the head was trained for
  - which YOLOX size the backbone is (nano/tiny/s/m/l/x)
  - whether it is a depthwise (nano) variant
  - what else the file carries (epoch, best AP, optimizer state, class names)

Needs torch. Nothing else here does, which is the point: run this before
installing the rest of the detector stack, so a mismatch shows up now.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# stem.conv output channels -> (width multiplier, YOLOX size name)
WIDTHS = {24: (0.375, "tiny / nano"), 32: (0.50, "s"), 48: (0.75, "m"),
          64: (1.00, "l"), 80: (1.25, "x")}
# dark2 CSPLayer bottleneck count -> depth multiplier
DEPTHS = {1: (0.33, "s / tiny / nano"), 2: (0.67, "m"), 3: (1.00, "l"), 4: (1.33, "x")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--show-keys", type=int, default=0,
                    help="print the first N state-dict keys")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed in this environment.\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
              "(CPU wheels are ~200MB; a YOLOX-s forward pass on CPU is fine for a "
              "demo-sized batch.)")
        return 1

    path = args.checkpoint
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    print(f"file        : {path}")
    print(f"size        : {path.stat().st_size:,} bytes")
    print(f"torch       : {torch.__version__}")
    print(f"cuda        : {torch.cuda.is_available()}"
          + (f" ({torch.cuda.device_count()} device(s))" if torch.cuda.is_available() else ""))
    print()

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"Unexpected top-level type: {type(ckpt)}")
        return 1

    print("top-level keys:")
    for key in ckpt:
        value = ckpt[key]
        if hasattr(value, "keys"):
            print(f"  {key:<18} dict with {len(value)} entries")
        elif hasattr(value, "shape"):
            print(f"  {key:<18} tensor {tuple(value.shape)}")
        else:
            print(f"  {key:<18} {value!r}"[:110])
    print()

    state = ckpt.get("model", ckpt)
    if not hasattr(state, "keys"):
        print("No state dict found under 'model'.")
        return 1
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    print(f"state dict  : {len(state)} tensors")

    # ── Classes, from the head ───────────────────────────────────────────────
    num_classes = None
    for key in state:
        if key.endswith("cls_preds.0.weight") or key.endswith("cls_preds.0.bias"):
            num_classes = state[key].shape[0]
            print(f"num_classes : {num_classes}   (from {key})")
            break
    if num_classes is None:
        print("num_classes : could not find a cls_preds head - is this a YOLOX "
              "checkpoint?")

    # ── Size, from the stem and dark2 ────────────────────────────────────────
    stem = next((state[k] for k in state if k.endswith("stem.conv.conv.weight")), None)
    if stem is not None:
        channels = stem.shape[0]
        width, name = WIDTHS.get(channels, (None, "unrecognised"))
        print(f"stem width  : {channels} channels -> width={width} ({name})")
        if stem.shape[1] != 12:
            print(f"              NOTE: stem input is {stem.shape[1]} channels, not the "
                  f"12 a Focus layer expects")
    else:
        print("stem width  : no stem.conv.conv.weight found")

    blocks = {k.split("dark2.1.m.")[1].split(".")[0]
              for k in state if "dark2.1.m." in k}
    if blocks:
        n = len(blocks)
        depth, name = DEPTHS.get(n, (None, "unrecognised"))
        print(f"dark2 depth : {n} bottleneck(s) -> depth={depth} ({name})")

    depthwise = any("dconv" in k or "pconv" in k for k in state)
    print(f"depthwise   : {depthwise}  ({'nano' if depthwise else 'standard convs'})")

    # ── Anything naming the classes ──────────────────────────────────────────
    named = [k for k in ckpt if any(w in k.lower()
                                    for w in ("class", "names", "label", "exp"))]
    if named:
        print(f"\nkeys that may name the classes: {named}")
        for k in named:
            print(f"  {k} = {ckpt[k]!r}"[:400])
    else:
        print("\nNo class names in the checkpoint - YOLOX does not store them. "
              "They have to be supplied separately, in training index order.")

    if args.show_keys:
        print(f"\nfirst {args.show_keys} state-dict keys:")
        for key in list(state)[:args.show_keys]:
            print(f"  {key}  {tuple(state[key].shape)}")

    print("\nWhat this means for wiring it up:")
    if num_classes is not None:
        print(f"  - the exp must set num_classes = {num_classes}")
        print(f"  - {num_classes} class name(s) needed, in training index order")
    if stem is not None and WIDTHS.get(stem.shape[0]):
        w = WIDTHS[stem.shape[0]][0]
        d = DEPTHS.get(len(blocks), (None,))[0] if blocks else None
        print(f"  - the exp must set width = {w}" + (f", depth = {d}" if d else ""))
    print("  - test_size is NOT stored in the checkpoint; if it was not trained at "
          "640x640, that value has to come from the training config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
