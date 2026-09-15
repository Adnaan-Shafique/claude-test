#!/usr/bin/env python3
"""
make_images.py — build a synthetic document corpus for the capacity benchmark.

Use this ONLY if you do not have 300 representative production images yet.
Real images are always better: vision-token count scales with image size and
visual complexity, and that count is the dominant term in VLM cost. A corpus of
blank white squares will produce throughput numbers you cannot ship a sourcing
decision on.

What it generates: invoice-like pages at a realistic scanned-document aspect
ratio, each with a header block, a line-item table, totals, and a footer — so
there is genuine text for the model to OCR and genuine structure to extract.
Each page carries different values, which keeps vLLM's prefix cache from
inflating throughput the way 300 copies of one image would.

Usage:
    python3 make_images.py --out ./corpus --count 300 --width 1700 --height 2200
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VENDORS = [
    "Northwind Logistics", "Contoso Manufacturing", "Fabrikam Industrial",
    "Adventure Works Supply", "Litware Components", "Proseware Chemicals",
    "Tailspin Freight", "Wide World Importers", "Graphic Design Institute",
    "Woodgrove Materials",
]
ITEMS = [
    "Hex bolt M12x40 galv", "Sealed bearing 6204-2RS", "Hydraulic hose 3/8 in",
    "Control relay 24VDC", "Steel plate 6mm A36", "Coupling flange DN50",
    "Filter cartridge 10um", "Gasket set EPDM", "Cable gland M20 brass",
    "Limit switch IP67", "Pressure gauge 0-16 bar", "Drive belt SPB-1800",
]
CITIES = ["Pune", "Chennai", "Hyderabad", "Ahmedabad", "Kochi", "Indore"]


def _font(size: int):
    """Find a real TTF; PIL's bitmap default is unreadably small on a 1700px page."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_page(idx: int, w: int, h: int, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (w, h), (252, 252, 250))
    d = ImageDraw.Draw(img)

    s = w / 1700.0                      # scale everything off a 1700px reference
    f_title = _font(int(46 * s))
    f_head  = _font(int(28 * s))
    f_body  = _font(int(24 * s))
    f_small = _font(int(20 * s))

    vendor = rng.choice(VENDORS)
    inv_no = f"INV-{2024 + idx % 3}-{100000 + idx * 37 % 900000:06d}"
    po_no  = f"PO-{500000 + idx * 91 % 400000:06d}"
    date   = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2025"
    gstin  = f"{rng.randint(10,37)}AABCU{rng.randint(1000,9999)}{rng.choice('ABCDEFGHJK')}1Z{rng.randint(0,9)}"

    y = int(70 * s)
    d.text((int(80 * s), y), vendor, fill=(20, 20, 30), font=f_title)
    y += int(62 * s)
    d.text((int(80 * s), y), f"{rng.choice(CITIES)}, India  |  GSTIN {gstin}",
           fill=(90, 90, 100), font=f_small)

    y += int(56 * s)
    d.line([(int(80 * s), y), (w - int(80 * s), y)], fill=(160, 160, 170), width=max(1, int(2 * s)))

    y += int(34 * s)
    for label, value in (("Invoice No", inv_no), ("PO Reference", po_no), ("Invoice Date", date)):
        d.text((int(80 * s), y),  f"{label}:", fill=(80, 80, 90), font=f_head)
        d.text((int(420 * s), y), value,       fill=(15, 15, 20), font=f_head)
        y += int(42 * s)

    # ── line-item table ──────────────────────────────────────────────────────
    y += int(40 * s)
    cols = [80, 760, 1000, 1240, 1500]
    heads = ["Description", "Qty", "Unit Price", "Amount"]
    d.rectangle([(int(70 * s), y - int(10 * s)), (w - int(70 * s), y + int(42 * s))],
                fill=(232, 234, 240))
    for cx, head in zip(cols, heads):
        d.text((int(cx * s), y), head, fill=(30, 30, 40), font=f_head)
    y += int(58 * s)

    n_lines = rng.randint(5, 11)
    subtotal = 0.0
    for _ in range(n_lines):
        desc = rng.choice(ITEMS)
        qty  = rng.randint(1, 240)
        unit = round(rng.uniform(12.5, 4800.0), 2)
        amt  = round(qty * unit, 2)
        subtotal += amt
        d.text((int(cols[0] * s), y), desc,            fill=(25, 25, 35), font=f_body)
        d.text((int(cols[1] * s), y), str(qty),        fill=(25, 25, 35), font=f_body)
        d.text((int(cols[2] * s), y), f"{unit:,.2f}",  fill=(25, 25, 35), font=f_body)
        d.text((int(cols[3] * s), y), f"{amt:,.2f}",   fill=(25, 25, 35), font=f_body)
        y += int(40 * s)
        d.line([(int(80 * s), y - int(8 * s)), (w - int(80 * s), y - int(8 * s))],
               fill=(225, 225, 232), width=1)

    # ── totals ───────────────────────────────────────────────────────────────
    y += int(30 * s)
    tax = round(subtotal * 0.18, 2)
    for label, value, bold in (
        ("Subtotal", subtotal, False),
        ("GST @ 18%", tax, False),
        ("Total Due (INR)", subtotal + tax, True),
    ):
        font = f_head if bold else f_body
        d.text((int(cols[2] * s), y), label,            fill=(60, 60, 70), font=font)
        d.text((int(cols[3] * s), y), f"{value:,.2f}",  fill=(10, 10, 15), font=font)
        y += int(46 * s)

    # ── footer ───────────────────────────────────────────────────────────────
    fy = h - int(190 * s)
    d.line([(int(80 * s), fy), (w - int(80 * s), fy)], fill=(160, 160, 170), width=max(1, int(2 * s)))
    d.text((int(80 * s), fy + int(24 * s)),
           "Payment terms: Net 45 days from invoice date.",
           fill=(80, 80, 90), font=f_small)
    d.text((int(80 * s), fy + int(56 * s)),
           f"Remit to: HDFC Bank  A/C {rng.randint(10**10, 10**11 - 1)}  IFSC HDFC000{rng.randint(1000,9999)}",
           fill=(80, 80, 90), font=f_small)
    d.text((int(80 * s), fy + int(88 * s)),
           f"Page 1 of 1  |  Document ID {idx:05d}",
           fill=(130, 130, 140), font=f_small)

    # Light scan noise — a perfectly clean synthetic render is easier to OCR
    # than anything that has been through a real scanner, which would flatter
    # the throughput numbers.
    px = img.load()
    for _ in range(int(w * h * 0.0006)):
        x, yy = rng.randrange(w), rng.randrange(h)
        v = rng.randint(180, 235)
        px[x, yy] = (v, v, v)

    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",     default="./corpus", help="output directory")
    ap.add_argument("--count",   type=int, default=300)
    ap.add_argument("--width",   type=int, default=1700, help="page width in px")
    ap.add_argument("--height",  type=int, default=2200, help="page height in px")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality")
    ap.add_argument("--seed",    type=int, default=20250915)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_bytes = 0
    for i in range(args.count):
        img = make_page(i, args.width, args.height, rng)
        path = out / f"doc_{i:05d}.jpg"
        img.save(path, "JPEG", quality=args.quality)
        total_bytes += path.stat().st_size
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.count} pages written")

    print(f"\n{args.count} pages in {out}")
    print(f"  page size : {args.width}x{args.height} px")
    print(f"  total     : {total_bytes / 1e6:.1f} MB "
          f"({total_bytes / args.count / 1e3:.0f} kB average)")
    print("\nNOTE: the GPU server downscales any image whose longest side exceeds "
          f"{2048} px (MAX_IMAGE_SIDE_PX), and Qwen3-VL's max_pixels caps it again "
          "at 1280*28*28 tokens' worth. Both apply to this corpus.")


if __name__ == "__main__":
    main()
