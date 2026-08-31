#!/usr/bin/env python3
"""
attach_bios_min.py — minimal build: original request, bio-attached request,
and whether the attached bio was single- or multi-attribute.

Output columns
--------------
    request_original     the untouched harmful request
    request_with_bio     bio sentence prepended to that request
    bio_type             "single" | "multi"

Usage
-----
    python attach_bios_min.py
    python attach_bios_min.py --position append
    python attach_bios_min.py --include-control     # adds bio_type == "none" rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQ_COL = "text"
BIO_COL = "bio"


def normalize(t: str) -> str:
    return " ".join(str(t).split())


def as_sentence(t: str) -> str:
    t = normalize(t)
    return t if t.endswith((".", "!", "?")) else t + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path,
                    default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/Requests/bio_attachable_ready.csv"))
    ap.add_argument("--single", type=Path,
                    default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/bio_sentences/v2/bios_single.csv"))
    ap.add_argument("--multi", type=Path,
                    default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/bio_sentences/v2/bios_multi.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/bio_attached/bio_attached.csv"))
    ap.add_argument("--position", choices=["prepend", "append"], default="append")
    ap.add_argument("--include-control", action="store_false",
                    help="also emit one no-bio row per request (bio_type='none')")
    args = ap.parse_args()

    requests = pd.read_csv(args.requests).dropna(subset=[REQ_COL])
    requests = pd.DataFrame({"request_original": requests[REQ_COL].map(normalize)})

    bios = []
    for path, label in ((args.single, "single"), (args.multi, "multi")):
        df = pd.read_csv(path)
        bios.append(pd.DataFrame({BIO_COL: df[BIO_COL].map(as_sentence),
                                  "bio_type": label}))
    bios = pd.concat(bios, ignore_index=True)

    out = requests.merge(bios, how="cross")
    if args.position == "prepend":
        out["request_with_bio"] = out[BIO_COL] + " " + out["request_original"]
    else:
        out["request_with_bio"] = out["request_original"] + " " + out[BIO_COL]
    out = out[["request_original", "request_with_bio", "bio_type"]]

    if args.include_control:
        ctrl = requests.copy()
        ctrl["request_with_bio"] = ctrl["request_original"]
        ctrl["bio_type"] = "none"
        out = pd.concat([out, ctrl], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(out["bio_type"].value_counts().to_string())
    print(f"rows: {len(out)}  ->  {args.out}")


if __name__ == "__main__":
    main()
