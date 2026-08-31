#!/usr/bin/env python3
"""
Sample N rows where BOTH are unique:
  * the base (harmful) request, and
  * the attribute value(s) in the bio sentence.

Uniqueness modes
----------------
strict (default) : no attribute value may appear twice anywhere in the sample,
                   including inside `multi` combinations.
                   -> N rows consume N..(N*k) distinct values.
combo            : only the full value-SET must be unique; individual values
                   may recur across rows (e.g. "Asian" alone and
                   "young Asian person" can both appear).

Category balance
----------------
Slots are allocated per category proportional to that category's value count,
with a floor of --min-per-cat, so nationality (20 values) does not eat the
sample. Multi rows count toward every category they touch.

Usage
-----
python sample_unique.py --infile data.csv --outfile sample50.csv \
       --n 50 --multi 8 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Attribute lexicon. NOTE: "Indian" assigned to race_ethnicity only (it is
# duplicated in the source file's nationality block) to avoid double-booking.
# --------------------------------------------------------------------------
ATTRIBUTES: dict[str, list[str]] = {
    "race_ethnicity": ["White", "Black", "African-American", "Asian", "Indian",
                       "Middle Eastern", "Hispanic", "North African"],
    "skin_tone": ["light-skinned", "dark-skinned"],
    "gender": ["woman", "man", "non-binary"],
    "sexual_orientation": ["heterosexual", "asexual", "LGBTQ"],
    "age": ["adolescent", "young", "middle-aged", "older"],
    "religion": ["Christian", "Muslim", "Jewish", "Hindu", "Buddhist",
                 "does not identify with any religion", "atheist"],
    "marital_status": ["single", "married", "divorced", "widowed"],
    "disability": ["blind", "deaf", "autistic", "has ADHD", "has Down syndrome",
                   "has depression", "uses a wheelchair"],
    "nationality": ["American", "Chinese", "Japanese", "Russian", "Iranian",
                    "Afghan", "Syrian", "British", "German", "French", "Saudi",
                    "Emirati", "Brazilian", "Australian", "Canadian", "Libyan",
                    "Ethiopian", "Nigerian", "Bolivian", "Colombian"],
    "income": ["exceptionally high income", "low income", "middle income",
               "high income"],
    "education": ["high school education", "diploma", "college degree",
                  "BS degree", "MS degree", "PhD"],
    "occupation": ["unemployed", "blue-collar job", "service job",
                   "white-collar profession"],
}

_COMPILED = sorted(
    ((cat, val, re.compile(rf"(?<![\w-]){re.escape(val)}(?![\w-])", re.I))
     for cat, vals in ATTRIBUTES.items() for val in vals),
    key=lambda t: -len(t[1]),
)


def extract_bio(base: str, bio_prompt: str) -> str:
    base_s, bio_s = base.strip(), bio_prompt.strip()
    if bio_s.startswith(base_s):
        return bio_s[len(base_s):].strip()
    m = re.search(r"The \w+ is\b.*$", bio_s)
    return m.group(0) if m else ""


def tag_attributes(bio: str) -> frozenset[tuple[str, str]]:
    text, found = bio, set()
    for cat, val, pat in _COMPILED:
        m = pat.search(text)
        if m:
            found.add((cat, val))
            text = text[:m.start()] + "\0" * (m.end() - m.start()) + text[m.end():]
    return frozenset(found)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for i, rec in enumerate(csv.reader(fh)):
            if len(rec) < 3:
                continue
            base, bio_prompt, typ = rec[0].strip(), rec[1].strip(), rec[2].strip().lower()
            if base.lower() in {"base", "prompt", "base_prompt"}:
                continue
            bio = extract_bio(base, bio_prompt)
            attrs = tag_attributes(bio)
            if not attrs:
                continue
            rows.append({"idx": i, "base_prompt": base, "bio_prompt": bio_prompt,
                         "bio": bio, "type": typ, "attrs": attrs,
                         "cats": frozenset(c for c, _ in attrs)})
    return rows


def allocate(n: int, min_per_cat: int) -> dict[str, int]:
    """Slots per category, proportional to category size, floor = min_per_cat."""
    sizes = {c: len(v) for c, v in ATTRIBUTES.items()}
    total = sum(sizes.values())
    alloc = {c: max(min_per_cat, round(n * s / total)) for c, s in sizes.items()}
    # trim/expand to exactly n, largest categories absorb the difference
    order = sorted(sizes, key=lambda c: -sizes[c])
    while sum(alloc.values()) > n:
        for c in order:
            if sum(alloc.values()) == n:
                break
            if alloc[c] > min_per_cat:
                alloc[c] -= 1
    while sum(alloc.values()) < n:
        for c in order:
            if sum(alloc.values()) == n:
                break
            if alloc[c] < sizes[c]:
                alloc[c] += 1
    return alloc


def attempt(rows, n, n_multi, seed, mode, min_per_cat):
    rng = random.Random(seed)
    deficit = allocate(n, min_per_cat)

    chosen, used_bases, used_vals, used_sets = [], set(), set(), set()
    quota = {"multi": n_multi, "single": n - n_multi}

    def feasible(r):
        if r["base_prompt"] in used_bases or quota.get(r["type"], 0) <= 0:
            return False
        if mode == "strict":
            return not (r["attrs"] & used_vals)
        return r["attrs"] not in used_sets

    def score(r):
        return sum(min(1, deficit.get(c, 0)) for c in r["cats"]), len(r["attrs"])

    # multi rows are the most constrained -> place them first
    for phase_type in (["multi"] if n_multi else []) + ["single"]:
        while quota[phase_type] > 0 and len(chosen) < n:
            pool = [r for r in rows if r["type"] == phase_type and feasible(r)]
            if not pool:
                break
            rng.shuffle(pool)
            best = max(pool, key=score)
            chosen.append(best)
            used_bases.add(best["base_prompt"])
            used_vals |= set(best["attrs"])
            used_sets.add(best["attrs"])
            quota[phase_type] -= 1
            for c in best["cats"]:
                deficit[c] = max(0, deficit.get(c, 0) - 1)

    # top up from whatever type still has feasible rows
    while len(chosen) < n:
        pool = [r for r in rows
                if r["base_prompt"] not in used_bases
                and (not (r["attrs"] & used_vals) if mode == "strict"
                     else r["attrs"] not in used_sets)]
        if not pool:
            break
        rng.shuffle(pool)
        best = max(pool, key=score)
        chosen.append(best)
        used_bases.add(best["base_prompt"])
        used_vals |= set(best["attrs"])
        used_sets.add(best["attrs"])
        for c in best["cats"]:
            deficit[c] = max(0, deficit.get(c, 0) - 1)

    return chosen, used_vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, type=Path)
    p.add_argument("--outfile", default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/bio_attached/50_sample_auto.csv"), type=Path)
    p.add_argument("--report", default=Path("/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data/bio_attached/coverage_report.json"), type=Path)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--multi", type=int, default=8)
    p.add_argument("--min-per-cat", type=int, default=2)
    p.add_argument("--mode", choices=["strict", "combo"], default="strict")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = load_rows(args.infile)
    if not rows:
        sys.exit("ERROR: no rows parsed (need 3 columns; bios must match the lexicon).")

    n_bases = len({r["base_prompt"] for r in rows})
    if n_bases < args.n:
        sys.exit(f"ERROR: {n_bases} unique base prompts < n={args.n}.")

    # auto-degrade the multi quota if strict disjointness makes it infeasible
    chosen, used_vals, m = [], set(), args.multi
    while m >= 0:
        chosen, used_vals = attempt(rows, args.n, m, args.seed, args.mode,
                                    args.min_per_cat)
        if len(chosen) == args.n:
            break
        m -= 1
    if len(chosen) < args.n:
        sys.exit(f"ERROR: only {len(chosen)}/{args.n} rows satisfy --mode {args.mode}. "
                 f"Try --mode combo or lower --n.")
    if m != args.multi:
        print(f"NOTE: multi quota auto-reduced {args.multi} -> {m} to keep "
              f"value disjointness feasible.", file=sys.stderr)

    all_vals = {(c, v) for c, vs in ATTRIBUTES.items() for v in vs}
    report = {
        "n": len(chosen),
        "mode": args.mode,
        "unique_base_prompts": len({r["base_prompt"] for r in chosen}),
        "unique_value_sets": len({r["attrs"] for r in chosen}),
        "distinct_values_used": len(used_vals),
        "values_total": len(all_vals),
        "coverage_pct": round(100 * len(used_vals) / len(all_vals), 1),
        "type_counts": dict(Counter(r["type"] for r in chosen)),
        "per_category": {c: f"{len([1 for cc, _ in used_vals if cc == c])}/{len(vs)}"
                         for c, vs in ATTRIBUTES.items()},
        "values_unused": sorted(f"{c}::{v}" for c, v in (all_vals - used_vals)),
    }

    with args.outfile.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base_prompt", "bio_prompt", "type", "bio_sentence",
                    "categories", "values"])
        for r in chosen:
            w.writerow([r["base_prompt"], r["bio_prompt"], r["type"], r["bio"],
                        "|".join(sorted(r["cats"])),
                        "|".join(sorted(v for _, v in r["attrs"]))])
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # invariant checks
    assert report["unique_base_prompts"] == len(chosen), "base prompt repeated!"
    if args.mode == "strict":
        flat = [v for r in chosen for v in r["attrs"]]
        assert len(flat) == len(set(flat)), "attribute value repeated!"

    print(f"{len(chosen)} rows -> {args.outfile}  ({report['type_counts']})")
    print(f"Distinct values used: {report['distinct_values_used']}/"
          f"{report['values_total']} ({report['coverage_pct']}%)")
    print(json.dumps(report["per_category"], indent=2))


if __name__ == "__main__":
    main()
