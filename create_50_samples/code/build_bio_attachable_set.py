"""
Build the bio-attachable evaluation set from MedSafetyBench.

Inputs : datasets/{train,test}/{gpt4,llama2}/med_safety_demonstrations_category_*.csv
         (only `harmful_medical_request` is read; `safe_response` is ignored)

Outputs:
  bio_attachable_ready.csv    P1 — usable as-is, no rewrite
  rewrite_worksheet.csv       P2 + P3 — blank `rewritten_stem` column for a human to fill
  placebo_set.csv             P0 — no patient referent; the control arm, never rewritten
  all_tagged.csv              everything, with slot / dedup / flag columns

Usage:
  python build_bio_attachable_set.py --repo ./med-safety-bench/datasets --outdir ./out
"""

import argparse
import glob
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------ patterns

SINGULAR = re.compile(
    r"\b(?:a|an|the|this|that|my|your|their|his|her|one|each|every)\s+"
    r"(?:\w+\s+){0,3}?patient(?:'s|’s)?\b", re.I)
PLURAL = re.compile(r"\bpatients(?:'|’)?\b", re.I)
BARE = re.compile(r"\bpatient(?:s|'s|’s|s'|s’)?\b", re.I)

ADJ = re.compile(
    r"\b(?:be|been|being|am|is|are|was|were|feel|feels|felt|remain|remains|stay|stays|"
    r"seem|seems|more|less|very|so|too|quite|extremely)\s+patient\b|"
    r"\bpatient\s+(?:enough|with\s+(?:me|you|him|her|them|us))\b", re.I)

REFERENT = re.compile(
    r"\b(?:individuals?|persons?|people|someone|somebody|clients?|users?|"
    r"child(?:ren)?|kids?|minors?|infants?|bab(?:y|ies)|newborns?|adolescents?|teens?|"
    r"teenagers?|elderly|seniors?|adults?|m[ae]n|wom[ae]n|mothers?|fathers?|parents?|"
    r"famil(?:y|ies)|caregivers?|residents?|inmates?|victims?|subjects?|participants?|"
    r"recipients?|donors?|pregnant\s+\w+|populations?|groups?\s+of\s+people)\b", re.I)

CLINICAL = re.compile(
    r"\b(?:diagnos\w*|prescrib\w*|treat(?:s|ed|ing|ment|ments)?|therap\w*|medicat\w*|"
    r"drugs?|doses?|dosages?|opioids?|antibiotics?|surger(?:y|ies)|surgical|operat\w*|"
    r"procedures?|symptoms?|conditions?|illness\w*|diseases?|prognosis|admit\w*|"
    r"discharg\w*|refer(?:r\w*|s)?|triage|screening|biops(?:y|ies)|consent|"
    r"confidential\w*|privacy|medical\s+record\w*|health\s+(?:record|information|data)|"
    r"care|clinical|hospitals?|clinics?|wards?|ICU|emergency\s+room|bedside|vaccin\w*|"
    r"immuniz\w*|pain\s+management|end[-\s]of[-\s]life|palliative|hospice|"
    r"resuscitat\w*|transplant\w*|examination|check[-\s]?up|consultation)\b", re.I)

# demographics already fixed in the stem -> collide with the attribute you manipulate
PREEXISTING_DEMO = {
    "sex/gender": re.compile(r"\b(?:wom[ae]n|m[ae]n|mothers?|fathers?|males?|females?|"
                             r"pregnant\w*|maternal|gynecolog\w*|obstetric\w*|"
                             r"transgender|nonbinary|non-binary)\b", re.I),
    "age": re.compile(r"\b(?:child(?:ren)?|kids?|minors?|infants?|bab(?:y|ies)|newborns?|"
                      r"adolescents?|teens?|teenagers?|elderly|seniors?|geriatric|"
                      r"pediatric|neonatal|\d{1,3}[-\s]year[-\s]old)\b", re.I),
    "race/ethnicity": re.compile(r"\b(?:races?|racial|ethnic\w*|Black|White|Asian|Hispanic|"
                                 r"Latino|Latina|Indigenous|Native\s+American|immigrants?|"
                                 r"undocumented)\b", re.I),
    "disability": re.compile(r"\b(?:disabled|disabilit\w*|impair\w*)\b", re.I),
    "SES/insurance": re.compile(r"\b(?:uninsured|insurance|medicaid|medicare|poor|poverty|"
                                r"low[-\s]income|financially\s+disadvantaged|homeless|wealthy|"
                                r"afford)\b", re.I),
}


def classify(text: str) -> str:
    """P1 individuated | P2 generic-plural | P3 implicit referent | P0 no patient."""
    n_noun, n_adj = len(BARE.findall(text)), len(ADJ.findall(text))
    if n_noun and n_adj < n_noun:
        if SINGULAR.search(text):
            return "P1"
        return "P2"
    if REFERENT.search(text) and CLINICAL.search(text):
        return "P3"
    return "P0"


def demo_collisions(text: str) -> str:
    return "; ".join(k for k, rx in PREEXISTING_DEMO.items() if rx.search(text))


# ------------------------------------------------------------------ load

def load_requests(repo: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for gen in ("gpt4", "llama2"):
            for f in sorted(glob.glob(os.path.join(repo, split, gen, "*.csv"))):
                m = re.search(r"category_(\d+)", os.path.basename(f))
                d = pd.read_csv(f)
                if "harmful_medical_request" not in d.columns:
                    raise KeyError(f"{f}: no 'harmful_medical_request' column")
                for i, t in enumerate(d["harmful_medical_request"].astype(str)):
                    rows.append({"split": split, "generator": gen,
                                 "category": int(m.group(1)), "src_row": i,
                                 "text": t.strip()})
    df = pd.DataFrame(rows)
    df.insert(0, "item_id", [f"MSB{i:04d}" for i in range(len(df))])
    return df


# ------------------------------------------------------------------ dedup

def add_dup_clusters(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Greedy single-link clustering on TF-IDF cosine similarity.

    TF-IDF is a lexical proxy. If you have network access, swap in
    sentence-transformers ('all-MiniLM-L6-v2') and cluster at ~0.90 — it will
    catch reworded near-duplicates that TF-IDF misses.
    """
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                        stop_words="english").fit_transform(df.text)
    S = cosine_similarity(X)
    n = len(df)
    cluster = [-1] * n
    cid = 0
    for i in range(n):
        if cluster[i] != -1:
            continue
        cluster[i] = cid
        for j in range(i + 1, n):
            if cluster[j] == -1 and S[i, j] >= threshold:
                cluster[j] = cid
        cid += 1
    df["dup_cluster"] = cluster
    sizes = pd.Series(cluster).value_counts()
    df["dup_cluster_size"] = df.dup_cluster.map(sizes)
    df["is_cluster_head"] = ~df.duplicated("dup_cluster", keep="first")
    return df


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/lkh/Documents/research/PennState/Project1/Experiments/med-safety-bench/datasets")
    ap.add_argument("--outdir", default="/Users/lkh/Documents/research/PennState/Project1/Experiments/create_50_samples/data/pre-processed_data")
    ap.add_argument("--dup-threshold", type=float, default=0.80)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_requests(args.repo)
    df["slot"] = df.text.apply(classify)
    df["needs_rewrite"] = df.slot.isin(["P2", "P3"])
    df["preexisting_demographics"] = df.text.apply(demo_collisions)
    df["rewrite_reason"] = df.slot.map({
        "P1": "",
        "P2": "'patient' is attributive/plural - singularise to create a referent",
        "P3": "referent is implicit - name it as 'a patient' explicitly",
        "P0": "no patient referent - PLACEBO ARM, do not rewrite",
    })
    df = add_dup_clusters(df, args.dup_threshold)

    ready = df[(df.slot == "P1") & df.is_cluster_head].copy()
    rework = df[df.needs_rewrite & df.is_cluster_head].copy()
    placebo = df[(df.slot == "P0") & df.is_cluster_head].copy()

    # blank columns for the human rewriter; rewritten_stem IS the no-bio control
    rework["rewritten_stem"] = ""
    rework["rewriter_initials"] = ""
    rework["exclude"] = ""

    ready.to_csv(f"{args.outdir}/bio_attachable_ready.csv", index=False)
    rework.to_csv(f"{args.outdir}/rewrite_worksheet.csv", index=False)
    placebo.to_csv(f"{args.outdir}/placebo_set.csv", index=False)
    df.to_csv(f"{args.outdir}/all_tagged.csv", index=False)

    print(f"loaded {len(df)} harmful requests")
    print(df.slot.value_counts().sort_index().to_string())
    print(f"\nnear-dup clusters @cos>={args.dup_threshold}: "
          f"{df.dup_cluster.nunique()} clusters, "
          f"{(~df.is_cluster_head).sum()} items suppressed")
    print(f"\nP1 ready (deduped)      : {len(ready)}")
    print(f"P2+P3 to rewrite        : {len(rework)}")
    print(f"  with demo collisions  : {(rework.preexisting_demographics != '').sum()}")
    print(f"P0 placebo (deduped)    : {len(placebo)}")
    print("\nP1 by category x generator:")
    print(ready.pivot_table(index="category", columns="generator",
                            values="item_id", aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
