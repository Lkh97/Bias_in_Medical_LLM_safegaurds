"""
MedSafetyBench -> patient-related subset, stratified by BIO-SLOT TYPE.

Loads only `harmful_medical_request` (safe_response is ignored).

Classes:
  P1  singular individuated patient  -> a bio sentence can be attached with no rewrite
  P2  plural / generic 'patients'    -> needs singularisation before a bio fits
  P3  implicit person referent       -> patient exists but is unnamed ('a minor', 'pregnant women')
  P0  no individual patient referent -> systemic/fraud/policy; bio insertion is incoherent -> EXCLUDE
"""

import re
import glob
import os
import pandas as pd
from pathlib import Path

REPO = "med-safety-bench/datasets"

# --------------------------------------------------------------------- patterns

# 'a patient', 'the patient's', 'my elderly patient' -> individuated
SINGULAR = re.compile(
    r"\b(?:a|an|the|this|that|my|your|their|his|her|one|each|every)\s+"
    r"(?:\w+\s+){0,3}?patient(?:'s|’s)?\b", re.I)
PLURAL = re.compile(r"\bpatients(?:'|’)?\b", re.I)
BARE = re.compile(r"\bpatient(?:s|'s|’s|s'|s’)?\b", re.I)

# adjective use: "be patient", "patient enough"
ADJ = re.compile(
    r"\b(?:be|been|being|am|is|are|was|were|feel|feels|felt|remain|remains|stay|stays|"
    r"seem|seems|more|less|very|so|too|quite|extremely)\s+patient\b|"
    r"\bpatient\s+(?:enough|with\s+(?:me|you|him|her|them|us))\b", re.I)

# implicit referents — NOTE the trailing s? on every count noun (easy bug to miss)
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


def classify(text: str) -> str:
    has_noun = bool(BARE.search(text)) and not (
        len(ADJ.findall(text)) >= len(BARE.findall(text)))
    if has_noun:
        if SINGULAR.search(text):
            return "P1"
        if PLURAL.search(text):
            return "P2"
        return "P2"                      # bare 'patient care', 'patient data'
    if REFERENT.search(text) and CLINICAL.search(text):
        return "P3"
    return "P0"


# --------------------------------------------------------------------- load

def load_requests(repo=REPO) -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for gen in ("gpt4", "llama2"):
            for f in sorted(glob.glob(f"{repo}/{split}/{gen}/*.csv")):
                cat = int(re.search(r"category_(\d+)", os.path.basename(f)).group(1))
                d = pd.read_csv(f)
                for i, t in enumerate(d["harmful_medical_request"].astype(str)):
                    rows.append({"split": split, "generator": gen, "category": cat,
                                 "row": i, "text": t.strip()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_requests()
    df["slot"] = df.text.apply(classify)
    df["patient_related"] = df.slot != "P0"

    print(f"N = {len(df)}")
    print(df.slot.value_counts().sort_index().to_string())
    print(f"\npatient-related (P1-P3): {df.patient_related.sum()} "
          f"({df.patient_related.mean():.1%})")
    print(f"drop-in ready (P1):      {(df.slot=='P1').sum()} "
          f"({(df.slot=='P1').mean():.1%})")

    print("\n% P1 by category")
    print((100 * df.assign(p1=df.slot == "P1").groupby("category").p1.mean()).round(1).to_string())
    print("\n% P1 by generator")
    print((100 * df.assign(p1=df.slot == "P1").groupby("generator").p1.mean()).round(1).to_string())

    df.to_csv("medsafety_requests_slot_tagged.csv", index=False)
    df[df.slot == "P1"].to_csv("medsafety_P1_bio_ready.csv", index=False)
    df.groupby("slot", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 30), random_state=0)
    ).to_csv("medsafety_audit_sample.csv", index=False)
