"""
Loads one random JSON file from NEW_DATASET_DIR and shows exactly what the
fine-tune data loaders extract from it:

  Part A — input_arguments  (mirrors load_new_dataset):
            Applicant/State arg_units → PREMISE, Non-Argument → NON_PREMISE
  Part B — facts_section    (mirrors load_facts_nonpremises):
            Flattened fact sentences  → NON_PREMISE

Run:
    python sample.py
"""
import json
import random
import re
from pathlib import Path

import config
from stage1_argument_mining.finetune_legalbert import (
    _load_exclusion_list, _flatten_facts_elements
)

DATASET_DIR = config.NEW_DATASET_DIR

# ---------------------------------------------------------------------------
# Pick one random file
# ---------------------------------------------------------------------------

all_files = list(DATASET_DIR.glob("*.json"))
if not all_files:
    raise FileNotFoundError(f"No JSON files found in {DATASET_DIR}")

path = random.choice(all_files)
print(f"File: {path.name}\n")

with open(path, "r", encoding="utf-8") as f:
    case = json.load(f)

print(f"case_id  : {case.get('case_id')}")
print(f"article  : {case.get('article')}")
print(f"judgment : {case.get('judgment')}")
print(f"parties  : {case.get('parties')}")
print()

excluded = _load_exclusion_list()
if case.get("case_id") in excluded:
    print("NOTE: this case_id is in the contamination exclusion list.")

# ---------------------------------------------------------------------------
# Part A — input_arguments  (mirrors load_new_dataset)
# ---------------------------------------------------------------------------

arg_rows = []

for para in case.get("input_arguments", []):
    agent = para.get("agent", "")
    if agent == "ECHR":
        continue
    label = 1 if agent in ("Applicant", "State") else 0
    label_name = "PREMISE" if label == 1 else "NON_PREMISE"
    for unit in para.get("arg_units", []):
        text = unit.get("word", "").strip()
        if text:
            arg_rows.append({
                "text":       text,
                "label":      label,
                "label_name": label_name,
                "agent":      agent,
                "claim":      unit.get("claim"),
            })

n_premise     = sum(1 for r in arg_rows if r["label"] == 1)
n_non_premise = sum(1 for r in arg_rows if r["label"] == 0)

print("=" * 70)
print("PART A — input_arguments  (load_new_dataset)")
print("=" * 70)
print(f"  Total    : {len(arg_rows)}")
print(f"  PREMISE  : {n_premise}  (Applicant / State arg_units)")
print(f"  NON_PREM : {n_non_premise}  (Non-Argument arg_units)")
print(f"  ECHR     : skipped")
print()
for i, row in enumerate(arg_rows):
    print(f"[{i+1:02d}] {row['label_name']:<12} agent={row['agent']:<14} claim={str(row['claim']):<5}")
    print(f"     {row['text'][:]}")
    print()

# ---------------------------------------------------------------------------
# Part B — facts_section  (mirrors load_facts_nonpremises)
# ---------------------------------------------------------------------------

fact_rows = []
facts = case.get("facts_section", {})
if isinstance(facts, dict):
    for elem in facts.get("elements", []):
        for text in _flatten_facts_elements(elem):
            fact_rows.append({"text": text, "label": 0, "label_name": "NON_PREMISE"})

print("=" * 70)
print("PART B — facts_section  (load_facts_nonpremises)")
print("=" * 70)
print(f"  Total NON_PREMISE sentences : {len(fact_rows)}")
print()
for i, row in enumerate(fact_rows):
    print(f"[{i+1:02d}] NON_PREMISE  {row['text'][:]}")
    print()
