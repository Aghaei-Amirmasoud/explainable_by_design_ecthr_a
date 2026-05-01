"""
Contamination check: all-data/ (12,947 cases) vs LexGLUE ecthr_a test+val splits.

Fingerprint-based matching checks BOTH input_arguments AND facts_section,
since both are used as training data (PREMISE and NON_PREMISE respectively).

Run:
    python check_contamination.py

Set config.NEW_DATASET_DIR to the folder containing the dataset JSON files.
"""
import json
import re
import config
from pathlib import Path
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 1. Load lex_glue/ecthr_a test + validation splits
# ---------------------------------------------------------------------------

print(f"Loading {config.LEXGLUE_DATASET}/{config.LEXGLUE_CONFIG} test + validation splits...")
ds = load_dataset(config.LEXGLUE_DATASET, config.LEXGLUE_CONFIG, trust_remote_code=True)
test_split = ds["test"]
val_split = ds["validation"]
print(f"  {len(test_split)} test cases + {len(val_split)} validation cases loaded")


# ---------------------------------------------------------------------------
# 2. Build paragraph fingerprint set from LexGLUE test + validation
# ---------------------------------------------------------------------------

def fingerprint(text: str) -> str:
    """Normalize whitespace, lowercase, take first 80 chars."""
    return re.sub(r"\s+", " ", text.lower().strip())[:500]


def _flatten_facts_elements(node: dict) -> list:
    """Recursively collect content strings from the nested facts_section tree."""
    texts = []
    content = re.sub(r"^\d+\.\s*", "", node.get("content", "")).strip()
    if content and len(content.split()) >= 5:
        texts.append(content)
    for child in node.get("elements", []):
        texts.extend(_flatten_facts_elements(child))
    return texts


test_fingerprints = set()
for example in list(test_split) + list(val_split):
    for para in example["text"]:
        fp = fingerprint(para)
        if len(fp) >= 20:          # skip very short / empty paragraphs
            test_fingerprints.add(fp)

print(f"  {len(test_fingerprints)} unique paragraph fingerprints indexed")


# ---------------------------------------------------------------------------
# 3. Scan all-data/ and check against fingerprints
# ---------------------------------------------------------------------------

new_dataset_dir = config.NEW_DATASET_DIR
contaminated_new: list[dict] = []
clean_count = 0

if not new_dataset_dir.exists():
    print(f"\n[ERROR] config.NEW_DATASET_DIR not found: {new_dataset_dir}")
    print("        Set the correct path in config.py and re-run.")
    exit(1)

json_files = list(new_dataset_dir.glob("*.json"))
print(f"\nScanning {len(json_files)} JSON files in all-data/...")

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
    except Exception:
        continue

    case_id = case.get("case_id", path.stem)
    matched_paras: list[str] = []

    # Check input_arguments (PREMISE training source)
    for para in case.get("input_arguments", []):
        text = para.get("text", "")
        if not text:
            continue
        fp = fingerprint(text)
        if fp in test_fingerprints:
            matched_paras.append(f"[arg] {text[:120]}")

    # Check facts_section (NON_PREMISE training source)
    facts = case.get("facts_section", {})
    if isinstance(facts, dict):
        for elem in facts.get("elements", []):
            for text in _flatten_facts_elements(elem):
                fp = fingerprint(text)
                if fp in test_fingerprints:
                    matched_paras.append(f"[fact] {text[:120]}")

    if matched_paras:
        contaminated_new.append({
            "file": path.name,
            "case_id": case_id,
            "article": case.get("article"),
            "matched_paragraphs": len(matched_paras),
            "sample": matched_paras[0],
        })
    else:
        clean_count += 1


# ---------------------------------------------------------------------------
# 4. Report results
# ---------------------------------------------------------------------------

total = len(json_files)
print("\n" + "=" * 60)
if not contaminated_new:
    print("RESULT: No contamination detected.")
    print(f"  All {total} cases are absent from LexGLUE test + validation splits.")
    print("  Safe to use all cases for Stage 1 fine-tuning.")
    print("  Checked: input_arguments (PREMISE) + facts_section (NON_PREMISE).")
else:
    print(f"WARNING: {len(contaminated_new)} contaminated file(s) out of {total}:")
    for c in contaminated_new[:20]:       # cap display at 20
        print(f"  {c['file']}  ({c['matched_paragraphs']} matching para(s))")
        print(f"    sample: \"{c['sample']}\"")
    if len(contaminated_new) > 20:
        print(f"  ... and {len(contaminated_new) - 20} more.")
    print(f"\n  {clean_count} clean cases  |  "
          f"{len(contaminated_new)} contaminated cases")
    print("\nContaminated case IDs (exclude from Stage 1 training):")
    contaminated_ids = sorted({c["case_id"] for c in contaminated_new})
    print("  " + ", ".join(str(i) for i in contaminated_ids[:50]))
    if len(contaminated_ids) > 50:
        print(f"  ... and {len(contaminated_ids) - 50} more.")

    # Write exclusion list to disk for use by the data loader
    exclusion_path = config.OUTPUT_DIR / "contaminated_case_ids.json"
    with open(exclusion_path, "w", encoding="utf-8") as f:
        json.dump(contaminated_ids, f, indent=2)
    print(f"\n  Exclusion list saved to: {exclusion_path}")

print("=" * 60)
