"""
Similarity-filtered factual negatives for LegalBERT fine-tuning.

For each case in all-data/, embeds premise sentences (Applicant/State arguments)
and facts_section sentences with all-mpnet-base-v2. Fact sentences whose max
cosine similarity to any premise in the same case falls below FACT_SIM_THRESHOLD
are kept as safe NON_PREMISE training examples.

This teaches LegalBERT that factual background text CAN be non-premise, without
introducing contradictory labels for legally relevant facts.
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


def _flatten_facts(node: dict) -> list[str]:
    texts = []
    content = re.sub(r"^\d+\.\s*", "", node.get("content", "")).strip()
    if content and len(content.split()) >= 5:
        texts.append(content)
    for child in node.get("elements", []):
        texts.extend(_flatten_facts(child))
    return texts


def _load_premises_and_facts(dataset_dir: Path, excluded: set) -> dict:
    """
    Returns case_id -> {"premises": [str], "facts": [str]}
    """
    cases: dict[str, dict] = {}

    for path in dataset_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception:
            continue

        case_id = case.get("case_id")
        if case_id is None or case_id in excluded:
            continue

        if case_id not in cases:
            cases[case_id] = {"premises": [], "facts": []}

        for para in case.get("input_arguments", []):
            agent = para.get("agent", "")
            if agent not in ("Applicant", "State"):
                continue
            for unit in para.get("arg_units", []):
                text = unit.get("word", "").strip()
                if text:
                    cases[case_id]["premises"].append(text)

        facts = case.get("facts_section", {})
        if isinstance(facts, dict):
            for elem in facts.get("elements", []):
                cases[case_id]["facts"].extend(_flatten_facts(elem))

    return cases


def filter_fact_negatives(
    dataset_dir: Path = config.NEW_DATASET_DIR,
    sim_threshold: float = config.FACT_SIM_THRESHOLD,
    cache_path: Path = config.FACT_NEGATIVES_CACHE,
    excluded: set = None,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """
    Returns case_id -> list of safe fact sentences (similarity < threshold).
    Caches result to disk; loads from cache if it exists.
    """
    if cache_path.exists():
        print(f"  Loading cached fact negatives from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if excluded is None:
        excluded = set()

    print("  Building similarity-filtered fact negatives...")
    cases = _load_premises_and_facts(dataset_dir, excluded)

    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    result: dict[str, list[str]] = {}
    stats = {"total_facts": 0, "kept": 0, "skipped_no_premises": 0}

    for case_id, data in tqdm(cases.items(), desc="Filtering facts"):
        premises = data["premises"]
        facts = data["facts"]

        if not premises or not facts:
            stats["skipped_no_premises"] += len(facts)
            continue

        stats["total_facts"] += len(facts)

        prem_embs = model.encode(premises, batch_size=batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
        fact_embs = model.encode(facts, batch_size=batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)

        prem_norms = prem_embs / (np.linalg.norm(prem_embs, axis=1, keepdims=True) + 1e-8)
        fact_norms = fact_embs / (np.linalg.norm(fact_embs, axis=1, keepdims=True) + 1e-8)
        sims = fact_norms @ prem_norms.T
        max_sims = sims.max(axis=1)

        safe = [facts[i] for i in range(len(facts)) if max_sims[i] < sim_threshold]
        if safe:
            result[case_id] = safe
            stats["kept"] += len(safe)

    print(f"  Fact negatives: {stats['total_facts']} total, "
          f"{stats['kept']} kept (sim < {sim_threshold}), "
          f"{stats['total_facts'] - stats['kept']} filtered out, "
          f"{stats['skipped_no_premises']} skipped (no premises in case)")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Cached to {cache_path}")

    return result


def inspect_fact_negatives(
    dataset_dir: Path = config.NEW_DATASET_DIR,
    sim_threshold: float = config.FACT_SIM_THRESHOLD,
    n_cases: int = 3,
    n_sentences: int = 5,
):
    """
    Print sample kept and filtered sentences for manual review.
    """
    from stage1_argument_mining.fact_filter import _load_premises_and_facts

    excluded = set()
    excl_path = config.OUTPUT_DIR / "contaminated_case_ids.json"
    if excl_path.exists():
        with open(excl_path, "r", encoding="utf-8") as f:
            excluded = set(json.load(f))

    cases = _load_premises_and_facts(dataset_dir, excluded)
    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    shown = 0
    for case_id, data in cases.items():
        if not data["premises"] or not data["facts"]:
            continue

        prem_embs = model.encode(data["premises"], show_progress_bar=False, convert_to_numpy=True)
        fact_embs = model.encode(data["facts"], show_progress_bar=False, convert_to_numpy=True)

        prem_norms = prem_embs / (np.linalg.norm(prem_embs, axis=1, keepdims=True) + 1e-8)
        fact_norms = fact_embs / (np.linalg.norm(fact_embs, axis=1, keepdims=True) + 1e-8)
        sims = fact_norms @ prem_norms.T
        max_sims = sims.max(axis=1)

        kept = [(data["facts"][i], max_sims[i]) for i in range(len(data["facts"])) if max_sims[i] < sim_threshold]
        filtered = [(data["facts"][i], max_sims[i]) for i in range(len(data["facts"])) if max_sims[i] >= sim_threshold]

        if not kept and not filtered:
            continue

        print(f"\n{'='*80}")
        print(f"Case: {case_id}  ({len(data['premises'])} premises, {len(data['facts'])} facts)")
        print(f"  KEPT as NON_PREMISE ({len(kept)}):")
        for text, sim in kept[:n_sentences]:
            print(f"    sim={sim:.3f}  {text[:]}")
        print(f"  FILTERED OUT ({len(filtered)}):")
        for text, sim in filtered[:n_sentences]:
            print(f"    sim={sim:.3f}  {text[:]}")

        shown += 1
        if shown >= n_cases:
            break


if __name__ == "__main__":
    inspect_fact_negatives()
