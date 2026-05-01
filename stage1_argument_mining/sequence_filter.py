import json
from tqdm import tqdm
import config
from stage1_argument_mining.argument_extractor import LegalBERTArgumentExtractor


def filter_dataset_split(split, extractor, split_name="split"):
    results = []
    for case_id, example in enumerate(tqdm(split, desc=f"Stage 1 [{split_name}]")):
        premises     = extractor.extract_premises(example["paragraphs"])
        premise_text = " ".join(p["sentence"] for p in premises)
        if not premise_text.strip():
            premise_text = " ".join(example["paragraphs"])
        results.append({
            "case_id":       case_id,
            "paragraphs":    example["paragraphs"],
            "labels_binary": example["labels_binary"],
            "premises":      premises,
            "premise_text":  premise_text,
            "used_fallback": len(premises) == 0,
        })
    avg = sum(len(r["premises"]) for r in results) / max(len(results), 1)
    print(f"[{split_name}] {len(results)} cases, avg {avg:.1f} premises/case")
    return results


def run_stage1(dataset, extractor=None, save_path=config.STAGE1_CACHE):
    """Run Stage 1 argument mining on the dataset."""
    if extractor is None:
        extractor = LegalBERTArgumentExtractor()

    output = {}
    for hf_split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        if hf_split in dataset:
            output[key] = filter_dataset_split(dataset[hf_split], extractor, hf_split)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def load_stage1(path=config.STAGE1_CACHE):
    """Load cached Stage 1 results from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_stage1_stats(output):
    print("\n" + "=" * 60)
    print("Stage 1 – Argument Mining Statistics")
    print("=" * 60)
    for split, cases in output.items():
        total   = sum(len(c["premises"]) for c in cases)
        avg     = total / max(len(cases), 1)
        zero    = sum(1 for c in cases if len(c["premises"]) == 0)
        print(f"  [{split:>5}] cases={len(cases):>5}  total_premises={total:>6}  "
              f"avg={avg:>5.1f}  zero_premise_cases={zero}")
    print("=" * 60)


if __name__ == "__main__":
    from data.data_loader import get_dataset
    import config as cfg
    cfg.MAX_TRAIN_SAMPLES = 20
    cfg.MAX_VAL_SAMPLES   = 5
    cfg.MAX_TEST_SAMPLES  = 5

    output = run_stage1(get_dataset())
    print_stage1_stats(output)
