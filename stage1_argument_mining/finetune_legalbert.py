import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

import config
from utils.helpers import seed_everything

ID2LABEL = {0: "NON_PREMISE", 1: "PREMISE"}
LABEL2ID = {"NON_PREMISE": 0, "PREMISE": 1}
BASE_MODEL = "nlpaueb/bert-base-uncased-echr"



def _load_exclusion_list() -> set:
    path = config.OUTPUT_DIR / "contaminated_case_ids.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()



def load_new_dataset(dataset_dir=config.NEW_DATASET_DIR, excluded=None) -> dict:
    if excluded is None:
        excluded = set()

    cases: dict[str, list] = {}

    for path in Path(dataset_dir).glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception:
            continue

        case_id = case.get("case_id")
        if case_id is None or case_id in excluded:
            continue

        if case_id not in cases:
            cases[case_id] = []

        for para in case.get("input_arguments", []):
            agent = para.get("agent", "")
            if agent == "ECHR":
                continue
            label = 1 if agent in ("Applicant", "State") else 0

            for unit in para.get("arg_units", []):
                text = unit.get("word", "").strip()
                if text:
                    cases[case_id].append({"text": text, "label": label})

    return cases


def make_new_dataset_splits(cases: dict, val_ratio=0.15, test_ratio=0.15,
                             seed=config.RANDOM_SEED):

    case_ids = list(cases.keys())
    random.seed(seed)
    random.shuffle(case_ids)

    n      = len(case_ids)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_ids  = case_ids[:n_test]
    val_ids   = case_ids[n_test:n_test + n_val]
    train_ids = case_ids[n_test + n_val:]

    train_rows = [sent for cid in train_ids for sent in cases[cid]]
    val_rows   = [sent for cid in val_ids   for sent in cases[cid]]
    test_rows  = [sent for cid in test_ids  for sent in cases[cid]]

    stats = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    return train_rows, val_rows, test_rows, stats


def balance_classes(rows: list, seed=config.RANDOM_SEED) -> list:
    random.seed(seed)
    premises     = [r for r in rows if r["label"] == 1]
    non_premises = [r for r in rows if r["label"] == 0]
    n = min(len(premises), len(non_premises))
    balanced = random.sample(premises, n) + random.sample(non_premises, n)
    random.shuffle(balanced)
    return balanced


def compute_metrics(eval_pred):
    preds  = np.argmax(eval_pred[0], axis=-1)
    labels = eval_pred[1]
    return {
        "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_binary": f1_score(labels, preds, average="binary", zero_division=0),
    }


def _print_split_stats(name, rows):
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    print(f"  {name:6s}: {len(rows):7d} sentences  "
          f"({n_pos} premises / {n_neg} non-premises)")


def prepare_data():
    """Load dataset, apply fact negatives, split, and balance."""
    excluded = _load_exclusion_list()
    if excluded:
        print(f"  Excluding {len(excluded)} contaminated case_id(s).")

    print(f"Loading new dataset from {config.NEW_DATASET_DIR} ...")
    cases = load_new_dataset(excluded=excluded)
    print(f"  {len(cases)} unique case_ids loaded (argument units)")

    if config.FACT_NEGATIVES:
        from stage1_argument_mining.fact_filter import filter_fact_negatives
        fact_negs = filter_fact_negatives(excluded=excluded)
        added = 0
        for cid, texts in fact_negs.items():
            if cid in cases:
                for t in texts:
                    cases[cid].append({"text": t, "label": 0})
                    added += 1
        print(f"  + {added} similarity-filtered fact negatives from {len(fact_negs)} cases")

    train_rows, val_rows, test_rows, stats = make_new_dataset_splits(cases)
    print(f"  Case-level split: {stats['train']} train / {stats['val']} val / {stats['test']} test cases")

    train_rows = balance_classes(train_rows)

    print("\nSplit sizes (train after balancing):")
    _print_split_stats("train", train_rows)
    _print_split_stats("val",   val_rows)
    _print_split_stats("test",  test_rows)

    return train_rows, val_rows, test_rows


def prepare_datasets(train_rows, val_rows, test_rows):
    """Tokenize and create HuggingFace datasets."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    ds = DatasetDict({
        "train": Dataset.from_dict({"text": [r["text"] for r in train_rows], "label": [r["label"] for r in train_rows]}),
        "validation": Dataset.from_dict({"text": [r["text"] for r in val_rows], "label": [r["label"] for r in val_rows]}),
        "test": Dataset.from_dict({"text": [r["text"] for r in test_rows], "label": [r["label"] for r in test_rows]}),
    })

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.S1_MAX_SEQ_LEN, padding="max_length")

    ds = ds.map(tokenize_batch, batched=True, desc="Tokenising")
    return ds, tokenizer


def train_model(ds, output_dir, epochs, batch_size, lr):
    """Train the model and return trainer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.06,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_binary",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            seed=config.RANDOM_SEED,
            report_to="none",
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    return trainer


def evaluate_and_save(trainer, ds, tokenizer, output_dir):
    """Evaluate on test set and save model."""
    out = trainer.predict(ds["test"])
    preds = np.argmax(out.predictions, axis=-1)
    print("\nTest set results:")
    print(classification_report(out.label_ids, preds, target_names=["NON_PREMISE", "PREMISE"]))

    best_dir = Path(output_dir) / "checkpoint-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(best_dir))
    trainer.save_model(str(best_dir))
    print(f"\nModel saved → {best_dir}")
    print("config.py will auto-detect it on the next run.")


def finetune(epochs=config.S1_EPOCHS,
             batch_size=config.S1_BATCH_SIZE,
             lr=config.S1_LR,
             output_dir=str(config.OUTPUT_DIR / "stage1_legalbert")):
    """Fine-tune LegalBERT for premise classification."""
    seed_everything()

    train_rows, val_rows, test_rows = prepare_data()
    ds, tokenizer = prepare_datasets(train_rows, val_rows, test_rows)
    trainer = train_model(ds, output_dir, epochs, batch_size, lr)
    evaluate_and_save(trainer, ds, tokenizer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=config.S1_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=config.S1_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=config.S1_LR)
    parser.add_argument("--output-dir", type=str,
                        default=str(config.OUTPUT_DIR / "stage1_legalbert"))
    args = parser.parse_args()
    finetune(args.epochs, args.batch_size, args.lr, args.output_dir)
