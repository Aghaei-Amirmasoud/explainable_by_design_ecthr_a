#!/usr/bin/env python
"""Run ONE seeded BERT training run and persist its result.

Each run takes hours, so this is a standalone process rather than a notebook cell:
the notebook kernel can die without losing anything, and — the point — four runs
can occupy the four GPUs of a g5.12xlarge at once. The runs are independent (no
gradient sync), so parallelising them changes nothing about the results.

    # one run per GPU, all four in parallel, ~1 run-length instead of 4
    for i in 0 1 2 3; do
      python run_seed_study.py --model fulltext --seed $((i+1)) --gpu $i \
        > logs/fulltext_s$((i+1)).log 2>&1 &
    done
    wait

Results land in outputs/stage2_bert/_results/<name>.json (a few KB: metrics +
test predictions). The notebook reads those; it never needs the weights.
"""
import argparse
import os
import sys
from pathlib import Path

MODELS = {                        # name -> text-extraction flags
    "fulltext": dict(use_premises=False, use_hybrid=False),
    "premises": dict(use_premises=True,  use_hybrid=False),
    "hybrid":   dict(use_premises=False, use_hybrid=True),
}

HP = dict(epochs=20, batch_size=8, lr=3e-5, physical_batch_size=2,
          seg_chunk_size=128, fp16=True, metric_for_best="macro",
          use_hierarchical=True, progress="heartbeat")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=None,
                    help="physical GPU index to pin this run to")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a finished checkpoint exists")
    ap.add_argument("--name", default=None,
                    help="checkpoint name (default: <model>_bert_s<seed>)")
    ap.add_argument("--discard-weights", action="store_true",
                    help="delete the ~330 MB checkpoint once its result JSON is "
                         "written. The study only needs metrics + predictions; "
                         "15 runs of weights is ~5 GB and will fill the disk.")
    ap.add_argument("--stage1-cache", default=None,
                    help="path to a stage1 extraction JSON other than the default "
                         "(config.STAGE1_CACHE). Use this to run on a different "
                         "extraction, e.g. --stage1-cache outputs/stage1_extracted_premises_thr090.json")
    args = ap.parse_args()

    # must happen before torch is imported anywhere
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import numpy as np
    import config
    from stage1_argument_mining.sequence_filter import load_stage1
    from stage2_outcome_prediction.bert_classifier import (
        train_or_load_bert, predict_bert, save_bert_result,
        bert_result_exists, bert_checkpoint_exists)
    from evaluation.metrics import compute_metrics, per_article_f1

    name  = args.name or f"{args.model}_bert_s{args.seed}"
    flags = MODELS[args.model]

    print("=" * 70)
    print(f"  run   : {name}")
    print(f"  model : {args.model}  {flags}")
    print(f"  seed  : {args.seed}   GPU: {args.gpu if args.gpu is not None else 'default'}")
    print("=" * 70, flush=True)

    if bert_result_exists(name) and not args.force:
        print(f"[skip] result already saved for {name}. Use --force to redo.")
        return 0

    cache_path = Path(args.stage1_cache) if args.stage1_cache else Path(config.STAGE1_CACHE)
    if not cache_path.exists():
        sys.exit(f"Stage 1 cache missing: {cache_path}\n"
                 f"Run the Stage 1 notebook cells first.")
    stage1 = load_stage1(cache_path)

    model, tok, meta = train_or_load_bert(
        name, stage1["train"], stage1["val"],
        force=args.force, seed=args.seed, **HP, **flags)

    y_true = np.array([c["labels_binary"] for c in stage1["test"]])
    y_pred = predict_bert(model, tok, stage1["test"],
                          use_premises=meta["use_premises"],
                          use_hybrid=meta["use_hybrid"],
                          use_hierarchical=meta["use_hierarchical"],
                          batch_size=HP["physical_batch_size"])

    metrics = compute_metrics(y_true, y_pred)
    metrics["per_article"] = per_article_f1(y_true, y_pred)
    save_bert_result(name, metrics, y_pred, meta)

    if args.discard_weights:
        import shutil
        from stage2_outcome_prediction.bert_classifier import BERT_MODEL_DIR
        shutil.rmtree(Path(BERT_MODEL_DIR) / name, ignore_errors=True)
        print(f"[cleanup] removed weights for {name}; result JSON kept")

    print("=" * 70)
    print(f"  {name}:  macro_f1={metrics['macro_f1']:.4f}  "
          f"micro_f1={metrics['micro_f1']:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
