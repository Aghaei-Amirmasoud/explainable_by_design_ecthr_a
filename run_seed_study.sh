#!/usr/bin/env bash
# Full LexGLUE protocol: 5 seeds x 3 models = 15 runs, serially on ONE GPU.
#
# Runs outside Jupyter on purpose: an 11-hour study should not depend on a
# notebook kernel staying alive. Each run writes its own result JSON, so this
# is resumable -- re-run the script and finished runs are skipped.
#
#   nohup bash run_seed_study.sh > logs/driver.log 2>&1 &
#   tail -f logs/driver.log
#
# If you ever do have multiple GPUs, add --gpu N and background the inner loop.
set -u
cd "$(dirname "$0")"
mkdir -p logs

SEEDS="1 2 3 4 5"
MODELS="fulltext hybrid premises"

start=$(date +%s)
for model in $MODELS; do
  for seed in $SEEDS; do
    echo "=== $model seed $seed  $(date '+%F %T') ==="
    # --discard-weights: 15 checkpoints is ~5 GB and fills the disk. The study
    # needs only the metrics + predictions in the result JSON, so the weights go
    # once the result is written. Drop the flag if you want the models kept.
    python run_seed_study.py --model "$model" --seed "$seed" --discard-weights \
      2>&1 | tee "logs/${model}_s${seed}.log" | grep -E '^\s{2}(epoch|\S+_bert)|Best val|Saved|cleanup'
  done
done
echo
echo "Total wall clock: $(( ($(date +%s) - start) / 60 )) min"
echo
python -c "
from stage2_outcome_prediction.bert_classifier import list_bert_runs
list_bert_runs()
"
