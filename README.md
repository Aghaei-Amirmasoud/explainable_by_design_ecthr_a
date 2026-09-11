# NLPPW — Explainable ECtHR Legal Outcome Prediction

**Two-stage NLP pipeline for predicting ECHR article violations with explainability by design.**

Fine-tuned LegalBERT extracts legally relevant sentences (premises) from case text; an interpretable SVM then predicts which of 10 ECHR articles were violated, attributing each prediction back to specific premises. A hierarchical LegalBERT classifier is included as a supplementary deep-learning comparison.

---

## Results (LexGLUE `ecthr_a` test set, 1,000 cases, 11-class protocol)

### Classical ML (SVM)

| Approach | Macro F1 | Micro F1 |
|---|---|---|
| Baseline (full text) | 0.5669 | 0.6396 |
| Premise-only | 0.5193 | 0.5989 |
| **Hybrid** | **0.5680** | **0.6485** |

The hybrid SVM matches the full-text baseline while providing sentence-level explanations.

### Deep Learning (Hierarchical LegalBERT, 5 seeds)

| Approach | Macro F1 (mean ± std) | Micro F1 (mean ± std) |
|---|---|---|
| Full-text | 0.6342 ± 0.0204 | 0.7021 ± 0.0099 |
| Hybrid (`[PREMISE]` markers) | 0.6308 ± 0.0196 | 0.7024 ± 0.0139 |
| Premises-only | 0.5910 ± 0.0174 | 0.6655 ± 0.0069 |

### Faithfulness (ERASER metrics, 1,180 predicted-article pairs, top-3 premises)

| Metric | Top-k | Random-k | Gap |
|---|---|---|---|
| Comprehensiveness (↑) | 0.2145 | 0.0483 | +0.1662 |
| Prediction held by rationale alone (↑) | 86.3% | 60.8% | +25.4% |
| Prediction flips when removed (↑) | 21.4% | 9.8% | +11.5% |

Cited premises are 4.4× more impactful than random ones. 86.3% of predictions are reconstructed from 3 cited sentences alone.

---

## Project Structure

```
NLPPW/
├── config.py                              # All hyperparameters and paths
├── run_pipeline.py                        # Main entry point (stages 1+2+eval)
├── run_seed_study.py                      # Single seeded BERT run (CLI, resumable)
├── check_contamination.py                 # LexGLUE vs all-data/ fingerprint check
├── sample.py                              # Inspect a random file from all-data/
│
├── all-data/                              # 12,947 annotated JSON files (case, article pairs)
├── echr_corpus/ECHR_Corpus.json           # Legacy 42-doc Poudyal dataset
│
├── data/
│   └── data_loader.py                     # LexGLUE ecthr_a loading & preprocessing
│
├── stage1_argument_mining/
│   ├── finetune_legalbert.py              # Fine-tune LegalBERT on all-data/
│   ├── fact_filter.py                     # Similarity-filtered factual negatives
│   ├── argument_extractor.py              # Sentence split → LegalBERT → premises
│   └── sequence_filter.py                 # Apply extractor to full LexGLUE dataset
│
├── stage2_outcome_prediction/
│   ├── embedder.py                        # SentenceTransformer + concat pooling
│   ├── embedder_hybrid.py                 # Hybrid embedder (full text + premise features)
│   ├── classifier.py                      # MultiOutputClassifier (SVM/DT/EBM)
│   ├── traceback.py                       # Premise-level attribution for SVM
│   └── bert_classifier.py                 # Hierarchical LegalBERT multi-label classifier
│
├── evaluation/
│   ├── metrics.py                         # Macro/micro F1, per-article F1
│   ├── extractor_controls.py              # Matched-budget controls (premise/complement/random/lead)
│   ├── faithfulness.py                    # ERASER comprehensiveness & sufficiency
│   ├── qualitative_review.py              # Manual inspection helpers
│   └── premise_count_analysis.py          # Performance vs premise count stratification
│
├── outputs/                               # Cached models, extracted premises, results
├── sagemaker_pipeline.ipynb               # AWS SageMaker end-to-end notebook
└── colab_pipeline.ipynb                   # Google Colab end-to-end notebook
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Check for contamination (optional but recommended)
```bash
python check_contamination.py
```
Outputs `outputs/contaminated_case_ids.json` — automatically excluded during Stage 1 training.

### 2. Fine-tune LegalBERT (Stage 1, run once)
```bash
python stage1_argument_mining/finetune_legalbert.py
```
Trains on `all-data/` with similarity-filtered factual negatives. Saves to `outputs/stage1_legalbert/checkpoint-best`.

### 3. Run the full pipeline
```bash
python run_pipeline.py
```

### 4. Run individual stages
```bash
python run_pipeline.py --stage 1      # Stage 1 only (premise extraction)
python run_pipeline.py --stage 2      # Stage 2 only (outcome prediction + eval)
python run_pipeline.py --stage eval   # Evaluation only (requires trained classifier)
python run_pipeline.py --force        # Force re-run (ignore cached premises)
```

### 5. Run the BERT seed study (5 seeds × 3 models, ~15 h)
```bash
mkdir -p logs
for model in fulltext premises hybrid; do
  for s in 1 2 3 4 5; do
    python run_seed_study.py \
      --model $model --seed $s --discard-weights \
      --stage1-cache outputs/stage1_extracted_premises_thr090.json \
      --name ${model}_thr090_s${s} \
      >> logs/${model}_thr090_s${s}.log 2>&1
  done
done
```
Each run writes a ~32 KB result JSON and deletes its ~330 MB checkpoint. Resumable: finished runs are skipped automatically.

---

## How It Works

### Stage 1 — Argument Mining

**Training data** (`all-data/`, 12,947 cases):
- **PREMISE (1)**: `arg_units` where `agent` ∈ {Applicant, State}
- **NON_PREMISE (0)**: `Non-Argument` units + similarity-filtered factual sentences (cosine similarity < 0.6 to any premise)
- **Excluded**: ECHR agent (court findings — outcome leakage)
- Case-level train/val/test split (70/15/15)

**Stage 1 test results** (on fine-tuning data):
- Macro F1: 0.9603 | Binary PREMISE F1: 0.9504

**Inference on LexGLUE:**
- Regex sentence splitter (`[.!?]` + whitespace + uppercase)
- Fixed threshold 0.90 (justified by bimodal score distribution: 63.8% of sentences score below 0.1, 27.0% above 0.9, natural valley at 0.9)
- Fallback: zero-premise cases use full paragraph text
- Output: 21.3 premises/case on average (test split)

### Stage 2 — Outcome Prediction

**Embedding** (`all-mpnet-base-v2`, 768-d per sentence):
- Concatenated pooling: max + mean + weighted_mean → 2,304-d
- Weighted mean uses LegalBERT premise confidence scores as weights
- + 5 handcrafted features → 2,309-d total

**Three conditions:**
1. **Baseline**: embeds full paragraph text (no premise extraction)
2. **Premise-only**: embeds only extracted premise sentences
3. **Hybrid**: embeds all paragraphs + per-paragraph premise features (has_premise, n_premises, premise_density, avg_confidence) with premise-aware pooling boost (2.0×)

**Classification:**
- `MultiOutputClassifier` wrapping `LinearSVC` (one binary classifier per article)
- `class_weight='balanced'` for label imbalance
- Per-article threshold tuning on validation set

**Explainability:**
- Project premise embeddings onto SVM weight vector
- Rank premises by signed contribution to each predicted article
- Verified faithful by ERASER metrics: comprehensiveness 4.4× better than random

### BERT Classifier (supplementary)

Hierarchical LegalBERT: 64 segments × 128 tokens per segment, 2-layer segment transformer over [CLS] embeddings, max pooling. Protocol: lr=3e-5, effective batch 8 (physical 2 × gradient accumulation 4), 20 epochs, fp16, early-stopping patience 3. Checkpoints selected on macro-F1 (note: LexGLUE uses micro — stated deviation).

---

## Extractor Validation

To verify the extractor selects better-than-arbitrary sentences, all four conditions are matched to the same word budget (~27% of full text):

| Condition | Description | Macro F1 |
|---|---|---|
| premises | LegalBERT-selected sentences | 0.5193 |
| complement | Sentences LegalBERT rejected | 0.5157 |
| random | Random sentences at same budget | 0.4935 |
| lead | First N sentences at same budget | 0.4713 |

Over 5 random seeds: premises beats random at z = +1.11 (modest positive signal); ties complement at z = −0.15. The extractor identifies argument-like sentences, but argument-likeness and article-predictive signal are not equivalent — this is a finding about the task, not a failure of the system.

---

## Key Configuration (`config.py`)

```python
PREMISE_THRESHOLD      = 0.90          # fixed sentence-score cutoff
FACT_NEGATIVES         = True          # similarity-filtered factual negatives
FACT_SIM_THRESHOLD     = 0.6           # max cosine sim to any premise → safe NON_PREMISE
SENTENCE_TRANSFORMER   = "all-mpnet-base-v2"
POOLING_STRATEGY       = "concat"      # max + mean + weighted_mean → 2304-d
CLASSIFIER_TYPE        = "svm"         # options: svm, decision_tree, ebm
RANDOM_SEED            = 42
```

---

## Datasets

| Source | Size | Used for |
|---|---|---|
| `all-data/` | 12,947 JSON files | Stage 1 fine-tuning |
| LexGLUE `ecthr_a` | 11,000 cases (9k/1k/1k) | Stage 2 train/val/test + Stage 1 inference |

### `all-data/` file structure
```json
{
  "case_id": "...",
  "article": "...",
  "judgment": "violation | no-violation",
  "input_arguments": [
    {
      "agent": "Applicant | State | Non-Argument | ECHR",
      "arg_units": [{"word": "...", "claim": true, ...}]
    }
  ],
  "facts_section": {"content": "...", "elements": [...]},
  "law_section": {...}
}
```
`all_arguments` (includes ECHR agent) must never be used for training — it contains court findings that leak the outcome.

---

## Caching

| File | Purpose |
|---|---|
| `outputs/stage1_legalbert/checkpoint-best` | Fine-tuned LegalBERT checkpoint |
| `outputs/stage1_extracted_premises_thr090.json` | Extracted premises (threshold 0.90) |
| `outputs/fact_negatives_filtered.json` | Similarity-filtered factual negatives |
| `outputs/contaminated_case_ids.json` | LexGLUE-contaminated case IDs |
| `outputs/extractor_controls_thr090.json` | Matched-budget control results |
| `outputs/stage2_bert/_results/*.json` | Per-seed BERT metrics + predictions |

Delete a cache file to rebuild it from scratch.

---

## Citations

```bibtex
@inproceedings{chalkidis-etal-2022-lexglue,
  title     = {{LexGLUE}: A Benchmark Dataset for Legal Language Understanding in {E}nglish},
  author    = {Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
               Bommarito, Michael and Androutsopoulos, Ion and
               Katz, Daniel and Aletras, Nikolaos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2022},
  pages     = {4310--4330},
  publisher = {Association for Computational Linguistics},
}

@inproceedings{chalkidis-etal-2020-legal,
  title     = {{LEGAL-BERT}: The Muppets straight out of Law School},
  author    = {Chalkidis, Ilias and Fergadiotis, Manos and
               Malakasiotis, Prodromos and Aletras, Nikolaos and
               Androutsopoulos, Ion},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020},
  pages     = {2898--2904},
  publisher = {Association for Computational Linguistics},
}

@inproceedings{reimers-gurevych-2019-sentence,
  title     = {Sentence-{BERT}: Sentence Embeddings using {S}iamese {BERT}-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year      = {2019},
  pages     = {3982--3992},
  publisher = {Association for Computational Linguistics},
}

@inproceedings{deyoung-etal-2020-eraser,
  title     = {{ERASER}: A Benchmark to Evaluate Rationalized {NLP} Models},
  author    = {DeYoung, Jay and Jain, Sarthak and Rajani, Nazneen Fatema and
               Lehman, Eric and Xiong, Caiming and Socher, Richard and
               Wallace, Byron C.},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020},
  pages     = {4443--4458},
  publisher = {Association for Computational Linguistics},
}

@misc{azminajid-echrargs,
  author = {azminajid},
  title  = {{ECHR} Arguments Dataset},
  year   = {2024},
  url    = {https://github.com/azminajid/echr-args-dataset},
  note   = {GitHub repository. Original annotation provenance unknown.}
}

@inproceedings{poudyal-etal-2020-echr,
  title     = {{ECHR}: Legal Corpus for Argument Mining},
  author    = {Poudyal, Prakash and Savelka, Jaromir and Ieven, Aagje and
               Moens, Marie Francine and Drummond, Tom and Wyner, Adam},
  booktitle = {Proceedings of the 7th Workshop on Argument Mining},
  year      = {2020},
  pages     = {67--75},
  publisher = {Association for Computational Linguistics},
  note      = {Legacy 42-document corpus (echr\_corpus/); superseded by azminajid-echrargs for Stage 1 fine-tuning.}
}

@article{pedregosa-etal-2011-sklearn,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and
             Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and
             Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and
             Dubourg, Vincent and others},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011},
}

@inproceedings{lou-etal-2012-ebm,
  title     = {Intelligible Models for Classification and Regression},
  author    = {Lou, Yin and Caruana, Rich and Gehrke, Johannes},
  booktitle = {Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year      = {2012},
  pages     = {150--158},
  publisher = {ACM},
}
```
