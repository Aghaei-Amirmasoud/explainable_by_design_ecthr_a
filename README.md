# NLPPW — Explainable ECtHR Legal Outcome Prediction

**Two-stage NLP pipeline for predicting ECHR article violations with explainability by design.**

Fine-tuned LegalBERT extracts legal premises from case text, then interpretable classifiers (SVM) or deep learning models (LegalBERT) predict which of 10 ECHR articles were violated, with per-premise attribution for explainability.

---

## Features

- **12,947 case dataset** (`all-data/`) with sentence-level argument annotations
- **Dynamic premise extraction** with adaptive per-paragraph thresholds
- **Similarity-filtered factual negatives** to improve Stage 1 generalization
- **Three embedding approaches**:
  - **Baseline**: Raw text embeddings (no premise extraction)
  - **Premise-only**: Only extracted premise sentences
  - **Hybrid**: Full paragraphs with premise-awareness features
- **Concatenated pooling** (max+mean+weighted_mean) for robust embeddings
- **Per-article threshold tuning** to handle class imbalance
- **Premise count analysis**: Performance stratification by extraction density
- **Multiple model architectures**:
  - Classical ML: SVM with explainability via weight vectors
  - Deep Learning: Fine-tuned LegalBERT (full-text, premises, hybrid with markers)
- **Full explainability**: trace predictions back to specific premises
- **Contamination checking** to prevent test leakage

---

## Project Structure

```
NLPPW/
├── README.md
├── requirements.txt
├── config.py                              # All hyperparameters and paths
├── run_pipeline.py                        # Main entry point (stages 1+2+eval)
├── check_contamination.py                 # LexGLUE vs all-data/ fingerprint check
├── visualize.py                           # Visualization utilities
│
├── all-data/                              # 12,500+ annotated JSON files (case, article pairs)
├── echr_corpus/ECHR_Corpus.json          # Legacy 42-doc Poudyal dataset
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
│   ├── embedder.py                        # SentenceTransformer + concat pooling (premise-only)
│   ├── embedder_hybrid.py                 # Hybrid embedder (full text + premise features)
│   ├── classifier.py                      # MultiOutputClassifier (SVM)
│   ├── traceback.py                       # Premise-level attribution
│   └── bert_classifier.py                 # LegalBERT multi-label classifier (3 modes)
│
├── evaluation/
│   ├── metrics.py                         # Macro/micro F1, per-article F1
│   ├── qualitative_review.py              # Manual inspection helpers
│   └── premise_count_analysis.py          # Performance vs premise count stratification
│
├── outputs/                               # Cached models, extracted premises, contamination lists
├── colab_pipeline.ipynb                   # Google Colab notebook
└── sagemaker_pipeline.ipynb               # AWS SageMaker notebook
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

### 2. Fine-tune LegalBERT (Stage 1 training)
```bash
python stage1_argument_mining/finetune_legalbert.py
```
Trains on `all-data/` with similarity-filtered factual negatives. Saves checkpoint to `outputs/stage1_legalbert/checkpoint-best`.

### 3. Run full pipeline
```bash
python run_pipeline.py
```
Runs Stage 1 (premise extraction), Stage 2 (outcome prediction), and evaluation.

### 4. Run individual stages
```bash
# Stage 1 only (argument extraction)
python run_pipeline.py --stage 1

# Stage 2 only (outcome prediction + evaluation)
python run_pipeline.py --stage 2

# Evaluation only (requires trained classifier)
python run_pipeline.py --stage eval

# Force re-run (ignore cached premises)
python run_pipeline.py --force
```

---

## How It Works

### Stage 1 — Argument Mining

**Training:**
- Fine-tune LegalBERT on `all-data/` (12,500+ cases)
- **PREMISE (label 1)**: sentences from Applicant/State `arg_units`
- **NON_PREMISE (label 0)**: 
  - `Non-Argument` arg_units (Convention article quotes)
  - Similarity-filtered facts from `facts_section` (cosine similarity < 0.6 to any premise)
- **Excluded**: ECHR agent arguments (outcome leakage)
- Case-level train/val/test split to prevent leakage

**Inference (on LexGLUE paragraphs):**
- Sentence split via regex (`[.!?]` + whitespace + uppercase)
- LegalBERT scores each sentence
- **Dynamic top-k**: adaptive per-paragraph threshold = mean + α×std of viable scores (≥ 0.50)
- Fallback: if no premises found, use full paragraph text

### Stage 2 — Outcome Prediction

**Three Approaches:**

1. **Baseline (Raw Text)**
   - Embeds full paragraph text without premise extraction
   - Useful for zero-premise cases and as performance baseline

2. **Premise-Only (Current Main Approach)**
   - Embeds only extracted premise sentences
   - Best for cases with many premises (10+)
   - Classical ML: 2304-d pooled + 5 handcrafted features = 2309-d
   - LegalBERT: Fine-tuned on premise text only

3. **Hybrid (Experimental)**
   - **Classical ML**: Embeds all paragraphs with per-paragraph premise features
     - Features: `has_premise`, `n_premises`, `premise_density`, `avg_confidence`
     - Premise-aware weighting (boost factor: 2.0x) during pooling
     - Best for moderate premise counts (3-5)
   - **LegalBERT**: Full text with `[PREMISE]` markers around premise paragraphs
     - Model learns to attend to marked regions
     - Preserves context while signaling relevance

**Classical ML Embedding:**
- `all-mpnet-base-v2` (768-d) encodes each premise/paragraph
- **Concatenated pooling**: max + mean + weighted_mean → 2304-d per case
- Weighted mean uses premise confidence scores as weights
- Result: 2308-2309 features (2304-d pooled + handcrafted)

**Classification:**
- **SVM** (`MultiOutputClassifier` with `LinearSVC`)
  - One binary classifier per article
  - `class_weight='balanced'` for imbalance handling
  - **Per-article threshold tuning**: maximize per-article F1 on validation set
- **LegalBERT** (deep learning alternative)
  - Fine-tuned on full text, premises, or hybrid (marked) text
  - End-to-end multi-label classification

**Explainability (SVM):**
- Project premise embeddings onto SVM weight vector
- Rank premises by contribution to prediction
- Trace each article violation back to supporting premises

**Performance Analysis:**
- Three-way comparison: Baseline vs Premise vs Hybrid
- Premise count stratification: performance across 0, 1, 2, 3-5, 6-10, 10+ bins
- Per-article F1 breakdown

---

## Key Configuration (`config.py`)

```python
# Stage 1 (Argument Mining)
LEGALBERT_MODEL        = "outputs/stage1_legalbert/checkpoint-best"
PREMISE_THRESHOLD      = 0.75        # fixed threshold (if dynamic disabled)
DYNAMIC_TOPK           = True        # adaptive per-paragraph threshold
DYNAMIC_TOPK_ALPHA     = 0.5         # mean + alpha*std
PREMISE_FLOOR          = 0.50        # minimum viable score
FACT_NEGATIVES         = True        # similarity-filtered factual negatives
FACT_SIM_THRESHOLD     = 0.6         # max cosine sim → safe NON_PREMISE

# Stage 2 (Outcome Prediction)
SENTENCE_TRANSFORMER   = "all-mpnet-base-v2"
POOLING_STRATEGY       = "concat"    # max, mean, weighted_mean, concat
CLASSIFIER_TYPE        = "svm"       # only SVM is supported

# Hybrid Experiment
USE_HYBRID_EMBEDDER          = False  # enable hybrid approach
HYBRID_PREMISE_WEIGHT_BOOST  = 2.0    # weight multiplier for premise paragraphs

# SVM Settings
SVM_C                  = 1.0
SVM_MAX_ITER           = 2000

# General
RANDOM_SEED            = 42
```

---

## Datasets

| Source | Used for |
|--------|----------|
| `all-data/` (12,947 JSON files) | Stage 1 fine-tuning |
| LexGLUE `ecthr_a` (HuggingFace) | Stage 2 labels + Stage 1 inference target |

### `all-data/` file structure
```json
{
  "case_id": "...",
  "article": "...",
  "judgment": "violation" | "no-violation",
  "input_arguments": [
    {
      "agent": "Applicant" | "State" | "Non-Argument" | "ECHR",
      "arg_units": [
        {"word": "...", "claim": true/false, ...}
      ]
    }
  ],
  "all_arguments": [...],  // includes ECHR — NEVER use for training
  "facts_section": {
    "content": "...",
    "elements": [...]
  },
  "law_section": {...}
}
```

---

## Caching

| Cache file | Regenerated when |
|------------|------------------|
| `outputs/stage1_extracted_premises.json` | `--force` flag or file missing |
| `outputs/stage1_legalbert/checkpoint-best` | Manually via `finetune_legalbert.py` |
| `outputs/stage2_classifier.joblib` | Premise-only classifier (premise embedder) |
| `outputs/stage2_classifier_hybrid.joblib` | Hybrid classifier (hybrid embedder) |
| `outputs/fact_negatives_filtered.json` | Deleted manually (rebuilds with new `FACT_SIM_THRESHOLD`) |
| `outputs/contaminated_case_ids.json` | `check_contamination.py` re-run |
| `outputs/premise_count_analysis_*.png` | Generated during evaluation |

---

## Notebooks

Both notebooks include:
- Stage 1 fine-tuning with similarity-filtered factual negatives
- Stage 2 classical ML (premise + hybrid) with three-way evaluation
- LegalBERT comparison (full-text vs premises vs hybrid with markers)
- Premise count analysis for both classical ML and BERT
- Qualitative review and demo inference

**Notebooks:**
- **`colab_pipeline.ipynb`**: Google Colab end-to-end pipeline (T4 GPU recommended)
- **`sagemaker_pipeline.ipynb`**: AWS SageMaker end-to-end pipeline (ml.g4dn.xlarge or ml.g5.xlarge)

---

## Evaluation

**Metrics (LexGLUE protocol):**
- Appends "No Violation" column (y=1 when all article labels are 0)
- Reports macro/micro F1, per-article F1, precision, recall, hamming loss

**Comparisons:**
1. **Three-way evaluation**: Baseline vs Premise vs Hybrid
   - Overall metrics table
   - Per-article F1 breakdown
   - Determines best approach

2. **Premise count analysis**:
   - Stratifies test cases by number of extracted premises (0, 1, 2, 3-5, 6-10, 10+)
   - Shows which approach works best at each density level
   - Generates plots: `premise_count_analysis_classical.png`, `premise_count_analysis_bert.png`

**Key Findings:**
- **Classical ML (SVM)**: Hybrid wins overall (macro F1: 0.56), excels at 3-5 premises
- **Deep Learning (LegalBERT)**: Premises-only wins (macro F1: 0.52), dominates with 10+ premises
- Architecture choice depends on premise extraction density distribution

---

## Citation


```bibtex

```
