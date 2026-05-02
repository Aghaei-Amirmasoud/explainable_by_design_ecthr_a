import os
from pathlib import Path

ROOT_DIR   = Path(__file__).parent

# Load .env if present
_env_file = ROOT_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login as _hf_login
    _hf_login(token=HF_TOKEN, add_to_git_credential=False)


OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STAGE1_CACHE = OUTPUT_DIR / "stage1_extracted_premises.json"
STAGE2_CACHE = OUTPUT_DIR / "stage2_embeddings.npy"
MODEL_CACHE  = OUTPUT_DIR / "stage2_classifier.joblib"

LEXGLUE_DATASET   = "lex_glue"
LEXGLUE_CONFIG    = "ecthr_a"
DATASET_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None
MAX_TEST_SAMPLES  = None

ECHR_CORPUS_PATH   = ROOT_DIR / "echr_corpus" / "ECHR_Corpus.json"
NEW_DATASET_DIR    = ROOT_DIR / "all-data"
FINETUNED_S1_MODEL = OUTPUT_DIR / "stage1_legalbert" / "checkpoint-best"

LEGALBERT_MODEL    = str(FINETUNED_S1_MODEL) if (OUTPUT_DIR / "stage1_legalbert" / "checkpoint-best").exists() \
                     else "nlpaueb/bert-base-uncased-echr"
PREMISE_LABEL      = "PREMISE"

S1_EPOCHS      = 3
S1_BATCH_SIZE  = 32
S1_LR          = 2e-5
S1_MAX_SEQ_LEN = 512
PREMISE_THRESHOLD = 0.75

DYNAMIC_TOPK       = True
DYNAMIC_TOPK_ALPHA = 0.5
DYNAMIC_TOPK_MIN   = 1
DYNAMIC_TOPK_MAX   = 10
PREMISE_FLOOR      = 0.50

FACT_NEGATIVES          = True
FACT_SIM_THRESHOLD      = 0.6
FACT_NEGATIVES_CACHE    = OUTPUT_DIR / "fact_negatives_filtered.json"

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
POOLING_STRATEGY = "concat"   # options: max, mean, weighted_mean, concat
CLASSIFIER_TYPE  = "svm"      # only SVM is supported

# Hybrid embedding experiment: embed full paragraphs with premise-aware features
# instead of only extracted premise sentences
USE_HYBRID_EMBEDDER = False
HYBRID_PREMISE_WEIGHT_BOOST = 2.0  # multiplier for paragraphs containing premises

# SVM classifier settings
SVM_C        = 1.0
SVM_MAX_ITER = 2000

NUM_LABELS         = 10
EVAL_AVERAGE       = "macro"
QUALITATIVE_N      = 10
RANDOM_SEED        = 42
