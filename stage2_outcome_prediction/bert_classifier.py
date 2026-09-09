import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, NamedTuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoConfig, AutoModel,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)
from sklearn.metrics import f1_score
from tqdm import tqdm
import config

BERT_MODEL_DIR = config.OUTPUT_DIR / "stage2_bert"


class ModelOutput(NamedTuple):
    loss: Optional[torch.Tensor]
    logits: torch.Tensor

# LexGLUE hierarchical config: 64 segments × 128 tokens = 8,192 token capacity
HIER_SEG_LEN  = 128
HIER_MAX_SEGS = 64


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_texts(cases, use_premises, use_hybrid=False):
    texts = []
    for c in cases:
        if use_hybrid:
            para_to_premises = {}
            for p in c.get("premises", []):
                para_id = p.get("paragraph_id", -1)
                para_to_premises.setdefault(para_id, []).append(p["sentence"])
            marked = []
            for para_idx, paragraph in enumerate(c.get("paragraphs", [])):
                if para_idx in para_to_premises:
                    marked.append(f"[PREMISE] {paragraph} [/PREMISE]")
                else:
                    marked.append(paragraph)
            texts.append(" ".join(marked))
        elif use_premises and c.get("premises"):
            texts.append(" ".join(p["sentence"] for p in c["premises"]))
        else:
            texts.append(" ".join(c["paragraphs"]))
    return texts


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CaseTextDataset(Dataset):
    """Flat dataset: single truncated sequence per case (512 tokens)."""

    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


class HierarchicalCaseDataset(Dataset):
    """Hierarchical dataset: up to max_segs segments of seg_len tokens each.

    Matches LexGLUE protocol: 64 segments × 128 tokens = 8,192 token capacity.
    Each segment gets its own [CLS]/[SEP] so BERT processes it independently.
    """

    def __init__(self, texts, labels, tokenizer,
                 seg_len=HIER_SEG_LEN, max_segs=HIER_MAX_SEGS):
        self.labels    = labels
        self.seg_len   = seg_len
        self.max_segs  = max_segs
        pad_id = tokenizer.pad_token_id
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        inner  = seg_len - 2  # tokens per segment excluding [CLS]/[SEP]

        self.all_input_ids      = []
        self.all_attention_masks = []

        for text in texts:   # silent: a bar here floods notebook output
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

            seg_ids, seg_masks = [], []
            for start in range(0, len(token_ids), inner):
                chunk = token_ids[start:start + inner]
                seq   = [cls_id] + chunk + [sep_id]
                pad_n = seg_len - len(seq)
                mask  = [1] * len(seq) + [0] * pad_n
                seq   = seq + [pad_id] * pad_n
                seg_ids.append(seq)
                seg_masks.append(mask)
                if len(seg_ids) == max_segs:
                    break

            # Pad document to max_segs with empty segments
            while len(seg_ids) < max_segs:
                seg_ids.append([pad_id] * seg_len)
                seg_masks.append([0] * seg_len)

            self.all_input_ids.append(seg_ids)
            self.all_attention_masks.append(seg_masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.all_input_ids[idx],       dtype=torch.long),
            "attention_mask": torch.tensor(self.all_attention_masks[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],               dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Hierarchical model wrapper (matches LexGLUE hierbert.py)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional embeddings with padding_idx=0 (matches LexGLUE)."""

    def __init__(self, max_positions, hidden_dim):
        super().__init__()
        pe = torch.zeros(max_positions + 1, hidden_dim)
        position = torch.arange(1, max_positions + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, seg_valid):
        B, S, H = x.shape
        positions = torch.zeros(B, S, dtype=torch.long, device=x.device)
        for b in range(B):
            n_valid = seg_valid[b].sum().item()
            positions[b, :int(n_valid)] = torch.arange(1, int(n_valid) + 1, device=x.device)
        return x + self.pe[positions]


class HierarchicalBertClassifier(nn.Module):
    """LexGLUE-faithful hierarchical BERT: segment encoder -> 2-layer segment
    transformer -> max pool -> linear head.

    Forward path (matches coastalcph/lex-glue models/hierbert.py):
      input_ids / attention_mask: (B, S, L)
      -> BERT encodes each segment in chunks -> [CLS] per segment -> (B, S, H)
      -> sinusoidal pos embeddings added
      -> 2-layer Transformer encoder over segments (padding mask applied)
      -> max pool over non-empty segments -> (B, H)
      -> linear -> (B, num_labels)
    """

    def __init__(self, encoder, hidden_dim, num_labels,
                 max_segs=HIER_MAX_SEGS, seg_chunk_size=16,
                 skip_empty_segments=True):
        super().__init__()
        self.encoder             = encoder
        self.seg_chunk_size      = seg_chunk_size
        self.skip_empty_segments = skip_empty_segments
        self.pos_encoder         = SinusoidalPositionalEncoding(max_segs, hidden_dim)

        cfg = encoder.config
        self.seg_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=cfg.num_attention_heads,
                dim_feedforward=cfg.intermediate_size,
                dropout=cfg.hidden_dropout_prob,
                activation=cfg.hidden_act,
                layer_norm_eps=cfg.layer_norm_eps,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        B, S, L = input_ids.shape
        flat_ids   = input_ids.view(B * S, L)
        flat_masks = attention_mask.view(B * S, L)

        # Cases are padded to max_segs, but the median case needs ~10 of 64 segments,
        # so ~75% of these rows are all-padding. Their [CLS] is discarded anyway --
        # src_key_padding_mask keeps them out of the segment transformer's attention
        # and they are -inf-masked before the max pool -- so encoding them is pure
        # waste. Run BERT on the real segments only and scatter back; the result for
        # every valid segment is unchanged.
        valid_flat = flat_masks.any(dim=-1)                      # (B*S,)
        if self.skip_empty_segments and not bool(valid_flat.all()):
            keep      = valid_flat.nonzero(as_tuple=True)[0]
            sel_ids   = flat_ids[keep]
            sel_masks = flat_masks[keep]
        else:
            keep      = None
            sel_ids   = flat_ids
            sel_masks = flat_masks

        cls_chunks = []
        for start in range(0, sel_ids.size(0), self.seg_chunk_size):
            chunk_ids   = sel_ids[start:start + self.seg_chunk_size]
            chunk_masks = sel_masks[start:start + self.seg_chunk_size]
            out = self.encoder(input_ids=chunk_ids, attention_mask=chunk_masks)
            cls_chunks.append(out.last_hidden_state[:, 0, :])
        cls_sel = torch.cat(cls_chunks, dim=0)

        if keep is None:
            cls_emb = cls_sel.view(B, S, -1)
        else:
            cls_flat = cls_sel.new_zeros(B * S, cls_sel.size(-1))
            cls_flat[keep] = cls_sel
            cls_emb = cls_flat.view(B, S, -1)

        seg_valid    = valid_flat.view(B, S)
        padding_mask = ~seg_valid

        cls_emb = self.pos_encoder(cls_emb, seg_valid)
        seg_out = self.seg_encoder(cls_emb, src_key_padding_mask=padding_mask)

        seg_out = seg_out.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        pooled, _ = seg_out.max(dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        return ModelOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bert_classifier(train_cases, val_cases,
                           use_premises=True, use_hybrid=False,
                           use_hierarchical=True,
                           model_name=None,
                           epochs=20,
                           batch_size=8,
                           lr=3e-5,
                           warmup_ratio=0.0,
                           weight_decay=0.0,
                           patience=3,
                           metric_for_best="macro",  # "macro" or "micro" (LexGLUE uses micro)
                           progress="heartbeat",     # "heartbeat" | "bar" | "none"
                           fp16=True,
                           seg_len=HIER_SEG_LEN,
                           max_segs=HIER_MAX_SEGS,
                           physical_batch_size=1,
                           seg_chunk_size=16,
                           seed=None,
                           checkpoint_dir=None):
    """Train LegalBERT classifier (single GPU).

    use_hierarchical=True  -> 64x128-token hierarchical encoding (LexGLUE protocol)
    use_hierarchical=False -> single 512-token truncated sequence (legacy)

    `seed` controls classifier-head init and batch shuffling (same as LexGLUE's
    `--seed`). cuDNN kernels stay nondeterministic, so runs are not bit-identical,
    but the seed is what varies across a multi-seed study. seed=None leaves the
    global RNG untouched (legacy behaviour, not reproducible).
    """
    if model_name is None:
        model_name = "nlpaueb/bert-base-uncased-echr"

    if seed is not None:
        set_seed(seed)
        print(f"[BERT] seed={seed}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts  = _extract_texts(train_cases, use_premises, use_hybrid)
    val_texts    = _extract_texts(val_cases,   use_premises, use_hybrid)
    train_labels = [c["labels_binary"] for c in train_cases]
    val_labels   = [c["labels_binary"] for c in val_cases]

    accum_steps = max(1, batch_size // physical_batch_size)

    if use_hierarchical:
        print(f"[BERT] Hierarchical mode: {max_segs} segments x {seg_len} tokens = "
              f"{max_segs * seg_len:,} token capacity")
        print(f"[BERT] physical_batch={physical_batch_size}, accum_steps={accum_steps} "
              f"-> effective_batch={physical_batch_size * accum_steps}")
        train_ds = HierarchicalCaseDataset(train_texts, train_labels, tokenizer, seg_len, max_segs)
        val_ds   = HierarchicalCaseDataset(val_texts,   val_labels,   tokenizer, seg_len, max_segs)
        encoder  = AutoModel.from_pretrained(model_name).to(device)
        model    = HierarchicalBertClassifier(encoder, encoder.config.hidden_size,
                                              config.NUM_LABELS,
                                              seg_chunk_size=seg_chunk_size).to(device)
    else:
        print("[BERT] Flat mode: single 512-token sequence (truncated)")
        train_ds = CaseTextDataset(train_texts, train_labels, tokenizer)
        val_ds   = CaseTextDataset(val_texts,   val_labels,   tokenizer)
        model    = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=config.NUM_LABELS,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        ).to(device)

    train_dl = DataLoader(train_ds, batch_size=physical_batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=physical_batch_size)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = (len(train_dl) // accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if fp16 and device.type == "cuda" else None
    sel_key = f"{metric_for_best}_f1"
    print(f"[BERT] lr={lr} | wd={weight_decay} | fp16={scaler is not None} | "
          f"warmup={warmup_steps}/{total_steps} | patience={patience} | "
          f"select_best_on={sel_key}")

    best_f1, best_state, no_improve = 0.0, None, 0

    n_steps  = len(train_dl)
    hb_every = max(1, n_steps // 4) if progress == "heartbeat" else 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        iterator = (tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
                    if progress == "bar" else train_dl)
        for step, batch in enumerate(iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(**batch)
                scaler.scale(out.loss / accum_steps).backward()
            else:
                out = model(**batch)
                (out.loss / accum_steps).backward()
            total_loss += out.loss.item()

            if (step + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if hb_every and (step + 1) % hb_every == 0:
                print(f"    epoch {epoch+1}/{epochs}  {100*(step+1)//n_steps:>3}%  "
                      f"({step+1}/{n_steps})  loss={total_loss/(step+1):.4f}", flush=True)

        val_metrics = _evaluate(model, val_dl, device)
        print(f"  loss={total_loss/len(train_dl):.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}")

        if val_metrics[sel_key] > best_f1:
            best_f1    = val_metrics[sel_key]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if checkpoint_dir is not None:
                # flush best-so-far to disk: a run killed at epoch 18/20 should
                # leave a usable model behind, not nothing
                _d = Path(checkpoint_dir)
                _d.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, _d / "pytorch_model.bin")
                (_d / "progress.json").write_text(json.dumps({
                    "epoch": epoch + 1, "epochs": epochs, "complete": False,
                    f"best_val_{sel_key}": round(float(best_f1), 6),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }, indent=2))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"[BERT] Best val {metric_for_best} F1: {best_f1:.4f}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def bert_checkpoint_exists(name, base_dir=BERT_MODEL_DIR):
    """True only for a run that finished.

    `train_meta.json` is written last, after training completes, so a run killed
    mid-training leaves weights behind but is still reported as not-done. Without
    this a crashed 160-minute run would silently masquerade as a finished one.
    """
    d = Path(base_dir) / name
    return (d / "pytorch_model.bin").exists() and (d / "train_meta.json").exists()


def bert_checkpoint_partial(name, base_dir=BERT_MODEL_DIR):
    """Weights from an interrupted run: best-so-far, but training never finished."""
    d = Path(base_dir) / name
    return (d / "pytorch_model.bin").exists() and not (d / "train_meta.json").exists()


# ---------------------------------------------------------------------------
# Result persistence  (metrics + test predictions, no model weights)
#
# A run costs hours; its *result* is a few KB. Keeping them separate means the
# aggregate/analysis cells reload instantly after a kernel restart and never
# need to touch the multi-hundred-MB checkpoints.
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def bert_result_path(name, base_dir=BERT_MODEL_DIR):
    return Path(base_dir) / "_results" / f"{name}.json"


def bert_result_exists(name, base_dir=BERT_MODEL_DIR):
    return bert_result_path(name, base_dir).exists()


def save_bert_result(name, metrics, y_pred, meta=None, base_dir=BERT_MODEL_DIR):
    p = bert_result_path(name, base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name":        name,
        "saved_at":    datetime.now().isoformat(timespec="seconds"),
        "metrics":     {k: v for k, v in metrics.items() if k != "per_article"},
        "per_article": metrics.get("per_article"),
        "meta":        meta or {},
        "y_pred":      np.asarray(y_pred).astype(int).tolist(),
    }
    p.write_text(json.dumps(payload, default=_json_default))
    print(f"[BERT] Saved result -> {p}")
    return p


def load_bert_result(name, base_dir=BERT_MODEL_DIR):
    """Return (metrics_dict, y_pred) for a finished run, without loading weights."""
    d       = json.loads(bert_result_path(name, base_dir).read_text())
    res     = dict(d["metrics"])
    res["per_article"]     = d.get("per_article")
    res["metric_for_best"] = d.get("meta", {}).get("metric_for_best", "micro")
    res["seed"]            = d.get("meta", {}).get("seed")
    res["saved_at"]        = d.get("saved_at")
    return res, np.array(d["y_pred"], dtype=int)


def list_bert_runs(base_dir=BERT_MODEL_DIR):
    """Status of every run on disk: result cached / weights present / interrupted."""
    base  = Path(base_dir)
    names = sorted({p.name for p in base.iterdir() if p.is_dir() and p.name != "_results"}
                   | {p.stem for p in (base / "_results").glob("*.json")}
                   if base.exists() else set())
    print(f"{'run':<24} {'result':<9} {'weights':<12} {'macro':>8} {'micro':>8}")
    print("-" * 65)
    for n in names:
        has_res = bert_result_exists(n, base_dir)
        weights = ("complete"    if bert_checkpoint_exists(n, base_dir)
                   else "INTERRUPTED" if bert_checkpoint_partial(n, base_dir)
                   else "-")
        ma = mi = "-"
        if has_res:
            r, _ = load_bert_result(n, base_dir)
            ma, mi = f"{r['macro_f1']:.4f}", f"{r['micro_f1']:.4f}"
        print(f"{n:<24} {'cached' if has_res else '-':<9} {weights:<12} {ma:>8} {mi:>8}")
    print("-" * 65)
    return names


def save_bert_model(model, tokenizer, name, meta, base_dir=BERT_MODEL_DIR):
    d = Path(base_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "pytorch_model.bin")
    tokenizer.save_pretrained(str(d))
    info = dict(meta)
    info["class"] = type(model).__name__
    (d / "train_meta.json").write_text(json.dumps(info, indent=2))
    print(f"[BERT] Saved -> {d}")
    return d


def load_bert_model(name, base_dir=BERT_MODEL_DIR, device=None):
    """Rebuild architecture from train_meta.json and load the saved weights.

    Returns (model, tokenizer, meta). `meta` carries use_premises / use_hybrid /
    use_hierarchical so evaluation reproduces the same text extraction.
    """
    d      = Path(base_dir) / name
    meta   = json.loads((d / "train_meta.json").read_text())
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(d))
    cfg       = AutoConfig.from_pretrained(meta["model_name"])

    if meta["class"] == "HierarchicalBertClassifier":
        encoder = AutoModel.from_config(cfg)          # architecture only, no download of weights
        model   = HierarchicalBertClassifier(
            encoder, cfg.hidden_size, meta["num_labels"],
            max_segs=meta.get("max_segs", HIER_MAX_SEGS),
            seg_chunk_size=meta.get("seg_chunk_size", 16))
    else:
        cfg.num_labels   = meta["num_labels"]
        cfg.problem_type = "multi_label_classification"
        model = AutoModelForSequenceClassification.from_config(cfg)

    state = torch.load(d / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[BERT] Loaded cached model <- {d}")
    return model, tokenizer, meta


def train_or_load_bert(name, train_cases, val_cases, force=False,
                       base_dir=BERT_MODEL_DIR, model_name=None, **kw):
    """Load `name` from disk if present, otherwise train it and save.

    Mirrors the Stage 1 checkpoint pattern: delete the folder to retrain.
    Returns (model, tokenizer, meta).
    """
    if not force and bert_checkpoint_exists(name, base_dir):
        return load_bert_model(name, base_dir)

    if model_name is None:
        model_name = "nlpaueb/bert-base-uncased-echr"

    d = Path(base_dir) / name
    if bert_checkpoint_partial(name, base_dir):
        print(f"[BERT] Found INTERRUPTED weights in {d} - retraining from scratch "
              f"(they will be overwritten). Move them aside first if you want them.")
    d.mkdir(parents=True, exist_ok=True)

    model, tokenizer = train_bert_classifier(train_cases, val_cases,
                                             model_name=model_name,
                                             checkpoint_dir=d, **kw)
    meta = {
        "model_name":       model_name,
        "num_labels":       config.NUM_LABELS,
        "use_premises":     kw.get("use_premises", True),
        "use_hybrid":       kw.get("use_hybrid", False),
        "use_hierarchical": kw.get("use_hierarchical", True),
        "seg_len":          kw.get("seg_len", HIER_SEG_LEN),
        "max_segs":         kw.get("max_segs", HIER_MAX_SEGS),
        "seg_chunk_size":   kw.get("seg_chunk_size", 16),
        "metric_for_best":  kw.get("metric_for_best", "macro"),
        "lr":               kw.get("lr", 3e-5),
        "batch_size":       kw.get("batch_size", 8),
        "seed":             kw.get("seed", None),
    }
    save_bert_model(model, tokenizer, name, meta, base_dir)
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().cpu().numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def predict_bert(model, tokenizer, cases,
                 use_premises=True, use_hybrid=False,
                 use_hierarchical=True, batch_size=8,
                 seg_len=HIER_SEG_LEN, max_segs=HIER_MAX_SEGS):
    device = next(model.parameters()).device
    texts  = _extract_texts(cases, use_premises, use_hybrid)
    labels = [c["labels_binary"] for c in cases]

    if use_hierarchical and isinstance(model, HierarchicalBertClassifier):
        ds = HierarchicalCaseDataset(texts, labels, tokenizer, seg_len, max_segs)
    else:
        ds = CaseTextDataset(texts, labels, tokenizer)

    dl = DataLoader(ds, batch_size=batch_size)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dl:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
    return np.vstack(all_preds)
