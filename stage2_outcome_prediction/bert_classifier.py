import numpy as np
from typing import Optional, NamedTuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
import config


class ModelOutput(NamedTuple):
    loss: Optional[torch.Tensor]
    logits: torch.Tensor

# LexGLUE hierarchical config: 64 segments × 128 tokens
# One PARAGRAPH = one SEGMENT (matches lex-glue experiments/ecthr.py)
HIER_SEG_LEN  = 128
HIER_MAX_SEGS = 64


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_segments(cases, use_premises, use_hybrid=False):
    """Per-case list of segments. LexGLUE treats each paragraph as one segment.

    fulltext : paragraphs as-is
    premises : premise sentences grouped by their source paragraph
    hybrid   : paragraphs as-is, premise-containing ones wrapped in [PREMISE] markers
    """
    all_segs = []
    for c in cases:
        paragraphs = c.get("paragraphs", []) or []
        premises   = c.get("premises", []) or []

        if use_hybrid:
            prem_paras = {p.get("paragraph_id", -1) for p in premises}
            segs = [f"[PREMISE] {para} [/PREMISE]" if i in prem_paras else para
                    for i, para in enumerate(paragraphs)]
        elif use_premises and premises:
            by_para = {}
            for p in premises:
                by_para.setdefault(p.get("paragraph_id", -1), []).append(p["sentence"])
            segs = [" ".join(by_para[k]) for k in sorted(by_para)]
        else:
            segs = list(paragraphs)

        if not segs:  # never emit an empty case
            segs = [" ".join(paragraphs)] if paragraphs else [""]
        all_segs.append(segs)
    return all_segs


def _extract_texts(cases, use_premises, use_hybrid=False):
    """Flat (non-hierarchical) mode: segments joined into one string."""
    return [" ".join(segs) for segs in
            _extract_segments(cases, use_premises, use_hybrid)]


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
    """One paragraph per segment, up to max_segs segments of seg_len tokens.

    Matches lex-glue experiments/ecthr.py:
        tokenizer(case[:max_segments], padding='max_length',
                  max_length=max_seg_length, truncation=True)
    Paragraphs past max_segs are dropped; paragraphs longer than seg_len are
    truncated (not split across segments).
    """

    def __init__(self, seg_lists, labels, tokenizer,
                 seg_len=HIER_SEG_LEN, max_segs=HIER_MAX_SEGS):
        self.labels   = labels
        self.seg_len  = seg_len
        self.max_segs = max_segs
        pad_id        = tokenizer.pad_token_id

        self.all_input_ids      = []
        self.all_attention_masks = []

        for segs in tqdm(seg_lists, desc="Tokenising (paragraph segments)", leave=False):
            enc = tokenizer(list(segs[:max_segs]), padding="max_length",
                            max_length=seg_len, truncation=True)
            seg_ids   = enc["input_ids"]
            seg_masks = enc["attention_mask"]

            # Pad document to max_segs with empty segments
            n_pad     = max_segs - len(seg_ids)
            seg_ids   = seg_ids   + [[pad_id] * seg_len] * n_pad
            seg_masks = seg_masks + [[0] * seg_len]      * n_pad

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
                 max_segs=HIER_MAX_SEGS, seg_chunk_size=16):
        super().__init__()
        self.encoder        = encoder
        self.seg_chunk_size = seg_chunk_size
        self.pos_encoder    = SinusoidalPositionalEncoding(max_segs, hidden_dim)

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

        cls_chunks = []
        for start in range(0, B * S, self.seg_chunk_size):
            chunk_ids   = flat_ids[start:start + self.seg_chunk_size]
            chunk_masks = flat_masks[start:start + self.seg_chunk_size]
            out = self.encoder(input_ids=chunk_ids, attention_mask=chunk_masks)
            cls_chunks.append(out.last_hidden_state[:, 0, :])
        cls_emb = torch.cat(cls_chunks, dim=0).view(B, S, -1)

        seg_valid    = flat_masks.view(B, S, L).any(dim=-1)
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
                           fp16=True,
                           seg_len=HIER_SEG_LEN,
                           max_segs=HIER_MAX_SEGS,
                           physical_batch_size=1,
                           seg_chunk_size=16):
    """Train LegalBERT classifier (single GPU).

    use_hierarchical=True  -> 64x128-token hierarchical encoding (LexGLUE protocol)
    use_hierarchical=False -> single 512-token truncated sequence (legacy)
    """
    if model_name is None:
        model_name = "nlpaueb/bert-base-uncased-echr"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_labels = [c["labels_binary"] for c in train_cases]
    val_labels   = [c["labels_binary"] for c in val_cases]

    accum_steps = max(1, batch_size // physical_batch_size)

    if use_hierarchical:
        print(f"[BERT] Hierarchical mode: 1 paragraph = 1 segment, "
              f"max {max_segs} segments x {seg_len} tokens")
        print(f"[BERT] physical_batch={physical_batch_size}, accum_steps={accum_steps} "
              f"-> effective_batch={physical_batch_size * accum_steps}")
        train_segs = _extract_segments(train_cases, use_premises, use_hybrid)
        val_segs   = _extract_segments(val_cases,   use_premises, use_hybrid)
        train_ds = HierarchicalCaseDataset(train_segs, train_labels, tokenizer, seg_len, max_segs)
        val_ds   = HierarchicalCaseDataset(val_segs,   val_labels,   tokenizer, seg_len, max_segs)
        encoder  = AutoModel.from_pretrained(model_name).to(device)
        model    = HierarchicalBertClassifier(encoder, encoder.config.hidden_size,
                                              config.NUM_LABELS,
                                              seg_chunk_size=seg_chunk_size).to(device)
    else:
        print("[BERT] Flat mode: single 512-token sequence (truncated)")
        train_texts = _extract_texts(train_cases, use_premises, use_hybrid)
        val_texts   = _extract_texts(val_cases,   use_premises, use_hybrid)
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
    print(f"[BERT] lr={lr} | wd={weight_decay} | fp16={scaler is not None} | "
          f"warmup={warmup_steps}/{total_steps} | patience={patience}")

    best_f1, best_state, no_improve = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")):
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

        val_metrics = _evaluate(model, val_dl, device)
        print(f"  loss={total_loss/len(train_dl):.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}")

        if val_metrics["micro_f1"] > best_f1:
            best_f1    = val_metrics["micro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"[BERT] Best val micro F1: {best_f1:.4f}")
    return model, tokenizer


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
    labels = [c["labels_binary"] for c in cases]

    if use_hierarchical and isinstance(model, HierarchicalBertClassifier):
        segs = _extract_segments(cases, use_premises, use_hybrid)
        ds   = HierarchicalCaseDataset(segs, labels, tokenizer, seg_len, max_segs)
    else:
        texts = _extract_texts(cases, use_premises, use_hybrid)
        ds    = CaseTextDataset(texts, labels, tokenizer)

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
