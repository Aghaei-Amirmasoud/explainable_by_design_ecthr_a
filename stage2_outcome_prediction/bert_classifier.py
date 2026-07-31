import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
import config

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

        for text in tqdm(texts, desc="Tokenising (hierarchical)", leave=False):
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
# Hierarchical model wrapper
# ---------------------------------------------------------------------------

class HierarchicalBertClassifier(nn.Module):
    """BERT segment encoder + mean-pool over segments + linear head.

    Forward path:
      input_ids / attention_mask: (B, S, L)  S=max_segs, L=seg_len
      → reshape to (B*S, L), run BERT encoder
      → take [CLS] token → (B*S, H)
      → reshape to (B, S, H)
      → mean-pool over non-empty segments → (B, H)
      → linear → (B, num_labels)
    """

    def __init__(self, encoder, hidden_dim, num_labels):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        B, S, L = input_ids.shape
        flat_ids   = input_ids.view(B * S, L)
        flat_masks = attention_mask.view(B * S, L)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_masks)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B*S, H)
        cls_emb = cls_emb.view(B, S, -1)              # (B, S, H)

        # Mask empty segments (all attention zeros)
        seg_valid = flat_masks.view(B, S, L).any(dim=-1).float()  # (B, S)
        seg_valid_exp = seg_valid.unsqueeze(-1)                    # (B, S, 1)
        pooled = (cls_emb * seg_valid_exp).sum(dim=1) / seg_valid_exp.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        return type("Out", (), {"loss": loss, "logits": logits})()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bert_classifier(train_cases, val_cases,
                           use_premises=True, use_hybrid=False,
                           use_hierarchical=True,
                           model_name=None,
                           epochs=20,          # LexGLUE: up to 20 epochs
                           batch_size=8,        # LexGLUE: batch_size=8
                           lr=1e-5,
                           warmup_ratio=0.1,    # LexGLUE: warmup
                           weight_decay=0.06,   # LexGLUE: weight decay
                           patience=3,          # early stopping
                           seg_len=HIER_SEG_LEN,
                           max_segs=HIER_MAX_SEGS):
    """Train LegalBERT classifier.

    use_hierarchical=True  → 64×128-token hierarchical encoding (LexGLUE protocol)
    use_hierarchical=False → single 512-token truncated sequence (legacy)
    """
    if model_name is None:
        model_name = "nlpaueb/bert-base-uncased-echr"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts  = _extract_texts(train_cases, use_premises, use_hybrid)
    val_texts    = _extract_texts(val_cases,   use_premises, use_hybrid)
    train_labels = [c["labels_binary"] for c in train_cases]
    val_labels   = [c["labels_binary"] for c in val_cases]

    if use_hierarchical:
        print(f"[BERT] Hierarchical mode: {max_segs} segments × {seg_len} tokens = "
              f"{max_segs * seg_len:,} token capacity")
        train_ds = HierarchicalCaseDataset(train_texts, train_labels, tokenizer, seg_len, max_segs)
        val_ds   = HierarchicalCaseDataset(val_texts,   val_labels,   tokenizer, seg_len, max_segs)
        encoder  = AutoModel.from_pretrained(model_name).to(device)
        model    = HierarchicalBertClassifier(encoder, encoder.config.hidden_size,
                                              config.NUM_LABELS).to(device)
    else:
        print("[BERT] Flat mode: single 512-token sequence (truncated)")
        train_ds = CaseTextDataset(train_texts, train_labels, tokenizer)
        val_ds   = CaseTextDataset(val_texts,   val_labels,   tokenizer)
        model    = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=config.NUM_LABELS,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        ).to(device)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps   = len(train_dl) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[BERT] Warmup: {warmup_steps}/{total_steps} steps | "
          f"lr={lr} | wd={weight_decay} | patience={patience}")

    best_f1, best_state, no_improve = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()

        val_metrics = _evaluate(model, val_dl, device)
        print(f"  loss={total_loss/len(train_dl):.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1    = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"[BERT] Best val macro F1: {best_f1:.4f}")
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
