import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from tqdm import tqdm
import config


class CaseTextDataset(Dataset):
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


def _extract_texts(cases, use_premises):
    texts = []
    for c in cases:
        if use_premises and c.get("premises"):
            texts.append(" ".join(p["sentence"] for p in c["premises"]))
        else:
            texts.append(" ".join(c["paragraphs"]))
    return texts


def train_bert_classifier(train_cases, val_cases, use_premises=True,
                          model_name=None, epochs=3, batch_size=16, lr=2e-5):
    if model_name is None:
        model_name = "nlpaueb/bert-base-uncased-echr"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    train_texts  = _extract_texts(train_cases, use_premises)
    val_texts    = _extract_texts(val_cases, use_premises)
    train_labels = [c["labels_binary"] for c in train_cases]
    val_labels   = [c["labels_binary"] for c in val_cases]

    train_ds = CaseTextDataset(train_texts, train_labels, tokenizer)
    val_ds   = CaseTextDataset(val_texts, val_labels, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()

        val_metrics = _evaluate(model, val_dl, device)
        print(f"  loss={total_loss/len(train_dl):.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_micro_f1={val_metrics['micro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, tokenizer


def _evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].int().cpu().numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


def predict_bert(model, tokenizer, cases, use_premises=True, batch_size=16):
    device = next(model.parameters()).device
    texts  = _extract_texts(cases, use_premises)
    labels = [c["labels_binary"] for c in cases]
    ds = CaseTextDataset(texts, labels, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
    return np.vstack(all_preds)
