import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PARA_NUM      = re.compile(r"^\d+\.?$")


def _split_sentences(text: str) -> list:
    """Split text into sentences, filtering out paragraph numbers and short fragments."""
    sentences = _SENT_BOUNDARY.split(text)
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and not _PARA_NUM.match(s) and len(s.split()) >= 3:
            filtered.append(s)
    return filtered


class LegalBERTArgumentExtractor:
    ID2LABEL = {0: "NON_PREMISE", 1: "PREMISE"}
    LABEL2ID = {"NON_PREMISE": 0, "PREMISE": 1}

    def __init__(self, model_name=config.LEGALBERT_MODEL, load_model=True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_model:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                id2label=self.ID2LABEL,
                label2id=self.LABEL2ID,
            ).to(self.device)
            self.model.eval()
        else:
            self.tokenizer = None
            self.model = None

    def predict_sentence(self, sentence: str):
        """Predict if a sentence is a premise. Returns (is_premise, confidence_score)."""
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=config.S1_MAX_SEQ_LEN
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        confidence = probs[1].item()
        return confidence >= config.PREMISE_THRESHOLD, confidence

    def extract_premises(self, paragraphs: list) -> list:
        """Extract premises from paragraphs using fixed or dynamic threshold."""
        premises = []

        for para_idx, paragraph in enumerate(paragraphs):
            sentences = _split_sentences(paragraph)
            if not sentences:
                continue

            # Score all sentences
            scored = []
            for sent_idx, sentence in enumerate(sentences):
                _, confidence = self.predict_sentence(sentence)
                scored.append((sent_idx, sentence, confidence))

            # Select based on threshold strategy
            if config.DYNAMIC_TOPK:
                selected = self._select_dynamic_topk(scored)
            else:
                selected = self._select_fixed_threshold(scored)

            # Add to results
            for sent_idx, sentence, confidence in selected:
                premises.append({
                    "sentence":     sentence,
                    "paragraph_id": para_idx,
                    "sentence_id":  sent_idx,
                    "confidence":   round(confidence, 4),
                })

        return premises

    def _select_fixed_threshold(self, scored):
        """Select sentences above fixed threshold."""
        return [(i, s, p) for i, s, p in scored if p >= config.PREMISE_THRESHOLD]

    def _select_dynamic_topk(self, scored):
        """Select sentences using adaptive threshold per paragraph."""
        # Filter by hard floor
        viable = [(i, s, p) for i, s, p in scored if p >= config.PREMISE_FLOOR]
        if not viable:
            return []

        # Calculate adaptive threshold
        viable_probs = np.array([v[2] for v in viable])
        if len(viable) > 1 and viable_probs.std() > 1e-6:
            threshold = viable_probs.mean() + config.DYNAMIC_TOPK_ALPHA * viable_probs.std()
        else:
            threshold = config.PREMISE_THRESHOLD

        selected = [(i, s, p) for i, s, p in viable if p >= threshold]

        # Enforce min/max bounds
        if len(selected) < config.DYNAMIC_TOPK_MIN:
            by_score = sorted(viable, key=lambda x: x[2], reverse=True)
            selected = by_score[:config.DYNAMIC_TOPK_MIN]
        elif len(selected) > config.DYNAMIC_TOPK_MAX:
            selected = sorted(selected, key=lambda x: x[2], reverse=True)[:config.DYNAMIC_TOPK_MAX]

        return selected


if __name__ == "__main__":
    sample = [
        "The applicant was born in 1975 and lives in Ankara.",
        "THE LAW\nThe applicant alleged ill-treatment in violation of Article 3. "
        "The Court finds that the state failed to conduct an effective investigation. "
        "It follows that there has been a violation of Article 13.",
        "The case was communicated to the Government on 12 March 2008.",
    ]
    extractor = LegalBERTArgumentExtractor()
    results = extractor.extract_premises(sample)
    if results:
        for p in results:
            print(f"  [para {p['paragraph_id']}] (conf={p['confidence']:.3f}) {p['sentence'][:120]}")
    else:
        print("No premises found — run finetune_legalbert.py first to train the classifier.")
