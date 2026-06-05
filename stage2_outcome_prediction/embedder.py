import numpy as np
from sentence_transformers import SentenceTransformer
import config


class PremiseEmbedder:
    def __init__(self, model_name=config.SENTENCE_TRANSFORMER_MODEL):
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim        = self.model.get_embedding_dimension()

    def embed_cases(self, cases, text_field="premise_text", batch_size=64):
        texts = [c[text_field] for c in cases]
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=True, convert_to_numpy=True)

    def _pool_single(self, case_embs, weights, strategy):
        if strategy == "max":
            return case_embs.max(axis=0)
        elif strategy == "mean":
            return case_embs.mean(axis=0)
        elif strategy == "weighted_mean":
            w = weights / (weights.sum() + 1e-8)
            return (case_embs * w[:, None]).sum(axis=0)

    def embed_premises_pooled(self, cases, batch_size=64,
                              strategy=config.POOLING_STRATEGY):
        is_concat = strategy == "concat"
        strategies = ["max", "mean", "weighted_mean"] if is_concat else [strategy]
        out_dim = self.dim * len(strategies)

        has_premises = [bool(c.get("premises")) for c in cases]

        all_sentences, all_indices, all_weights = [], [], []
        for i, case in enumerate(cases):
            if has_premises[i]:
                for p in case["premises"]:
                    all_sentences.append(p["sentence"])
                    all_indices.append(i)
                    all_weights.append(p["confidence"])

        result = np.zeros((len(cases), out_dim))

        if all_sentences:
            embs    = self.model.encode(all_sentences, batch_size=batch_size,
                                        show_progress_bar=True, convert_to_numpy=True)
            indices = np.array(all_indices)
            weights = np.array(all_weights)
            for i in range(len(cases)):
                if has_premises[i]:
                    mask = indices == i
                    case_embs = embs[mask]
                    w = weights[mask]
                    parts = [self._pool_single(case_embs, w, s) for s in strategies]
                    result[i] = np.concatenate(parts)

        fallback_idxs = [i for i, h in enumerate(has_premises) if not h]
        if fallback_idxs:
            fallback_texts = [" ".join(cases[i]["paragraphs"]) for i in fallback_idxs]
            fallback_embs  = self.model.encode(fallback_texts, batch_size=batch_size,
                                               show_progress_bar=False, convert_to_numpy=True)
            for i, emb in zip(fallback_idxs, fallback_embs):
                if is_concat:
                    result[i] = np.tile(emb, len(strategies))
                else:
                    result[i] = emb

        return result

    @staticmethod
    def extract_labels(cases):
        return np.array([c["labels_binary"] for c in cases], dtype=int)

    def prepare_split(self, cases):
        X = self.embed_premises_pooled(cases)

        handcrafted = np.zeros((len(cases), 5))
        for i, c in enumerate(cases):
            premises = c.get("premises", [])
            n_prem = len(premises)
            confs = [p["confidence"] for p in premises] if premises else [0.0]
            text_len = sum(len(para) for para in c.get("paragraphs", []))
            n_sents = max(text_len // 80, 1)

            handcrafted[i, 0] = float(c.get("used_fallback", False))
            handcrafted[i, 1] = n_prem
            handcrafted[i, 2] = np.mean(confs)
            handcrafted[i, 3] = np.max(confs)
            handcrafted[i, 4] = n_prem / n_sents

        return np.hstack([X, handcrafted]), self.extract_labels(cases)


def save_embeddings(X, path=str(config.STAGE2_CACHE)):
    np.save(path, X)


def load_embeddings(path=str(config.STAGE2_CACHE)):
    return np.load(path)
