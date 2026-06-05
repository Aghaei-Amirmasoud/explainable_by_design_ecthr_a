import re
import numpy as np
from sentence_transformers import SentenceTransformer
import config

# Reuse sentence splitter from Stage 1 for consistency
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PARA_NUM = re.compile(r"^\d+\.?$")


def _split_sentences(text: str) -> list:
    """Split text into sentences (same logic as Stage 1)."""
    sentences = _SENT_BOUNDARY.split(text)
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and not _PARA_NUM.match(s) and len(s.split()) >= 3:
            filtered.append(s)
    return filtered


class HybridPremiseEmbedder:
    """Embeds full paragraphs with premise-awareness features."""

    def __init__(self, model_name=config.SENTENCE_TRANSFORMER_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_embedding_dimension()

    def _pool_single(self, para_embs, weights, strategy):
        """Pool paragraph embeddings using specified strategy."""
        if strategy == "max":
            return para_embs.max(axis=0)
        elif strategy == "mean":
            return para_embs.mean(axis=0)
        elif strategy == "weighted_mean":
            w = weights / (weights.sum() + 1e-8)
            return (para_embs * w[:, None]).sum(axis=0)

    def _build_paragraph_premise_map(self, case):
        para_map = {}
        for p in case.get("premises", []):
            para_id = p.get("paragraph_id", -1)
            if para_id not in para_map:
                para_map[para_id] = []
            para_map[para_id].append(p)
        return para_map

    def _extract_premise_features(self, paragraph_text, premises_in_para):
        has_premise = float(len(premises_in_para) > 0)
        n_premises = len(premises_in_para)

        # Count sentences in paragraph using same splitter as Stage 1
        n_sentences = max(len(_split_sentences(paragraph_text)), 1)
        premise_density = n_premises / n_sentences

        avg_confidence = (np.mean([p["confidence"] for p in premises_in_para])
                         if premises_in_para else 0.0)

        return np.array([has_premise, n_premises, premise_density, avg_confidence])

    def embed_paragraphs_with_premise_features(self, cases, batch_size=64,
                                               strategy=config.POOLING_STRATEGY,
                                               premise_weight_boost=None):
        if premise_weight_boost is None:
            premise_weight_boost = config.HYBRID_PREMISE_WEIGHT_BOOST

        is_concat = strategy == "concat"
        strategies = ["max", "mean", "weighted_mean"] if is_concat else [strategy]
        base_dim = self.dim * len(strategies)

        # Batch embed all paragraphs across all cases
        all_paragraphs = []
        all_para_features = []
        all_para_indices = []  # which case each paragraph belongs to

        for case_idx, case in enumerate(cases):
            paragraphs = case.get("paragraphs", [])
            para_premise_map = self._build_paragraph_premise_map(case)

            for para_idx, paragraph in enumerate(paragraphs):
                all_paragraphs.append(paragraph)
                all_para_indices.append(case_idx)

                # Extract premise features for this paragraph
                premises_in_para = para_premise_map.get(para_idx, [])
                features = self._extract_premise_features(paragraph, premises_in_para)
                all_para_features.append(features)

        # Embed all paragraphs in batch
        if len(all_paragraphs) == 0:
            # Edge case: no paragraphs at all
            return np.zeros((len(cases), base_dim + 4))

        print(f"[Hybrid] Embedding {len(all_paragraphs)} paragraphs from {len(cases)} cases...")
        para_embeddings = self.model.encode(all_paragraphs, batch_size=batch_size,
                                           show_progress_bar=True, convert_to_numpy=True)

        all_para_features = np.array(all_para_features)  # (n_paragraphs, 4)
        all_para_indices = np.array(all_para_indices)

        # Pool paragraphs per case
        result_embeddings = np.zeros((len(cases), base_dim))
        result_features = np.zeros((len(cases), 4))

        for case_idx in range(len(cases)):
            mask = all_para_indices == case_idx
            if not mask.any():
                continue

            case_para_embs = para_embeddings[mask]
            case_para_feats = all_para_features[mask]

            # Weights for pooling: boost paragraphs with premises
            weights = np.ones(len(case_para_embs))
            if strategy in ["weighted_mean", "concat"]:
                # Boost paragraphs that contain premises
                has_premise_flags = case_para_feats[:, 0]  # first feature is has_premise
                weights = weights + has_premise_flags * (premise_weight_boost - 1)

            # Pool embeddings
            parts = [self._pool_single(case_para_embs, weights, s) for s in strategies]
            result_embeddings[case_idx] = np.concatenate(parts)

            # Aggregate premise features across paragraphs (case-level summary)
            result_features[case_idx, 0] = case_para_feats[:, 0].max()  # has_any_premise
            result_features[case_idx, 1] = case_para_feats[:, 1].sum()  # total_premises
            result_features[case_idx, 2] = case_para_feats[:, 2].mean()  # avg_density
            result_features[case_idx, 3] = case_para_feats[:, 3].mean()  # avg_confidence

        # Concatenate embeddings + case-level premise features
        return np.hstack([result_embeddings, result_features])

    @staticmethod
    def extract_labels(cases):
        """Extract binary labels from cases."""
        return np.array([c["labels_binary"] for c in cases], dtype=int)

    def prepare_split(self, cases, **kwargs):
        """Prepare (X, y) for a dataset split using hybrid approach."""
        X_base = self.embed_paragraphs_with_premise_features(cases, **kwargs)

        # Add additional handcrafted features (same as original embedder for consistency)
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

        X = np.hstack([X_base, handcrafted])
        y = self.extract_labels(cases)

        return X, y
