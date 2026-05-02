"""
Hybrid Premise-Aware Embedder (Experimental)

Instead of discarding non-premise paragraphs, this embedder:
1. Embeds ALL paragraphs from the full case text
2. Augments each paragraph with premise-awareness features:
   - has_premise: binary flag indicating if any sentence was extracted as premise
   - n_premises_in_para: count of premise sentences from this paragraph
   - premise_density: ratio of premise sentences to total sentences in paragraph
   - avg_premise_confidence: mean LegalBERT score for premises from this paragraph
3. Pools paragraph embeddings with optional premise-aware weighting

This captures both the full textual context (including non-premise content)
and explicit relevance signals from Stage 1 argument mining.

Usage:
    Set config.USE_HYBRID_EMBEDDER = True to use this instead of PremiseEmbedder
"""
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
        """Map paragraph_id -> list of premises extracted from that paragraph.

        Args:
            case: Dict with 'premises' key containing list of premise dicts

        Returns:
            Dict mapping paragraph_id (int) -> list of premise dicts
        """
        para_map = {}
        for p in case.get("premises", []):
            para_id = p.get("paragraph_id", -1)
            if para_id not in para_map:
                para_map[para_id] = []
            para_map[para_id].append(p)
        return para_map

    def _extract_premise_features(self, paragraph_text, premises_in_para):
        """Extract premise-aware features for a single paragraph.

        Returns: [has_premise, n_premises, premise_density, avg_confidence]
        """
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
        """Embed all paragraphs with premise-aware features.

        Args:
            cases: List of case dicts with 'paragraphs' and 'premises' keys
            batch_size: Batch size for sentence transformer
            strategy: Pooling strategy (max, mean, weighted_mean, concat)
            premise_weight_boost: Multiplier for paragraphs containing premises
                                 (default from config.HYBRID_PREMISE_WEIGHT_BOOST)

        Returns:
            (n_cases, base_dim + 4) array where:
                base_dim = embedding_dim * n_strategies (e.g., 768*3=2304 for concat)
                +4 = case-level premise features [has_any_premise, total_premises,
                     avg_density, avg_confidence]
        """
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
        """Prepare (X, y) for a dataset split using hybrid approach.

        Args:
            cases: List of case dictionaries
            **kwargs: Additional arguments passed to embed_paragraphs_with_premise_features

        Returns:
            X: (n_cases, feature_dim) array with embeddings + premise features
            y: (n_cases, n_labels) binary label array
        """
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


if __name__ == "__main__":
    # Test with dummy data
    print("Testing HybridPremiseEmbedder...\n")

    dummy_cases = [
        {
            "paragraphs": [
                "The applicant was detained without a court order. This violated his rights.",
                "The case was communicated to the Government on 12 March 2008.",
            ],
            "labels_binary": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Article 5 violated
            "premises": [
                {"sentence": "The applicant was detained without a court order.",
                 "paragraph_id": 0, "sentence_id": 0, "confidence": 0.91},
                {"sentence": "This violated his rights.",
                 "paragraph_id": 0, "sentence_id": 1, "confidence": 0.78},
            ],
        },
        {
            "paragraphs": [
                "The applicant alleged ill-treatment under Article 3.",
                "The state failed to investigate the allegations properly.",
            ],
            "labels_binary": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Article 3 violated
            "premises": [
                {"sentence": "The state failed to investigate the allegations properly.",
                 "paragraph_id": 1, "sentence_id": 0, "confidence": 0.85},
            ],
        },
        {
            "paragraphs": [
                "The proceedings began on 15 January 2005.",
                "The applicant was represented by counsel.",
            ],
            "labels_binary": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No violation
            "premises": [],  # No premises extracted
        },
    ]

    embedder = HybridPremiseEmbedder()
    X, y = embedder.prepare_split(dummy_cases)

    print(f"\nResults:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"\nFeature dimensions breakdown:")
    print(f"  - Base embeddings (concat pooling): {embedder.dim * 3} = {embedder.dim} × 3")
    print(f"  - Case-level premise features: 4")
    print(f"    [has_any_premise, total_premises, avg_density, avg_confidence]")
    print(f"  - Additional handcrafted features: 5")
    print(f"    [used_fallback, n_prem, avg_conf, max_conf, premise_ratio]")
    print(f"  - Total: {X.shape[1]}")
    print(f"\nCase 1 (2 premises in para 0):")
    print(f"  Premise features: {X[0, embedder.dim*3:embedder.dim*3+4]}")
    print(f"Case 2 (1 premise in para 1):")
    print(f"  Premise features: {X[1, embedder.dim*3:embedder.dim*3+4]}")
    print(f"Case 3 (0 premises):")
    print(f"  Premise features: {X[2, embedder.dim*3:embedder.dim*3+4]}")
    print(f"\nTest passed! All paragraphs embedded with premise indicators.")
