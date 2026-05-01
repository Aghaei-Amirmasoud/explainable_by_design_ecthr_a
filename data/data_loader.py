from datasets import load_dataset
import config

ARTICLE_NAMES = [
    "Article 2",  "Article 3",  "Article 5",  "Article 6",
    "Article 8",  "Article 9",  "Article 10", "Article 11",
    "Article 14", "P1-1",
]

assert len(ARTICLE_NAMES) == config.NUM_LABELS


def label_indices_to_binary(label_indices):
    vec = [0] * config.NUM_LABELS
    for idx in label_indices:
        if 0 <= idx < config.NUM_LABELS:
            vec[idx] = 1
    return vec


def binary_to_label_names(binary_vec):
    return [ARTICLE_NAMES[i] for i, v in enumerate(binary_vec) if v == 1]


def get_dataset(max_train=config.MAX_TRAIN_SAMPLES,
                max_val=config.MAX_VAL_SAMPLES,
                max_test=config.MAX_TEST_SAMPLES):
    """Load ECtHR dataset with binary labels and paragraph fields."""
    ds = load_dataset(config.LEXGLUE_DATASET, config.LEXGLUE_CONFIG, trust_remote_code=True)

    # Limit dataset splits
    limits = {"train": max_train, "validation": max_val, "test": max_test}
    for split, limit in limits.items():
        if split in ds and limit is not None:
            ds[split] = ds[split].select(range(min(limit, len(ds[split]))))

    # Add binary labels and paragraph fields in one pass
    def _preprocess(batch):
        batch["labels_binary"] = [label_indices_to_binary(idx) for idx in batch["labels"]]
        batch["paragraphs"] = batch["text"]
        batch["full_text"] = [" ".join(text) for text in batch["text"]]
        return batch

    return ds.map(_preprocess, batched=True, desc="Preprocessing dataset")


if __name__ == "__main__":
    dataset = get_dataset()
    sample  = dataset["train"][0]
    print("Paragraphs (first 2):", sample["paragraphs"][:2])
    print("Violated articles:",    binary_to_label_names(sample["labels_binary"]))
