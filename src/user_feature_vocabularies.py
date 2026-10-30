import json
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customers.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "user_vocabularies"

# Customer categorical features
USER_CATEGORICAL_FEATURES = [
    "club_member_status",
    "fashion_news_frequency",
]

UNKNOWN_TOKEN = "[UNK]"
MISSING_TOKEN = "[MISSING]"


def build_vocabulary(series):
    """Create a deterministic category-to-ID mapping."""
    values = series.astype("string").str.strip()
    values = values.mask(values.eq(""))
    categories = sorted(values.dropna().unique().tolist())

    vocabulary = {UNKNOWN_TOKEN: 0}

    if values.isna().any():
        vocabulary[MISSING_TOKEN] = 1

    for category in categories:
        if category not in vocabulary:
            vocabulary[category] = len(vocabulary)

    return vocabulary


def save_vocabulary(vocabulary, feature, output_dir=OUTPUT_DIR):
    """Save a categorical vocabulary as a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{feature}_vocabulary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file, indent=2, ensure_ascii=False)

    return output_path


def load_vocabulary(feature, output_dir=OUTPUT_DIR):
    """Load and validate a saved categorical vocabulary."""
    output_dir = Path(output_dir)
    input_path = output_dir / f"{feature}_vocabulary.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise ValueError("Vocabulary must be a JSON object.")

    if vocabulary.get(UNKNOWN_TOKEN) != 0:
        raise ValueError("[UNK] must have ID 0.")

    if len(vocabulary.values()) != len(set(vocabulary.values())):
        raise ValueError("Vocabulary IDs must be unique.")

    if not all(isinstance(value, int) for value in vocabulary.values()):
        raise ValueError("Vocabulary IDs must be integers.")

    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Vocabulary IDs must be contiguous from 0.")

    return vocabulary


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset not found: {DATA_PATH}\n"
            "Place the processed customer CSV at data/customers.csv."
        )

    customers = pd.read_csv(DATA_PATH)

    missing_columns = [
        feature
        for feature in USER_CATEGORICAL_FEATURES
        if feature not in customers.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required customer columns are missing: {missing_columns}"
        )

    for feature in USER_CATEGORICAL_FEATURES:
        vocabulary = build_vocabulary(customers[feature])
        output_path = save_vocabulary(vocabulary, feature)

        print(f"Created vocabulary for {feature}: {len(vocabulary)} entries")
        print(f"Saved to: {output_path}")
if __name__ == "__main__":
    main()