
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

    # Reserve ID 0 for categories not seen during vocabulary creation.
    vocabulary = {UNKNOWN_TOKEN: 0}

    # Missing values receive their own ID when present in the data.
    if values.isna().any():
        vocabulary[MISSING_TOKEN] = 1

    for category in categories:
        # Prevent special tokens from being assigned twice.
        if category not in vocabulary:
            vocabulary[category] = len(vocabulary)

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for feature in USER_CATEGORICAL_FEATURES:
        vocabulary = build_vocabulary(customers[feature])
        output_path = OUTPUT_DIR / f"{feature}_vocabulary.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(vocabulary, file, indent=2, ensure_ascii=False)

        print(f"Created vocabulary for {feature}: {len(vocabulary)} entries")
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
