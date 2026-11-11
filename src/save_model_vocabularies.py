
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "serving_vocabularies"

VOCABULARIES = {
    "customer_id": {
        "file": DATA_DIR / "customers.csv",
        "expected_count": 1_371_980,
    },
    "article_id": {
        "file": DATA_DIR / "articles.csv",
        "expected_count": 105_542,
    },
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column, config in VOCABULARIES.items():
        print(f"\nLoading {column} from {config['file'].name}...")

        df = pd.read_csv(
            config["file"],
            usecols=[column],
            dtype={column: str},
        )

        # Preserve the exact original training order.
        ids = df[column].dropna().unique().tolist()

        print(f"Unique IDs found: {len(ids):,}")

        if len(ids) != config["expected_count"]:
            raise ValueError(
                f"{column} count mismatch! "
                f"Expected {config['expected_count']:,}, "
                f"found {len(ids):,}. "
                "Vocabulary was not saved."
            )

        output_path = OUTPUT_DIR / f"{column}_vocabulary.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(ids, file)

        print(f"Saved: {output_path}")

    print("\nAll original model vocabularies saved successfully!")


if __name__ == "__main__":
    main()
