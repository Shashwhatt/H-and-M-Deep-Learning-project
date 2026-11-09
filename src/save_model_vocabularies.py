
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "two_tower"

FILES = [
    DATA_DIR / "train_interactions.csv",
    DATA_DIR / "validation_interactions.csv",
]

CHUNK_SIZE = 500_000


def collect_vocabularies():
    customer_ids = set()
    article_ids = set()

    for file_path in FILES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required interaction file not found: {file_path}"
            )

        print(f"\nReading: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            dtype={
                "customer_id": str,
                "article_id": str,
            },
            chunksize=CHUNK_SIZE,
        ):
            chunk = chunk.dropna(
                subset=["customer_id", "article_id"]
            )

            customer_ids.update(chunk["customer_id"].unique())
            article_ids.update(chunk["article_id"].unique())

        print("Unique customers collected:", f"{len(customer_ids):,}")
        print("Unique articles collected:", f"{len(article_ids):,}")

    return sorted(customer_ids), sorted(article_ids)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building customer and article vocabularies...")

    customer_vocabulary, article_vocabulary = collect_vocabularies()

    customer_path = OUTPUT_DIR / "customer_id_vocabulary.json"
    article_path = OUTPUT_DIR / "article_id_vocabulary.json"

    with open(customer_path, "w", encoding="utf-8") as file:
        json.dump(customer_vocabulary, file)

    with open(article_path, "w", encoding="utf-8") as file:
        json.dump(article_vocabulary, file)

    print("\n========== VOCABULARIES SAVED ==========")
    print("Customers:", f"{len(customer_vocabulary):,}")
    print("Articles:", f"{len(article_vocabulary):,}")
    print("Customer vocabulary:", customer_path)
    print("Article vocabulary:", article_path)
    print("========================================")


if __name__ == "__main__":
    main()
