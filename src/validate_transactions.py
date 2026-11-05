
import pandas as pd
from pathlib import Path

# Dataset path
file_path = Path("data/raw/transactions_train.csv")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

print("Checking H&M transaction dataset...\n")

total_rows = 0
unique_customers = set()
unique_articles = set()

missing_counts = {
    "t_dat": 0,
    "customer_id": 0,
    "article_id": 0
}

min_date = None
max_date = None

# Read the large CSV in chunks to reduce memory usage
for chunk in pd.read_csv(
    file_path,
    usecols=["t_dat", "customer_id", "article_id"],
    chunksize=500_000
):
    total_rows += len(chunk)

    unique_customers.update(
        chunk["customer_id"].dropna().unique()
    )

    unique_articles.update(
        chunk["article_id"].dropna().unique()
    )

    for column in missing_counts:
        missing_counts[column] += chunk[column].isna().sum()

    dates = pd.to_datetime(
        chunk["t_dat"],
        errors="coerce"
    ).dropna()

    if not dates.empty:
        chunk_min = dates.min()
        chunk_max = dates.max()

        min_date = (
            chunk_min if min_date is None
            else min(min_date, chunk_min)
        )

        max_date = (
            chunk_max if max_date is None
            else max(max_date, chunk_max)
        )

# Display validation results
print("\n========== DATASET VALIDATION ==========")

print("Total transactions:", total_rows)
print("Unique customers:", len(unique_customers))
print("Unique articles:", len(unique_articles))

print("\nMissing values:")
for column, count in missing_counts.items():
    print(f"{column}: {count}")

print("\nDate range:", min_date, "to", max_date)

print("\n========================================")
print("Transaction dataset validation completed!")
