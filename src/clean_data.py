
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["customers.csv", "articles.csv"]

for filename in FILES:
    input_path = RAW_DIR / filename
    output_path = PROCESSED_DIR / filename

    print(f"\nCleaning {filename}...")

    df = pd.read_csv(input_path, dtype=str)

    original_rows = len(df)
    original_duplicates = int(df.duplicated().sum())

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove records without their primary ID
    id_column = "customer_id" if filename == "customers.csv" else "article_id"
    missing_ids = int(df[id_column].isna().sum() + (df[id_column].str.strip() == "").sum())
    df = df[df[id_column].notna() & (df[id_column].str.strip() != "")]

    # Standardize empty strings as missing values
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    df.to_csv(output_path, index=False)

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicate rows found: {original_duplicates:,}")
    print(f"Rows missing ID: {missing_ids:,}")
    print(f"Cleaned rows: {len(df):,}")
    print(f"Saved to: {output_path}")

print("\nCleaning completed!")
