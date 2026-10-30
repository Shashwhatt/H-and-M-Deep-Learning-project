
import pandas as pd

files = {
    "customers": "data/processed/customers.csv",
    "articles": "data/processed/articles.csv",
}

for name, path in files.items():
    print(f"\n--- {name.upper()} ---")

    total_rows = 0
    missing_counts = None

    for chunk in pd.read_csv(path, chunksize=50000):
        total_rows += len(chunk)

        current_missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = current_missing
        else:
            missing_counts += current_missing

    print("Total rows:", total_rows)
    print("Missing values by column:")
    print(missing_counts[missing_counts > 0].to_string()
          if missing_counts is not None and
          (missing_counts > 0).any()
          else "No missing values")
