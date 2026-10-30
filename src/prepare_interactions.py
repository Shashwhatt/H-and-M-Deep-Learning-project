
import pandas as pd
from pathlib import Path

# Input and output paths
input_file = Path("data/raw/transactions_train.csv")
output_dir = Path("data/processed")

train_file = output_dir / "train_interactions.csv"
validation_file = output_dir / "validation_interactions.csv"

# Date-based split
split_date = pd.Timestamp("2020-09-15")

if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

output_dir.mkdir(parents=True, exist_ok=True)

# Remove old outputs to avoid appending duplicate data
for file in [train_file, validation_file]:
    if file.exists():
        file.unlink()

train_count = 0
validation_count = 0

print("Preparing transaction interactions...")

for chunk in pd.read_csv(
    input_file,
    usecols=["t_dat", "customer_id", "article_id"],
    chunksize=500_000
):
    chunk["t_dat"] = pd.to_datetime(
        chunk["t_dat"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["t_dat", "customer_id", "article_id"]
    )

    train_data = chunk[chunk["t_dat"] <= split_date]
    validation_data = chunk[chunk["t_dat"] > split_date]

    columns = ["customer_id", "article_id"]

    if not train_data.empty:
        train_data[columns].to_csv(
            train_file,
            mode="a",
            header=not train_file.exists(),
            index=False
        )
        train_count += len(train_data)

    if not validation_data.empty:
        validation_data[columns].to_csv(
            validation_file,
            mode="a",
            header=not validation_file.exists(),
            index=False
        )
        validation_count += len(validation_data)

    print(
        f"Processed training: {train_count:,} | "
        f"validation: {validation_count:,}"
    )

print("\n========== PREPARATION COMPLETE ==========")
print("Training interactions:", f"{train_count:,}")
print("Validation interactions:", f"{validation_count:,}")
print("Training file:", train_file)
print("Validation file:", validation_file)
print("==========================================")
