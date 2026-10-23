
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES = ["customers.csv", "articles.csv"]
CHUNK_SIZE = 100_000

for filename in FILES:
    file_path = DATA_DIR / filename

    print(f"\n{'=' * 50}")
    print(f"Dataset: {filename}")

    if not file_path.exists():
        print("File not found!")
        continue

    print(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    row_count = 0
    missing_counts = {}

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        columns = next(reader)

        print("Columns:", columns)

        for row in reader:
            row_count += 1

            for index, value in enumerate(row):
                if value.strip() == "":
                    column = columns[index]
                    missing_counts[column] = missing_counts.get(column, 0) + 1

            if row_count % CHUNK_SIZE == 0:
                print(f"Rows inspected: {row_count:,}", flush=True)

    print(f"Total rows: {row_count:,}")
    print("Missing values:", missing_counts)
