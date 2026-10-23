
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

customers_path = DATA_DIR / "customers.csv"
articles_path = DATA_DIR / "articles.csv"

customers = pd.read_csv(customers_path, usecols=["customer_id"])
articles = pd.read_csv(
    articles_path,
    usecols=["article_id", "product_type_name", "product_group_name"]
)

print("CUSTOMER ANALYSIS")
print("Total customer records:", len(customers))
print("Unique customer IDs:", customers["customer_id"].nunique())
print("Duplicate customer IDs:", customers["customer_id"].duplicated().sum())

print("\nPRODUCT ANALYSIS")
print("Total article records:", len(articles))
print("Unique article IDs:", articles["article_id"].nunique())
print("Duplicate article IDs:", articles["article_id"].duplicated().sum())

print("\nTop 10 product groups:")
print(articles["product_group_name"].value_counts().head(10))

report = [
    "H&M Cold-Start Preparation Report",
    f"Customer records: {len(customers)}",
    f"Unique customer IDs: {customers['customer_id'].nunique()}",
    f"Duplicate customer IDs: {customers['customer_id'].duplicated().sum()}",
    f"Article records: {len(articles)}",
    f"Unique article IDs: {articles['article_id'].nunique()}",
    f"Duplicate article IDs: {articles['article_id'].duplicated().sum()}",
    "",
    "Note: Interaction-based cold-start counts require transactions_train.csv.",
    "",
    "Top product groups:",
    articles["product_group_name"].value_counts().head(10).to_string()
]

report_path = REPORT_DIR / "cold_start_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print(f"\nReport saved to: {report_path}")
print("Analysis completed!")
