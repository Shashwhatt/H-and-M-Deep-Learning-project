import pandas as pd

from .item_feature_config import ITEM_CATEGORICAL_FEATURES, ITEM_ID_COLUMN


def load_articles(file_path):
    df = pd.read_csv(file_path)

    required_columns = [ITEM_ID_COLUMN] + ITEM_CATEGORICAL_FEATURES
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    return df


if __name__ == "__main__":
    file_path = r"C:\Users\TULASI SAI\Downloads\articles.csv\articles.csv"

    articles = load_articles(file_path)
    print("Dataset loaded successfully.")
    print("Rows:", len(articles))
    print("Required columns are present.")