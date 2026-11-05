from .item_feature_config import ITEM_ID_COLUMN, ITEM_CATEGORICAL_FEATURES


def validate_dataset(df):
    required_columns = [ITEM_ID_COLUMN] + ITEM_CATEGORICAL_FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[ITEM_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate article IDs found.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("Dataset validation passed.")