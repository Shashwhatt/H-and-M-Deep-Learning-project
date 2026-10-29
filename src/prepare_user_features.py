
import pandas as pd

from src.user_feature_vocabularies import (
    USER_CATEGORICAL_FEATURES,
    UNKNOWN_TOKEN,
    MISSING_TOKEN,
    build_vocabulary,
)

REQUIRED_COLUMNS = [
    "customer_id",
    "club_member_status",
    "fashion_news_frequency",
    "age",
]


def prepare_user_features(customers: pd.DataFrame):
    """Validate and encode customer features for model input."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in customers.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required customer columns are missing: {missing_columns}"
        )

    result = customers[REQUIRED_COLUMNS].copy()

    # Preserve customer IDs as strings and reject invalid or duplicate IDs.
    result["customer_id"] = (
        result["customer_id"].astype("string").str.strip()
    )

    invalid_ids = (
        result["customer_id"].isna()
        | result["customer_id"].eq("")
    )

    if invalid_ids.any():
        raise ValueError("Customer IDs cannot be missing or blank.")

    if result["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs were found.")

    vocabularies = {}

    # Build deterministic mappings for categorical customer features.
    for feature in USER_CATEGORICAL_FEATURES:
        values = result[feature].astype("string").str.strip()
        values = values.mask(values.eq(""))

        vocabulary = build_vocabulary(values)
        vocabularies[feature] = vocabulary

        values_for_encoding = values.fillna(MISSING_TOKEN)

        result[f"{feature}_id"] = (
            values_for_encoding
            .map(vocabulary)
            .fillna(vocabulary[UNKNOWN_TOKEN])
            .astype("int64")
        )

    # Convert age to numeric and identify missing/invalid values.
    age = pd.to_numeric(result["age"], errors="coerce")
    result["age_missing"] = age.isna().astype("int64")

    median_age = age.median()
    if pd.isna(median_age):
        median_age = 0.0

    result["age"] = age.fillna(median_age).astype("float64")

    # Return only model-ready features and the mappings used to encode them.
    model_features = result[
        [
            "customer_id",
            "age",
            "age_missing",
            "club_member_status_id",
            "fashion_news_frequency_id",
        ]
    ].copy()

    return model_features, vocabularies
