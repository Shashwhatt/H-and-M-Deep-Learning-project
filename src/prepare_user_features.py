import pandas as pd

from src.user_feature_vocabularies import (
    OUTPUT_DIR,
    USER_CATEGORICAL_FEATURES,
    UNKNOWN_TOKEN,
    MISSING_TOKEN,
    build_vocabulary,
    load_vocabulary,
    save_vocabulary,
)


REQUIRED_COLUMNS = [
    "customer_id",
    "club_member_status",
    "fashion_news_frequency",
    "age",
]


def prepare_user_features(customers, vocabularies=None):
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

    if vocabularies is None:
        vocabularies = {}

        for feature in USER_CATEGORICAL_FEATURES:
            vocabulary = build_vocabulary(result[feature])
            vocabularies[feature] = vocabulary

    for feature in USER_CATEGORICAL_FEATURES:
        if feature not in vocabularies:
            raise ValueError(
                f"Vocabulary missing for feature: {feature}"
            )

        values = result[feature].astype("string").str.strip()
        values = values.mask(values.eq(""))

        values_for_encoding = values.fillna(MISSING_TOKEN)

        result[f"{feature}_id"] = (
            values_for_encoding
            .map(vocabularies[feature])
            .fillna(vocabularies[feature][UNKNOWN_TOKEN])
            .astype("int64")
        )

    age = pd.to_numeric(result["age"], errors="coerce")

    result["age_missing"] = age.isna().astype("int64")

    median_age = age.median()

    if pd.isna(median_age):
        median_age = 0.0

    result["age"] = age.fillna(median_age).astype("float64")

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


def prepare_training_features(customers):
    """Build and save vocabularies using training customer data."""

    features, vocabularies = prepare_user_features(customers)

    for feature, vocabulary in vocabularies.items():
        save_vocabulary(vocabulary, feature, OUTPUT_DIR)

    return features, vocabularies

def prepare_inference_features(customers):
    """Load saved training vocabularies and encode new customers."""

    vocabularies = {
        feature: load_vocabulary(feature, OUTPUT_DIR)
        for feature in USER_CATEGORICAL_FEATURES
    }

    return prepare_user_features(customers, vocabularies)