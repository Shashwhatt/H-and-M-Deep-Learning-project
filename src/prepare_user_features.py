
import json
import math

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

AGE_MEDIAN_PATH = OUTPUT_DIR / "age_median.json"


def calculate_age_median(age_values):
    """Calculate the median age using training customer data."""

    age = pd.to_numeric(age_values, errors="coerce")
    median_age = age.median()

    if pd.isna(median_age):
        median_age = 0.0

    return float(median_age)


def prepare_user_features(customers, vocabularies=None, age_median=None):
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
            vocabularies[feature] = build_vocabulary(result[feature])

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

    if age_median is None:
        age_median = calculate_age_median(result["age"])
    else:
        age_median = float(age_median)

        if not math.isfinite(age_median):
            raise ValueError("Age median must be a finite number.")

    age = pd.to_numeric(result["age"], errors="coerce")

    result["age_missing"] = age.isna().astype("int64")
    result["age"] = age.fillna(age_median).astype("float64")

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
    """Build and save training vocabularies and age preprocessing."""

    age_median = calculate_age_median(customers["age"])

    features, vocabularies = prepare_user_features(
        customers,
        age_median=age_median,
    )

    for feature, vocabulary in vocabularies.items():
        save_vocabulary(vocabulary, feature, OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with AGE_MEDIAN_PATH.open("w", encoding="utf-8") as file:
        json.dump({"age_median": age_median}, file, indent=2)

    print(f"Saved training age median: {age_median}")
    print(f"Saved age configuration: {AGE_MEDIAN_PATH}")

    return features, vocabularies


def prepare_inference_features(customers):
    """Load training preprocessing configuration for inference."""

    vocabularies = {
        feature: load_vocabulary(feature, OUTPUT_DIR)
        for feature in USER_CATEGORICAL_FEATURES
    }

    if not AGE_MEDIAN_PATH.exists():
        raise FileNotFoundError(
            f"Training age configuration not found: {AGE_MEDIAN_PATH}. "
            "Run training feature preparation first."
        )

    with AGE_MEDIAN_PATH.open("r", encoding="utf-8") as file:
        age_config = json.load(file)

    if "age_median" not in age_config:
        raise ValueError("Saved age configuration is invalid.")

    return prepare_user_features(
        customers,
        vocabularies=vocabularies,
        age_median=age_config["age_median"],
    )
