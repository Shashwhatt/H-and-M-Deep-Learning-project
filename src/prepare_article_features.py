
"""Prepare article metadata for the H&M recommendation model."""

import pandas as pd


ARTICLE_REQUIRED_COLUMNS = [
    "article_id",
    "product_code",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]

ARTICLE_CATEGORICAL_FEATURES = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]

MISSING_TOKEN = "[MISSING]"


def prepare_article_features(articles):
    """Clean article metadata and return model-ready features."""

    missing_columns = [
        column
        for column in ARTICLE_REQUIRED_COLUMNS
        if column not in articles.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required article columns: {missing_columns}"
        )

    if articles["article_id"].isna().any():
        raise ValueError("article_id cannot contain missing values")

    if articles["article_id"].duplicated().any():
        raise ValueError("article_id must be unique")

    features = articles[
        ["article_id", "product_code"] + ARTICLE_CATEGORICAL_FEATURES
    ].copy()

    features["article_id"] = features["article_id"].astype("int64")

    features["product_code"] = (
        pd.to_numeric(features["product_code"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    for column in ARTICLE_CATEGORICAL_FEATURES:
        features[column] = (
            features[column]
            .fillna(MISSING_TOKEN)
            .astype(str)
            .str.strip()
            .replace("", MISSING_TOKEN)
        )

    return features
