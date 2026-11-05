
import pandas as pd
import pytest

from src.prepare_article_features import (
    ARTICLE_CATEGORICAL_FEATURES,
    prepare_article_features,
)


def sample_articles():
    return pd.DataFrame({
        "article_id": [1001, 1002],
        "product_code": [101, 102],
        "product_type_name": ["T-shirt", "Jeans"],
        "product_group_name": ["Garment Upper body", "Garment Lower body"],
        "graphical_appearance_name": ["Solid", "Stripe"],
        "colour_group_name": ["Black", "Blue"],
        "perceived_colour_value_name": ["Dark", "Medium"],
        "perceived_colour_master_name": ["Black", "Blue"],
        "department_name": ["Jersey", "Denim"],
        "index_name": ["Ladieswear", "Menswear"],
        "index_group_name": ["Ladieswear", "Menswear"],
        "section_name": ["Basics", "Casual"],
        "garment_group_name": ["Jersey Basic", "Trousers"],
    })


def test_prepare_article_features_returns_expected_columns():
    result = prepare_article_features(sample_articles())

    assert len(result.columns) == 13
    assert len(result) == 2


def test_article_id_is_preserved():
    result = prepare_article_features(sample_articles())

    assert result["article_id"].tolist() == [1001, 1002]


def test_missing_categories_are_handled():
    articles = sample_articles()
    articles.loc[0, "colour_group_name"] = None

    result = prepare_article_features(articles)

    assert result.loc[0, "colour_group_name"] == "[MISSING]"


def test_empty_categories_are_handled():
    articles = sample_articles()
    articles.loc[0, "product_type_name"] = " "

    result = prepare_article_features(articles)

    assert result.loc[0, "product_type_name"] == "[MISSING]"


def test_duplicate_article_ids_raise_error():
    articles = sample_articles()
    articles.loc[1, "article_id"] = 1001

    with pytest.raises(ValueError, match="unique"):
        prepare_article_features(articles)


def test_missing_article_id_raises_error():
    articles = sample_articles()
    articles.loc[0, "article_id"] = None

    with pytest.raises(ValueError, match="missing"):
        prepare_article_features(articles)


def test_missing_required_column_raises_error():
    articles = sample_articles().drop(columns=["department_name"])

    with pytest.raises(ValueError, match="Missing required article columns"):
        prepare_article_features(articles)


def test_categorical_features_have_no_missing_values():
    result = prepare_article_features(sample_articles())

    for column in ARTICLE_CATEGORICAL_FEATURES:
        assert result[column].isna().sum() == 0
