
import pandas as pd
import pytest

from src.prepare_user_features import prepare_user_features
from src.prepare_user_features import (
    prepare_training_features,
    prepare_inference_features,
)


def sample_customers():
    return pd.DataFrame({
        "customer_id": ["customer_1", "customer_2", "customer_3"],
        "club_member_status": ["ACTIVE", "ACTIVE", None],
        "fashion_news_frequency": ["Regularly", "NONE", ""],
        "age": [25, None, 40],
    })


def test_output_rows_and_columns():
    features, _ = prepare_user_features(sample_customers())

    assert len(features) == 3
    assert features.columns.tolist() == [
        "customer_id",
        "age",
        "age_missing",
        "club_member_status_id",
        "fashion_news_frequency_id",
    ]


def test_customer_ids_remain_strings():
    features, _ = prepare_user_features(sample_customers())

    assert features["customer_id"].tolist() == [
        "customer_1", "customer_2", "customer_3"
    ]


def test_missing_age_is_imputed_and_flagged():
    features, _ = prepare_user_features(sample_customers())

    assert features["age"].notna().all()
    assert features["age_missing"].tolist() == [0, 1, 0]


def test_missing_categories_are_encoded():
    features, vocabularies = prepare_user_features(sample_customers())

    assert features["club_member_status_id"].notna().all()
    assert features["fashion_news_frequency_id"].notna().all()
    assert "[MISSING]" in vocabularies["club_member_status"]
    assert "[MISSING]" in vocabularies["fashion_news_frequency"]


def test_duplicate_customer_ids_are_rejected():
    customers = sample_customers()
    customers.loc[1, "customer_id"] = "customer_1"

    with pytest.raises(ValueError, match="Duplicate customer IDs"):
        prepare_user_features(customers)


def test_missing_required_column_is_rejected():
    customers = sample_customers().drop(columns=["age"])

    with pytest.raises(ValueError, match="Required customer columns"):
        prepare_user_features(customers)
def test_unseen_categories_use_unknown_id(tmp_path, monkeypatch):
    import src.prepare_user_features as module

    monkeypatch.setattr(
        module,
        "OUTPUT_DIR",
        tmp_path,
    )

    training_customers = sample_customers()
    prepare_training_features(training_customers)

    inference_customers = pd.DataFrame({
        "customer_id": ["customer_new"],
        "club_member_status": ["NEW-STATUS"],
        "fashion_news_frequency": ["Weekly"],
        "age": [30],
    })

    features = prepare_inference_features(inference_customers)

    assert features[0]["club_member_status_id"].iloc[0] == 0
    assert features[0]["fashion_news_frequency_id"].iloc[0] == 0


def test_inference_preserves_known_category_ids(tmp_path, monkeypatch):
    import src.prepare_user_features as module

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)

    training_customers = sample_customers()
    _, vocabularies = prepare_training_features(training_customers)

    inference_customers = sample_customers().iloc[[0]].copy()

    features, loaded_vocabularies = prepare_inference_features(
        inference_customers
    )

    assert loaded_vocabularies == vocabularies
    assert (
        features["club_member_status_id"].iloc[0]
        == vocabularies["club_member_status"]["ACTIVE"]
    )


def test_inference_preserves_missing_category_id(tmp_path, monkeypatch):
    import src.prepare_user_features as module

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)

    training_customers = sample_customers()
    _, vocabularies = prepare_training_features(training_customers)

    inference_customers = sample_customers().iloc[[2]].copy()

    features, _ = prepare_inference_features(inference_customers)

    assert (
        features["club_member_status_id"].iloc[0]
        == vocabularies["club_member_status"]["[MISSING]"]
    )