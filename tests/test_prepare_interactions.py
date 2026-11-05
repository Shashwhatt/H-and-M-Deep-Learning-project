
import pandas as pd
import pytest

from src.prepare_interactions import (
    generate_sample_interactions,
    split_interactions,
    generate_negative_samples,
)


def test_generate_sample_interactions():
    interactions = generate_sample_interactions(
        interactions_per_user=3,
        random_state=42,
    )

    assert not interactions.empty
    assert list(interactions.columns) == [
        "customer_id", "article_id", "label"
    ]
    assert interactions["label"].eq(1).all()


def test_generate_sample_interactions_reproducible():
    first = generate_sample_interactions(random_state=42)
    second = generate_sample_interactions(random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_split_interactions():
    interactions = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C3"],
        "article_id": ["A1", "A2", "A3", "A4", "A5"],
        "label": [1, 1, 1, 1, 1],
    })

    train, validation = split_interactions(
        interactions,
        validation_size=0.2,
    )

    assert len(train) + len(validation) == len(interactions)
    assert len(validation) == 1


def test_negative_samples_exclude_all_known_positives():
    train = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "article_id": ["A1", "A2", "A3"],
        "label": [1, 1, 1],
    })

    all_positives = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2"],
        "article_id": ["A1", "A2", "A4", "A3", "A5"],
        "label": [1, 1, 1, 1, 1],
    })

    negatives = generate_negative_samples(
        train,
        ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        all_positive_interactions=all_positives,
        random_state=42,
    )

    overlap = all_positives.merge(
        negatives,
        on=["customer_id", "article_id"],
    )

    assert overlap.empty
    assert negatives["label"].eq(0).all()


def test_negative_sampling_rejects_invalid_labels():
    interactions = pd.DataFrame({
        "customer_id": ["C1"],
        "article_id": ["A1"],
        "label": [0],
    })

    with pytest.raises(ValueError):
        generate_negative_samples(interactions, ["A1", "A2"])
