
import json
import pytest

from item_features.load_vocabularies import load_vocabularies
from item_features.item_feature_config import (
    ITEM_CATEGORICAL_FEATURES,
)


def test_load_all_item_vocabularies(tmp_path):
    for feature in ITEM_CATEGORICAL_FEATURES:
        vocabulary = {
            "<UNK>": 0,
            "CATEGORY_A": 1,
        }

        with (tmp_path / f"{feature}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(vocabulary, file)

    vocabularies = load_vocabularies(tmp_path)

    assert len(vocabularies) == 9
    assert vocabularies["product_type_name"]["<UNK>"] == 0


def test_missing_vocabulary_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocabularies(tmp_path)


def test_missing_unknown_token_raises_error(tmp_path):
    for feature in ITEM_CATEGORICAL_FEATURES:
        vocabulary = {"CATEGORY_A": 0}

        with (tmp_path / f"{feature}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(vocabulary, file)

    with pytest.raises(ValueError):
        load_vocabularies(tmp_path)


def test_duplicate_vocabulary_ids_raise_error(tmp_path):
    for feature in ITEM_CATEGORICAL_FEATURES:
        vocabulary = {
            "<UNK>": 0,
            "CATEGORY_A": 1,
            "CATEGORY_B": 1,
        }

        with (tmp_path / f"{feature}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(vocabulary, file)

    with pytest.raises(ValueError):
        load_vocabularies(tmp_path)
