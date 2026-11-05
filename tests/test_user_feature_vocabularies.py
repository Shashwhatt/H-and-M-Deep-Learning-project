
import json
import pandas as pd
import pytest

from src.user_feature_vocabularies import (
    build_vocabulary,
    save_vocabulary,
    load_vocabulary,
)


def test_duplicate_categories_share_one_id():
    vocabulary_with_duplicates = build_vocabulary(
        pd.Series(["ACTIVE", "ACTIVE", "PRE-CREATE"])
    )
    vocabulary_without_duplicates = build_vocabulary(
        pd.Series(["ACTIVE", "PRE-CREATE"])
    )

    assert vocabulary_with_duplicates == vocabulary_without_duplicates
    assert len(vocabulary_with_duplicates) == 3


def test_unknown_token_is_reserved():
    vocabulary = build_vocabulary(pd.Series(["ACTIVE"]))

    assert vocabulary["[UNK]"] == 0


def test_missing_values_are_handled():
    vocabulary = build_vocabulary(
        pd.Series(["ACTIVE", None, ""])
    )

    assert vocabulary["[MISSING]"] == 1


def test_category_ids_are_unique():
    vocabulary = build_vocabulary(
        pd.Series(["ACTIVE", "PRE-CREATE", "ACTIVE"])
    )

    category_ids = list(vocabulary.values())
    assert len(category_ids) == len(set(category_ids))


def test_vocabulary_is_deterministic():
    first = build_vocabulary(
        pd.Series(["ACTIVE", "PRE-CREATE", "NONE"])
    )
    second = build_vocabulary(
        pd.Series(["NONE", "ACTIVE", "PRE-CREATE"])
    )

    assert first == second


def test_vocabulary_is_saved_and_loaded_correctly(tmp_path):
    vocabulary = build_vocabulary(
        pd.Series(["ACTIVE", "PRE-CREATE", None])
    )

    save_vocabulary(vocabulary, "club_member_status", tmp_path)

    loaded_vocabulary = load_vocabulary(
        "club_member_status", tmp_path
    )

    assert loaded_vocabulary == vocabulary


def test_missing_vocabulary_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocabulary("club_member_status", tmp_path)


def test_invalid_vocabulary_ids_are_rejected(tmp_path):
    invalid_vocabulary = {
        "[UNK]": 0,
        "ACTIVE": 2,
    }

    file_path = tmp_path / "club_member_status_vocabulary.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(invalid_vocabulary, file)

    with pytest.raises(ValueError, match="contiguous"):
        load_vocabulary("club_member_status", tmp_path)
