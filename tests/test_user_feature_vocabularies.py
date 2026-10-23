
import pandas as pd

from src.user_feature_vocabularies import build_vocabulary


def test_duplicate_categories_share_one_id():
    vocabulary_with_duplicates = build_vocabulary(
        pd.Series(["ACTIVE", "ACTIVE", "PRE-CREATE"])
    )
    vocabulary_without_duplicates = build_vocabulary(
        pd.Series(["ACTIVE", "PRE-CREATE"])
    )

    assert vocabulary_with_duplicates == vocabulary_without_duplicates
    assert len(vocabulary_with_duplicates) == 3  # [UNK] plus two categories


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
