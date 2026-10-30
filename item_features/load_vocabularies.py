import json
from pathlib import Path

from .item_feature_config import (
    ITEM_CATEGORICAL_FEATURES,
    UNKNOWN_TOKEN,
)


def load_vocabularies(input_dir):
    """Load and validate saved item feature vocabularies."""

    input_dir = Path(input_dir)
    vocabularies = {}

    for feature in ITEM_CATEGORICAL_FEATURES:
        file_path = input_dir / f"{feature}.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        if not isinstance(vocabulary, dict):
            raise ValueError(f"{feature} vocabulary must be a dictionary")

        if UNKNOWN_TOKEN not in vocabulary:
            raise ValueError(
                f"{feature} vocabulary is missing {UNKNOWN_TOKEN}"
            )

        if vocabulary[UNKNOWN_TOKEN] != 0:
            raise ValueError(
                f"{feature} must reserve ID 0 for {UNKNOWN_TOKEN}"
            )

        ids = list(vocabulary.values())

        if any(type(value) is not int for value in ids):
            raise ValueError(f"{feature} vocabulary IDs must be integers")

        if len(set(ids)) != len(ids):
            raise ValueError(
                f"Duplicate IDs found in {feature} vocabulary"
            )

        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                f"{feature} vocabulary IDs must be contiguous from 0"
            )

        vocabularies[feature] = vocabulary

    return vocabularies