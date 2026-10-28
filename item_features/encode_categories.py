import pandas as pd

from item_features.item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN


def encode_categories(df, vocabularies):
    df = df.copy()

    for column in ITEM_CATEGORICAL_FEATURES:
        vocabulary = vocabularies[column]

        df[column] = (
            df[column]
            .fillna(UNKNOWN_TOKEN)
            .astype(str)
            .str.strip()
        )

        df[column] = df[column].map(
            lambda value: vocabulary.get(value, 0)
        )

    return df
