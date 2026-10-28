from .item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN


def encode_categories(df, vocabularies):
    df = df.copy()

    for column in ITEM_CATEGORICAL_FEATURES:
        if column not in df.columns:
            continue

        if column not in vocabularies:
            raise ValueError(
                f"Missing vocabulary for categorical feature: {column}"
            )

        vocabulary = vocabularies[column]
        unknown_id = vocabulary.get(UNKNOWN_TOKEN, 0)

        df[column] = (
            df[column]
            .fillna(UNKNOWN_TOKEN)
            .astype(str)
            .str.strip()
        )

        df.loc[df[column] == "", column] = UNKNOWN_TOKEN

        df[column] = df[column].map(
            lambda value: vocabulary.get(value, unknown_id)
        )

    return df