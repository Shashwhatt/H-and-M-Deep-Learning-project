from .item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN


def handle_missing(df):
    df = df.copy()

    for column in ITEM_CATEGORICAL_FEATURES:
        if column not in df.columns:
            continue

        df[column] = df[column].fillna(UNKNOWN_TOKEN)
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column] == "", column] = UNKNOWN_TOKEN

    return df


# Keep compatibility with code using the newer function name.
handle_missing_values = handle_missing