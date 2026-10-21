from item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN


def handle_missing_values(df):
    df = df.copy()

    for column in ITEM_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna(UNKNOWN_TOKEN)
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column] == "", column] = UNKNOWN_TOKEN

    return df