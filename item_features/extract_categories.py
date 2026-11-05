from .item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN


def extract_unique_categories(df):
    categories = {}

    for column in ITEM_CATEGORICAL_FEATURES:
        unique_values = df[column].unique().tolist()

        if UNKNOWN_TOKEN not in unique_values:
            unique_values.insert(0, UNKNOWN_TOKEN)

        categories[column] = unique_values

    return categories