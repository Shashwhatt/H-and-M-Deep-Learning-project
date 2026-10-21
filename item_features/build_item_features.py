from .load_articles import load_articles
from .handle_missing import handle_missing
from .extract_categories import extract_unique_categories
from .create_vocabularies import create_vocabularies
from .encode_categories import encode_categories


def build_item_features(csv_path):
    df = load_articles(csv_path)
    df = handle_missing(df)

    categories = extract_unique_categories(df)
    vocabularies = create_vocabularies(categories)

    encoded_df = encode_categories(df, vocabularies)

    return encoded_df, vocabularies