from item_features.item_feature_config import ITEM_CATEGORICAL_FEATURES, UNKNOWN_TOKEN
from item_features.load_articles import load_articles
from item_features.extract_categories import extract_unique_categories
from .save_vocabularies import save_vocabularies


def create_vocabularies(categories):
    vocabularies = {}

    for features, values in categories.items():
        unique_values = sorted(
            value for value in values
            if value != UNKNOWN_TOKEN
        )

        vocabulary = {UNKNOWN_TOKEN: 0}

        for index, value in enumerate(unique_values, start=1):
            vocabulary[value] = index

        vocabularies[features] = vocabulary

    return vocabularies
    


if __name__ == "__main__":
    file_path = r"C:\Users\TULASI SAI\Downloads\articles.csv\articles.csv"
    articles = load_articles(file_path)
    categories = extract_unique_categories(articles)
    vocabularies = create_vocabularies(categories)

    save_vocabularies(vocabularies, "item_features/vocabularies")

    print("Vocabularies created successfully.")

    for features, vocabulary in vocabularies.items():
        print(features, ":", len(vocabulary))