from item_feature_config import UNKNOWN_TOKEN
from load_articles import load_articles
from extract_categories import extract_unique_categories


def create_vocabularies(categories):
    vocabularies = {}

    for feature, values in categories.items():
        unique_values = sorted(
            value for value in values
            if value != UNKNOWN_TOKEN
        )

        vocabulary = {UNKNOWN_TOKEN: 0}

        for index, value in enumerate(unique_values, start=1):
            vocabulary[value] = index

        vocabularies[feature] = vocabulary

    return vocabularies
    return vocabularies


if __name__ == "__main__":
    file_path = r"C:\Users\TULASI SAI\Downloads\articles.csv\articles.csv"
    articles = load_articles(file_path)
    categories = extract_unique_categories(articles)
    vocabularies = create_vocabularies(categories)

    print("Vocabularies created successfully.")

    for feature, vocabulary in vocabularies.items():
        print(feature, ":", len(vocabulary))