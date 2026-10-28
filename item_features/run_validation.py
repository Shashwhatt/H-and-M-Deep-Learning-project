from .load_articles import load_articles
from .validate_dataset import validate_dataset
from .validate_vocabularies import validate_vocabularies


def run_validation(csv_path, vocabulary_dir):
    df = load_articles(csv_path)

    validate_dataset(df)
    validate_vocabularies(vocabulary_dir)
    

    print("All item feature validations passed.")