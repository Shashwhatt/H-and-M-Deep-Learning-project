import pandas as pd

from .encode_categories import encode_categories


def test_encode_categories():
    df = pd.DataFrame({
        "product_type_name": ["Trousers", "UnknownValue"],
        "product_group_name": ["Garment Lower body", "UnknownValue"],
        "colour_group_name": ["Black", "UnknownValue"],
        "department_name": ["Ladieswear", "UnknownValue"],
        "index_name": ["Ladieswear", "UnknownValue"],
        "index_group_name": ["Ladieswear", "UnknownValue"],
        "section_name": ["Womens Everyday Basics", "UnknownValue"],
        "garment_group_name": ["Trousers", "UnknownValue"],
    })

    vocabularies = {
        column: {"<UNK>": 0, "Trousers": 1, "Garment Lower body": 2,
                 "Black": 3, "Ladieswear": 4,
                 "Womens Everyday Basics": 5}
        for column in df.columns
    }

    encoded = encode_categories(df, vocabularies)

    assert encoded.shape == df.shape
    assert (encoded >= 0).all().all()