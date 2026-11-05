import json
import os

def validate_vocabularies(vocabulary_dir):
    for filename in os.listdir(vocabulary_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(vocabulary_dir, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        if "<UNK>" not in vocabulary:
            raise ValueError(f"<UNK> missing in {filename}")

        if vocabulary["<UNK>"] != 0:
            raise ValueError(f"<UNK> must have ID 0 in {filename}")

    print("All vocabularies are valid.")