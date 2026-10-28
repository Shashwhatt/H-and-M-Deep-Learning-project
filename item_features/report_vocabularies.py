import json
import os


def report_vocabulary_sizes(vocabulary_dir):
    for filename in sorted(os.listdir(vocabulary_dir)):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(vocabulary_dir, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        print(f"{filename}: {len(vocabulary)} categories")