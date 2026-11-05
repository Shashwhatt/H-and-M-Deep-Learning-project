import json
import os


def save_vocabularies(vocabularies, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for feature, vocabulary in vocabularies.items():
        file_path = os.path.join(output_dir, f"{feature}.json")

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(vocabulary, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(vocabularies)} vocabularies to {output_dir}")