
import os
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.two_tower_model import HMQueryTower, HMCandidateTower, HMTwoTowerModel
except ImportError:
    from two_tower_model import HMQueryTower, HMCandidateTower, HMTwoTowerModel


# ==========================================
# CONFIGURATION
# ==========================================

CUSTOMERS_FILE = BASE_DIR / "data" / "processed" / "customers.csv"
ARTICLES_FILE = BASE_DIR / "data" / "processed" / "articles.csv"

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train_interactions.csv"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation_interactions.csv"

MODEL_DIR = Path(os.environ.get("MODEL_OUTPUT_DIR", str(BASE_DIR / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 2

# Training dataset parameters
MAX_TRAIN_ROWS = 200_000
MAX_VALIDATION_ROWS = 10_000

CHUNK_SIZE = 100_000


# ==========================================
# LOAD CUSTOMER AND ARTICLE VOCABULARIES
# ==========================================

def load_id_vocabulary(file_path, column):
    print(f"Loading {column} IDs...")

    df = pd.read_csv(
        file_path,
        usecols=[column],
        dtype={column: str}
    )

    ids = df[column].dropna().unique().tolist()

    print(f"Unique {column} IDs: {len(ids):,}")

    return ids


# ==========================================
# READ INTERACTIONS IN CHUNKS
# ==========================================

def interaction_generator(file_path, max_rows, random_sample=False, recent_first=True):
    rows_read = 0
    total_train_rows = 31_548_013

    if recent_first and not random_sample:
        # Seek directly near the end of train_interactions.csv
        # This focuses training on the most recent Autumn 2020 transactions (14.7x higher item overlap with validation!)
        byte_estimate = int(max_rows * 80 * 1.3)
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            seek_pos = max(0, file_size - byte_estimate)
            f.seek(seek_pos)
            if seek_pos > 0:
                f.readline()  # Discard partial line

            lines = []
            for line in f:
                lines.append(line)
                if len(lines) >= max_rows:
                    break

        for i in range(0, len(lines), CHUNK_SIZE):
            batch = lines[i:i + CHUNK_SIZE]
            cids, aids = [], []
            for b in batch:
                row = b.decode("utf-8", errors="ignore").strip().split(",")
                if len(row) >= 2:
                    c = row[0].strip()
                    a = row[1].strip().zfill(10)
                    if c and a:
                        cids.append(c)
                        aids.append(a)
            if cids:
                yield {
                    "customer_id": np.array(cids, dtype=object),
                    "article_id": np.array(aids, dtype=object)
                }

    elif random_sample:
        sample_fraction = min(max_rows / total_train_rows, 1.0)

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                file_path,
                usecols=["customer_id", "article_id"],
                dtype={
                    "customer_id": str,
                    "article_id": str
                },
                chunksize=CHUNK_SIZE
            )
        ):
            remaining = max_rows - rows_read

            if remaining <= 0:
                break

            chunk = chunk.dropna(
                subset=["customer_id", "article_id"]
            )

            sample_size = min(
                remaining,
                round(len(chunk) * sample_fraction)
            )

            if sample_size == 0:
                continue

            sampled = chunk.sample(
                n=sample_size,
                random_state=42 + chunk_number
            )

            rows_read += len(sampled)

            yield {
                "customer_id": sampled["customer_id"].to_numpy(),
                "article_id": (
                    sampled["article_id"]
                    .str.strip()
                    .str.zfill(10)
                    .to_numpy()
                )
            }

    else:
        for chunk in pd.read_csv(
            file_path,
            usecols=["customer_id", "article_id"],
            dtype={
                "customer_id": str,
                "article_id": str
            },
            chunksize=CHUNK_SIZE
        ):
            remaining = max_rows - rows_read

            if remaining <= 0:
                break

            chunk = chunk.iloc[:remaining].dropna(
                subset=["customer_id", "article_id"]
            )

            rows_read += len(chunk)

            if not chunk.empty:
                yield {
                    "customer_id": chunk["customer_id"].to_numpy(),
                    "article_id": (
                        chunk["article_id"]
                        .str.strip()
                        .str.zfill(10)
                        .to_numpy()
                    )
                }


def create_dataset(file_path, max_rows, random_sample=False, recent_first=False):
    output_signature = {
        "customer_id": tf.TensorSpec(
            shape=(None,), dtype=tf.string
        ),
        "article_id": tf.TensorSpec(
            shape=(None,), dtype=tf.string
        )
    }

    dataset = tf.data.Dataset.from_generator(
        lambda: interaction_generator(
            file_path,
            max_rows,
            random_sample=random_sample,
            recent_first=recent_first
        ),
        output_signature=output_signature
    )

    return (
        dataset
        .unbatch()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

# ==========================================
# TWO-TOWER RETRIEVAL MODEL
# ==========================================

class HMRetrievalModel(tfrs.Model):

    def __init__(self, customer_ids, article_ids):

        super().__init__()

        self.query_tower = HMQueryTower(
            customer_ids,
            EMBEDDING_DIM
        )

        self.candidate_tower = HMCandidateTower(
            article_ids,
            EMBEDDING_DIM
        )

        # Candidate article embeddings for retrieval evaluation
        candidate_dataset = (
            tf.data.Dataset
            .from_tensor_slices(article_ids)
            .batch(1024)
            .map(
                lambda ids: self.candidate_tower({
                    "article_id": ids
                })
            )
        )

        # Use temperature scaling (tau=0.05) on L2-normalized embeddings for sharp contrastive separation
        self.retrieval_task = tfrs.tasks.Retrieval(
            temperature=0.05,
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset
            )
        )

    def compute_loss(self, features, training=False):

        customer_embeddings = self.query_tower({
            "customer_id": features["customer_id"]
        })

        article_embeddings = self.candidate_tower({
            "article_id": features["article_id"]
        })

        # In-batch negatives are used by the retrieval task.
        return self.retrieval_task(
            customer_embeddings,
            article_embeddings
        )


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("\n========== H&M TWO-TOWER TRAINING ==========\n")

    print("TensorFlow:", tf.__version__)
    print("TFRS:", tfrs.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    # Load vocabularies
    customer_ids = load_id_vocabulary(
        CUSTOMERS_FILE,
        "customer_id"
    )

    article_ids = load_id_vocabulary(
        ARTICLES_FILE,
        "article_id"
    )

    # Prepare training and validation datasets
    print("\nPreparing training dataset (Recent Transactions Focus)...")

    train_dataset = create_dataset(
        TRAIN_FILE,
        MAX_TRAIN_ROWS,
        random_sample=False,
        recent_first=True
    )

    print("Preparing validation dataset...")

    validation_dataset = create_dataset(
        VALIDATION_FILE,
        MAX_VALIDATION_ROWS,
        random_sample=False,
        recent_first=False
    )

    # Create model
    print("\nCreating Two-Tower model...")

    model = HMRetrievalModel(
        customer_ids,
        article_ids
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        )
    )

    # Save best checkpoint & decay LR on plateau
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            MODEL_DIR / "best_model.weights.h5"
        ),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        verbose=1
    )

    # Train
    print("\nStarting model training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, lr_callback]
    )

    # Save final weights in serving-compatible format.
    print("\nPreparing serving model...")

    serving_model = HMTwoTowerModel(
        customer_ids,
        article_ids,
        embedding_dim=EMBEDDING_DIM,
    )

    sample = {
        "customer_id": tf.constant([customer_ids[0]]),
        "article_id": tf.constant([article_ids[0]]),
    }

    serving_model(sample)

    serving_model.query_tower.set_weights(
        model.query_tower.get_weights()
    )

    serving_model.candidate_tower.set_weights(
        model.candidate_tower.get_weights()
    )

    serving_model.save_weights(
        str(MODEL_DIR / "two_tower.weights.h5")
    )

    print("\n========== TRAINING COMPLETE ==========")

    print("Model saved at:")
    print(MODEL_DIR / "two_tower.weights.h5")

    print("\nTraining history:")

    for metric, values in history.history.items():
        print(f"{metric}: {values}")

    print("\n========================================")


if __name__ == "__main__":
    main()
