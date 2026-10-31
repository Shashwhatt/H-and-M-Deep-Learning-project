
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from train_model import HMRetrievalModel


# ==========================================
# CONFIGURATION
# ==========================================

CUSTOMERS_FILE = "data/processed/customers.csv"
ARTICLES_FILE = "data/processed/articles.csv"

TRAIN_FILE = "data/processed/train_interactions.csv"
VALIDATION_FILE = "data/processed/validation_interactions.csv"

MODEL_FILE = "models/best_model.weights.h5"

SAMPLE_USERS = 200
TOP_K = 20
CHUNK_SIZE = 500_000
EMBEDDING_BATCH_SIZE = 2048

RANDOM_SEED = 42


# ==========================================
# EVALUATION METRICS
# ==========================================

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)

    ideal_dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


# ==========================================
# LOAD VALIDATION CUSTOMERS
# ==========================================


def load_validation_users():
    print("\nLoading FULL validation dataset...")

    validation = pd.read_csv(
        VALIDATION_FILE,
        dtype={
            "customer_id": str,
            "article_id": str
        }
    )

    validation = validation.dropna(
        subset=["customer_id", "article_id"]
    )

    validation["customer_id"] = validation["customer_id"].str.strip()
    validation["article_id"] = (
        validation["article_id"].str.strip().str.zfill(10)
    )

    relevant_by_user = (
        validation.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    selected_users = list(relevant_by_user.keys())

    print("Total validation interactions:", len(validation))
    print("Total validation customers:", len(selected_users))

    return selected_users, relevant_by_user



# ==========================================
# FIND PREVIOUSLY PURCHASED ARTICLES
# ==========================================

def load_seen_articles(selected_users):
    print("\nFinding previously purchased articles...")

    selected_set = set(selected_users)

    seen_by_user = {
        user: set() for user in selected_users
    }

    for chunk in pd.read_csv(
        TRAIN_FILE,
        dtype={
            "customer_id": str,
            "article_id": str
        },
        chunksize=CHUNK_SIZE
    ):

        filtered = chunk[
            chunk["customer_id"].isin(selected_set)
        ].copy()

        # Restore leading zeros in article IDs
        filtered["article_id"] = (
            filtered["article_id"].str.strip().str.zfill(10)
        )

        for user, article in zip(
            filtered["customer_id"],
            filtered["article_id"]
        ):
            seen_by_user[user].add(article)

    print("Previous purchase history loaded.")

    return seen_by_user


# ==========================================
# LOAD MODEL AND RESTORE CHECKPOINT
# ==========================================

def load_trained_model(customer_ids, article_ids):

    print("\nLoading trained Two-Tower model...")

    model = HMRetrievalModel(
        customer_ids,
        article_ids
    )

    # Build both towers before restoring weights.
    model.query_tower({
        "customer_id": tf.constant([customer_ids[0]])
    })

    model.candidate_tower({
        "article_id": tf.constant([article_ids[0]])
    })

    model.load_weights(MODEL_FILE)

    print("Trained weights restored successfully.")

    return model


# ==========================================
# GENERATE ALL ARTICLE EMBEDDINGS
# ==========================================

def generate_candidate_embeddings(model, article_ids):

    print("\nGenerating article embeddings...")

    all_embeddings = []

    for start in range(
        0,
        len(article_ids),
        EMBEDDING_BATCH_SIZE
    ):

        batch_ids = article_ids[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        embeddings = model.candidate_tower({
            "article_id": tf.constant(batch_ids)
        })

        all_embeddings.append(embeddings.numpy())

    candidate_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    print("Article embeddings shape:", candidate_embeddings.shape)

    return candidate_embeddings


# ==========================================
# GENERATE CUSTOMER RECOMMENDATIONS
# ==========================================

def evaluate_recommendations(
    model,
    selected_users,
    relevant_by_user,
    seen_by_user,
    article_ids
):
    print("\nGenerating article embeddings...")

    candidate_embeddings = generate_candidate_embeddings(
        model,
        article_ids
    )

    article_to_index = {
        article: index
        for index, article in enumerate(article_ids)
    }

    metric_results = {
        5: {"recall": [], "ndcg": []},
        10: {"recall": [], "ndcg": []},
        20: {"recall": [], "ndcg": []}
    }

    evaluated_users = 0
    skipped_users = 0
    batch_size = 64
    total_users = len(selected_users)

    print("\nEvaluating all validation customers...")

    for start in range(0, total_users, batch_size):

        batch_users = selected_users[start:start + batch_size]

        user_embeddings = model.query_tower({
            "customer_id": tf.constant(batch_users)
        }).numpy()

        scores = user_embeddings @ candidate_embeddings.T

        for row, user in enumerate(batch_users):

            # Remove previously purchased articles
            seen_indices = [
                article_to_index[a]
                for a in seen_by_user[user]
                if a in article_to_index
            ]

            scores[row, seen_indices] = -np.inf

            # Relevant unseen validation articles
            relevant = (
                relevant_by_user[user] - seen_by_user[user]
            )

            relevant = {
                article for article in relevant
                if article in article_to_index
            }

            if not relevant:
                skipped_users += 1
                continue

            # Get top 20 recommendations
            top_indices = np.argpartition(
                scores[row],
                -20
            )[-20:]

            top_indices = top_indices[
                np.argsort(scores[row, top_indices])[::-1]
            ]

            recommendations = [
                article_ids[index]
                for index in top_indices
            ]

            evaluated_users += 1

            for k in [5, 10, 20]:

                recall = recall_at_k(
                    recommendations,
                    relevant,
                    k
                )

                ndcg = ndcg_at_k(
                    recommendations,
                    relevant,
                    k
                )

                metric_results[k]["recall"].append(recall)
                metric_results[k]["ndcg"].append(ndcg)

        completed = min(start + batch_size, total_users)

        if completed % 1024 == 0 or completed == total_users:
            print(
                f"Progress: {completed:,}/{total_users:,} customers"
            )

    print("\n========== FULL DATASET EVALUATION ==========")

    print("Total validation customers:", total_users)
    print("Users evaluated:", evaluated_users)
    print("Users skipped:", skipped_users)

    if evaluated_users == 0:
        print("No eligible users found.")
        return

    for k in [5, 10, 20]:

        recall = np.mean(metric_results[k]["recall"])
        ndcg = np.mean(metric_results[k]["ndcg"])

        print(f"\nRecall@{k}: {recall:.6f} ({recall * 100:.4f}%)")
        print(f"NDCG@{k}:   {ndcg:.6f} ({ndcg * 100:.4f}%)")

    print("\n=============================================")

# ==========================================
# MAIN
# ==========================================

def main():
    print("\n========== FULL H&M MODEL EVALUATION ==========\n")

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}"
        )

    print("Loading customer vocabulary...")

    customers = pd.read_csv(
        CUSTOMERS_FILE,
        usecols=["customer_id"],
        dtype={"customer_id": str}
    )

    customer_ids = (
        customers["customer_id"]
        .dropna()
        .str.strip()
        .unique()
        .tolist()
    )

    print("Loading article vocabulary...")

    articles = pd.read_csv(
        ARTICLES_FILE,
        usecols=["article_id"],
        dtype={"article_id": str}
    )

    articles["article_id"] = (
        articles["article_id"]
        .str.strip()
        .str.zfill(10)
    )

    article_ids = (
        articles["article_id"]
        .dropna()
        .unique()
        .tolist()
    )

    customer_vocab = set(customer_ids)
    article_vocab = set(article_ids)

    print("Total customers:", len(customer_ids))
    print("Total articles:", len(article_ids))

    # Load all validation customers
    selected_users, relevant_by_user = load_validation_users()

    # Keep customers present in model vocabulary
    selected_users = [
        user for user in selected_users
        if user in customer_vocab
    ]

    relevant_by_user = {
        user: relevant_by_user[user] & article_vocab
        for user in selected_users
    }

    print("\nCustomers matching vocabulary:", len(selected_users))

    if not selected_users:
        print("No matching validation customers.")
        return

    # Load training history for all validation customers
    seen_by_user = load_seen_articles(selected_users)

    # Keep users with at least one unseen validation article
    eligible_users = [
        user for user in selected_users
        if relevant_by_user[user] - seen_by_user[user]
    ]

    print("\nEligible customers:", len(eligible_users))

    if not eligible_users:
        print("No eligible customers found.")
        return

    # Keep all eligible customers. No random sampling.
    selected_users = eligible_users

    print("Customers going into evaluation:", len(selected_users))

    # Load trained model
    model = load_trained_model(
        customer_ids,
        article_ids
    )

    # Evaluate all eligible customers in batches
    evaluate_recommendations(
        model,
        selected_users,
        relevant_by_user,
        seen_by_user,
        article_ids
    )

    print("\n========== EVALUATION COMPLETED ==========")


if __name__ == "__main__":
    main()
