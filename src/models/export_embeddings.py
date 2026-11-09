import os
import numpy as np
import tensorflow as tf
from two_tower_model import HMTTwoTowerModel

def export_item_embeddings():
    print("--- Loading Trained Two-Tower Model ---")
    # Placeholder lists - in production, load actual unique IDs from vocabulary artifacts
    unique_customers = ["C1", "C2", "C3"]
    unique_articles = ["A001", "A002", "A003", "A004"]
    
    candidates_ds = tf.data.Dataset.from_tensor_slices(unique_articles)
    
    model = HMTTwoTowerModel(
        unique_customer_ids=unique_customers,
        unique_article_ids=unique_articles,
        parsed_candidates_ds=candidates_ds,
        embedding_dim=32
    )
    
    # Load trained weights
    if os.path.exists("models/two_tower_weights.index"):
        model.load_weights("models/two_tower_weights")
    else:
        print("Warning: Saved weights not found. Using initialized weights for demonstration.")

    print("--- Generating Pre-computed Item Embeddings ---")
    item_embeddings = {}
    for article_id in unique_articles:
        tensor_id = tf.constant([article_id])
        embedding = model.candidate_model(tensor_id).numpy()[0]
        item_embeddings[article_id] = embedding.tolist()

    os.makedirs("models", exist_ok=True)
    np.save("models/item_embeddings.npy", item_embeddings)
    print("Item embeddings successfully exported to models/item_embeddings.npy!")

if __name__ == "__main__":
    export_item_embeddings()