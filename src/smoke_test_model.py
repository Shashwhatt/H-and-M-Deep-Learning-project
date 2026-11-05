
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import tensorflow as tf

from two_tower_model import HMTwoTowerModel


# Load sample customer and article IDs
customers = pd.read_csv(
    "data/processed/customers.csv",
    usecols=["customer_id"],
    dtype={"customer_id": str},
    nrows=100
)

articles = pd.read_csv(
    "data/processed/articles.csv",
    usecols=["article_id"],
    dtype={"article_id": str},
    nrows=100
)

customer_ids = customers["customer_id"].dropna().unique().tolist()
article_ids = articles["article_id"].dropna().unique().tolist()

# Initialize model
model = HMTwoTowerModel(
    customer_ids,
    article_ids,
    embedding_dim=32
)

# Test two customers and two products
test_inputs = {
    "customer_id": tf.constant(customer_ids[:2]),
    "article_id": tf.constant(article_ids[:2])
}

scores = model(test_inputs)

print("Customer count:", len(customer_ids))
print("Article count:", len(article_ids))
print("Output scores:", scores.numpy())
print("Output shape:", scores.shape)
print("Two-Tower model architecture test successful!")
