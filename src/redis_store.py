import redis
import numpy as np
import json
import os

def populate_redis_cache():
    print("--- Connecting to Redis Feature Store ---")
    # Connect to local Redis instance
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    try:
        r.ping()
        print("Successfully connected to Redis!")
    except redis.ConnectionError:
        print("Redis server is not running. Please start Redis before caching vectors.")
        return

    # Load exported embeddings
    embeddings_path = "models/item_embeddings.npy"
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file not found at {embeddings_path}. Run export_embeddings.py first.")
        return
        
    item_embeddings = np.load(embeddings_path, allow_pickle=True).item()

    print("--- Caching Item Embeddings into Redis ---")
    for article_id, vector in item_embeddings.items():
        # Store vector as JSON string under hash or key format
        r.set(f"item_emb:{article_id}", json.dumps(vector))

    print(f"Successfully cached {len(item_embeddings)} item vectors into Redis!")

if __name__ == "__main__":
    populate_redis_cache()