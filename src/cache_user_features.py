
import json

import pandas as pd

from api.services.redis_service import RedisService
from src.prepare_user_features import prepare_inference_features


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "sample" / "customers_sample.csv"


def cache_user_features():
    # Load customer data
    customers = pd.read_csv(CUSTOMERS_PATH)

    # Prepare customer features using saved vocabularies
    features, _ = prepare_inference_features(customers)

    # Connect to Redis
    redis_service = RedisService()

    if not redis_service.ping():
        raise ConnectionError("Redis server is not available.")

    cached_count = 0

    try:
        # Store each customer's features as JSON
        for record in features.to_dict(orient="records"):
            customer_id = record["customer_id"]
            redis_key = f"user_features:{customer_id}"

            redis_service.client.set(
                redis_key,
                json.dumps(record)
            )

            cached_count += 1

        print(f"Customers cached successfully: {cached_count}")

    finally:
        redis_service.close()


if __name__ == "__main__":
    cache_user_features()
