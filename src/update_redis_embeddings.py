
from src.redis_feature_store import (
    connect_redis,
    store_article_embeddings,
)


def main():
    client = connect_redis()

    print("Starting daily article embedding update...")

    store_article_embeddings(client)

    print("Daily Redis embedding update completed!")


if __name__ == "__main__":
    main()
