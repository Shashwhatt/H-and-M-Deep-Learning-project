
import os

import redis
from redis.exceptions import RedisError


class RedisService:
    def __init__(self):
        self.redis_url = os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )
        self.client = redis.Redis.from_url(
            self.redis_url,
            decode_responses=True
        )

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except RedisError:
            return False

    def get_user_features(self, key: str):
        try:
            return self.client.get(key)
        except RedisError as exc:
            raise RuntimeError("Unable to retrieve user features from Redis") from exc

    def close(self):
        self.client.close()
