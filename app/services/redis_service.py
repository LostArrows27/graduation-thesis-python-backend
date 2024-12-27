import redis
from app.core.config import settings


class RedisService:
    def __init__(self):
        self.client = redis.StrictRedis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )

    def push_to_stream(self, stream_name: str, data: dict):
        self.client.xadd(stream_name, data)

    def read_from_stream(self, stream_name: str, group_name: str, consumer_name: str, count: int = 1, block: int = 0):
        return self.client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: '>'},
            count=count,
            block=block
        )

    def ack_stream(self, stream_name: str, group_name: str, entry_id: str):
        self.client.xack(stream_name, group_name, entry_id)

    def update_hash(self, hash_name: str, data: dict):
        self.client.hset(hash_name, mapping=data)
