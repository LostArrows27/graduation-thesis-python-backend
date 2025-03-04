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

    def read_from_stream(self, stream_name: str, group_name: str, consumer_name: str, count: int = 1, block: int = 0, start_id='0'):
        return self.client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: start_id},
            count=count,
            block=block
        )

    def ack_stream(self, stream_name: str, group_name: str, entry_id: str):
        self.client.xack(stream_name, group_name, entry_id)

    def update_hash(self, hash_name: str, data: dict):
        self.client.hset(hash_name, mapping=data)

    def delete_stream_entry(self, stream_name, entry_id):
        self.client.xdel(stream_name, entry_id)

    def set_ttl(self, key, ttl):
        self.client.expire(key, ttl)

    def create_consumer_group(self, stream_name: str, group_name: str):
        try:
            self.client.xgroup_create(
                stream_name, group_name, id='0', mkstream=True)

        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Consumer group already exists
                pass
            else:
                raise e

    def update_image_label_job(self, image_id: str, image_bucket_id: str, image_name: str):
        self.update_hash(
            f"image_job:{image_id}",
            {
                "image_bucket_id": image_bucket_id,
                "image_name": image_name,
                "label_status": "processing"
            }
        )

        self.set_ttl(f"image_job:{image_id}", 10800)
