import json
import threading
import psycopg2
import select
import logging
from threading import Thread
from app.core.config import settings
from app.services.redis_service import RedisService

logging.basicConfig(level=logging.INFO)

conn_params = {
    'dbname': settings.db_name,
    'user': settings.db_user,
    'password': settings.db_password,
    'host': settings.db_host,
    'port': settings.db_port,
    "sslmode": "require"
}

listener_thread = None
stop_event = None


def start_listener(redis_service: RedisService):
    global listener_thread, stop_event
    if listener_thread is None:
        stop_event = threading.Event()
        listener_thread = Thread(
            target=listen_to_notifications, args=(redis_service,))
        listener_thread.daemon = True
        listener_thread.start()


def stop_listener():
    global listener_thread, stop_event
    if listener_thread:
        stop_event.set()
        listener_thread.join()
        listener_thread = None


def listen_to_notifications(redis_service: RedisService):
    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute("LISTEN image_meta_data_insert;")
        logging.info(
            "Listening for insert events on public.image_meta_data...")

        while not stop_event.is_set():
            if select.select([conn], [], [], 1) == ([], [], []):
                continue  # timeout, check the stop flag again
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                logging.info(f"Received notification: {notify.payload}")
                try:
                    payload = json.loads(notify.payload)
                    image_id = payload["id"]
                    image_bucket_id = payload["image_bucket_id"]
                    image_name = payload["image_name"]

                    # add image_id to the stream
                    redis_service.push_to_stream(
                        'image_label_stream',
                        {
                            'image_id': image_id,
                            'image_bucket_id': image_bucket_id,
                            'image_name': image_name,
                        }
                    )
                except RuntimeError as e:
                    logging.error(f"Error processing notification: {e}")

    except Exception as e:
        logging.error(f"Database listener error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            logging.info("Database listener stopped.")
