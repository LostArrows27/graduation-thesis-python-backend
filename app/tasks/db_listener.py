import json
import psycopg2
import select
import logging
from threading import Thread
from app.core.config import settings
from app.services.ai_services import AIService
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


def start_listener(ai_service: AIService):
    global listener_thread, stop_event
    if listener_thread is None:
        import threading
        stop_event = threading.Event()
        listener_thread = Thread(
            target=listen_to_notifications, args=(ai_service,))
        listener_thread.start()


def stop_listener():
    global listener_thread, stop_event
    if listener_thread:
        stop_event.set()
        listener_thread.join()
        listener_thread = None


def listen_to_notifications(ai_service: AIService):
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
                continue  # Timeout, check the stop flag again
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                logging.info(f"Received notification: {notify.payload}")
                try:
                    ai_service.inference_service
                    payload = json.loads(notify.payload)
                    image_bucket_id = payload["image_bucket_id"]
                    image_name = payload["image_name"]
                    image_id = payload["id"]

                    # process image + update on database
                    image_labels = ai_service.classify_image(
                        image_bucket_id, image_name, image_id)

                    ai_service.update_image_labels(image_id, image_labels)

                except RuntimeError as e:
                    logging.error(f"Error processing notification: {e}")

    except Exception as e:
        logging.error(f"Database listener error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            logging.info("Database listener stopped.")
