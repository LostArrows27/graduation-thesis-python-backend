import psycopg2
import select
import logging
from threading import Thread
from app.core.config import settings

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

def listen_to_notifications():
    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute("LISTEN image_meta_data_insert;")
        logging.info("Listening for insert events on public.image_meta_data...")

        while not stop_event.is_set():
            if select.select([conn], [], [], 1) == ([], [], []):
                continue  # Timeout, check the stop flag again
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                logging.info(f"Received notification: {notify.payload}")

    except Exception as e:
        logging.error(f"Database listener error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            logging.info("Database listener stopped.")

def start_listener():
    global listener_thread, stop_event
    if listener_thread is None:
        import threading
        stop_event = threading.Event()
        listener_thread = Thread(target=listen_to_notifications)
        listener_thread.start()

def stop_listener():
    global listener_thread, stop_event
    if listener_thread:
        stop_event.set()
        listener_thread.join()
        listener_thread = None