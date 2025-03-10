import json
import threading
import traceback
import psycopg2
import select
from threading import Thread
from app.core.config import settings
from app.libs.logger.log import log_error, log_info
from app.services.ai_services import AIService
from app.services.supabase_service import SupabaseService


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


def start_listener(ai_service: AIService, supabase_service: SupabaseService):
    global listener_thread, stop_event
    if listener_thread is None:
        stop_event = threading.Event()
        listener_thread = Thread(
            target=listen_to_notifications, args=(ai_service, supabase_service))
        listener_thread.daemon = True
        listener_thread.start()


def stop_listener():
    global listener_thread, stop_event
    if listener_thread:
        stop_event.set()
        listener_thread.join()
        listener_thread = None


# listen
def listen_to_notifications(ai_service: AIService, supabase_service: SupabaseService):
    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute("LISTEN image_face_detection;")
        log_info(
            "Listening for insert events on public.image...")

        while not stop_event.is_set():
            if select.select([conn], [], [], 1) == ([], [], []):
                continue  # timeout, check the stop flag again
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    payload = json.loads(notify.payload)
                    image_id = payload["id"]
                    image_bucket_id = payload["image_bucket_id"]
                    image_name = payload["image_name"]
                    user_id = payload["uploader_id"]

                    image_url = supabase_service.get_image_public_url(
                        image_bucket_id, image_name)

                    face_locations, face_encodings = ai_service.category_image_face(
                        image_url)

                    supabase_service.update_person_table(
                        face_encodings, face_locations, image_id, user_id, image_name)

                    supabase_service.mark_image_done_face_detection(image_id)
                    log_info(f"Face detection done for image: {image_name}")

                except RuntimeError as e:
                    log_error(
                        f"Error categorize image: {e}\n{traceback.format_exc()}")

    except Exception as e:
        log_error(f"Database listener error: {e}\n{traceback.format_exc()}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            log_info("Database listener stopped.")

# payload = json.loads(notify.payload)
# image_id = payload["id"]
# image_bucket_id = payload["image_bucket_id"]
# image_name = payload["image_name"]
# labels = payload["labels"]
# log_info(f"[DB] Received image from: {image_name}")

# # labels null / not null -> flag to distinguish the process
# # db listener -> only new image insert to the db without labels
# # endpoint -> classify image with labels not null -> update image labels
# if labels is None:
#     # add image_id to the stream
#     redis_service.push_to_stream(
#         'image_label_stream',
#         {
#             'image_id': image_id,
#             'image_bucket_id': image_bucket_id,
#             'image_name': image_name,
#         }
#     )
