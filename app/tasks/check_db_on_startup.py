import asyncio
import json
import threading
import gc
from app.libs.logger.log import log_error, log_info
from app.services.ai_services import AIService
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
import traceback

from app.utils.process_image_concurrently import process_image_concurrently

# Global thread variable for the coordinator
coordinator_thread = None

# function run 1 time on startup
# checking unlabeled images in table -> process


def process_person_images(ai_service: AIService, supabase_service: SupabaseService):
    try:
        uncategory_face_images = supabase_service.client.table(
            'image').select('*').is_('is_face_detection', False).execute().data

        log_info(
            f"Found {len(uncategory_face_images)} uncategory face images !")

        for image in uncategory_face_images:
            image_id = image['id']
            image_bucket_id = image['image_bucket_id']
            image_name = image['image_name']
            user_id = image['uploader_id']

            image_url = supabase_service.get_image_public_url(
                image_bucket_id, image_name)

            face_locations, face_encodings = ai_service.category_image_face(
                image_url)

            supabase_service.update_person_table(
                face_encodings, face_locations, image_id, user_id, image_name)

            supabase_service.mark_image_done_face_detection(image_id)

            log_info(f"Face detection done for image: {image_name}")
    except Exception as e:
        log_error(
            f"Error category images face: {e}\n{traceback.format_exc()}")


def process_unlabeled_images(ai_service: AIService, supabase_service: SupabaseService, redis_service: RedisService):
    try:
        unlabeled_images = supabase_service.client.table(
            'image').select('*').is_('labels', None).execute().data

        log_info(f"Found {len(unlabeled_images)} unlabeled images !")

        for image in unlabeled_images:
            image_id = image['id']
            image_bucket_id = image['image_bucket_id']
            image_name = image['image_name']

            log_info(f"Processing unlabeled image {image_name}")

            # update redis label job -> processing
            redis_service.update_image_label_job(
                image_id, image_bucket_id, image_name
            )

            image_labels, image_features = process_image_concurrently(
                ai_service, image_bucket_id, image_name)

            # update redis label job -> completed
            redis_service.update_hash(
                f"image_job:{image_id}",
                {
                    "labels": json.dumps(image_labels),
                    "label_status": "completed"
                }
            )

            supabase_service: SupabaseService = ai_service.inference_service.supabase_service
            image_row = supabase_service.save_image_features_and_labels(
                image_bucket_id, image_name, image_labels, image_features.squeeze(0).tolist())

            if image_row:
                log_info(f"Labels for image {image_name} updated successfully")
            else:
                raise Exception(
                    f"Error updating labels for image {image_name}")

    except Exception as e:
        log_error(
            f"Error processing unlabeled images: {e}\n{traceback.format_exc()}")

# Sequential processing function that runs both tasks one after another


def sequential_processor(ai_service: AIService, supabase_service: SupabaseService, redis_service: RedisService):
    log_info("Starting sequential background processing")

    try:
        # First, process unlabeled images
        log_info("Starting image labeling task")
        process_unlabeled_images(ai_service, supabase_service, redis_service)
        log_info("Image labeling task completed - cleaning up resources")
        # Explicit garbage collection after image processing
        gc.collect()

        # Then, process face detection
        log_info("Starting face detection task")
        process_person_images(ai_service, supabase_service)
        log_info("Face detection task completed - cleaning up resources")
        # Explicit garbage collection after face processing
        gc.collect()

    except Exception as e:
        log_error(
            f"Error in sequential processor: {e}\n{traceback.format_exc()}")
    finally:
        log_info("Sequential background processing completed")

# Start the background processor with sequential execution


def start_background_processor(ai_service: AIService, supabase_service: SupabaseService, redis_service: RedisService):
    global coordinator_thread

    # Create and start the coordinator thread
    coordinator_thread = threading.Thread(
        target=sequential_processor,
        args=(ai_service, supabase_service, redis_service)
    )
    coordinator_thread.daemon = True
    coordinator_thread.start()

    log_info("Background processor started in sequential mode")

# Function to clean up the coordinator thread if needed


def cleanup_background_thread(timeout=5):
    global coordinator_thread

    if coordinator_thread and coordinator_thread.is_alive():
        log_info("Waiting for background processor to complete...")
        coordinator_thread.join(timeout=timeout)

        if coordinator_thread.is_alive():
            log_error(
                "Background processor did not complete within timeout period")
            return False
        else:
            log_info("Background processor completed successfully")

    # Clear the thread reference
    coordinator_thread = None
    gc.collect()

    return True
