import asyncio
import json
import threading
from app.libs.logger.log import log_error, log_info
from app.services.ai_services import AIService
from app.services.redis_service import RedisService
from app.services.supabase_service import SupabaseService
import traceback

from app.utils.process_image_concurrently import process_image_concurrently

thread = None
face_thread = None

# function run 1 time on startup
# checking unlabeled images in table -> process


def process_person_images(ai_service: AIService, supabase_service: SupabaseService):
    try: 
        uncategory_face_images = supabase_service.client.table('image').select('*').is_('is_face_detection', False).execute().data
        
        log_info(f"Found {len(uncategory_face_images)} uncategory face images !")
        
        for image in uncategory_face_images:
            image_id = image['id']
            image_bucket_id = image['image_bucket_id']
            image_name = image['image_name']
            user_id = image['uploader_id']
            
            image_url = supabase_service.get_image_public_url(
                        image_bucket_id, image_name)
            
            face_locations, face_encodings = ai_service.category_image_face(image_url)
            
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

            image_url = ai_service.inference_service.supabase_service.get_image_public_url(
                image_bucket_id, image_name)

            # update redis label job -> processing
            redis_service.update_image_label_job(
                image_id, image_bucket_id, image_name
            )

            image_labels, image_features = process_image_concurrently(
                ai_service, image_bucket_id, image_name, image_url)

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


def start_background_processor(ai_service: AIService, supabase_service: SupabaseService, redis_service: RedisService):
    thread = threading.Thread(
        target=process_unlabeled_images, args=(ai_service, supabase_service, redis_service))
    thread.daemon = True
    thread.start()
    
    face_thread = threading.Thread(
        target=process_person_images, args=(ai_service, supabase_service))
    face_thread.daemon = True
    face_thread.start()
