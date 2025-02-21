import asyncio
import threading
from app.libs.logger.log import log_error, log_info
from app.services.ai_services import AIService
from app.services.supabase_service import SupabaseService
import traceback

from app.utils.process_image_concurrently import process_image_concurrently

thread = None

# function run 1 time on startup
# checking unlabeled images in table -> process


def process_unlabeled_images(ai_service: AIService, supabase_service: SupabaseService):
    try:
        unlabeled_images = supabase_service.client.table(
            'image').select('*').is_('labels', None).execute().data

        log_info(f"Found {len(unlabeled_images)} unlabeled images !")

        for image in unlabeled_images:
            image_bucket_id = image['image_bucket_id']
            image_name = image['image_name']

            log_info(f"Processing unlabeled image {image_name}")

            image_url = ai_service.inference_service.supabase_service.get_image_public_url(
                image_bucket_id, image_name)

            # Get labels, features, and description concurrently
            image_labels, image_features, description = process_image_concurrently(
                ai_service, image_bucket_id, image_name, image_url)

            # Save labels, features, and description to Supabase
            supabase_service: SupabaseService = ai_service.inference_service.supabase_service
            image_row = supabase_service.save_image_features_and_labels(
                image_bucket_id, image_name, image_labels, image_features.squeeze(0).tolist(), description)

            if image_row:
                log_info(f"Labels for image {image_name} updated successfully")
            else:
                raise Exception(
                    f"Error updating labels for image {image_name}")

    except Exception as e:
        log_error(
            f"Error processing unlabeled images: {e}\n{traceback.format_exc()}")


def start_background_processor(ai_service: AIService, supabase_service: SupabaseService):
    thread = threading.Thread(
        target=process_unlabeled_images, args=(ai_service, supabase_service))
    thread.daemon = True
    thread.start()
