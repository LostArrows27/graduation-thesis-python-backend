import logging
import threading
from app.services.ai_services import AIService
from app.services.supabase_service import SupabaseService

logging.basicConfig(level=logging.INFO)

thread = None

# function run 1 time on startup
# checking unlabeled images in table -> process


def process_unlabeled_images(ai_service: AIService, supabase_service: SupabaseService):
    try:
        unlabeled_images = supabase_service.client.table(
            'image_meta_data').select('*').is_('labels', None).execute().data

        logging.info(f"Found {len(unlabeled_images)} unlabeled images !")

        for image in unlabeled_images:
            image_id = image['id']
            image_bucket_id = image['image_bucket_id']
            image_name = image['image_name']

            logging.info(f"Processing unlabeled image {image_id}")

            # Classify the image
            image_labels = ai_service.classify_image(
                image_bucket_id, image_name, image_id)

            # Update the image labels
            response_data = ai_service.update_image_labels(
                image_id, image_labels)

            if response_data:
                logging.info(
                    f"Labels for image {image_id} updated successfully.")
            else:
                raise Exception(f"Error updating labels for image {image_id}")

    except Exception as e:
        logging.error(f"Error processing unlabeled images: {e}")


def start_background_processor(ai_service: AIService, supabase_service: SupabaseService):
    thread = threading.Thread(
        target=process_unlabeled_images, args=(ai_service, supabase_service))
    thread.daemon = True
    thread.start()
