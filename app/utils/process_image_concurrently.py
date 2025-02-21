from concurrent.futures import ThreadPoolExecutor

from app.libs.logger.log import log_error
from app.services.ai_services import AIService


def process_image_concurrently(ai_service: AIService, image_bucket_id, image_name, image_url):
    # with ThreadPoolExecutor() as executor:
    #     future_classify = executor.submit(
    #         ai_service.classify_image, image_bucket_id, image_name, None)
    #     future_description = executor.submit(
    #         ai_service.generate_image_description, image_url)

    #     image_labels, image_features = future_classify.result()
    #     description = future_description.result()

    #     return image_labels, image_features, description
    image_labels, image_features = ai_service.classify_image(image_bucket_id, image_name, None);
     
    return image_labels, image_features, None
