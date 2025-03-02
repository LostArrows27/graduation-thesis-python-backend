from app.services.ai_services import AIService


def process_image_concurrently(ai_service: AIService, image_bucket_id, image_name, image_url):
    image_labels, image_features = ai_service.classify_image(
        image_bucket_id, image_name, None)

    return image_labels, image_features
