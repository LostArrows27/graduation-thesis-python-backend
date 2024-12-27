from app.models.model import AIModel
from app.models.inference import AIInferenceService
from app.services.supabase_service import SupabaseService
from app.services.redis_service import RedisService


class AIService:
    def __init__(self, supabase_service: SupabaseService, redis_service=None):
        self.model = AIModel()
        self.inference_service = AIInferenceService(
            self.model, supabase_service)

    def classify_image(self, image_bucket_id: str, image_name: str, image_id: str):
        return self.inference_service.classify_image(image_bucket_id, image_name, image_id)

    def update_image_labels(self, image_id: str, labels: dict):
        response_data = self.inference_service.supabase_service.update_image_labels(
            image_id, labels)

        return response_data


def get_ai_service(supabase_service: SupabaseService):
    return AIService(supabase_service)
