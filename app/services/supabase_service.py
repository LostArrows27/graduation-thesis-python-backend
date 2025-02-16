from supabase import create_client, Client
import torch
from app.core.config import settings


# NOTE: add more methods later
class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url, settings.supabase_key)

    def get_image_metadata(self, image_id: str):
        response = self.client.table('image').select(
            '*').eq('id', image_id).execute()
        return response.data

    def update_image_labels(self, image_id: str, labels: dict):
        response = self.client.table('image').update(
            labels).eq('id', image_id).execute()
        return response.data

    def get_image_public_url(self, image_bucket_id: str, image_name: str):
        return self.client.storage.from_(image_bucket_id).get_public_url(image_name)

    def save_image_features_and_labels(self, image_bucket_id: str, image_name: str, labels: dict, image_features: torch.Tensor):
        response = self.client.table('image').update({
            'labels': labels,
            'image_features': image_features
        }).eq('image_bucket_id', image_bucket_id).eq('image_name', image_name).execute()
        return response.data[0]


def get_supabase_service():
    return SupabaseService()
