from supabase import create_client, Client
from app.core.config import settings


# NOTE: add more methods later
class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url, settings.supabase_key)

    def get_image_metadata(self, image_id: str):
        response = self.client.table('image_meta_data').select(
            '*').eq('id', image_id).execute()
        return response.data

    def update_image_labels(self, image_id: str, labels: dict):
        response = self.client.table('image_meta_data').update(
            labels).eq('id', image_id).execute()
        return response.data

    def get_image_public_url(self, image_bucket_id: str, image_name: str):
        return self.client.storage.from_(image_bucket_id).get_public_url(image_name)


def get_supabase_service():
    return SupabaseService()
