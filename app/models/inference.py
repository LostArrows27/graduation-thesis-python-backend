from io import BytesIO
import traceback
import torch
from collections import defaultdict
from app.libs.logger.log import log_error, log_info
from app.models.model import FaceCategoryModel
from app.models.preprocess import load_features_parallel, load_labels_parallel, read_grouped_items, load_filter_items
from app.models.config import CONFIG
from app.services.supabase_service import SupabaseService
from app.utils.image_utils import load_image_file_from_url, load_image_from_url


class AIInferenceService:
    def __init__(self, model, supabase_service: SupabaseService, face_model: FaceCategoryModel):
        self.model = model
        self.face_model = face_model
        self.supabase_service = supabase_service

        # load text features
        (
            self.location_filter_text_features,
            self.location_text_features,
            self.action_text_features,
            self.event_text_features,
        ) = load_features_parallel(CONFIG)

        # load labels
        (
            self.location_labels,
            self.action_labels,
            self.event_labels,
        ) = load_labels_parallel(CONFIG)
        self.location_filter_labels = load_filter_items(
            CONFIG["labels"]["location_filter"])
        self.location_group_labels = read_grouped_items(
            CONFIG["labels"]["location_group"])

    # face model
    def category_face(self, image_url: str):
        try:
            image_file = load_image_file_from_url(image_url)
            face_locations, face_encoding = self.face_model.category_image(
                image_file)
            return face_locations, face_encoding
        except Exception as e:
            log_error(
                f"Error category face: {e}\n{traceback.format_exc()}")

    # image_label model

    def is_relate_image(self, image_url: str):
        try:
            image = self.model.preprocess(load_image_from_url(
                image_url)).unsqueeze(0)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                image_features = self.model.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                relate_probs = (100.0 * image_features @
                                self.location_filter_text_features.T).softmax(dim=-1)
            _, top_index = torch.topk(relate_probs[0], 1)
            is_relate = self.location_filter_labels[top_index.item(
            )]["is_relate"]
            return is_relate, image_features
        except RuntimeError as e:
            raise RuntimeError(f"Error in is_relate_image: {e}")

    def get_top_labels(self, labels, text_features, image_features):
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                probs = (100.0 * image_features @
                         text_features.T).softmax(dim=-1)
                labels_probs, labels_indices = torch.topk(probs[0], 2)
                return [{labels[labels_indices[i].item()]: labels_probs[i].item()}for i in range(2)]
        except Exception as e:
            raise RuntimeError(f"Error in get_top_labels: {e}")

    def classify_image(self, image_bucket_id: str, image_name: str, image_id: str):
        results = defaultdict(list)
        try:
            # use supabase service to get the image_url
            image_url = self.supabase_service.get_image_public_url(
                image_bucket_id, image_name)

            log_info(f"Classifying image: {image_name}")

            is_relate, image_features = self.is_relate_image(image_url)
            if is_relate:
                location_labels = self.get_top_labels(
                    self.location_labels, self.location_text_features, image_features)
                action_labels = self.get_top_labels(
                    self.action_labels, self.action_text_features, image_features)
                event_labels = self.get_top_labels(
                    self.event_labels, self.event_text_features, image_features)
                results = {
                    "location_labels": location_labels,
                    "action_labels": action_labels,
                    "event_labels": event_labels,
                }
            else:
                results = {
                    "location_labels": [],
                    "action_labels": [],
                    "event_labels": [],
                }

            return results, image_features
        except RuntimeError as e:
            raise RuntimeError(f"Error in classify_image: {e}")
