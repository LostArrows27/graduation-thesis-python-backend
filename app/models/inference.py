import torch
from collections import defaultdict
from PIL import Image
from app.models.preprocess import load_features_parallel, load_labels_parallel, read_grouped_items, load_filter_items
from app.models.config import CONFIG


class AIInferenceService:
    def __init__(self, model):
        self.model = model

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

    def is_relate_image(self, image_path):
        image = self.model.preprocess(Image.open(
            image_path)).unsqueeze(0).to(CONFIG["device"])
        with torch.no_grad():
            image_features = self.model.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            relate_probs = (100.0 * image_features @
                            self.location_filter_text_features.T).softmax(dim=-1)
        _, top_index = torch.topk(relate_probs[0], 1)
        is_relate = self.location_filter_labels[top_index.item()]["is_relate"]
        return is_relate, image_features if is_relate else None

    def get_top_labels(self, labels, text_features, image_features):
        with torch.no_grad():
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            labels_probs, labels_indices = torch.topk(probs[0], 2)
            return [(labels[labels_indices[i].item()], labels_probs[i].item()) for i in range(2)]

    def classify_images(self, image_paths):
        results = defaultdict(list)
        for image_path in image_paths:
            is_relate, image_features = self.is_relate_image(image_path)
            if is_relate:
                location_labels = self.get_top_labels(
                    self.location_labels, self.location_text_features, image_features)
                action_labels = self.get_top_labels(
                    self.action_labels, self.action_text_features, image_features)
                event_labels = self.get_top_labels(
                    self.event_labels, self.event_text_features, image_features)
                results[image_path] = {
                    "location_labels": location_labels,
                    "action_labels": action_labels,
                    "event_labels": event_labels,
                }
            else:
                results[image_path] = {
                    "location_labels": [],
                    "action_labels": [],
                    "event_labels": [],
                }
        return results
