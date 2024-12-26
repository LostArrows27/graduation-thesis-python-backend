import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "labels": {
        "location_filter": os.path.join(BASE_DIR, "labels", "location", "location_filter.txt"),
        "location_group": os.path.join(BASE_DIR, "labels", "location", "location_label_with_group.txt"),
        "location_label": os.path.join(BASE_DIR, "labels", "location", "location_label.txt"),
        "action_label": os.path.join(BASE_DIR, "labels", "action", "activity_labels.txt"),
        "event_label": os.path.join(BASE_DIR, "labels", "event", "event_labels.txt"),
    },
    "features": {
        "location_filter": os.path.join(BASE_DIR, "features", "location", "location_filter.pt"),
        "location": os.path.join(BASE_DIR, "features", "location", "location_final.pt"),
        "action": os.path.join(BASE_DIR, "features", "action", "text_features_action.pt"),
        "event": os.path.join(BASE_DIR, "features", "event", "text_features_event.pt"),
    },
}
