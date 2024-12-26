import os
import torch
from concurrent.futures import ThreadPoolExecutor


def load_text_features(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return torch.load(file_path)


def load_labels(label_file_path):
    with open(label_file_path, "r") as file:
        return [line.strip() for line in file]


def read_grouped_items(filename):
    categories = {}
    current_category = None
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*") and line.endswith("*"):
                current_category = line[1:-1]
                categories[current_category] = []
            elif current_category is not None:
                categories[current_category].append(line)
    return categories


def load_filter_items(filename):
    categories = []
    current_category = "relate"
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*"):
                current_category = "non_relate"
                continue
            categories.append(
                {"name": line, "is_relate": current_category == "relate"})
    return categories


def load_features_parallel(config):
    with ThreadPoolExecutor() as executor:
        future_location_filter = executor.submit(
            load_text_features, config["features"]["location_filter"])
        future_location = executor.submit(
            load_text_features, config["features"]["location"])
        future_action = executor.submit(
            load_text_features, config["features"]["action"])
        future_event = executor.submit(
            load_text_features, config["features"]["event"])

        return (
            future_location_filter.result(),
            future_location.result(),
            future_action.result(),
            future_event.result(),
        )


def load_labels_parallel(config):
    with ThreadPoolExecutor() as executor:
        future_location_labels = executor.submit(
            load_labels, config["labels"]["location_label"])
        future_action_labels = executor.submit(
            load_labels, config["labels"]["action_label"])
        future_event_labels = executor.submit(
            load_labels, config["labels"]["event_label"])

        return (
            future_location_labels.result(),
            future_action_labels.result(),
            future_event_labels.result(),
        )
