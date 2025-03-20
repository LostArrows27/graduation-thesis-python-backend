import pytest
import os
import torch
from unittest.mock import patch, mock_open, MagicMock
from concurrent.futures import Future

from app.models.preprocess import (
    load_text_features,
    load_labels,
    read_grouped_items,
    load_filter_items,
    load_features_parallel,
    load_labels_parallel
)


@pytest.fixture
def mock_tensor():
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def mock_future():
    future = Future()
    future.set_result(torch.tensor([1.0, 2.0, 3.0]))
    return future


@pytest.fixture
def mock_labels_future():
    future = Future()
    future.set_result(["label1", "label2", "label3"])
    return future


@pytest.fixture
def config_dict():
    return {
        "features": {
            "location_filter": "path/to/location_filter.pt",
            "location": "path/to/location.pt",
            "action": "path/to/action.pt",
            "event": "path/to/event.pt"
        },
        "labels": {
            "location_label": "path/to/location_labels.txt",
            "action_label": "path/to/action_labels.txt",
            "event_label": "path/to/event_labels.txt"
        }
    }


# Tests for load_text_features
def test_load_text_features_file_not_found():
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_text_features("nonexistent_file.pt")
        assert "File not found" in str(excinfo.value)


def test_load_text_features_success(mock_tensor):
    with patch('os.path.exists', return_value=True), \
            patch('torch.load', return_value=mock_tensor):
        result = load_text_features("fake_path.pt")
        assert torch.equal(result, mock_tensor)


# Tests for load_labels
def test_load_labels():
    mock_data = "label1\nlabel2\nlabel3"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_labels("fake_labels.txt")
        assert result == ["label1", "label2", "label3"]


def test_load_labels_empty_file():
    with patch("builtins.open", mock_open(read_data="")):
        result = load_labels("empty_file.txt")
        assert result == []


def test_load_labels_with_whitespace():
    mock_data = "  label1  \n  label2  \n  label3  "
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_labels("fake_labels.txt")
        assert result == ["label1", "label2", "label3"]


# Tests for read_grouped_items
def test_read_grouped_items_empty_file():
    with patch("builtins.open", mock_open(read_data="")):
        result = read_grouped_items("empty_file.txt")
        assert result == {}


def test_read_grouped_items_with_categories():
    mock_data = "*Category1*\nitem1\nitem2\n\n*Category2*\nitem3\nitem4"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = read_grouped_items("fake_file.txt")
        assert result == {
            "Category1": ["item1", "item2"],
            "Category2": ["item3", "item4"]
        }


def test_read_grouped_items_with_empty_categories():
    mock_data = "*Category1*\n*Category2*\nitem1\nitem2"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = read_grouped_items("fake_file.txt")
        assert result == {
            "Category1": [],
            "Category2": ["item1", "item2"]
        }


# Tests for load_filter_items
def test_load_filter_items_empty_file():
    with patch("builtins.open", mock_open(read_data="")):
        result = load_filter_items("empty_file.txt")
        assert result == []


def test_load_filter_items_only_related():
    mock_data = "item1\nitem2\nitem3"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_filter_items("fake_file.txt")
        assert result == [
            {"name": "item1", "is_relate": True},
            {"name": "item2", "is_relate": True},
            {"name": "item3", "is_relate": True}
        ]


def test_load_filter_items_with_non_related():
    mock_data = "item1\nitem2\n*\nitem3\nitem4"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_filter_items("fake_file.txt")
        assert result == [
            {"name": "item1", "is_relate": True},
            {"name": "item2", "is_relate": True},
            {"name": "item3", "is_relate": False},
            {"name": "item4", "is_relate": False}
        ]


def test_load_filter_items_with_blank_lines():
    mock_data = "item1\n\nitem2\n*\n\nitem3\n\nitem4"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = load_filter_items("fake_file.txt")
        assert result == [
            {"name": "item1", "is_relate": True},
            {"name": "item2", "is_relate": True},
            {"name": "item3", "is_relate": False},
            {"name": "item4", "is_relate": False}
        ]


def test_load_features_parallel(config_dict, mock_future):
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.return_value = mock_future

    # Get the full import path that's actually used in preprocess.py
    with patch('app.models.preprocess.ThreadPoolExecutor', return_value=mock_executor):
        result = load_features_parallel(config_dict)

        # Check that submit was called 4 times
        assert mock_executor.__enter__.return_value.submit.call_count == 4

        # Check that we got 4 tensors back
        assert len(result) == 4
        for tensor in result:
            assert torch.is_tensor(tensor)


def test_load_features_parallel_error_handling(config_dict):
    # Create a future that will raise an exception
    error_future = Future()
    error_future.set_exception(FileNotFoundError("Test exception"))

    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.return_value = error_future

    with patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_executor):
        with pytest.raises(FileNotFoundError):
            load_features_parallel(config_dict)


def test_load_labels_parallel(config_dict, mock_labels_future):
    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.return_value = mock_labels_future

    # Change this to patch ThreadPoolExecutor where it's imported in preprocess.py
    with patch('app.models.preprocess.ThreadPoolExecutor', return_value=mock_executor):
        result = load_labels_parallel(config_dict)

        # Check that submit was called 3 times (for each label set)
        assert mock_executor.__enter__.return_value.submit.call_count == 3

        # Check that we got 3 label lists back
        assert len(result) == 3
        for labels in result:
            assert isinstance(labels, list)
            assert len(labels) == 3  # From our mock_labels_future


def test_load_labels_parallel_error_handling(config_dict):
    # Create a future that will raise an exception
    error_future = Future()
    error_future.set_exception(FileNotFoundError("Test exception"))

    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.return_value = error_future

    with patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_executor):
        with pytest.raises(FileNotFoundError):
            load_labels_parallel(config_dict)
