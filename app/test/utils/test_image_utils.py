import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import requests

from app.utils.image_utils import load_image_from_url, load_image_file_from_url


@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.content = b'fake image content'
    return mock


def test_load_image_from_url(mock_response):
    # mock response
    with patch('requests.get', return_value=mock_response), \
            patch('PIL.Image.open', return_value=MagicMock(spec=Image.Image)):
        image = load_image_from_url(
            'http://example.com/image.jpg')
        assert image is not None


def test_load_image_from_url_request_exception():
    with patch('requests.get', side_effect=requests.RequestException):
        with pytest.raises(RuntimeError) as excinfo:
            load_image_from_url('http://example.com/image.jpg')
        assert "Failed to download image from URL" in str(excinfo.value)


def test_load_image_file_from_url(mock_response):
    mock_image = MagicMock()

    with patch('requests.get', return_value=mock_response), \
            patch('PIL.Image.open', return_value=mock_image):
        image_file = load_image_file_from_url(
            'https://placehold.co/600x400/EEE/31343C')
        assert isinstance(image_file, BytesIO)
        assert image_file.getvalue() == b'fake image content'


def test_load_image_file_from_url_request_exception():
    with patch('requests.get', side_effect=requests.RequestException):
        with pytest.raises(RuntimeError) as excinfo:
            load_image_file_from_url('http://example.com/image.jpg')
        assert "Failed to download image from URL" in str(excinfo.value)
