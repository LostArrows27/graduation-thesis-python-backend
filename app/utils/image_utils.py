import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException:
        raise RuntimeError(f"Failed to download image from URL: {url}")
    except UnidentifiedImageError:
        raise RuntimeError(f"Failed to identify image from URL: {url}")


def load_image_file_from_url(url: str) -> BytesIO:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.RequestException:
        raise RuntimeError(f"Failed to download image from URL: {url}")
