import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    except requests.RequestException:
        raise RuntimeError(f"Failed to download image from URL: {url}")


def load_image_file_from_url(url: str) -> BytesIO:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        # Validate it's an actual image
        Image.open(image_data).verify()
        # Reset file pointer after verification
        image_data.seek(0)
        return image_data
    except requests.RequestException:
        raise RuntimeError(f"Failed to download image from URL: {url}")
