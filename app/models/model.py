from io import BytesIO
import traceback
import open_clip
import torch
from app.libs.logger.log import log_error, log_info
from app.models.config import CONFIG
import time
import face_recognition
import dlib
import os
from PIL import Image, ImageDraw
import datetime
import os


class AIModel:
    def __init__(self):
        self.device = CONFIG["device"]
        log_info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_info(f"CUDA Device: {torch.cuda.get_device_name()}")
        else:
            log_info("CUDA is not available. Check your installation.")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'convnext_base', pretrained='laion400m_s13b_b51k')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        log_info("CLIP mode done loading")

    def get_text_features(self, text: str):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            tokenizer_text = self.tokenizer(text)
            text_features = self.model.encode_text(tokenizer_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


class FaceCategoryModel:
    def __init__(self):
        if dlib.DLIB_USE_CUDA:
            log_info("DLIB is using CUDA")
        else:
            log_info("DLIB is using CPU")

        self.model = face_recognition

        # warm up model
        # try:
        #     start = time.time()
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     image = self.model.load_image_file(
        #         os.path.join(current_dir, "../utils/image/warm_up.jpg"))

        #     face_locations = self.model.face_locations(image, model="cnn")

        #     if face_locations.__len__() == 0:
        #         raise Exception("No face found in the image")
        # except Exception as e:
        #     log_error(e)
        #     log_error(
        #         f"Error loading face_regconition model: {e}\n{traceback.format_exc()}")
        #     raise Exception(e)
        # finally:
        #     end = time.time()
        #     log_info(
        #         f"Face recognition model warm up time: {end - start} seconds")

    def category_image(self, image_file: BytesIO):
        try:
            # # Save the current position in the BytesIO object
            # if hasattr(image_file, 'seek') and hasattr(image_file, 'tell'):
            #     pos = image_file.tell()
            #     image_file.seek(0)

            # Use the improved image loader
            image, ratio = load_image_file(image_file)
            log_info(f"Image loaded and processed with resize ratio: {ratio}")

            # Face detection
            face_locations = self.model.face_locations(
                image, model="cnn", number_of_times_to_upsample=2)
            log_info(f"Found {len(face_locations)} faces in image")

            # Face encoding
            face_encodings = self.model.face_encodings(
                image, face_locations, model="large")

            # If ratio was applied, scale face locations back to original dimensions
            if ratio > 0 and ratio != 1:
                if ratio < 1:  # We scaled down
                    inverse_ratio = 1/ratio
                else:  # We divided by ratio
                    inverse_ratio = ratio

                # Scale face locations back to original dimensions
                face_locations = [(int(top*inverse_ratio), int(right*inverse_ratio),
                                   int(bottom*inverse_ratio), int(left*inverse_ratio))
                                  for top, right, bottom, left in face_locations]

            # # If faces were detected, save the image with bounding boxes
            # if face_locations and len(face_locations) > 0:
            #     # Reset file position to beginning for saving
            #     if hasattr(image_file, 'seek'):
            #         image_file.seek(0)
            #     # Save image with face bounding boxes
            #     save_image_with_faces(image_file, face_locations)
            #     # Reset file position
            #     if hasattr(image_file, 'seek'):
            #         image_file.seek(pos)

            return face_locations, face_encodings
        except Exception as e:
            log_error(f"Error in face categorization: {e}")
            log_error(traceback.format_exc())
            raise Exception(e)


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array with smart resizing
    to avoid memory issues with large images.

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' and 'L' (grayscale) are supported.
    :return: tuple of (image contents as numpy array, resize ratio)
    """
    from PIL import Image
    import numpy as np

    try:
        im = Image.open(file)
        width, height = im.size
        w, h = width, height

        log_info(f"Loading image with dimensions: {width}x{height}")

        ratio = -1
        # Determine appropriate resize ratio based on dimensions
        if width > 3600 or height > 3600:
            # Very large images
            if width > height:
                ratio = width / 800
            else:
                ratio = height / 800
        elif 1200 <= width <= 1600 or 1200 <= height <= 1600:
            ratio = 1 / 2
        elif 1600 <= width <= 3600 or 1600 <= height <= 3600:
            ratio = 1 / 3

        if ratio > 0:
            if ratio < 1:
                # Scale down slightly large images
                w = int(width * ratio)
                h = int(height * ratio)
            else:
                # Scale down very large images
                w = int(width / ratio)
                h = int(height / ratio)

            log_info(
                f"Resizing image from {width}x{height} to {w}x{h} (ratio: {ratio:.2f})")

            # Use modern resampling method (LANCZOS replaces deprecated ANTIALIAS)
            im = im.resize((w, h), Image.Resampling.LANCZOS)

        if mode:
            im = im.convert(mode)

        # Convert to numpy array
        return np.array(im), ratio

    except Exception as e:
        log_error(f"Error loading image: {e}")
        log_error(traceback.format_exc())
        raise


def save_image_with_faces(image_file, face_locations, output_dir="detected_faces"):
    """
    Saves the original image with bounding boxes drawn around detected faces

    :param image_file: Original image file (BytesIO or path)
    :param face_locations: List of face locations as (top, right, bottom, left) tuples
    :param output_dir: Directory where images will be saved
    :return: Path to the saved image
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open the original image (not the resized one)
        original_image = Image.open(image_file)

        # Create a draw object
        draw = ImageDraw.Draw(original_image)

        # Draw rectangle around each face
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw bounding box
            draw.rectangle([(left, top), (right, bottom)],
                           outline="red", width=3)

            # Optionally add face number
            draw.text((left, top - 20), f"Face #{i+1}", fill="red")

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Get original filename if possible, otherwise use timestamp
        if hasattr(image_file, 'name'):
            base_name = os.path.splitext(os.path.basename(image_file.name))[0]
        else:
            base_name = "image"

        filename = f"{base_name}-faces-{timestamp}.jpg"
        save_path = os.path.join(output_dir, filename)

        # Save the image
        original_image.save(save_path, "JPEG", quality=95)
        log_info(
            f"Saved image with {len(face_locations)} detected faces to {save_path}")

        return save_path

    except Exception as e:
        log_error(f"Error saving image with face boxes: {e}")
        log_error(traceback.format_exc())
        return None
