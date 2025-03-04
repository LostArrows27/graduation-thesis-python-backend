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


class FaceCategoryModel:
    def __init__(self):
        if dlib.DLIB_USE_CUDA:
            log_info("DLIB is using CUDA")
        else:
            log_info("DLIB is using CPU")

        self.model = face_recognition

        # warm up model
        try:
            start = time.time()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image = self.model.load_image_file(
                os.path.join(current_dir, "../utils/image/warm_up.jpg"))

            face_locations = self.model.face_locations(image, model="cnn")

            if face_locations.__len__() == 0:
                raise Exception("No face found in the image")
        except Exception as e:
            log_error(e)
            log_error(
                f"Error loading face_regconition model: {e}\n{traceback.format_exc()}")
            raise Exception(e)
        finally:
            end = time.time()
            log_info(
                f"Face recognition model warm up time: {end - start} seconds")

    def category_image(self, image_file: BytesIO):
        try:
            image = self.model.load_image_file(image_file)
            face_locations = self.model.face_locations(image, model="cnn")
            face_encodings = self.model.face_encodings(
                image, face_locations, model="large")
            return face_locations, face_encodings
        except Exception as e:
            log_error(e)
            log_error(
                f"Error category face: {e}\n{traceback.format_exc()}")
            raise Exception(e)
