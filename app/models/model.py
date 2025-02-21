import traceback
import open_clip
import torch
from app.libs.logger.log import log_error, log_info
from app.models.config import CONFIG
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO


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


class BLIPModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large")
        self.model.eval()
        log_info("BCLIP mode done loading")

    def generate_description(self, image_url: str, text: str = "a photography of"):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = self.processor(image, text, return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=50)
            description = self.processor.decode(
                out[0], skip_special_tokens=True)
            return description
        except Exception as e:
            log_error(
                f"Error generate description: {e}\n{traceback.format_exc()}")
