import open_clip
import torch
from app.models.config import CONFIG


class AIModel:
    def __init__(self):
        self.device = CONFIG["device"]
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
        else:
            print("CUDA is not available. Check your installation.")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'convnext_base', pretrained='laion400m_s13b_b51k')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
