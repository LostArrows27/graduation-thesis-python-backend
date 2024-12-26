import open_clip
from app.models.config import CONFIG


class AIModel:
    def __init__(self):
        self.device = CONFIG["device"]
        print(f"Using device: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'convnext_base', pretrained='laion400m_s13b_b51k')
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
