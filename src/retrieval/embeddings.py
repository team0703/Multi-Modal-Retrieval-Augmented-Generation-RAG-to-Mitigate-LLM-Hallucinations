import torch
from PIL import Image


def embed_image(image_path, clip_model, clip_processor, device):
    """Embeds a single image using CLIP. Returns a list of floats."""
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding[0].cpu().tolist()


def embed_text_clip(text, clip_model, clip_processor, device):
    """Embeds text using CLIP's text tower (for cross-modal search against image vectors)."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    return embedding[0].cpu().tolist()


def embed_text_dense(text, text_model):
    """Embeds text using the dedicated text-dense model (MiniLM)."""
    return text_model.encode(text).tolist()
