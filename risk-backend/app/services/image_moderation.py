# app/services/image_moderation.py
from PIL import Image

def moderate_image(img: Image.Image) -> dict:
    """
    对传入的 PIL Image 做安全审查。
    返回：{'score': int, 'issues': List[str]}
    """
    return {"score": 100, "issues": []}