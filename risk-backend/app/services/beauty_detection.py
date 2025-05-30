# app/services/beauty_detection.py
from PIL import Image

def detect_makeup(img: Image.Image) -> dict:
    """
    对传入的 PIL Image 做美妆检测。
    返回：{'type': str, 'tone': str, 'issues': List[str]}
    """
    return {"type": "unknown", "tone": "neutral", "issues": []}