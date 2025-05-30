# app/services/image_service.py
import io, asyncio
from PIL import Image
from .image_moderation import moderate_image
from .beauty_detection import detect_makeup

async def evaluate_image(image_bytes: bytes) -> dict:
    """
    - moderate_image:  {'score': int, 'issues': [str,...]}
    - detect_makeup:  {'type': str, 'tone': str, 'issues': [...]}
    """
    loop = asyncio.get_event_loop()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    safety_task = loop.run_in_executor(None, moderate_image, img)
    beauty_task = loop.run_in_executor(None, detect_makeup, img)

    safety, beauty = await asyncio.gather(safety_task, beauty_task)
    return {"safety": safety, "beauty": beauty}