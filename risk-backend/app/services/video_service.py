# app/services/video_service.py
import cv2, tempfile, os, asyncio
from .image_service import evaluate_image

async def evaluate_video(video_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps)
    results = []

    loop = asyncio.get_event_loop()
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if i % interval == 0:
            _, buf = cv2.imencode(".jpg", frame)
            task = loop.create_task(evaluate_image(buf.tobytes()))
            results.append(task)
        i += 1

    await asyncio.gather(*results)
    cap.release()
    os.unlink(path)

    safeties = [r.result()["safety"]["score"] for r in results]
    beauties = [r.result()["beauty"] for r in results]
    avg_safety = sum(safeties) / len(safeties) if safeties else 0

    from collections import Counter
    tone = Counter([b["tone"] for b in beauties]).most_common(1)[0][0]
    typ  = Counter([b["type"] for b in beauties]).most_common(1)[0][0]

    return {
        "avg_safety": avg_safety,
        "beauty": {"type": typ, "tone": tone},
        "frames_analyzed": len(results),
    }