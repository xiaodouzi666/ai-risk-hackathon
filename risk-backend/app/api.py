from fastapi import APIRouter, Response, BackgroundTasks
from .schemas import EvalRequest, EvalResponse
from .services.orchestrator import run_evaluation
from .services.report import build_pdf, build_json
from fastapi import UploadFile, File
from .services import image_service, video_service

router = APIRouter()

@router.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    results = run_evaluation(req.model_url, req.dims)
    return {"results": results}

@router.post("/download/pdf")
def download_pdf(req: EvalRequest):
    pdf_bytes = build_pdf(run_evaluation(req.model_url, req.dims), model_url=req.model_url)
    return Response(pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=risk_report.pdf"})

@router.post("/download/json")
def download_json(req: EvalRequest):
    json_bytes = build_json(run_evaluation(req.model_url, req.dims))
    return Response(json_bytes, media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=risk_report.json"})

@router.post("/evaluate/image")
async def evaluate_image_endpoint(file: UploadFile = File(...)):
    """
    接收上传的图像文件，返回美妆场景下的安全评估结果
    """
    image_bytes = await file.read()
    result = await image_service.evaluate_image(image_bytes)
    return {"results": result}

@router.post("/evaluate/video")
async def evaluate_video_endpoint(file: UploadFile = File(...)):
    """
    接收上传的视频文件，返回美妆场景下的安全评估结果
    """
    video_bytes = await file.read()
    result = await video_service.evaluate_video(video_bytes)
    return {"results": result}