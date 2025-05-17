from fastapi import APIRouter, Response, BackgroundTasks
from .schemas import EvalRequest, EvalResponse
from .services.orchestrator import run_evaluation
from .services.report import build_pdf, build_json

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
