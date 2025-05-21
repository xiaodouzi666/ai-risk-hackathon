from pydantic import BaseModel, Field
from typing import List, Dict, Literal

DIM = Literal["Fairness", "Robustness", "Privacy", "Transparency"]

class EvalRequest(BaseModel):
    model_url: str = Field(..., example="https://example.com/model")
    dims: List[DIM] = Field(default_factory=lambda: ["Fairness", "Robustness", "Privacy", "Transparency"])

class DimResult(BaseModel):
    score: int            # 0–100
    summary: str
    issues: List[Dict]

class EvalResponse(BaseModel):
    results: Dict[str, DimResult]