# app/services/fairness.py
"""
Fairness evaluation module
~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates group‑based performance gaps (currently accuracy) and maps the worst
inter‑group gap to a 0‑100 score.  Works in black‑box mode by calling an arbitrary
remote prediction API through a *RestAPIAdapter* abstraction.

Key features
------------
* **Black‑box by default** – only requires a REST endpoint that returns labels or
  class‑probabilities.
* **Optional automatic attribute inference** – if the input data do not carry
  `sex` / `skin_tone` columns and `auto_attr=True`, a lightweight CLIP+MLP model
  (PyTorch) is loaded to infer those attributes.  If torch is not installed or
  GPU is unavailable, the code falls back to CPU.
* **Configurable scoring rule** – worst‑case accuracy gap of ≥20 pp ⇒ 0 points;
  perfectly equal ⇒ 100 points.

Returned JSON structure
-----------------------
```
{
  "score": 78,
  "summary": "Overall acc 0.82, worst gap 0.07 (dark_female)",
  "issues": [
      {"group": "dark_female", "acc": 0.71, "overall": 0.82,
       "gap": 0.11, "severity": "high"}
  ]
}
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame

# ---- internal imports -------------------------------------------------------
from .data_utils import load_dataset, infer_attributes_with_clip  # you will add
from .openai_adapter import OpenAIAdapter as RestAPIAdapter

PROTECTED_COLS = ["sex", "skin_tone"]
GAP_CAP = 0.20        # gap >= 0.20 ⇒ score 0
GAP_THRESHOLD = 0.05  # gap > 0.05 ⇒ mark as issue

# -----------------------------------------------------------------------------

def _metricframe_summary(y_true: np.ndarray, y_pred: np.ndarray, sens_df: pd.DataFrame) -> Tuple[int, List[Dict[str, Any]], float, float]:
    """Compute overall & group accuracies and map to score.

    Returns
    -------
    fairness_score : int
        0‑100 mapping based on worst accuracy gap.
    issues : list[dict]
        Groups whose gap exceeds *GAP_THRESHOLD*.
    overall : float
        Overall accuracy.
    worst_gap : float
        Largest (overall‑minus‑group) gap.
    """
    groups = sens_df.astype(str).agg("-".join, axis=1)
    mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    overall = float(mf.overall)
    by_group = mf.by_group
    gaps = overall - by_group
    worst_gap = float(gaps.max())

    issues = []
    for g, gap in gaps.items():
        if gap > GAP_THRESHOLD:
            issues.append({
                "group": g,
                "acc": float(by_group[g]),
                "overall": overall,
                "gap": float(gap),
                "severity": "high" if gap > 0.10 else "medium"
            })

    fairness_score = int(max(0, 100 * (1 - min(GAP_CAP, worst_gap) / GAP_CAP)))
    return fairness_score, issues, overall, worst_gap


# Public API ------------------------------------------------------------------

def evaluate_fairness(model_url: str,
                      task: str = "text",
                      *,
                      auto_attr: bool = False,
                      adapter_kwargs: dict | None = None) -> Dict[str, Any]:
    """Run fairness evaluation on *model_url*.

    Parameters
    ----------
    model_url : str
        Base URL or endpoint of the model prediction service.
    task : {"image", "text"}
        Determines which builtin test‑set/adapter is used.
    auto_attr : bool, default False
        If **True** and the dataset lacks sensitive columns, load a tiny CLIP
        classifier to infer `sex` & `skin_tone` on the fly.  Requires torch.
    adapter_kwargs : dict, optional
        Extra kwargs forwarded to ``RestAPIAdapter`` (e.g. auth headers).
    """
    # 1) load data ------------------------------------------------------------
    X, y_true, sens_df = load_dataset(task, protected_cols=PROTECTED_COLS)
    if sens_df.empty and auto_attr:
        sens_df = infer_attributes_with_clip(X)  # returns DataFrame

    # 2) predictions via user API --------------------------------------------
    adapter = RestAPIAdapter(model_url=model_url, task=task, **(adapter_kwargs or {}))
    y_pred = adapter.batch_predict(X)  # <- generic: returns list[int|str]

    # 3) score and issues -----------------------------------------------------
    score, issues, overall, worst_gap = _metricframe_summary(np.asarray(y_true), np.asarray(y_pred), sens_df)
    summary = f"Overall acc {overall:.2f}, worst gap {worst_gap:.2f}"

    result = {"score": score, "summary": summary, "issues": issues}
    return result
