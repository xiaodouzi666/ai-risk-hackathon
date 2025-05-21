# app/services/robustness.py
"""
Robustness MVP – zero-dependency version
---------------------------------------
* Text task only.
* Randomly masks 20 % of tokens with '*' to simulate typos/occlusion.
* Score = 100 * (acc_adv / acc_clean).
"""

from __future__ import annotations
import random, re
import numpy as np
from sklearn.metrics import accuracy_score

from .data_utils import load_dataset
from .openai_adapter import OpenAIAdapter as RestAPIAdapter  # or generic adapter


_MASK_TOKEN = "*"
_MASK_PCT   = 0.2          # 20 % of word tokens


def _random_mask(text: str, pct: float = _MASK_PCT) -> str:
    """Return text with `pct` of word tokens replaced by _MASK_TOKEN."""
    # split by non-word characters, keep delimiters
    tokens = re.split(r"(\W+)", text)
    word_idx = [i for i in range(0, len(tokens), 2) if tokens[i].strip()]
    k = max(1, int(len(word_idx) * pct))
    for i in random.sample(word_idx, k):
        tokens[i] = _MASK_TOKEN
    return "".join(tokens)


# --------------------------------------------------------------------------- #
def evaluate_robustness(model_url: str,
                        *,
                        task: str = "text",
                        adapter_kwargs: dict | None = None) -> dict:
    """Minimal robustness evaluation for text models."""
    if task != "text":
        raise NotImplementedError("Robustness MVP supports text task only.")

    # 1) load data
    X, y_true, _ = load_dataset("text")

    # 2) generate adversarial (masked) texts
    X_adv = [_random_mask(x) for x in X]

    # 3) prediction via adapter
    adapter = RestAPIAdapter(model_url=model_url, task="text",
                             **(adapter_kwargs or {}))
    y_pred_clean = adapter.batch_predict(X)
    y_pred_adv   = adapter.batch_predict(X_adv)

    acc_clean = accuracy_score(y_true, y_pred_clean)
    acc_adv   = accuracy_score(y_true, y_pred_adv)

    ratio = acc_adv / max(acc_clean, 1e-3)

    score = int(min(1.0, ratio) * 100)  
    summary = f"Acc clean {acc_clean:.2f} → adv {acc_adv:.2f}"

    issues: list[dict] = []        # 没有高风险样本就保持空列表

    result = {"score": score, "summary": summary, "issues": issues}

    return result
