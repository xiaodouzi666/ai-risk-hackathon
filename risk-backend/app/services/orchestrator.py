from .fairness import evaluate_fairness
from .robustness import evaluate_robustness
from .privacy import evaluate_privacy
from .transparency import evaluate_transparency

FUNC_MAP = {
    "Fairness": evaluate_fairness,
    "Robustness": evaluate_robustness,
    "Privacy": evaluate_privacy,
    "Transparency": evaluate_transparency,
}

def run_evaluation(model_url: str, dims: list[str]):
    out = {}
    for d in dims:
        fn = FUNC_MAP[d]
        out[d] = fn(model_url)  # 返回 {"score":…, "summary":…, "issues":[…]}
    return out
