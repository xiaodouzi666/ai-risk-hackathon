from .fairness import evaluate_fairness
from .robustness import evaluate_robustness
from .privacy   import evaluate_privacy
from .transparency import evaluate_transparency

FUNC_MAP = {
    "Fairness":      evaluate_fairness,
    "Robustness":    evaluate_robustness,
    "Privacy":       evaluate_privacy,
    "Transparency":  evaluate_transparency,
}

def run_evaluation(model_url: str, dims: list[str]):
    results: dict[str, dict] = {}

    for d in dims:
        key = d.capitalize()                # 允许前端传 "fairness" 或 "Fairness"
        fn  = FUNC_MAP[key]
        results[key] = fn(model_url)        # {"score": …, "summary": …, "issues": […]}

    results.update({k.lower(): v for k, v in results.items()})

    return results