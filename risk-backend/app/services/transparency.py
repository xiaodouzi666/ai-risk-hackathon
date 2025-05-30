# app/services/transparency.py
import re, numpy as np
from .data_utils import load_dataset
from .openai_adapter import OpenAIAdapter as RestAPIAdapter

_MASK = "<MASK>"
TOP_K = 10                       # 返回前 k 重要词
MAX_TOKENS = 80                  # 截断，控制花费

def _mask_tokenise(text):
    tokens = re.split(r"(\W+)", text)[:MAX_TOKENS]
    idx = [i for i in range(0, len(tokens), 2) if tokens[i].strip()]
    return tokens, idx

def _importance(adapter, text):
    tokens, word_idx = _mask_tokenise(text)

    # baseline
    try:
        full_p = adapter.batch_predict([text], proba=True)[0]
        proba_mode = True
    except TypeError:
        full_lbl = adapter.batch_predict([text])[0]
        proba_mode = False

    scores = []
    for i in word_idx:
        orig = tokens[i]
        tokens[i] = _MASK
        masked = "".join(tokens)
        try:
            p = adapter.batch_predict([masked], proba=True)[0] if proba_mode else None
        except TypeError:
            proba_mode = False
            p = None

        if proba_mode:
            scores.append(full_p - p)
        else:
            masked_lbl = adapter.batch_predict([masked])[0]
            scores.append(1.0 if masked_lbl != full_lbl else 0.0)

        tokens[i] = orig
    return [tokens[i] for i in word_idx], np.array(scores)


def evaluate_transparency(model_url: str,
                          *, task="text", adapter_kwargs=None):
    X, _, _ = load_dataset("text")
    adapter = RestAPIAdapter(model_url, task="text", **(adapter_kwargs or {}))

    tokens, contrib = _importance(adapter, X[0])
    order = np.argsort(-np.abs(contrib))[:TOP_K]
    highlights = [{"token": tokens[i], "score": float(contrib[i])} for i in order]

    raw = 100 * (1 - max(0.0, contrib.mean()))
    score = int(max(0, min(100, raw)))
    summary = f"Top tokens: {', '.join(t['token'] for t in highlights[:5])}"
    result = {"score": score, "summary": summary, "highlights": highlights, "issues": []}
    return result