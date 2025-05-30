from __future__ import annotations
import os
from typing import Iterable, List
import requests
import numpy as np

TOP_LOGPROBS = 5

class QianwenAdapter:
    def __init__(
        self,
        model_url: str | None = None,
        task: str = "text",
        *,
        system_prompt: str | None = None,
        batch_size: int = 8,
        **_,
    ):
        self.api_key = os.getenv("QIANWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("QIANWEN_API_KEY not set")

        self.api_url = (model_url or os.getenv("QIANWEN_API_URL", "")).strip()
        if not self.api_url:
            raise RuntimeError("QIANWEN_API_URL not set")

        self.system_prompt = system_prompt or "你是一个分类模型，只返回 0 或 1。"

        self.task = task
        self.batch_size = batch_size
        self.supports_grad = False

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _classify_batch(self, prompts: List[str], *, proba: bool = False) -> list:
        """
        proba=True → 返回正例概率（float 0–1），否则返回离散 0/1 标签
        """
        outputs: list[float|int] = []

        for prompt in prompts:
            payload = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 3,
            }
            if proba:
                payload["logprobs"] = TOP_LOGPROBS

            resp = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            text = choice["message"]["content"].strip()

            if proba:
                logprobs = choice["logprobs"]["token_logprobs"]
                p1 = 0.0
                for tkn, lp in zip(choice["logprobs"]["tokens"], logprobs):
                    if tkn.strip() == "1":
                        p1 = np.exp(lp)
                        break
                outputs.append(p1)
            else:
                outputs.append(int("".join(filter(str.isdigit, text)) or 0))

        return outputs

    def predict_batch(self, inputs: Iterable, *, proba: bool = False):
        if self.task == "text":
            inputs = list(inputs)
            results: list[int|float] = []
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i : i + self.batch_size]
                results.extend(self._classify_batch(batch, proba=proba))
            return results
        else:
            raise NotImplementedError("QianwenAdapter only supports text classification")

    # alias
    batch_predict = predict_batch
