"""
Adapter for OpenAI Chat/Completions → 返回离散标签
------------------------------------------------
* 仅依赖   pip install openai>=1.14.0
* 读取  OPENAI_API_KEY 环境变量
* 默认用 gpt-3.5-turbo；想换 gpt-4o 传 model_name
"""
from __future__ import annotations
import os, itertools, base64, requests, json
from typing import Iterable, List
import openai


class OpenAIAdapter:
    def __init__(
        self,
        model_url: str | None = None,          # 不用；保持签名一致
        task: str = "text",
        *,
        model_name: str = "gpt-4.1",
        system_prompt: str | None = None,
        batch_size: int = 8,
        **_
    ):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.model_name = model_name
        self.task = task
        self.system_prompt = (
            system_prompt
            or "You are a classification model. "
               "Respond ONLY with the class label (0 or 1)."
        )
        self.batch_size = batch_size
        self.supports_grad = False  # 纯黑盒

    # ------------------------------------------------------------------
    def _classify_batch(self, prompts: List[str]) -> List[int]:
        """Zero-shot二分类示例：返回 0 或 1。"""
        labels = []
        for p in prompts:
            resp = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": p},
                ],
                temperature=0,
                max_tokens=3,
            )
            txt = resp.choices[0].message.content.strip()
            lbl = int("".join(filter(str.isdigit, txt)) or 0)
            labels.append(lbl)
        return labels


    # ------------------------------------------------------------------
    def predict_batch(self, inputs: Iterable):
        if self.task == "text":
            return self._classify_batch(list(inputs))
        elif self.task == "image":
            raise NotImplementedError(
                "OpenAIAdapter currently only supports text classification demo."
            )
    batch_predict = predict_batch