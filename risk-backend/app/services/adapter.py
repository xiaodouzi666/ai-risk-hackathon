from __future__ import annotations
import os, itertools, base64, requests, json
from typing import Iterable, List
import openai


class OpenAIAdapter:
    def __init__(
        self,
        model_url: str | None = None, 
        task: str = "text",
        *,
        model_name: str = "gpt-3.5-turbo",
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
        self.supports_grad = False

    # ------------------------------------------------------------------
    def _classify_batch(self, prompts: List[str]) -> List[int]:
        """Zero-shot binary classification: label 0 or 1."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": p},
        ]

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
