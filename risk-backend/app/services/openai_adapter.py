from __future__ import annotations
import os, itertools
from typing import Iterable, List
import openai
import numpy as np

TOP_LOGPROBS = 5

class OpenAIAdapter:
    def __init__(
        self,
        model_url: str | None = None,
        task: str = "text",
        *,
        model_name: str = "gpt-4.1",
        system_prompt: str | None = None,
        batch_size: int = 8,
        **_,
    ):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.model_name   = (model_url or model_name).strip()
        self.task         = task
        self.batch_size   = batch_size
        self.system_prompt = (
            system_prompt
            or "You are a classification model. Respond ONLY with class label 0 or 1."
        )
        self.supports_grad = False

    # ------------------------------------------------------------------ #
    def _classify_batch(self, prompts: List[str], *, proba: bool = False) -> list:
        """
        如果 proba=True → 返回 “p(正类)” (float 0-1)，否则返回离散标签。
        """
        outputs = []
        for p in prompts:
            kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": p},
                ],
                temperature=0,
                max_tokens=3,
            )

            if proba:
                kwargs.update(
                    logprobs=True,
                    top_logprobs=TOP_LOGPROBS,
                )
            # ---------------------------------------------------

            resp = openai.chat.completions.create(**kwargs)

            txt = resp.choices[0].message.content.strip()

            if proba:
                token_info = resp.choices[0].logprobs.content
                p1 = 0.5
                for tok in token_info:
                    if tok["token"].strip() == "1":
                        p1 = np.exp(tok["logprob"])
                        break
                outputs.append(p1)
            else:
                outputs.append(int("".join(filter(str.isdigit, txt)) or 0))

        return outputs

    # ------------------------------------------------------------------ #
    def predict_batch(self, inputs: Iterable, *, proba: bool = False):
        if self.task == "text":
            return self._classify_batch(list(inputs), proba=proba)
        raise NotImplementedError("Only text classification demo supported.")

    batch_predict = predict_batch