"""Optional LLM rewrite of playbook responses via an OpenAI-compatible endpoint.

OFF by default. Enabled only when SALES_COPILOT_LLM_REWRITE=1 and SALES_COPILOT_LLM_BASE_URL
is set (e.g. a local ModelMux gateway). Any failure falls back to the playbook text, so the
copilot, tests and evaluations never depend on it.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Mapping

Poster = Callable[[str, dict[str, object], dict[str, str], float], dict[str, object]]


def _http_post(
    url: str, body: dict[str, object], headers: dict[str, str], timeout: float
) -> dict[str, object]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, object] = json.loads(resp.read().decode())
        return data


class ResponseRewriter:
    def __init__(self, env: Mapping[str, str] | None = None, poster: Poster | None = None) -> None:
        e = os.environ if env is None else env
        self.enabled = e.get("SALES_COPILOT_LLM_REWRITE", "0") == "1" and bool(
            e.get("SALES_COPILOT_LLM_BASE_URL")
        )
        self.base_url = e.get("SALES_COPILOT_LLM_BASE_URL", "").rstrip("/")
        self.model = e.get("SALES_COPILOT_LLM_MODEL", "default")
        self.api_key = e.get("SALES_COPILOT_LLM_API_KEY", "")
        self.timeout = float(e.get("SALES_COPILOT_LLM_TIMEOUT_S", "3"))
        self._post = poster or _http_post

    def rewrite(self, response: str, utterance: str) -> str:
        if not self.enabled or not response:
            return response
        body: dict[str, object] = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 80,
            "messages": [
                {
                    "role": "system",
                    "content": "Rewrite the sales coaching tip so it fits what the prospect "
                    "just said. One sentence, under 30 words. Do not invent facts or prices.",
                },
                {"role": "user", "content": f"Prospect: {utterance}\nTip: {response}"},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            data = self._post(f"{self.base_url}/chat/completions", body, headers, self.timeout)
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                text = str(choices[0]["message"]["content"]).strip()
                if text:
                    return text
        except Exception:
            pass
        return response
