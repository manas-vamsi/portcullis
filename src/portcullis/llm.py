"""LLM test doubles. An LLM here is any callable: messages -> response dict."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

Messages = list[dict[str, Any]]
Response = dict[str, Any]
LLM = Callable[[Messages], Response]


class ScriptExhausted(Exception):
    """FakeLLM was called more times than it has scripted responses."""


class CassetteMiss(Exception):
    """No recording for this request and recording is off."""


class FakeLLM:
    """Returns scripted responses in order. Records every call for assertions."""

    def __init__(self, script: list[Response]):
        self._script = list(script)
        self.calls: list[Messages] = []

    def __call__(self, messages: Messages) -> Response:
        self.calls.append(messages)
        if len(self.calls) > len(self._script):
            raise ScriptExhausted(f"call {len(self.calls)} but only {len(self._script)} scripted")
        return self._script[len(self.calls) - 1]


class CassetteLLM:
    """Replays recorded responses keyed by a hash of (model, messages).

    Default is replay-only so tests and CI can never reach a real model.
    With record=True, a miss calls `inner` once and saves the answer.
    """

    def __init__(self, inner: LLM, path: Path, record: bool = False, model: str = ""):
        self._inner = inner
        self._path = Path(path)
        self._record = record
        self._model = model

    def __call__(self, messages: Messages) -> Response:
        key = self._key(messages)
        store = self._load()
        if key in store:
            return store[key]
        if not self._record:
            raise CassetteMiss(f"no recording for {key[:12]} in {self._path}")
        response = self._inner(messages)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({**store, key: response}, indent=2, sort_keys=True))
        return response

    def _key(self, messages: Messages) -> str:
        canonical = json.dumps(
            {"model": self._model, "messages": messages}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load(self) -> dict[str, Response]:
        return json.loads(self._path.read_text()) if self._path.exists() else {}
