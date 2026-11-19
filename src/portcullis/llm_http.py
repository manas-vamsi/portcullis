"""An LLM adapter for any OpenAI-compatible chat API.

Ollama (local) and several hosted providers say they accept this format, so one adapter should cover
the free options. It has only been tested against a fake server, not against any real provider. It turns the agent's
message history into a chat request and the answer back into the agent's reply dict.

Rules it keeps, because the model's output and the network are untrusted:
- The API key is read from an environment variable you name, per call. It is never stored on the
  object, shown in repr(), written to an error message, or sent over plain http.
- Plain http is allowed only to loopback (a local Ollama). Redirects are not followed.
- Token counts must come from the server's `usage`. A missing or odd count is an error, because the
  agent's token budget would silently stop working with a made-up number.
- Tool-call arguments that are not a JSON object are passed on as they are, so the agent refuses the
  call (`invalid_tool_call`) instead of this adapter guessing what the model meant.
- Errors report the HTTP status or the kind of failure only, never the response body or the URL.
"""

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx

from portcullis.llm import Messages, Response

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class LLMError(Exception):
    """The model could not be used for this call. The message never contains the key, URL or response body."""


def _check_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url must be an http(s) URL with a host")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials; put the key in an environment variable")
    if parsed.scheme == "http" and parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError("plain http is only allowed to localhost; use https")
    return url.rstrip("/")


def _wire_call(call: Mapping[str, Any]) -> dict[str, Any]:
    if call.get("invalid"):  # a malformed call the agent kept only as a stub; its refusal message still needs a call to answer
        name, arguments = "invalid_call", "{}"
    else:
        name, arguments = call["name"], json.dumps(call["args"])
    return {"id": call["id"], "type": "function", "function": {"name": name, "arguments": arguments}}


def _wire_messages(messages: Messages, system_prompt: str | None) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}] if system_prompt else []
    for message in messages:
        if message["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = [_wire_call(c) for c in message.get("tool_calls") or []]
            if calls:
                entry["tool_calls"] = calls
            wire.append(entry)
        elif message["role"] == "tool":
            wire.append({"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]})
        else:
            wire.append({"role": message["role"], "content": message["content"]})
    return wire


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return raw
        return value if isinstance(value, dict) else raw
    return raw


def _parse_call(raw: Any, index: int) -> dict[str, Any]:
    function = raw.get("function") if isinstance(raw, dict) else None
    call_id = raw.get("id") if isinstance(raw, dict) else None
    call_id = call_id if isinstance(call_id, str) and call_id else f"call-{index}"  # some servers omit ids
    if not isinstance(function, dict):
        return {"id": call_id, "name": None, "args": None}  # malformed: the agent refuses it
    return {"id": call_id, "name": function.get("name"), "args": _parse_arguments(function.get("arguments"))}


def _parse_answer(body: Any) -> Response:
    try:
        message = body["choices"][0]["message"]
        tokens = body["usage"]["total_tokens"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("the answer has no message or no token usage") from exc
    if not isinstance(message, dict):
        raise LLMError("the answer's message is not an object")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise LLMError("the answer's token usage is not a non-negative integer")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise LLMError("the answer's tool_calls is not a list")
    content = message.get("content")
    return {
        "content": content if isinstance(content, str) else "",
        "tool_calls": [_parse_call(raw, index) for index, raw in enumerate(calls)],
        "tokens": tokens,
    }


class OpenAICompatLLM:
    """`llm(messages) -> reply`, for the guarded agent. `tools` are {name, description, parameters (JSON schema)}."""

    def __init__(
        self,
        base_url: str,
        model: str,
        tools: Sequence[Mapping[str, Any]] = (),
        api_key_env: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = _check_base_url(base_url)
        self._model, self._tools = model, [dict(t) for t in tools]
        self._key_env, self._system_prompt = api_key_env, system_prompt
        self._temperature, self._max_tokens = temperature, max_tokens
        self._client = client or httpx.Client(timeout=timeout)  # follow_redirects stays off: the key must not follow a redirect

    def __repr__(self) -> str:
        return f"OpenAICompatLLM(base_url={self._url!r}, model={self._model!r}, api_key_env={self._key_env!r})"

    def _headers(self) -> dict[str, str]:
        if self._key_env is None:
            return {}
        key = os.environ.get(self._key_env)
        if not key:
            raise LLMError(f"the environment variable {self._key_env} is not set")
        return {"Authorization": f"Bearer {key}"}

    def __call__(self, messages: Messages) -> Response:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _wire_messages(messages, self._system_prompt),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._tools:
            payload["tools"] = [{"type": "function", "function": t} for t in self._tools]
            payload["tool_choice"] = "auto"
        try:
            response = self._client.post(f"{self._url}/chat/completions", json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMError("the request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach the server ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise LLMError(f"the server answered HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError("the server's answer is not JSON") from exc
        return _parse_answer(body)
