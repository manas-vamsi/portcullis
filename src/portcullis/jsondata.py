"""Strict JSON-data check for values that come from outside (a model, an HTTP client)."""

from typing import Any

MAX_INT, MIN_INT = 2**64 - 1, -(2**63)  # LangGraph's checkpoint serializer fails outside this range
MAX_DEPTH = 32  # ...and on data nested about 250 levels deep; tool arguments never need more than this


def check_json(value: Any, depth: int = 0) -> None:
    """Only plain JSON data may enter the checkpointed state."""
    if depth > MAX_DEPTH:
        raise ValueError("nested too deeply")
    if value is None or isinstance(value, (bool, str, float)):
        return
    if isinstance(value, int):
        if not MIN_INT <= value <= MAX_INT:
            raise ValueError("integer outside the 64-bit range")
    elif isinstance(value, list):
        for item in value:
            check_json(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            check_json(item, depth + 1)
    else:
        raise ValueError(f"{type(value).__name__} is not JSON data")
