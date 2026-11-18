"""API keys for the server. A key is random and shown once; only its SHA-256 is stored.

Keys are long random strings, so a fast hash is enough (there is nothing to guess). A key
belongs to one tenant and has one role: `agent` (the runtime that asks for decisions) or
`approver` (a human or tool that answers approval requests). The roles never overlap, so a
key that can ask for tool calls cannot approve them.
"""

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class KeyEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant: str = Field(min_length=1, max_length=128)
    role: Literal["agent", "approver"]
    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")  # recorded as the approver in the audit log


class Principal(NamedTuple):
    tenant: str
    role: str
    name: str


def new_key() -> str:
    return "pc_" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_keys(path: Path) -> tuple[KeyEntry, ...]:
    """Read the key file. Any problem is an error: a server with no usable keys must not start open."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = tuple(KeyEntry.model_validate(item) for item in raw["keys"])
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise ValueError(f"cannot use the key file {path}: {type(exc).__name__}") from exc
    if not entries:
        raise ValueError(f"the key file {path} has no keys")
    if len({e.sha256 for e in entries}) != len(entries):
        raise ValueError(f"the key file {path} has a duplicate key")
    return entries


def authenticate(entries: tuple[KeyEntry, ...] | list[KeyEntry], presented: str | None) -> Principal | None:
    """The principal for a presented key, or None. Compares against every entry, in constant time per entry."""
    if not presented:
        return None
    digest = hash_key(presented)
    found = None
    for entry in entries:
        if hmac.compare_digest(entry.sha256, digest):
            found = Principal(entry.tenant, entry.role, entry.name)
    return found
