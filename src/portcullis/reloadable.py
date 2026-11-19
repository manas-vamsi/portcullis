"""A configuration file that is reloaded when it changes, so a policy or a key list can be edited without a restart.

It fails closed. If the new version cannot be read or is invalid, every request is refused until a
good version appears. The old version is deliberately not kept: if someone tightened a policy or
revoked a key and made a typo, quietly carrying on with the looser old one is the worst outcome.
The error names the kind of failure only, never the file's contents.

Changes are noticed by the file's modification time, size and inode, checked on each request (a stat call).
While a file is bad it is read again on every request, so a transient failure cannot stick.
"""

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class ConfigUnavailable(Exception):
    """The configuration file is unreadable or invalid right now; refuse until it is fixed."""


class Reloadable(Generic[T]):
    def __init__(self, path: Path, load: Callable[[Path], T]) -> None:
        self._path, self._load = Path(path), load
        self._lock = threading.Lock()
        self._stamp = self._stat()  # before loading: a change during the load is then noticed on the next request
        self._value: T = load(self._path)  # a bad file at startup is an error, not a degraded server
        self._error: str | None = None

    def _stat(self) -> tuple[int, int, int] | None:
        try:
            info = os.stat(self._path)
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size, info.st_ino)  # the inode: an atomic replace changes it

    def get(self) -> T:
        stamp = self._stat()
        with self._lock:
            if stamp != self._stamp:
                self._stamp = stamp
                try:
                    self._value, self._error = self._load(self._path), None
                except Exception as exc:  # deliberate: any failure to load means refuse
                    self._error = type(exc).__name__
                    self._stamp = None  # try again on the next request: a transient failure must not stick
            if self._error is not None:
                raise ConfigUnavailable(f"{self._path.name} cannot be loaded ({self._error})")
            return self._value
