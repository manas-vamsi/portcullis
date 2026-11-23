"""Fail-closed launcher for the portcullis PreToolUse hook. Standard library only.

Claude Code treats exit code 1 as a non-blocking error and lets the tool call through, and it
also lets a call through when a hook times out. A Python import failure or an uncaught exception
exits with 1. So this launcher turns every failure into exit 2, which blocks the call, and a
watchdog kills a hung hook with exit 2 before Claude Code's own timeout can let the call through.

It runs portcullis.quickhook in this Python if it can be imported. That decides plainly valid
policies without loading pydantic and hands everything else to the full hook, portcullis.hook
(set PORTCULLIS_HOOK_MODULE=portcullis.hook to always use the full one). If portcullis is not importable
here, it delegates to the Python named by PORTCULLIS_PYTHON (the interpreter of the environment
where portcullis is installed). With neither, it blocks.

One gap it cannot close from inside: if the `python` command itself cannot be started, Claude
Code reports a non-blocking error. Make sure `python` is on PATH.
"""

import importlib
import os
import subprocess
import sys
import threading

MODULE = os.environ.get("PORTCULLIS_HOOK_MODULE", "portcullis.quickhook")
# Both must stay below the hook timeout in hooks.json (30): this launcher has to give up first.
DEFAULT_DEADLINE_SECONDS = 20  # the whole hook, any path
DELEGATE_TIMEOUT_SECONDS = 15  # the delegated Python, inside that


def blocked(message: str) -> int:
    try:
        sys.stderr.write(f"portcullis hook: {message}\n")
        sys.stderr.flush()
    except OSError:
        pass  # nothing more we can do; the exit code still blocks
    return 2


def start_watchdog() -> None:
    try:
        deadline = float(os.environ.get("PORTCULLIS_HOOK_DEADLINE", DEFAULT_DEADLINE_SECONDS))
    except ValueError:
        deadline = DEFAULT_DEADLINE_SECONDS

    def expire() -> None:
        blocked("deadline exceeded, blocking the call")
        os._exit(2)

    timer = threading.Timer(deadline, expire)
    timer.daemon = True
    timer.start()


def delegate(data: bytes) -> int:
    python = os.environ.get("PORTCULLIS_PYTHON")
    if not python:
        return blocked("portcullis is not importable here; install it, or set PORTCULLIS_PYTHON to a Python that has it")
    try:
        child = subprocess.run([python, "-m", MODULE], input=data, capture_output=True, timeout=DELEGATE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        return blocked(f"cannot run {python}: {type(exc).__name__}")
    sys.stdout.buffer.write(child.stdout)
    sys.stderr.buffer.write(child.stderr)
    return child.returncode if child.returncode in (0, 2) else 2


def main() -> int:
    start_watchdog()
    try:
        data = sys.stdin.buffer.read()
        try:
            module = importlib.import_module(MODULE)
        except ImportError:
            return delegate(data)
        code, out, err = module.run(data, os.environ)
        sys.stdout.write(out)
        sys.stdout.flush()
        sys.stderr.write(err)
        sys.stderr.flush()
        return code if code in (0, 2) else 2
    except BaseException as exc:  # includes KeyboardInterrupt and SystemExit: exit 1 would let the call through
        return blocked(f"unexpected {type(exc).__name__}")


if __name__ == "__main__":
    sys.exit(main())
