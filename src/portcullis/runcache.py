"""Reads a run from its audit log, reusing earlier verification only while the stored history is unchanged.

Every decision needs the run's history, and the history is checked against its hash chain first. Doing both
from scratch on every request costs more the longer the run gets. This reader remembers, per run, the calls it
folded, the chain's head and a digest of every stored field of every row it has verified. On the next load the
database recomputes that digest over the same rows (one query; no rows sent):

- Same digest: those rows are unchanged to the byte, so they are not verified again. Only rows added since are
  fetched, checked against the remembered head, and folded on.
- Different digest, a different row count, or any failure: the remembered state is thrown away and the whole run
  is read and verified from the start, exactly as without a cache.

So a cache hit never skips the check that history was not edited; it replaces re-hashing every row with
having the database confirm that nothing changed. (A cache cannot detect an attacker who rewrites the whole
chain consistently, which plain verification cannot either; see the audit log's documented limits.)

The checksum and the rows are read in one REPEATABLE READ transaction, so they describe the same
snapshot of the log. Callers get copies of the records, so what they do to them (they record an answer in memory after writing it)
never reaches the cache. Callers must hold the run's lock while loading it, which the server and hook do.
"""

import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import psycopg

from portcullis.audit import GENESIS, check_rows, fetch_rows, history_checksums, row_dicts
from portcullis.runstate import CallRecord, InconsistentRun, fold


class ChainBroken(Exception):
    """The run's audit chain does not verify, or its rows tell an impossible story: no decision can be made from it."""


@dataclass(frozen=True)
class _Entry:
    length: int  # rows verified and folded
    head: str  # hash of the last of them
    digest: str  # digest of all of them, as the database computes it
    calls: dict[str, CallRecord]


class RunReader:
    def __init__(self, capacity: int = 256) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._guard = threading.Lock()

    def _recall(self, run: str) -> _Entry | None:
        with self._guard:
            entry = self._entries.get(run)
            if entry is not None:
                self._entries.move_to_end(run)
            return entry

    def _remember(self, run: str, entry: _Entry | None) -> None:
        with self._guard:
            self._entries.pop(run, None)
            if entry is not None and self._capacity > 0:
                self._entries[run] = entry
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)

    def load(self, conn: psycopg.Connection, run: str) -> dict[str, CallRecord]:
        entry = self._recall(run)
        # One snapshot for both reads: otherwise an edit that is in place for the checksum and undone before the
        # fetch (or the reverse) would pair a digest with rows it does not describe, and poison the cache.
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            total, digest, prefix_rows, prefix_digest = history_checksums(conn, run, entry.length if entry else 0)
            if entry is not None and prefix_rows == entry.length and prefix_digest == entry.digest:
                start, calls, head = entry.length, entry.calls, entry.head  # unchanged: continue where it was verified
            else:
                start, calls, head = 0, {}, GENESIS
            rows = fetch_rows(conn, run, start)
        checked = check_rows(run, rows, start, head)
        if not checked.ok:
            raise ChainBroken(run)  # the entry (if any) is untouched: the new rows were not folded into it
        try:
            fold(row_dicts(rows), calls)
        except InconsistentRun as exc:
            self._remember(run, None)
            raise ChainBroken(run) from exc
        self._remember(run, _Entry(total, checked.head, digest, calls))  # same snapshot: the digest describes these rows
        return {call_id: replace(record) for call_id, record in calls.items()}

