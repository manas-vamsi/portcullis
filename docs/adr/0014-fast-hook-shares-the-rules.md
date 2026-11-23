# ADR-0014: A faster hook that reads plain policies without pydantic, shares the rules, and hands everything else on

Status: accepted. Date: 2026-11-23.

## Context

ADR-0013 kept the hook a plain process per call and named the honest fix for its cost: a faster-starting
implementation of the same logic. Profiling showed the decision itself takes microseconds; most of each call is
loading pydantic and building the policy's validation models. A second engine in another language would be fast,
but two copies of the rules can drift apart, and in a guard drift is a hole.

## Decision

- The checks move to `portcullis.rules`, which uses the standard library only and reads its inputs by attribute.
  `portcullis.policy.decide` (pydantic models) and the fast hook both call it, so there is one copy of the rules.
- `portcullis.quickhook` reads the policy file with a strict standard-library reader that accepts only plainly valid
  policies (exact keys, exact types, no duplicate keys, finite numbers, compiling regexes) and converts values as
  pydantic does (an integer limit becomes a float, so messages match). Anything else (a database, an odd event, an
  unusual policy, any error) goes to the full hook unchanged, which validates with pydantic and fails closed.
- It can only be faster, never looser. Tests check that on generated policies and calls its output equals the full
  hook's byte for byte, that every policy it accepts is one pydantic accepts with the same content, and that each of
  eleven planted bugs in it is caught.

## Consequences

- Good: without a database, a hook call no longer loads pydantic. Measured once on a loaded laptop, interleaved with
  the full hook: about 0.22 to 0.29 s against 0.84 to 0.86 s. Rerun `benchmarks/run.py` (`hook`) for a cleaner figure.
- Bad: the strict reader is a second piece of code that must track the policy models. When a field is added to the
  models, the reader must learn it too; until it does, policies using the field simply take the slower full path
  (the reader refuses unknown keys), so the gap costs speed, not safety.
- The Claude Code launcher runs the fast hook by default (`PORTCULLIS_HOOK_MODULE=portcullis.hook` restores the full
  one). With a database the full hook still runs every call (it needs psycopg and the audit chain).

## Where

`src/portcullis/rules.py`, `src/portcullis/quickhook.py`, `tests/test_quickhook.py`, `tests/test_surface_parity.py`,
`benchmarks/run.py`, `plugin/scripts/policy_hook.py`
