# Architecture decision records

Short records of the decisions that shape portcullis: what was decided, why, and what it costs.
Each names where it is built and tested. They describe the code as it is; where a decision has a
known cost or gap, the record says so.

| # | Decision |
|---|---|
| 1 | [Decisions are made by a deterministic function, never by a model](0001-deterministic-policy-no-llm.md) |
| 2 | [Every failure refuses the call](0002-fail-closed.md) |
| 3 | [One hash-chained audit log per run, in Postgres](0003-hash-chained-audit-log.md) |
| 4 | [A run's state is rebuilt from its own audit chain](0004-state-from-the-audit-chain.md) |
| 5 | [Side effects are at-least-once, deduplicated by a key the model cannot choose](0005-at-least-once-not-exactly-once.md) |
| 6 | [Spend is counted when a call is allowed, under two caps that approval cannot exceed](0006-spend-caps.md) |
| 7 | [A human approves through one of three channels, each with a stated strength](0007-approval-channels.md) |
| 8 | [Tool definitions are pinned by a hash of what the model reads, and re-read on every call](0008-pin-what-the-model-reads.md) |
| 9 | [Who is acting comes from the credential, never from the request](0009-tenant-from-the-credential.md) |
| 10 | [Policy and key files reload on change, and a bad file refuses everything](0010-reload-fails-closed.md) |
| 11 | [The dashboard is plain HTML and JavaScript served by the server](0011-dashboard-as-static-files.md) |
| 12 | [Nothing is claimed that is not built, tested or measured, and weaknesses are pinned by tests](0012-evidence-before-claims.md) |
| 13 | [The Claude Code hook does not use a background process to go faster](0013-no-background-process-for-the-hook.md) |
| 14 | [A faster hook that reads plain policies without pydantic, shares the rules, and hands everything else on](0014-fast-hook-shares-the-rules.md) |
