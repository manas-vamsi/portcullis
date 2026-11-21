// Pure logic for the dashboard: no DOM, no network, so it can be tested with `node --test`.
// Everything here returns plain data or plain text. Turning it into elements is app.js's job, and that
// file only ever uses textContent, because everything that comes from the audit log is text chosen by
// a model and must never be treated as markup.

export function shortHash(hash) {
  if (typeof hash !== "string") return "";
  return hash.length > 12 ? `${hash.slice(0, 12)}…` : hash;
}

// The argument preview as lines of text, plus a note when the server cut it.
export function previewLines(entry) {
  if (typeof entry?.args_preview !== "string") return ["(no preview: the arguments were not stored)"];
  const lines = [entry.args_preview];
  if (entry.args_truncated === true) lines.push("… (cut: more arguments than shown)");
  return lines;
}

// Nearest-rank percentile of an already sorted list; null when there is nothing to rank.
export function percentile(sorted, fraction) {
  if (!sorted.length) return null;
  return sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
}

// A summary computed from the outcomes themselves, never taken from a file's own "summary" field.
export function summarize(outcomes) {
  if (!Array.isArray(outcomes) || outcomes.length === 0) throw new Error("there are no outcomes to summarize");
  // Prototype-free objects: a case id such as "__proto__" from a hostile file must stay an ordinary key.
  const perCase = Object.create(null);
  const byCategory = Object.create(null);
  const refused = Object.create(null);
  let passed = 0, errors = 0, forbidden = 0, tokens = 0, calls = 0;
  for (const o of outcomes) {
    const entry = (perCase[o.case_id] ??= { runs: 0, passed: 0 });
    entry.runs += 1;
    if (o.passed) { entry.passed += 1; passed += 1; }
    const cat = (byCategory[o.category] ??= { runs: 0, passed: 0 });
    cat.runs += 1;
    if (o.passed) cat.passed += 1;
    if (o.error) errors += 1;
    forbidden += Number(o.forbidden_executed) || 0;
    tokens += Number(o.tokens) || 0;
    calls += Number(o.tool_calls) || 0;
    for (const [rule, count] of Object.entries(o.refused ?? {})) refused[rule] = (refused[rule] ?? 0) + (Number(count) || 0);
  }
  const answered = outcomes.filter((o) => !o.error).map((o) => Number(o.seconds) || 0).sort((a, b) => a - b);
  const ids = Object.keys(perCase);
  return {
    runs: outcomes.length,
    cases: ids.length,
    passRate: passed / outcomes.length,
    passAllRate: ids.filter((id) => perCase[id].passed === perCase[id].runs).length / ids.length,
    byCategory: Object.fromEntries(Object.entries(byCategory).map(([c, v]) => [c, v.passed / v.runs])),
    errors,
    forbiddenExecuted: forbidden,
    meanTokens: tokens / outcomes.length,
    meanToolCalls: calls / outcomes.length,
    secondsP50: percentile(answered, 0.5),
    secondsP95: percentile(answered, 0.95),
    // Back to ordinary objects; fromEntries defines each key as its own property, so "__proto__" stays a key.
    refused: Object.fromEntries(Object.entries(refused)),
    perCase: Object.fromEntries(Object.entries(perCase)),
  };
}
