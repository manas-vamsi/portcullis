// Tests for the dashboard's pure logic. Run with: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { percentile, previewLines, shortHash, summarize } from "../../src/portcullis/web/lib.js";

const outcome = (caseId, run, passed, extra = {}) => ({
  case_id: caseId, category: "task", run, passed, error: null, tool_calls: 2, tokens: 100,
  seconds: 1, forbidden_executed: 0, refused: {}, ...extra,
});

test("a summary counts pass rate, pass-all rate, errors, breaches and means", () => {
  const s = summarize([
    outcome("a", 0, true), outcome("a", 1, true),
    outcome("b", 0, true), outcome("b", 1, false),
    outcome("s", 0, false, { category: "safety", error: "LLMError", forbidden_executed: 1, refused: { tool_not_allowed: 2 } }),
  ]);
  assert.equal(s.runs, 5);
  assert.equal(s.cases, 3);
  assert.equal(s.passRate, 3 / 5);
  assert.equal(s.passAllRate, 1 / 3);
  assert.equal(s.errors, 1);
  assert.equal(s.forbiddenExecuted, 1);
  assert.deepEqual(s.refused, { tool_not_allowed: 2 });
  assert.deepEqual(s.perCase.b, { runs: 2, passed: 1 });
  assert.equal(s.byCategory.safety, 0);
  assert.equal(s.meanTokens, 100);
});

test("an empty list of outcomes is an error, not a row of zeros", () => {
  assert.throws(() => summarize([]), /no outcomes/);
  assert.throws(() => summarize("nope"), /no outcomes/);
});

test("percentiles use the nearest rank and ignore runs that got no answer", () => {
  const values = Array.from({ length: 10 }, (_, i) => i + 1);
  assert.equal(percentile(values, 0.5), 5);
  assert.equal(percentile(values, 0.95), 10);
  const s = summarize([...values.map((v, i) => outcome("a", i, true, { seconds: v })), outcome("e", 0, false, { error: "x", seconds: 99 })]);
  assert.equal(s.secondsP50, 5);
  assert.equal(s.secondsP95, 10);
  assert.equal(percentile([], 0.5), null);
});

test("a summary whose every run errored has no latency", () => {
  const s = summarize([outcome("a", 0, false, { error: "x" })]);
  assert.equal(s.secondsP50, null);
});

test("hashes are shortened for display and tolerate junk", () => {
  assert.equal(shortHash("a".repeat(64)), "aaaaaaaaaaaa…");
  assert.equal(shortHash("abc"), "abc");
  assert.equal(shortHash(undefined), "");
  assert.equal(shortHash(12345), "");
});

test("an arguments preview becomes plain lines of text and says when it was cut", () => {
  assert.deepEqual(previewLines({ args_preview: '{"cost":900}', args_truncated: false }), ['{"cost":900}']);
  assert.deepEqual(previewLines({ args_preview: '{"cost":9', args_truncated: true }), ['{"cost":9', "… (cut: more arguments than shown)"]);
  assert.deepEqual(previewLines({}), ["(no preview: the arguments were not stored)"]);
  assert.deepEqual(previewLines({ args_preview: 5 }), ["(no preview: the arguments were not stored)"]);
});

test("a hostile case id such as __proto__ stays an ordinary key and pollutes nothing", () => {
  const s = summarize([outcome("__proto__", 0, true), outcome("constructor", 0, false), outcome("a", 0, true)]);
  assert.equal(s.cases, 3);
  assert.deepEqual(Object.keys(s.perCase).sort(), ["__proto__", "a", "constructor"]);
  assert.equal(({}).runs, undefined);
  assert.equal(s.passRate, 2 / 3);
  const r = summarize([outcome("a", 0, true, { refused: { __proto__: 5 } })]);
  assert.equal(({}).passed, undefined);
  assert.ok(r);
});
