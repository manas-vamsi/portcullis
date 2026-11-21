// The dashboard's DOM code. Rules this file keeps:
//  - Text from the server (and from an eval results file) is only ever put on the page with textContent.
//    It is chosen by a model, so it is never HTML, never an attribute value that could run, never a class name.
//  - The API key lives in a variable in this tab. It is not stored anywhere and not put in a URL.
//  - Requests go to this same server only, with the key in an Authorization header.
import { previewLines, shortHash, summarize } from "./lib.js";

let apiKey = "";
let timer = null;
let pendingOffset = 0;
const PAGE = 100; // the server sends at most this many pending calls at a time
const MAX_CASE_ROWS = 500; // a results file with more cases is summarised, not drawn row by row

const $ = (id) => document.getElementById(id);

/** Make an element with text content (never markup). */
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function say(text, isError = false) {
  const note = $("message");
  note.textContent = text;
  note.className = isError ? "error" : "";
}

async function api(method, path, body) {
  if (!apiKey) throw new Error("Enter an API key first.");
  const response = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${apiKey}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
    credentials: "omit",
    cache: "no-store",
  });
  let data = null;
  try { data = await response.json(); } catch { /* not JSON: leave null */ }
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : "the server refused the request";
    throw new Error(`${response.status}: ${detail}`);
  }
  return data;
}

function table(headers, rows) {
  const t = el("table");
  const head = el("tr");
  headers.forEach((h) => head.append(el("th", h)));
  t.append(head);
  rows.forEach((cells) => {
    const row = el("tr");
    cells.forEach((cell) => row.append(cell instanceof Node ? cell : el("td", cell)));
    t.append(row);
  });
  return t;
}

function cell(child) {
  const td = el("td");
  td.append(child);
  return td;
}

// ---- approvals ----

async function answer(entry, approved, row) {
  row.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  try {
    const path = `/v1/runs/${encodeURIComponent(entry.run_id)}/calls/${encodeURIComponent(entry.call_id)}/approval`;
    const result = await api("POST", path, { approved });
    const blocked = result.blocked_by ? ` (blocked: ${result.blocked_by})` : "";
    say(`${entry.call_id}: ${result.status}${blocked}`, result.status === "approval_denied" && approved);
  } catch (error) {
    say(error.message, true);
  }
  await loadPending();
}

async function loadPending() {
  const box = $("pending");
  try {
    const { pending } = await api("GET", `/v1/approvals?limit=${PAGE}&offset=${pendingOffset}`);
    box.replaceChildren();
    // Paging, so a flood of requests cannot hide newer ones behind the oldest hundred.
    $("prev-page").disabled = pendingOffset === 0;
    $("next-page").disabled = pending.length < PAGE;
    $("page-note").textContent = pending.length
      ? `Showing ${pendingOffset + 1} to ${pendingOffset + pending.length}${pending.length === PAGE ? " (there may be more)" : ""}`
      : "";
    if (!pending.length) { box.append(el("p", "Nothing is waiting.", "note")); return; }
    const rows = pending.map((entry) => {
      const preview = el("pre", previewLines(entry).join("\n"));
      const approve = el("button", "Approve", "ok");
      const deny = el("button", "Deny", "bad");
      approve.type = deny.type = "button";
      const actions = el("td");
      actions.append(approve, " ", deny);
      const row = [entry.run_id, entry.call_id, entry.tool, `${entry.rule}: ${entry.reason}`, cell(preview), shortHash(entry.request_sha256), actions];
      return { row, entry, approve, deny, actions };
    });
    box.append(table(["Run", "Call", "Tool", "Why it waits", "What was asked", "Request hash", ""], rows.map((r) => r.row)));
    const trs = box.querySelectorAll("tr");
    rows.forEach((r, i) => {
      const tr = trs[i + 1];
      r.approve.addEventListener("click", () => answer(r.entry, true, tr));
      r.deny.addEventListener("click", () => answer(r.entry, false, tr));
    });
  } catch (error) {
    say(error.message, true);
  }
}

// ---- audit log ----

async function showAudit(event) {
  event.preventDefault();
  const box = $("audit-result");
  try {
    const run = $("run").value.trim();
    const audit = await api("GET", `/v1/runs/${encodeURIComponent(run)}/audit`);
    box.replaceChildren();
    const v = audit.verified;
    box.append(el("p", v.ok ? `Chain verified: ${v.rows} rows, head ${v.head}` : `CHAIN BROKEN: ${v.problem}`, v.ok ? "ok" : "bad"));
    box.append(table(
      ["#", "Time", "Event", "Event id", "Detail"],
      audit.events.map((e) => [e.idx, e.ts, e.event, e.event_id, cell(el("pre", JSON.stringify(e.payload, null, 1)))]),
    ));
    say("");
  } catch (error) {
    say(error.message, true);
  }
}

// ---- eval report ----

function percent(x) {
  return `${(x * 100).toFixed(1)}%`;
}

async function showResults(event) {
  const file = event.target.files[0];
  const box = $("eval-result");
  if (!file) return;
  try {
    if (file.size > 20_000_000) throw new Error("That file is too large to be a results file.");
    const document_ = JSON.parse(await file.text());
    const s = summarize(document_.outcomes);
    const meta = document_.meta ?? {};
    box.replaceChildren();
    box.append(el("p", `Model ${meta.model ?? "?"}, ${meta.runs_per_case ?? "?"} runs per case, cases file ${shortHash(meta.cases_sha256)}, finished ${meta.finished_at ?? "?"}`, "note"));
    box.append(table(["Measure", "Value"], [
      ["Runs / cases", `${s.runs} / ${s.cases}`],
      ["Pass rate", percent(s.passRate)],
      ["Pass-all rate (every run of a case passed)", percent(s.passAllRate)],
      ["By category", Object.entries(s.byCategory).map(([c, r]) => `${c} ${percent(r)}`).join(", ")],
      ["Errors", s.errors],
      ["Breaches (forbidden tool ran, or cost over a limit in a safety case)", s.forbiddenExecuted],
      ["Seconds per run p50 / p95", s.secondsP50 === null ? "no run got an answer" : `${s.secondsP50.toFixed(2)} / ${s.secondsP95.toFixed(2)}`],
      ["Mean tokens / tool calls", `${s.meanTokens.toFixed(0)} / ${s.meanToolCalls.toFixed(1)}`],
      ["Calls refused, by rule", Object.keys(s.refused).length ? Object.entries(s.refused).map(([r, n]) => `${r} ${n}`).join(", ") : "none"],
    ]));
    const cases = Object.entries(s.perCase);
    box.append(table(["Case", "Passed"], cases.slice(0, MAX_CASE_ROWS).map(([id, c]) => [id, `${c.passed}/${c.runs}`])));
    if (cases.length > MAX_CASE_ROWS) box.append(el("p", `${cases.length - MAX_CASE_ROWS} more cases not shown.`, "note"));
    if (s.forbiddenExecuted > 0) say("A forbidden action ran in this results file.", true); else say("");
  } catch (error) {
    box.replaceChildren();
    say(`Cannot read that file: ${error.message}`, true);
  }
}

// ---- wiring ----

function showTab(name) {
  document.querySelectorAll("main section").forEach((s) => { s.hidden = s.id !== name; });
  document.querySelectorAll("nav button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tab === name)));
}

function start() {
  $("key").addEventListener("input", (e) => { apiKey = e.target.value.trim(); });
  $("key-form").addEventListener("submit", (e) => e.preventDefault());
  document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("load-pending").addEventListener("click", loadPending);
  $("prev-page").addEventListener("click", () => { pendingOffset = Math.max(0, pendingOffset - PAGE); loadPending(); });
  $("next-page").addEventListener("click", () => { pendingOffset += PAGE; loadPending(); });
  $("auto").addEventListener("change", (e) => {
    clearInterval(timer);
    timer = e.target.checked ? setInterval(loadPending, 10_000) : null;
  });
  $("audit-form").addEventListener("submit", showAudit);
  $("results-file").addEventListener("change", showResults);
}

start();
