"use strict";
const $ = (id) => document.getElementById(id);
const LETTERS = "ABCD";
const get = (u) => fetch(u).then((r) => r.json());
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%");
const opt = (v, label) => `<option value="${esc(v)}">${esc(label ?? v)}</option>`;
const vbadge = (v) => `<span class="vbadge ${String(v).toLowerCase()}">${esc(v)}</span>`;
const empty = (cols, msg) => `<tr><td class="empty" colspan="${cols}">${esc(msg)}</td></tr>`;

let META = { variants: [], splits: [], default_model: "" };
const SPLIT_SIZE = {};

document.querySelectorAll("nav button").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("on", s.id === b.dataset.tab));
    if (b.dataset.tab === "results") loadMetrics();
    if (b.dataset.tab === "batch") pollRuns();
    if (b.dataset.tab === "arch") renderArch();
  };
});

(async function boot() {
  META = await get("/api/variants");
  const vopts = META.variants.map((v) => opt(v.id, `${v.id} — ${v.label}`)).join("");
  $("askVariant").innerHTML = vopts + opt("__all__", "compare all six");
  $("bVariant").innerHTML = vopts;
  $("aVariant").innerHTML = vopts;
  const sopts = META.splits.map((s) => opt(s)).join("");
  $("askSplit").innerHTML = sopts;
  $("bSplit").innerHTML = sopts;
  $("askSplit").value = $("bSplit").value = "test";
  $("askModel").value = $("bModel").value = META.default_model;
  loadItems(0);
  describeRange();
  pollRuns();
})();

// ── Ask ─────────────────────────────────────────────────────────────────
$("askSource").onchange = () => {
  const custom = $("askSource").value === "custom";
  $("askCustom").style.display = custom ? "" : "none";
  ["askSplitWrap", "askSearchWrap", "askItemWrap", "askPageWrap"]
    .forEach((id) => ($(id).style.display = custom ? "none" : ""));
  showAskItem();
};
// Changing the split or the search term restarts paging from the first page.
$("askSplit").onchange = () => loadItems(0);
let searchTimer;
$("askSearch").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => loadItems(0), 250); };
$("askItem").onchange = showAskItem;
["cQ", "cA", "cB", "cC", "cD", "cGold"].forEach((id) => ($(id).oninput = showAskItem));

const PAGE = 50;
let ITEMS = [], offset = 0, matched = 0;

async function loadItems(start = offset) {
  const q = encodeURIComponent($("askSearch").value.trim());
  const r = await get(`/api/items?split=${$("askSplit").value}&offset=${start}&limit=${PAGE}&q=${q}`);
  ITEMS = r.items; offset = start; matched = r.total;
  SPLIT_SIZE[$("askSplit").value] = q ? SPLIT_SIZE[$("askSplit").value] : r.total;
  $("askItem").innerHTML = ITEMS.map((it) =>
    opt(it.id, `${it.id} · ${it.question.slice(0, 80)}…`)).join("") || opt("", "no match");
  const last = offset + ITEMS.length;
  $("askPage").textContent = matched ? `${offset + 1}–${last} of ${matched}` : "no match";
  $("askPrev").disabled = offset === 0;
  $("askNext").disabled = last >= matched;
  showAskItem();
}
$("askPrev").onclick = () => loadItems(Math.max(0, offset - PAGE));
$("askNext").onclick = () => loadItems(offset + PAGE);

function currentItem() {
  if ($("askSource").value === "custom") {
    const g = $("cGold").value;
    return {
      question: $("cQ").value.trim(),
      options: ["cA", "cB", "cC", "cD"].map((i) => $(i).value.trim()),
      answer_idx: g === "" ? null : Number(g),
    };
  }
  return ITEMS.find((i) => i.id === $("askItem").value) || null;
}

const optionList = (options, gold, pred) => `<ul class="opts">${options.map((o, i) => `
  <li class="${i === gold ? "gold" : ""} ${i === pred ? "pred" : ""}">
    <span class="ltr">${LETTERS[i]}</span><span>${esc(o)}</span>
    ${i === gold ? '<span class="mark">correct</span>'
      : i === pred ? '<span class="mark">model chose</span>' : ""}
  </li>`).join("")}</ul>`;

function showAskItem() {
  const it = currentItem(), card = $("askItemCard");
  if (!it || !it.question) { card.style.display = "none"; return; }
  card.style.display = "";
  card.innerHTML = `<div class="panel-head"><h2>${esc(it.id || "Custom question")}</h2></div>
    <p class="qcard">${esc(it.question)}</p>
    ${optionList(it.options, it.answer_idx ?? -1, -1)}`;
}

$("askGo").onclick = async () => {
  const it = currentItem();
  if (!it || !it.question) return alert("Pick or type a question first.");
  const pick = $("askVariant").value;
  const ids = pick === "__all__" ? META.variants.map((v) => v.id) : [pick];
  const body = { variants: ids, model: $("askModel").value.trim() || null };
  if ($("askSource").value === "custom")
    Object.assign(body, { question: it.question, options: it.options, answer_idx: it.answer_idx });
  else Object.assign(body, { split: $("askSplit").value, item_id: it.id });

  $("askGo").disabled = true;
  $("askGo").textContent = "Running…";
  $("askCols").innerHTML = ids.map(colShell).join("");
  try {
    await streamSSE("/api/ask", body, onAskEvent);
  } catch (e) {
    alert("Run failed: " + e.message);
  } finally {
    $("askGo").disabled = false;
    $("askGo").textContent = "Run";
  }
};

const labelOf = (v) => (META.variants.find((x) => x.id === v) || {}).label || "";
const colShell = (v) => `<div class="col" id="col-${v}">
  <div class="col-head">${vbadge(v)}<span class="label">${esc(labelOf(v))}</span>
    <span class="kv" id="kv-${v}" style="margin-left:auto">queued</span></div>
  <div class="chips" id="chips-${v}"></div>
  <div id="detail-${v}"></div>
  <div id="verdict-${v}"></div></div>`;

const TRACES = {};

function onAskEvent(ev, d) {
  const v = d.variant;
  if (ev === "start") {
    TRACES[v] = {};
    $(`kv-${v}`).textContent = "running…";
    $(`chips-${v}`).innerHTML = d.nodes.map((n) =>
      `<span class="chip pending" id="chip-${v}-${n}">${esc(n)}</span>`).join("");
    // V1 retrieves inside its agent tool loop, not in a node delta, so no raw hits appear.
    if (v === "V1") $(`detail-${v}`).innerHTML = `<p class="note">V1 searches inside the agent
      loop, so its retrieved passages are not part of the node trace.</p>`;
  } else if (ev === "node") {
    TRACES[v][d.node] = d.update;
    const chip = $(`chip-${v}-${d.node}`);
    if (chip) {
      chip.className = "chip done";
      chip.textContent = `${d.node} · ${d.elapsed_s}s`;
      chip.onclick = () => showNode(v, d.node);
    }
  } else if (ev === "done") {
    const mark = d.correct === null ? '<span class="kv">no gold answer</span>'
      : d.correct ? '<span class="ok" style="font-weight:600">correct</span>'
      : `<span class="bad" style="font-weight:600">wrong — gold ${d.gold_letter}</span>`;
    $(`kv-${v}`).textContent = `${d.latency_s}s · ${d.tokens_in + d.tokens_out} tok`;
    $(`verdict-${v}`).innerHTML = `<div class="verdict">
        <span class="letter ${d.correct === false ? "bad" : d.correct ? "ok" : ""}">${d.answer_letter ?? "—"}</span>
        ${mark}</div>
      <pre class="out">${esc(d.rationale)}</pre>`;
  } else if (ev === "error") {
    $(`kv-${v}`).innerHTML = `<span class="bad">${esc(d.message)}</span>`;
    document.querySelectorAll(`#chips-${v} .chip.pending`).forEach((c) => (c.style.opacity = ".3"));
  }
}

function showNode(v, node) {
  document.querySelectorAll(`#chips-${v} .chip`).forEach((c) =>
    c.classList.toggle("on", c.id === `chip-${v}-${node}`));
  $(`detail-${v}`).innerHTML = renderUpdate(TRACES[v][node]);
}

// `retrieved` (raw passages) and `evidence` (the digest that replaced them) are shown as
// separate blocks — seeing both is the whole point of the search node.
const FIELD_LABEL = {
  case_understanding: "case understanding + search query",
  query: "search query", retrieved: "retrieved passages (raw)",
  evidence: "evidence digest", clinical_report: "clinical reasoning (no evidence seen)",
  rationale: "rationale", answer: "answer index",
};
function renderUpdate(u) {
  if (!u || !Object.keys(u).length) return `<p class="note">This step wrote nothing to state.</p>`;
  return Object.entries(u).map(([k, val]) =>
    `<h3>${esc(FIELD_LABEL[k] || k)}</h3><pre class="out">${esc(val)}</pre>`).join("");
}

/** POST + read an SSE body. `EventSource` is GET-only, so the stream is parsed by hand. */
async function streamSSE(url, body, onEvent) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop();
    for (const b of blocks) {
      const ev = (b.match(/^event: (.*)$/m) || [])[1];
      const data = (b.match(/^data: ([\s\S]*)$/m) || [])[1];
      if (ev && data) onEvent(ev, JSON.parse(data));
    }
  }
}

// ── Batch ───────────────────────────────────────────────────────────────
["bSplit", "bStart", "bLimit"].forEach((id) => ($(id).oninput = describeRange));
$("bSplit").onchange = describeRange;

function describeRange() {
  const start = Math.max(0, Number($("bStart").value) || 0);
  const n = Math.max(0, Number($("bLimit").value) || 0);
  const size = SPLIT_SIZE[$("bSplit").value];
  const end = n ? start + n - 1 : (size ? size - 1 : null);
  const split = $("bSplit").value;
  const pad = (i) => `${split}-${String(i).padStart(5, "0")}`;
  $("bRange").textContent = n
    ? `Runs ${n} question(s): ${pad(start)} → ${pad(end)}.`
    : `Runs every question from ${pad(start)} to the end of the split.`;
}

$("bGo").onclick = async () => {
  const res = await fetch("/api/runs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      variant: $("bVariant").value, split: $("bSplit").value,
      model: $("bModel").value.trim() || null,
      limit: Number($("bLimit").value), start: Number($("bStart").value),
      trace: $("bTrace").checked,
    }),
  });
  const j = await res.json();
  if (!res.ok) return alert(j.detail || "could not start");
  pollRuns();
};

$("bRefresh").onclick = () => pollRuns();
$("bClearAll").onclick = async () => {
  const { runs } = await get("/api/runs");
  const done = runs.filter((r) => r.status !== "running");
  if (!done.length) return alert("Nothing to clear — no finished runs.");
  if (!confirm(`Delete ${done.length} finished run(s) and their result files? This cannot be undone.`)) return;
  for (const r of done) await fetch(`/api/runs/${encodeURIComponent(r.run_id)}`, { method: "DELETE" });
  pollRuns();
};

let pollTimer;
async function pollRuns() {
  clearTimeout(pollTimer);
  const { runs } = await get("/api/runs");
  $("bTable").innerHTML = `<tr><th>version</th><th>run</th><th>progress</th><th class="num">accuracy</th>
    <th>status</th><th>traces</th><th></th></tr>` + (runs.length ? runs.map((r) => {
    const frac = r.total ? r.done / r.total : 0;
    return `<tr>
      <td>${vbadge(r.variant)}</td>
      <td class="mono dim">${esc(r.split)} · ${esc(r.model)}</td>
      <td style="min-width:190px">
        <div class="row" style="gap:9px; align-items:center; flex-wrap:nowrap">
          <div class="bar" style="flex:1"><i style="width:${(frac * 100).toFixed(1)}%"></i></div>
          <span class="kv" style="white-space:nowrap">${r.done}/${r.total || "?"}</span></div></td>
      <td class="num"><b>${pct(r.accuracy)}</b></td>
      <td><span class="tag ${r.status}">${r.status}</span></td>
      <td class="kv">${r.has_trace ? "yes" : "—"}</td>
      <td style="text-align:right; white-space:nowrap">
        <button class="btn sm" onclick="showLog('${esc(r.run_id)}')">log</button>
        ${r.status === "running"
          ? `<button class="btn sm danger" onclick="stopRun('${esc(r.run_id)}')">stop</button>`
          : `<button class="btn sm danger" onclick="deleteRun('${esc(r.run_id)}')">clear</button>`}
      </td></tr>`;
  }).join("") : empty(7, "No runs yet — start one above."));
  // Only keep polling while something is actually moving.
  if (runs.some((r) => r.status === "running")) pollTimer = setTimeout(pollRuns, 2000);
}

async function showLog(id) {
  const r = await get(`/api/runs/${encodeURIComponent(id)}`);
  $("bLogWrap").style.display = "";
  const lines = r.log_tail || [];
  $("bLogId").dataset.runId = id;   // the label carries a line count, so match on this
  $("bLogId").textContent = `${id} · last ${lines.length} lines`;
  $("bLog").textContent = lines.join("\n") || "(no output yet)";
  $("bLog").scrollTop = $("bLog").scrollHeight;   // newest output, like `tail -f`
  $("bLogWrap").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
async function stopRun(id) {
  await fetch(`/api/runs/${encodeURIComponent(id)}/stop`, { method: "POST" });
  pollRuns();
}
async function deleteRun(id) {
  if (!confirm(`Delete ${id} and its result files? This cannot be undone.`)) return;
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) return alert((await r.json()).detail || "could not delete");
  if ($("bLogId").dataset.runId === id) $("bLogWrap").style.display = "none";
  pollRuns();
}

// ── Results ─────────────────────────────────────────────────────────────
$("rReload").onclick = loadMetrics;
$("rBaseline").onchange = loadMetrics;
$("rRun").onchange = loadRunItems;
$("rFilter").onchange = loadRunItems;
$("rDelete").onclick = async () => {
  const id = $("rRun").value;
  if (!id) return;
  if (!confirm(`Delete ${id} and its result files? This cannot be undone.`)) return;
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) return alert((await r.json()).detail || "could not delete");
  loadMetrics();
};

async function loadMetrics() {
  if (!$("rBaseline").options.length)
    $("rBaseline").innerHTML = META.variants.map((v) => opt(v.id)).join("");
  const m = await get(`/api/metrics?baseline=${$("rBaseline").value || "V0"}`);
  $("rBaseLabel").textContent = m.baseline;

  $("rBoard").innerHTML = `<tr><th>version</th><th class="num">n</th><th class="num">accuracy</th>
    <th class="num">95% CI</th><th class="num">invalid</th><th class="num">latency</th>
    <th class="num">tokens</th></tr>` + (m.leaderboard.length ? m.leaderboard.map((r) =>
    `<tr><td>${vbadge(r.variant)}</td><td class="num kv">${r.n}</td>
      <td class="num"><b>${pct(r.accuracy)}</b></td>
      <td class="num kv">${pct(r.ci[0])} – ${pct(r.ci[1])}</td>
      <td class="num kv">${pct(r.invalid_rate)}</td>
      <td class="num kv">${r.mean_latency.toFixed(1)}s</td>
      <td class="num kv">${r.mean_tokens.toFixed(0)}</td></tr>`).join("")
    : empty(7, "No prediction files yet — launch a run from the Batch tab."));

  $("rPaired").innerHTML = `<tr><th>version</th><th class="num">gain</th><th class="num">win/loss/tie</th>
    <th class="num">paired</th><th class="num">McNemar p</th></tr>` + (m.paired.length ? m.paired.map((p) =>
    `<tr><td>${vbadge(p.variant)}</td>
      <td class="num ${p.gain >= 0 ? "ok" : "bad"}"><b>${p.gain >= 0 ? "+" : ""}${pct(p.gain)}</b></td>
      <td class="num kv">${p.wins}/${p.losses}/${p.ties}</td>
      <td class="num kv">${p.n_paired}</td>
      <td class="num ${p.p_value < 0.05 ? "ok" : "kv"}">${p.p_value.toFixed(4)}</td></tr>`).join("")
    : empty(5, `Nothing to pair against ${m.baseline}.`));

  const { runs } = await get("/api/runs");
  const keep = $("rRun").value;
  $("rRun").innerHTML = runs.map((r) => opt(r.run_id)).join("") || opt("", "no runs");
  if (keep && runs.some((r) => r.run_id === keep)) $("rRun").value = keep;
  $("rDelete").disabled = !runs.length;
  loadRunItems();
}

async function loadRunItems() {
  const id = $("rRun").value;
  if (!id) { $("rItems").innerHTML = empty(6, "No run selected."); $("rSummary").innerHTML = ""; return; }
  const r = await get(`/api/runs/${encodeURIComponent(id)}/items?limit=1000&filter=${$("rFilter").value}`);
  const hits = r.items.filter((i) => i.correct).length;
  const tok = r.items.reduce((a, i) => a + i.tokens_in + i.tokens_out, 0);
  const sec = r.items.reduce((a, i) => a + i.latency_s, 0);
  $("rSummary").innerHTML = r.items.length ? `
    <div class="stat"><div class="n">${r.total}</div><div class="l">questions</div></div>
    <div class="stat"><div class="n">${hits}</div><div class="l">correct</div></div>
    <div class="stat"><div class="n">${(sec / r.items.length).toFixed(1)}s</div><div class="l">avg latency</div></div>
    <div class="stat"><div class="n">${Math.round(tok / r.items.length)}</div><div class="l">avg tokens</div></div>` : "";
  $("rItems").innerHTML = `<tr><th>item</th><th>question</th><th>pred</th><th>gold</th>
    <th class="num">latency</th><th class="num">tokens</th></tr>` + (r.items.length ? r.items.map((it) =>
    `<tr class="click" onclick="openItem('${esc(id)}','${esc(it.item_id)}')">
      <td class="mono dim">${esc(it.item_id)}</td>
      <td>${esc((it.question || "").slice(0, 100))}…</td>
      <td class="mono ${it.correct ? "ok" : "bad"}"><b>${it.pred_letter ?? "—"}</b></td>
      <td class="mono kv">${it.gold_letter}</td>
      <td class="num kv">${it.latency_s.toFixed(1)}s</td>
      <td class="num kv">${it.tokens_in + it.tokens_out}</td></tr>`).join("")
    : empty(6, "No matching questions."));
}

async function openItem(runId, itemId) {
  const r = await get(`/api/runs/${encodeURIComponent(runId)}/items/${encodeURIComponent(itemId)}`);
  const it = r.item, rec = r.record;
  $("drawerTitle").innerHTML = `${esc(itemId)} <span class="mono dim">${esc(runId)}</span>`;
  $("drawerBody").innerHTML = `
    <p class="qcard">${esc(it ? it.question : "")}</p>
    ${it ? optionList(it.options, rec.gold, rec.pred) : ""}
    <div class="stats" style="margin:16px 0">
      <div class="stat"><div class="n ${rec.correct ? "ok" : "bad"}">${rec.pred_letter ?? "—"}</div>
        <div class="l">${rec.correct ? "correct" : "wrong"}</div></div>
      <div class="stat"><div class="n">${rec.latency_s.toFixed(1)}s</div><div class="l">latency</div></div>
      <div class="stat"><div class="n">${rec.tokens_in + rec.tokens_out}</div><div class="l">tokens</div></div>
    </div>
    <button class="btn sm" onclick="compareItem('${esc(itemId)}')">Compare across versions</button>
    <h3>Final rationale</h3><pre class="out">${esc(rec.rationale)}</pre>
    ${r.trace ? `<h3>Node trace</h3>` + r.trace.map((n) =>
      `<h3>${esc(n.node)} <span class="kv">${n.elapsed_s}s</span></h3>${renderUpdate(n.update)}`).join("")
      : `<p class="note" style="margin-top:14px">This run was recorded without traces. Tick
         “record traces” in the Batch tab to capture per-step detail.</p>`}`;
  openDrawer();
}

async function compareItem(itemId) {
  const r = await get(`/api/items/${encodeURIComponent(itemId)}/compare`);
  $("drawerTitle").innerHTML = `${esc(itemId)} <span class="dim">across versions</span>`;
  $("drawerBody").innerHTML = `<p class="qcard">${esc(r.item ? r.item.question : "")}</p>
    ${r.item ? optionList(r.item.options, r.item.answer_idx, -1) : ""}
    ${r.runs.length ? r.runs.map((x) => `
      <h3>${vbadge(x.variant)}
        <span class="mono ${x.correct ? "ok" : "bad"}"><b>${x.pred_letter ?? "—"}</b></span>
        <span class="kv">${x.latency_s.toFixed(1)}s</span></h3>
      <pre class="out">${esc(x.rationale)}</pre>`).join("")
      : `<p class="note" style="margin-top:14px">No runs on disk contain this question.</p>`}`;
  openDrawer();
}

const openDrawer = () => { $("drawer").classList.add("on"); $("scrim").classList.add("on"); };
const closeDrawer = () => { $("drawer").classList.remove("on"); $("scrim").classList.remove("on"); };
document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());

// ── Architecture ────────────────────────────────────────────────────────
let ARCH = null;
$("aVariant").onchange = renderArch;

async function renderArch() {
  if (!ARCH) ARCH = await get("/api/architecture");
  const id = $("aVariant").value;
  const i = ARCH.variants.findIndex((v) => v.id === id);
  const v = ARCH.variants[i], prev = i > 0 ? ARCH.variants[i - 1] : null;

  // What this rung adds is the ladder's whole story, so diff the configs explicitly.
  let added = "";
  if (prev) {
    const keys = Object.keys(v.config).filter((k) =>
      JSON.stringify(v.config[k]) !== JSON.stringify(prev.config[k]));
    added = keys.length
      ? `<div class="tw"><table><tr><th>setting</th><th>${prev.id}</th><th>${v.id}</th></tr>` +
        keys.map((k) => `<tr><td class="mono">${esc(k)}</td>
          <td class="mono dim">${esc(JSON.stringify(prev.config[k]))}</td>
          <td class="mono"><b>${esc(JSON.stringify(v.config[k]))}</b></td></tr>`).join("") + "</table></div>"
      : `<p class="note">Same configuration as ${prev.id}.</p>`;
  }

  $("aBody").innerHTML = `
    <div class="panel">
      <div class="panel-head">${vbadge(v.id)}<h2>${esc(v.label)}</h2></div>
      <div class="chips">${v.nodes.map((n) => `<span class="chip done">${esc(n)}</span>`).join("")}</div>
      <div class="tw"><table><tr><th>step</th><th>agent / role</th></tr>${v.nodes.map((n) =>
        `<tr><td class="mono">${esc(n)}</td><td class="dim">${esc(ARCH.node_roles[n] || "—")}</td></tr>`
      ).join("")}</table></div>
    </div>
    ${prev ? `<div class="panel"><h2>What ${v.id} adds over ${prev.id}</h2>${added}</div>` : ""}
    <div class="panel"><h2>Graph</h2>
      <pre class="mermaid out">${esc(v.mermaid)}</pre>
      <h3>Diagram from docs/</h3>
      <img src="${esc(v.diagram)}" style="max-width:100%"
        onerror="this.replaceWith(Object.assign(document.createElement('p'),
          {className:'note',textContent:'No SVG for this version in docs/diagrams/.'}))"></div>
    <div class="panel"><h2>Role prompts</h2>${Object.entries(ARCH.roles).map(([k, p]) =>
      `<h3>${esc(k)}</h3><pre class="out">${esc(p)}</pre>`).join("")}</div>`;

  // Render the mermaid source if the CDN is reachable; otherwise the <pre> above stands.
  if (!window.mermaid) {
    await new Promise((ok) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
      s.onload = ok; s.onerror = ok;
      document.head.appendChild(s);
    });
    if (window.mermaid) mermaid.initialize({ startOnLoad: false, theme: "neutral" });
  }
  if (window.mermaid) {
    try { await mermaid.run({ nodes: document.querySelectorAll(".mermaid") }); } catch (e) { /* keep source */ }
  }
}
