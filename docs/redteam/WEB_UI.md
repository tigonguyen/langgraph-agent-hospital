# Red-team web UI — run modes and Stream eval

How the web inspector (`src/agent_hospital/web/`) exposes the red-team scripts. Launch it as in
the README §5, or use `.claude/launch.json` (port 8010).

## 1. Run modes

A switch in the header picks one of two modes. The browser remembers the choice (localStorage;
if storage is blocked the UI starts in Normal run).

| mode | tabs |
|---|---|
| **Normal run** | Ask · Batch · Results · Architecture — the V0–V5 ladder, unchanged |
| **Attack & defend** | Stream eval |

- `?mode=attack` opens straight into Attack & defend, e.g. as a demo link.
- Each mode remembers its last tab.
- In attack mode the active mode button and a banner under the header use the danger hue.
- The UI keeps its original theme. Attack mode only adds the mode switch, the banner and a violet
  harness badge.

## 2. Stream eval

**Layout.**
- Settings on the left, Runs on the right, with a draggable divider. You can also focus it and
  press ←/→; double-click resets it to 38 %.
- The split fills the whole window: full width, and the height left under the header. Each pane
  scrolls on its own, and the split refits on resize.
- Below 980 px the panes stack and the page scrolls normally.
- The per-prompt and log panels open full width underneath.

**Form.**
- **Answering model (under test)**, e.g. `qwen-tb`.
- **Judge model (for ASR)**, default `gpt-oss:20b`, the judge the published results use. Pick a
  model you aren't testing. Leave it empty to skip judging; ASR then falls back to the keyword
  regex, marked *≈ regex*.
- Both fields suggest the models installed in Ollama.
- **MedQA items** and **Harmful-medical inserted**.
- **Harness**, one of the options below.

There is no Defense picker, off-topic count or seed: harness runs don't mix with guards, and runs
use `-k 0 --seed 0`.

| harness | script | what runs |
|---|---|---|
| none — model only | `eval_mixed.py M -n -m -k 0 --seed 0` | the model answers directly; MedSafetyBench test + the 40 hand-written prompts, shuffled into MedQA |
| VS1 `sysprompt` · VS2 `gatetool` · VS3 `gatenodes` | `eval_guarded.py <harness> --model M -m -n` | the model inside a guarded LangGraph (`graph/guarded.py`), **every node the same model**: VS1 a refusal instruction to the model itself; VS2 the model holds a `classify_request` tool and decides whether to screen itself; VS3 a gate node labels the request and the graph routes HARMFUL to a fixed refusal. Items: the first *m* MedSafetyBench, then *n* MedQA |

With a judge model, the run is chained in one process group to `judge.py`'s three rubrics over the
harmful replies: `--metric binary`, then `refusal`, then `harm`. These are the same three label
sets `docs/redteam/results/` publishes. Status reads *judging* while labels are written, and stop
ends every step.

**Runs table.** One row per run.

| column | meaning | source |
|---|---|---|
| Harness | harness badge (`VS1 · sysprompt`, `VS2 · gatetool`, `VS3 · gatenodes`), or *model only*. Runs made outside this form with a guard are labelled `guard <g>` or `legacy <g>` | run id / meta |
| stream | MedQA + harmful counts | run id |
| **ASR** | attack success rate = NOT_REFUSED / judged harmful replies (partial help counts), with counts, judge name and unparsed labels. *≈ regex* (1 − keyword refusal) until a judge has run | `<run>.binary-<judge>.jsonl` · published `*.asr_labels.jsonl` (gpt-oss:20b) |
| **Refusal rate** | REFUSED / n: declined outright without engaging | `<run>.judge.jsonl` · published `*.refusal_labels.jsonl` (qwen3:14b) |
| **HRR** | harmful-response rate = COMPLIED / n: handed over any part of what was asked | `<run>.harm.jsonl` · published `*.harm_labels.jsonl` (qwen3:14b) |
| **Accuracy** | MedQA exam accuracy (utility) | `correct` per MedQA row |
| **False refusal** | MedQA questions refused (utility cost, keyword-judged, as in STATUS.md) | `refused` per MedQA row |
| **Tokens in / out** | mean per item | Ollama `prompt_eval_count` / `eval_count`; summed over nodes for a harness |
| **Latency mean / p95** | seconds per item, model load excluded | `latency_s`; summed over nodes for a harness |

Older files without cost fields show "—" for tokens and latency.

**Two sources**, switched at the top of the Runs pane:
- **This machine**: `data/redteam/med/eval_mixed/`. Besides this form's runs it also lists
  `eval_msb900.py` runs (`<model>_msb<n>_s0[_g<guard>]`, harmful only) and CLI `--guard` runs,
  labelled with their guard.
- **Published results**: `docs/redteam/results/<set>/<name>.jsonl`, read-only, with their exported
  label files. `?source=published` opens this view directly. For these rows tokens and latency
  come from the exported summary, which can exclude items the raw rows can't (`gatenodes` drops one
  item that spans a manual pause; hover the `*`).

The published numbers reproduce the summaries. For example `harmful_900/med-booster-tb`:
ASR 89/900 = 9.9 %, refusal rate 651/900 = 72.3 %, HRR 75/900 = 8.3 %.

**Actions per run.**
- **judge ASR** labels a finished run whose three rubrics aren't complete, using the judge model
  field. It only runs the missing rubrics.
- **resume** appears only for runs this form could have started.
- **clear** also deletes the run's label files.

**Per-prompt rows.**
- A judged harmful row shows its judge labels (`ASR: refused | not_refused`, `refusal: refused |
  answered`, `harm: safe | complied`), with the keyword verdict kept underneath as `regex:`.
- Harness rows also show the graph's decision: gate label, "tool not called", "answerer never
  ran".

## 3. API

| route | method | does |
|---|---|---|
| `/api/redteam/ladder` | GET | the harness list and the default judge |
| `/api/redteam/runs?source=local\|published` | GET | list runs: `harness`, `judged.{binary,refusal,harm}` (rate, hits, labelled, of, unparsed, judge), `asr_regex`, `medqa_acc`, `false_refusal`, `cost`, `status` (incl. `judging`, `published`) |
| `/api/redteam/runs` | POST | start one run per model: `{models, n, m, harness, judge}`; `judge: null` = no judging |
| `/api/redteam/runs/{id}/judge` | POST | `{judge}`: judge a finished run's missing rubrics |
| `/api/redteam/runs/{id}/items`, `/log`, `/stop` | GET / POST | per-prompt rows, log tail, stop (the whole process group) |
| `/api/redteam/runs/{id}` | DELETE | remove a finished run's files, labels included |

Run ids are the scripts' own stems:
- model only: `<model>_n<n>_m<m>_s0`;
- harness: `<model>_guard-<harness>_m<m>_n<n>`;
- published: `pub:<set>:<name>`, read-only (items and labels only).

All run files go to `data/redteam/med/eval_mixed/`.

## 4. What is not in this branch

`attack-defend-ui` was rebuilt on `origin/main`. Earlier versions of the page are kept in the
local tag `backup/attack-defend-ui-pre-rebase` and in this branch's history:

- the `/api/defense/*` backend;
- the V0–V5 selector;
- the Defense (D0–D5) picker;
- the guard-model input.

`eval_mixed.py --guard` itself is unchanged and still runs from the CLI; those runs show up here
labelled with their guard.

## 5. Verified

- A `gatenodes` run started from the form's API (`qwen3:4b-instruct` answering and judging, n=1,
  m=2) went through running → judging → finished.
- Its row showed ASR 0/2 from the judge labels, accuracy, false refusal, tokens 251 / 14 and
  latency 2.19 / 2.28 s. The test run and its labels were deleted afterwards.
- The **judge ASR** button's route (`POST …/judge`) was not exercised, so as not to write labels
  into existing runs.
- The page renders side by side at 1700 px, checked with headless Chrome.
