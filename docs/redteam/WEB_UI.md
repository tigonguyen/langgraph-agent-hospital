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
- The UI keeps its original theme. Attack mode only adds the mode switch, the banner, the D0–D5
  badge ramp (grey → deep green) and a violet harness badge.

## 2. Stream eval

**Layout.**
- Settings on the left, Runs on the right, with a draggable divider. You can also focus it and
  press ←/→; double-click resets it to 38 %.
- The split fills the whole window: full width, and the height left under the header. Each pane
  scrolls on its own, and the split refits on resize.
- Below 980 px the panes stack and the page scrolls normally.
- The per-prompt and log panels open full width underneath.

**Form.** Models (Ollama tags, one run per model), MedQA items, harmful-medical items, then one
of the two choices below. The non-medical count and the seed are no longer in the form; runs use
`-k 0 --seed 0`.

| choice | options | script | what runs |
|---|---|---|---|
| **Harness (H)** | none · `sysprompt` · `gatetool` · `gatenodes` | `eval_guarded.py <harness> --model M -m -n` | the model inside a guarded LangGraph (`graph/guarded.py`), **every node the same model**: S1 a refusal instruction to the model itself; S2 the model holds a `classify_request` tool and decides whether to screen itself; S3 a gate node labels the request and the graph routes HARMFUL to a fixed refusal |
| **Defense (D)** | D0 none · D1 system · D2 gate · D3 verify · D4 gate+verify · D5 memory | `eval_mixed.py M -n -m --guard <g>` | the bare model behind a guard run by a separate un-attacked model (`--gate-model`, default `qwen3:14b`) |

- **Exclusive.** Harness and defense don't stack: picking one resets the other to none. This
  matches the scripts, since `eval_guarded.py` has no `--guard`.
- **Model only.** Harness = none and Defense = D0 is the baseline.
- **Ladder rows.** Both lists are shown as clickable rungs with one line on what each adds.
- **Different item order.** A harness run takes the first *m* MedSafetyBench items and then *n*
  MedQA items. A defense run shuffles MedSafetyBench + the 40 hand-written prompts into MedQA.

**Runs table.**
- An H or D badge per run, then MedQA accuracy, false refusal, harmful refused and the
  harmful-response rate.
- **blocked**: what the gate, verifier or memory stopped, how often the `gatetool` agent never
  called its tool, and how often the answerer never ran.
- Runs of the earlier clean-judge guard (`graph/guard.py`, recognised by `judges` in their meta)
  show as *legacy* and are not placed on the D ladder.
- Resume restarts the same run; both scripts skip items already in their file.

**Per-prompt rows** add the guard's decision under the verdict: gate label, verifier label,
memory hit, "answerer never ran".

## 3. API

| route | method | does |
|---|---|---|
| `/api/redteam/ladder` | GET | the harness and defense lists above |
| `/api/redteam/runs` | GET / POST | list runs (with `harness`, `defense`, `gate` stats) / start one run per model: `{models, n, m, harness, guard}`; `harness ≠ none` overrides `guard` |
| `/api/redteam/runs/{id}/items`, `/log`, `/stop` | GET / POST | per-prompt rows, log tail, stop |
| `/api/redteam/runs/{id}` | DELETE | remove a finished run's files |

Run ids are the scripts' own stems:
- harness: `<model>_guard-<harness>_m<m>_n<n>`;
- defense: `<model>[_hw]_n<n>_m<m>_s<seed>[_g<guard>]`.

All run files go to `data/redteam/med/eval_mixed/`.

## 4. What is not in this branch

`attack-defend-ui` was rebuilt on `origin/main`. Two things from the earlier version of the
branch are not here; they're kept in the local tag `backup/attack-defend-ui-pre-rebase`:

- the `/api/defense/*` backend (`web/defense.py`: live guard trace, injection, poisoning, jobs);
- the V0–V5 "system under test" selector.

The defense backend depended on the old `scripts/redteam/med/` layout and had no page. The V
selector was removed at your request.

## 5. Verified

- A `gatenodes` harness run started from `POST /api/redteam/runs` (`qwen3:4b-instruct`, m=1,
  n=1) finished. The gate blocked the harmful prompt and the answerer never ran on it. The test
  run was deleted afterwards.
- The Stream eval page renders at 1600 px (side by side) and 860 px (stacked), checked with
  headless Chrome.
