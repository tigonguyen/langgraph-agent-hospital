# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes (from forrestchang/andrej-karpathy-skills, derived from Andrej Karpathy's observations on LLM coding pitfalls). Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Answer Style

**Keep answers onpoint — concise but not too short.** Lead with the answer. Cut preamble, hedging, and restating the question. Give enough context and reasoning to be genuinely useful; don't trim to the point of being cryptic. Expand further when asked or when correctness requires it.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Git Commits

**Format — conventional-commit style: `<type>: <summary>`.**

- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`, `build`, `ci`, `style`. Optional scope: `feat(spec): …`.
- Summary: lowercase, imperative mood, no trailing period, brief (~50 chars).
- One subject line is the norm. Add a short body **only** when the *why* isn't obvious from the diff — don't pad.

**Attribution — commits are the user's.**

- Never include `Co-Authored-By: Claude ...` or any AI-attribution trailer.
- Commits show only the user's git identity.
