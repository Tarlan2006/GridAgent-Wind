---
name: hackalem-code-quality
description: Keep the HackAlem AI codebase simple and readable during hackathon development, and audit diffs for complexity, dead code, and missing sanity tests before a commit/demo checkpoint.
metadata:
  short-description: Simplicity check and audit pass before committing code
---

# skill-code-quality

Purpose: keep the codebase simple and readable during hackathon development — favor the simplest working solution over clever or heavily abstracted code.

Invoke this when writing new code, reviewing a diff, or before a commit/demo checkpoint.

## Context

You are writing or editing code for a hackathon submission on a tight deadline, judged partly on how clearly the solution is understood, not on how sophisticated the implementation looks. The stated project value is: solving a hard problem with simple code, not solving a simple problem with hard code.

## Rules

1. **Before adding a new abstraction** (base class, interface, plugin system, config-driven dispatch, new module), ask whether the task actually needs it today. If the answer is "it might later," do not build it now. Write the concrete version first.

2. **Prefer one obvious code path** over multiple configurable ones. A judge or teammate reading a file cold should be able to trace what happens without opening three other files, unless the task genuinely requires that split (e.g. `src/agents/tools.py` separated from `src/core/` because they are swapped independently).

3. **Every function and class needs a one-line docstring** stating what it does and why it exists, not restating its name. If you cannot write that line in one sentence, the function is doing too much — split it.

4. **No dead code, no commented-out blocks, no unused imports or parameters left "just in case."** Delete them. Git history keeps the old version if it's ever needed.

5. **Match the agreed file structure**: application code lives in `src/agents/`, `src/core/`, `src/weather/`, `src/app.py`, and `src/backtest.py`; project assets live in `data/`, `training/`, `models/`, `outputs/`, `tests/`, and `docs/`. Do not invent new top-level folders mid-hackathon without flagging it to the user first.

6. **Naming**: functions and variables say what they do in plain words (`fetch_energy_readings`, not `proc1` or `handle_data`). No cleverness in naming — a tired teammate at 3am should understand it without asking.

7. **Every new tool or agent capability gets at least one sanity test before being called "done"**: a call with realistic input, an assertion on the shape of the output. Not full coverage — just proof it runs and returns something sane.

8. **When two solutions solve the same problem, pick the one with fewer moving parts** (fewer files, fewer dependencies, fewer configuration knobs), even if the more complex one is marginally more "correct" in the general case. This is a hackathon constraint, not a production system.

9. **Be brief in comments and commit messages.** State what changed and why in one line, no filler.

## Works with

- After a meaningful diff, run a `/code-reviewer`-style pass over the changed files: flag correctness risks and simplification opportunities, not style nitpicks. Report findings, don't silently rewrite unless asked.
- If the code under review is the agent's core loop (planning, tool selection, termination), defer to `hackalem-agentic-loop` for the rules specific to that part — this file governs general code hygiene, not loop design.
