---
name: hackalem-agentic-loop
description: Design or review the HackAlem AI agent's core orchestration loop (src/agents/orchestrator.py) — the input to agent to tools to output cycle — for simplicity, bounded termination, and demo-safety.
metadata:
  short-description: Review or design the agent's core reasoning loop
---

# skill-agentic-loop

Purpose: design or review the agent's core reasoning loop (the input → agent → tools → output cycle) so it is simple, bounded, debuggable, and demoable — not just working once in a demo run.

Invoke this when writing or editing the orchestration code that decides what the agent does at each step (typically `src/agents/orchestrator.py` in this project), not for general application code — use `hackalem-code-quality` for that.

## Context

Judges and teammates need to be able to point at the loop and say what it does without reading every branch. A hackathon agent that "sometimes works" because the loop has unclear termination or silent failure modes will lose more points than one with fewer features but a loop that is obviously correct.

## Rules

1. **One loop, one shape.** Input → plan/decide → call tool(s) → observe result → decide again → output. Do not build multiple loop variants (e.g. a "fast path" and a "fallback path") unless the task genuinely has two distinct flows — and if it does, name and document both explicitly rather than letting them fall out of implicit branching.

2. **Explicit termination condition.** Every loop needs a hard max-iteration cap and a clear success condition, both visible in one place in the code (not buried across files). An agent that can loop indefinitely is not demo-safe.

3. **Every tool call is inspectable.** Log (to console or a structured log, not just print-and-forget) what tool was called, with what arguments, and what came back, at a level a teammate can read during a live demo to explain what just happened.

4. **Fail loud, not silent.** If a tool call errors or returns something malformed, the loop must not swallow it and continue on bad data. Either retry with a bound, or surface the failure up to the output with a clear message — never guess and proceed.

5. **No speculative generality in tool selection.** Don't build a dynamic tool-registry/plugin-discovery system for 3-4 known tools. A short, explicit if/dispatch over the actual tool set is more debuggable and is what the codebase actually needs today (see `hackalem-code-quality` rule 1).

6. **The system/agent prompt driving the loop gets the same rigor as the code.** Before finalizing it, apply `/prompt-engineer`-style discipline: state the task, constraints, and output format explicitly; test it against a few realistic and a few edge-case inputs; don't ship a prompt that was only ever tried on the happy path.

7. **State stays minimal and visible.** If the loop carries state between iterations (conversation history, intermediate results), keep it in one obvious structure, not scattered across globals or hidden object attributes. A teammate should be able to print the state at any iteration and understand it.

8. **Every loop gets one end-to-end sanity test**: a realistic input goes in, the loop runs to completion within the iteration cap, and the output has the expected shape. This is the single highest-value test in the repo — it proves the demo will not fall over.

9. **Be brief in loop comments.** Mark each phase (plan / act / observe) with a one-line comment, not a paragraph — the structure should be visible from the code, not explained around it.

## Works with

- `hackalem-code-quality` governs the rest of the codebase; this file governs only the loop/orchestration logic.
- `hackalem-readme` depends on this loop being settled before the Architecture section is written — flag it if the README is being written against a loop that's still changing shape.
- For any UI/demo surface built around the loop's output (dashboard, results page), check `/design:design-system` conventions if one exists for the project, so the demo doesn't look inconsistent with the rest of the submission.
