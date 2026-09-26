---
name: hackalem-readme
description: Write or rewrite README.md so a judge understands the HackAlem AI project in under 2 minutes with zero verbal explanation from the team.
metadata:
  short-description: Write the hackathon README a judge can skim in 2 minutes
---

# skill-readme

Purpose: write or rewrite README.md so a judge understands the project in under 2 minutes with zero verbal explanation from the team.

Invoke this whenever asked to write, update, or review the hackathon README.

## Context

You are writing the README for a hackathon submission. A judge opens it cold, spends 60-120 seconds, and moves on. There is no verbal pitch attached to this file. Everything the judge needs to understand what the project does, why it matters, and that it actually works has to be on this page.

## Rules

1. **Fixed structure.** Use exactly these sections, in this order. Do not add sections beyond this list, do not skip one — if a section has nothing real to say yet, leave a one-line placeholder marked `TODO` rather than deleting the heading.
   - Title + one-line pitch (what it does, for whom — not "an AI agent for energy", the actual outcome)
   - Problem (2-3 sentences max, tied to the actual case brief, no team-internal jargon)
   - Demo (screenshot or GIF goes here, above any explanation text — if no visual asset exists yet, this is the top-priority TODO)
   - SMART goal — one short block stating what the project does in Specific / Measurable / Achievable / Relevant / Time-bound terms. Measurable must name a real number or metric, not "improves efficiency"
   - Architecture (one diagram or one short flow line: input → agent → tools → output; no paragraphs of prose)
   - Quickstart (3-5 shell commands, copy-pasteable, nothing implicit)
   - Why simple (one explicit paragraph: state the design choice to keep the solution simple rather than complex, and the concrete reason — this is a stated differentiator, not filler)
   - Team (names + roles, one line each)

2. **No unexplained jargon**, acronyms, or internal team shorthand. If a term needs the team present to explain it, either define it inline in one clause or cut it.

3. **Every claim must be checkable from the repo as it stands at read time.** Never write "supports X" or "handles Y" for something that is stubbed, mocked, or not yet wired up — mark it as a roadmap item explicitly instead of implying it works.

4. **Quickstart commands must be exact and complete**: paths, exact flags, env vars named. Test them yourself against the actual repo state before writing them down — do not write commands you assume would work.

5. **No walls of text.** Prefer short lines, one idea per line, over paragraphs. A judge skimming should get the shape of the project without reading every word.

6. **Before finishing, re-read the diff between what the README claims and what the code in the repo actually does.** Flag any mismatch to the user rather than silently softening the claim or silently making the code match.

7. **Be brief.** No filler, no hedging, no emojis unless the user's own draft already used them, no marketing tone ("revolutionary", "cutting-edge", "seamless").

## Works with

- Run `hackalem-code-quality` first (or in the same pass) if the codebase itself is still shifting — the README's Architecture and claims sections depend on the code being settled, not the other way around.
- If wording needs sharpening (the one-line pitch, the Problem paragraph), apply `/prompt-engineer`-style tightening: cut vague verbs, name the actual number, one idea per sentence.
