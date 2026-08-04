---
name: report
description: Plain-English status/results report for any project. Use whenever the user
  asks for "a report", "what ran", "status", "what happened", "summary", "explain the
  results", or wants work explained simply for a non-specialist. Reads the project's own
  durable records (logs, status files, trackers, git history).
---

# Plain-English report

The reader is smart and busy, wants the truth fast, and hates jargon. Write so a
non-specialist on this project can follow it.

## Rules
- Plain English. Short sentences, common words. If a technical term is unavoidable,
  define it in the same breath, once (e.g. "gated = run through a strict pass/fail test").
- Facts only. Every claim ties to a source: a run/commit id, a number, a file, a ticket.
  Never invent. If something is unknown, say "unknown".
- No spin. Negative, null, and failed results get equal weight. If an earlier claim was
  reversed, say so plainly under Corrections.
- Brief. Group many items by theme; headline per theme, not a play-by-play.

## Figures & diagrams (a picture often beats a paragraph)
When a figure makes the report clearer, include one — and add a one-line caption saying
the single thing it shows:
- Concept / architecture / flow / decision -> an ASCII diagram inline.
- Data / results (trends, comparisons, distributions) -> generate a REAL plot from the
  data with a small script (never hand-draw numbers); save it under the project's figures
  area and reference it. Follow any "figures regenerated from data" rule the project has.
- Prefer a figure over a long paragraph when it compresses the idea. Skip decorative ones;
  every figure must earn its place.

## Find the sources first (use whatever the project actually keeps)
experiment logs, LOG/STATE/STATUS/CHANGELOG/NOTES, lab notebooks; trackers / results
tables (csv, json, dashboards); recent git commits, open PRs/issues, CI; README/docs for
baseline and goal. For large reads use a subagent; bring back only conclusions. If there
is no durable record, say so and report from what's observable.

## Format
One opening line: the big-picture arc.
Then per theme (or per item if asked):
- Task — what was tried, one sentence.
- Baseline — the number/behaviour being compared against.
- Result — real numbers (value ± spread, sample size, conditions); a figure if it helps.
- What it means — the takeaway, plain.
- How we know — why it's trustworthy (method, controls, sample size, fair test).
Close with:
- Still running / open — each live item: where it is and when it lands.
- Corrections — anything that reversed an earlier belief (integrity first).

## Scope
Honor any window ("last 3 days") or sub-area named. Default: since the last report, or
the currently-active work.