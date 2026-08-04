---
name: generate-leaderboard
description: Reads the Current Best pinned block from every per-task log and produces a ranked leaderboard table across all tasks. Use when the user says "show me the leaderboard", "what is the best result per task", "update the summary table", or "what should I work on next".
allowed-tools: Read, Write, Bash, Glob
---

# Generate Cross-Task Leaderboard

## Step 1a — Fast path: aggregate run.json sidecars (preferred)

Every promoted run has a flat `run.json`. Aggregate them without touching markdown:

```bash
python3 -c "
import json, pathlib
from collections import defaultdict
best = defaultdict(lambda: (None, None))  # task -> (value, record)
for p in pathlib.Path('experiments').rglob('run.json'):
    try:
        r = json.loads(p.read_text())
    except Exception:
        continue
    task = r.get('task'); v = r.get('key_value'); km = r.get('key_metric')
    if task is None or v is None: continue
    # Lower-is-better for ppl/loss/mae/mse; higher-is-better otherwise
    lower_better = km in {'test_ppl','val_ppl','mean_ppl','loss','mae','mse'}
    cur_v, _ = best[task]
    if cur_v is None or (lower_better and v < cur_v) or (not lower_better and v > cur_v):
        best[task] = (v, r)
for task, (v, r) in sorted(best.items()):
    print(f\"{task:<30} {r.get('key_metric',''):<10} {v:>10}  run={r.get('run_id')}  {r.get('date','')}\")
"
```

If a task has no sidecars, fall back to Step 1b.

## Step 1b — Fallback: pinned blocks from STATUS / LOG

```bash
for f in logs/STATUS.*.md logs/LOG.*.md; do
  echo "=== $f ==="
  awk '/^## 📌 Current Best/{found=1} found{print} /^---/{if(found) exit}' "$f"
  echo ""
done
```

Prefer `logs/STATUS.*.md` — smaller and authoritative for the pinned block.

## Step 2 — Build the leaderboard table

For each task, extract the 🥇 row from its pinned block.
Produce a single markdown table sorted by metric value (best first).
Group tasks by metric type if they use different metrics.

Output format:

```markdown
## Leaderboard — <date>

### <metric group> (higher is better / lower is better)

| Rank | Task | Run | Date | Metric | Value | Config |
|------|------|-----|------|--------|-------|--------|
| 1 | mnist_cnn | Run 057 | 2025-04-10 | acc | 98.3% | lr=1e-3 |
| 2 | cifar10 | Run 031 | 2025-03-22 | acc | 94.1% | lr=3e-4 |
...

### No result yet

Tasks with no quantitative result: <comma-separated list>
```

## Step 3 — Write output

Save to `experiments/_leaderboard/leaderboard_<YYYY-MM-DD>.md`.
Also print to stdout for immediate review.

## Step 4 — Embed in PAPER.md (optional)

If the user asks to update PAPER.md, replace the content under
`## Summary` or `## Current Results` with the leaderboard table.
Do not touch other sections.

## Step 5 — Bi-directional sync

No LOG entry needed for leaderboard generation unless results changed.
If you updated PAPER.md, note it in the relevant task's log entry.
