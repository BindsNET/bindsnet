---
name: archive-search
description: Searches archived log entries across all tasks or a specific task. Use when the user says "what did we try before", "search the archive", "find old runs with X", or when the agent needs historical context beyond the 5 hot entries in the active log.
allowed-tools: Bash, Read
---

# Search Archived Log Entries

## Step 1 — Identify scope

Determine: is this a single-task search or cross-task?

```bash
# List all archive files
ls logs/archive/

# For a specific task:
ls logs/archive/LOG.<task>.*.md

# Total archive size
wc -l logs/archive/*.md | tail -1
```

## Step 1.5 — Check the search index (fast path)

If the search can be answered by metadata alone (run number, date, metric, title keyword, task),
use the structured index first:

```bash
python3 -c "
import json
for line in open('logs/archive/INDEX.jsonl'):
    e = json.loads(line)
    if '<keyword>' in e.get('title','').lower() or '<keyword>' in str(e.get('tags',[])).lower():
        metric = f\"{e.get('key_metric','')}: {e.get('key_value','')}\" if e.get('key_value') else ''
        print(f\"{e['date']}  Run {e.get('run','?'):>3}  [{e['task']}]  {e['title']}  {metric}\")
"
```

For metric-based queries (e.g. "accuracy > 90%"):

```bash
python3 -c "
import json
for line in open('logs/archive/INDEX.jsonl'):
    e = json.loads(line)
    if e.get('key_value') is not None and e['key_value'] > 90:
        print(f\"{e['date']}  Run {e.get('run','?'):>3}  [{e['task']}]  {e.get('key_metric','')}: {e['key_value']}\")
"
```

If this returns sufficient results, you can skip the full-text grep in Steps 2-3.

## Step 2 — Search

```bash
# Search for a keyword across all archives
grep -l "<keyword>" logs/archive/*.md

# Search within a specific task's archives
grep -n "<keyword>" logs/archive/LOG.<task>.*.md

# Find runs matching a config value (e.g. learning rate)
grep -n "lr=1e-3\|learning_rate: 0.001" logs/archive/LOG.<task>.*.md

# Find runs by approximate date range
ls logs/archive/LOG.<task>.2025-0[1-3]*.md
```

## Step 3 — Read matching entries

Once you have a filename and line number, read the surrounding context:

```bash
# Read a specific archive file
cat logs/archive/LOG.<task>.<date>.<seq>.md

# Or just the matching section (N lines around match)
grep -A 20 "<keyword>" logs/archive/LOG.<task>.*.md | head -60
```

## Step 4 — Summarize findings

Report to the user:
- Which archive files matched
- Run numbers and dates of matching entries
- The key content (config, result, notes) from each match
- Whether any matching run is a candidate to resume or replicate

## Notes

- Archive files are read-only. Never write to them.
- If a keyword matches many entries, filter by date or metric value.
- Active log entries (logs/LOG.<task>.md) are not searched here —
  the agent can read those directly.
