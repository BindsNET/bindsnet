---
name: archive-log
description: Archives old entries from a per-task log file when it grows too large. Moves entries beyond the keep-last threshold to logs/archive/. Also use when the user says "archive the log", "trim the log", or "the log is too long".
allowed-tools: Bash, Read, Edit
---

# Archive Old Log Entries

## When to invoke

- Manually, when a task log is getting unwieldy
- Automatically, called by `/log-entry` when entry count > 10
- Automatically, called by `/intake-results` Step 8
- During migration from the old monolithic LOG.md

## Step 1 -- Check the current state

```bash
# Count entries
grep -c "^## " logs/LOG.<task>.md

# See the oldest entries that will be archived
grep "^## " logs/LOG.<task>.md | tail -10
```

## Step 2 -- Dry-run the archive script

```bash
python scripts/archive_log.py --task <task> --keep-last 5 --dry-run
```

Read the output carefully:
- How many entries will be archived?
- What is the archive filename?
- What pointer line will be written into the active log?

## Step 3 -- Execute

```bash
python scripts/archive_log.py --task <task> --keep-last 5
```

## Step 4 -- Verify

```bash
# Active log should now have 5 entries + 1 pointer line near the bottom
grep -c "^## " logs/LOG.<task>.md
grep "^> Archived entries" logs/LOG.<task>.md

# Archive file should exist
ls logs/archive/LOG.<task>.*.md
```

### Pinned block check

```bash
head -5 logs/LOG.<task>.md | grep "📌"
```

The `## 📌 Current Best` block must still be present at the top after archival.
If it is missing, the `split_entries` function did not handle it correctly —
restore from the archive and fix the script before continuing.

## Step 5 -- Update global index

Open `LOG.md` and confirm the row for `<task>` still points to `logs/LOG.<task>.md` (the active file -- not the archive).

## Step 6 -- Rebuild the search index

The archive script auto-triggers this, but if it didn't run, do it manually:

```bash
python scripts/build_search_index.py
```

This regenerates `logs/archive/INDEX.jsonl` with all entries (active + archived).

## Step 7 -- Refresh STATUS

The hot log just changed; regenerate the compact STATUS file (Claude's default
read surface):

```bash
python scripts/generate_status.py --task <task>
```

## Archive file naming

`logs/archive/LOG.<task>.<YYYY-MM-DD>.<seq>.md`

The `<seq>` suffix (01, 02, ...) prevents collisions if you archive the same task multiple times in one day.

## Pointer line format (written into the active log)

```markdown
> Archived entries 001--NNN -> logs/archive/LOG.<task>.<date>.<seq>.md
```

This line is always kept at the **bottom** of the active log, below all `## ` entries.
Multiple pointer lines accumulate as more archives are created -- do not delete them.
