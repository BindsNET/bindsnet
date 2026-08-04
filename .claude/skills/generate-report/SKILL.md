---
name: generate-report
description: Analyzes a promoted experiment directory and produces a structured markdown report under experiments/_reports/. Use after intaking and promoting HPC results to summarize findings. Also use when the user says "generate a report", "summarize results", or "write the experiment report".
allowed-tools: Read, Write, Bash, Glob, Grep
---

# Generate Experiment Report

## Step 1 -- Identify the experiment and task

Ask (or infer from context): what is the promoted experiment path and which task does it belong to?

```bash
ls experiments/<task>/
```

The task name determines which log file to update: `logs/LOG.<task>.md`.

## Step 2 -- Inventory results

```bash
# Check what's in the experiment root
ls experiments/<task>/<experiment_name>/

# Find all result/metric files
find experiments/<task>/<experiment_name>/ -name "*.json" -o -name "*.csv" | head -30

# Check existing reports
ls experiments/<task>/<experiment_name>/_reports/ 2>/dev/null || echo "No reports yet"
```

## Step 3 -- Read key files

- `status.json` -- final state and task metadata
- `manifest.json` -- expected vs observed jobs
- Per-run metric files (e.g. `results.json`, `metrics.csv`, `summary.json`)
- Existing `README.md` inside the package

## Step 4 -- Write the report

Output path: `experiments/<task>/<experiment_name>/_reports/<report_name>/<report_name>.md`

### Report structure

```markdown
# <Experiment Name> -- Report

**Date:** YYYY-MM-DD
**Task:** <task>
**Status:** complete / incomplete (N/M runs)

## Summary

One paragraph: what was tested, key result, headline number.

## Configuration

Table of key hyperparameters swept.

## Results

### Main metric table

| Method | Metric | Mean | P10 | Std |
|--------|--------|------|-----|-----|

### Figures

Embed any generated plots (use relative paths).

### Key findings

Bullet list of the 3-5 most important observations.

## Failure / Incomplete Jobs (if any)

List missing jobs and their impact on conclusions.

## Next Steps

What should be done based on these results?
```

## Step 5 -- Update canonical summary

Ensure `_reports/<report_name>/summary.json` exists with required fields:
- `experiment_id`, `source_packages`, `expected_jobs`, `observed_jobs`, `missing_jobs`, `status`

## Step 6 -- Bi-directional sync (required by AGENTS.md)

### PAPER.md

1. In the **Results** section: update the current best result for this task.
2. If the previous best result is being displaced, move it (with its config) to **Appendix: Ablation History -- \<task\>**.
3. In **Ablation Summary**: add or update the row for this variant in the per-task table.

```markdown
<!-- Ablation Summary table format -->
| Run | Key change | Metric | vs. previous |
|-----|------------|--------|--------------|
```

After adding the new row to the Ablation History table, re-sort it:

```bash
conda activate normal
python scripts/sort_ablation_tables.py
```

### README.md

Update the "Current Results" row for this task (one line per task, overwrite -- do not append).

### logs/LOG.\<task\>.md

Invoke `/log-entry` for this task. The entry should reference the report path and headline number.

### LOG.md (global index)

Update the row for `<task>` with the new Last Run, Status, and link.
