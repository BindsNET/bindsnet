---
name: log-entry
description: Appends a new structured entry to LOG.md following the project's experimental log format. Use after completing an experiment run, implementing a feature, fixing a bug, or any change that should be recorded. Also use when the user says "update the log", "add a log entry", or "record this run".
allowed-tools: Read, Edit
---

# Add a Log Entry

## Step 0 — Check and update the pinned block

After writing the entry (Step 3), decide whether this run is a new best or
second-best for this task:

- Read the existing `## 📌 Current Best` block (if any)
- Compare the new result against the current 🥇 and 🥈
- If the new result is better than 🥇: demote current 🥇 to 🥈, set new 🥇
- If the new result is better than 🥈 only: replace 🥈
- If the new result is not top-2: leave the block unchanged
- If no block exists yet: create it with this run as 🥇 (and 🥈 empty)

**The pinned block must always appear before the first `## ` entry in the file.**
When in doubt about ranking (e.g. the metric changed), leave the block as-is
and note "metric changed — manual review needed" in the Notes row.

### PAPER.md sync (required when pinned block changes)

If the pinned block was updated (🥇 changed):

1. Open `PAPER.md` and locate the Results section for this task.
   It will be under `## Results` or a subsection like `### <task>`.
2. Replace the current best result line with the new 🥇 values.
3. Move the displaced result (the old 🥇) into the task's
   `## Appendix: Ablation History — <task>` table as a new row.
   Format: `| YYYY-MM-DD | Run NNN | <what changed> | <metric> | <value> |`
4. Do not rewrite the whole Results section — surgical edit only.

If the metric or task does not appear in PAPER.md Results yet, add it.
If PAPER.md does not exist, skip this step and note it in the log entry.

## Step 1 -- Identify the task

Determine the task name from context (e.g. the experiment directory, user instruction,
or the most recently modified code). The task name is the subdirectory under `experiments/`.

The target file is `logs/LOG.<task>.md`.

## Step 2 -- Read the current log

```bash
head -10 logs/LOG.<task>.md
```

Find the current run number from the first `## ` heading. New run number = last + 1.

## Step 3 -- Write the entry

Use today's date (YYYY-MM-DD). **Prepend** the new entry immediately after the file
header and before the first existing `## ` heading -- logs are reverse-chronological
(newest first).

### Entry format

```markdown
## YYYY-MM-DD Run NNN (Short Title -- Key Distinction)

```yaml meta
run: NNN
date: "YYYY-MM-DD"
task: <task_name>
type: experiment          # experiment | infrastructure | bugfix | intake
slurm_job: <job_id>       # omit if not a SLURM run
key_metric: <metric_name> # omit if no quantitative result
key_value: <number>       # the single most important result
key_unit: "%"             # omit if dimensionless
walltime_mean_s: <number> # mean wallclock seconds across all tasks; omit if not a SLURM run
walltime_std_s: <number>  # std of wallclock seconds across all tasks; omit if not a SLURM run
params:
  model: <model_name>
  rank: <rank>
  method: <method>
  epochs: <N>
  seeds: <N>
tags: [<tag1>, <tag2>]
```

**SLURM job:** `<job_id>` | **Array:** `<array_range>` | **Cluster:** `<cluster_name>`

### Objective

One paragraph explaining what this run/change set out to do and why.

### Differential Update

**`path/to/changed/file.py`** (brief scope summary):
- Bullet explaining each specific change and rationale

**`configs/task/config.yaml`**:
- What changed and why

### Wallclock (required for SLURM runs)

For any entry that corresponds to one or more SLURM jobs, always include a
`### Wallclock` section in the entry body. Extract elapsed seconds from the
`.out` files (the `Elapsed: <N>s` line written by the runner) and compute
mean and std across all tasks in the sweep:

```bash
grep -h "Elapsed:" experiments/from_HPC/<task>/<prefix>.*.out \
  | awk '{print $2}' | sed 's/s//' \
  | python3 -c "
import sys, statistics
vals=[int(l) for l in sys.stdin]
print(f'N={len(vals)}  mean={statistics.mean(vals):.0f}s  std={statistics.stdev(vals):.0f}s  min={min(vals)}s  max={max(vals)}s')
"
```

Format the section as:

```markdown
### Wallclock

**Overall (N=<N> tasks):** mean=<mean>s (<mean_h>h) ± <std>s (<std_h>h) | min=<min>s | max=<max>s

| Config | N | mean (s) | std (s) | mean (h) |
|---|---|---:|---:|---:|
| <config> | <n> | <mean> | <std> | <h>h |
```

Include per-config breakdown whenever the sweep spans multiple configurations.
Omit if all tasks share the same config (report overall only).

### Smoke Test (if applicable)

```
result_1: shape OK
result_2: value OK
```

Brief interpretation.

---
```

### Metadata block rules

The ` ```yaml meta ``` ` block is **required** for every entry:
- Always include `run`, `date`, `task`, `type`
- Include `key_metric`/`key_value`/`key_unit` when the entry reports a quantitative result
- Include `slurm_job` when applicable
- Include relevant hyperparameters in `params` (free-form dict, task-specific)
- Add descriptive `tags` for filtering (e.g. `rank-sweep`, `promoted`, `negative-result`)

The `**SLURM job:**` line is optional — omit it for non-SLURM runs (local tests, debug runs).
When present it must be the first line of the entry body, before Objective.

## Step 4 -- Check if archiving is needed

```bash
count=$(grep -c "^## " logs/LOG.<task>.md)
echo "$count entries"
```

If count > 10, invoke `/archive-log` with `--task <task> --keep-last 5`.
(Hot logs stay small so STATUS + grep remain fast.)

## Step 5 -- Update the global index

Open `LOG.md` and update the row for this task with the new Last Run and Status.

## Step 6 -- Bi-directional sync (required)

After writing the log entry, verify that PAPER.md, README.md, and code are
consistent with what was logged. Any methodological change in code must be
reflected in PAPER.md in the same session.

## Step 7 — Never archive the pinned block

The `## 📌 Current Best` block is permanent. Confirm it is still present at
the top of the file after any archival operation:

```bash
head -5 logs/LOG.<task>.md | grep "📌"
```

If it is missing, restore it from the archive file's header or from memory.

## Step 8 — Refresh STATUS, global index, and search index

After every log entry, regenerate the STATUS file (what Claude loads by default),
the global index, and the search index from ground truth:

```bash
conda activate normal
python scripts/generate_status.py --task <task>
python scripts/refresh_index.py
python scripts/build_search_index.py
```

This replaces the manual row-update instruction. Do not hand-edit LOG.md or
STATUS.<task>.md — both are regenerated from the per-task log.

## Image embedding

If embedding images, copy to `log/img/[original_name]_[YYYY-MM-DD-H_M_S].[ext]`
first, then embed with a relative path.
