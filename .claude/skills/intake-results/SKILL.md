---
name: intake-results
description: Runs the full HPC result intake pipeline for this project. Use when experiment results have been transferred from the cluster into experiments/from_HPC and need to be validated, promoted, and archived. Also use when the user says "intake", "promote results", or "process HPC output".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Intake Results from HPC

Full policy is in [docs/policy/HPC_RESULT_INTAKE_POLICY.md](docs/policy/HPC_RESULT_INTAKE_POLICY.md).

## Step 1 -- Check what arrived

```bash
ls experiments/from_HPC/
```

If packages landed under a staging subdirectory (`hpc_staging`), normalize first:

```bash
bash scripts/hpc/intake_from_hpc_staging.sh --dry-run
# If dry-run looks correct:
bash scripts/hpc/intake_from_hpc_staging.sh --execute --run-intake --interactive
```

```bash
# Extract SLURM job IDs from stdout logs in the incoming package
grep -r "SLURM job:" experiments/from_HPC/ 2>/dev/null | head -20
```

Copy the job ID(s) into the log entry SLURM field when writing the entry.

## Step 2 -- Run intake (dry-run first, always)

```bash
python scripts/hpc/hpc_result_intake.py --interactive --dry-run
```

Review the output, then run live:

```bash
python scripts/hpc/hpc_result_intake.py --interactive
```

## Step 3 -- Decide on each package

| Classification | Condition | Action |
|----------------|-----------|--------|
| `COMPLETE` | `status.json` valid, `state=finished`, all required files present | Auto-promote |
| `INCOMPLETE` | `state=partial/failed` or required files missing | Choose: rerun missing, rerun all, accept partial, or skip |
| `UNKNOWN` | `status.json` missing or malformed | Do NOT silently promote -- investigate first |

Quarantine path (if promotion fails): `experiments/_quarantine/<name>_<timestamp>/`
Audit log: `experiments/from_HPC/intake_events.jsonl`

## Step 4 -- Unify results

```bash
python scripts/hpc/unify_hpc_results.py --dry-run
python scripts/hpc/unify_hpc_results.py --archive-complete
```

## Step 5 -- Identify the task

From the promoted package, determine the task name. This is the subdirectory under `experiments/` where the package landed. All subsequent log writes target `logs/LOG.<task>.md`.

## Step 6 -- Generate report

After promotion, invoke `/generate-report` with the promoted experiment path. That skill handles PAPER.md, README.md, and the per-task log in one step.

## Step 7 -- Post-intake refresh (single command)

Run the wrapper. It writes `run.json` sidecars for the task's runs, archives
`LOG.<task>.md` if it exceeds 10 active entries, regenerates STATUS, rebuilds
`LOG.md`, and rebuilds the archive search index.

```bash
python scripts/post_intake_refresh.py --task <task>
```

For a multi-task intake pass the flag more than once, or omit `--task` to
refresh every active task:

```bash
python scripts/post_intake_refresh.py --task wikitext --task surrogate
python scripts/post_intake_refresh.py       # all tasks
```

This replaces the previous manual sequence of `write_run_sidecar.py` +
`archive_log.py` + `generate_status.py` + `refresh_index.py` +
`build_search_index.py`.

## Step 8 -- Bi-directional sync (required by AGENTS.md)

The `/generate-report` skill covers this. Confirm before closing the session:
- [ ] `logs/LOG.<task>.md` has a new entry
- [ ] `logs/STATUS.<task>.md` regenerated
- [ ] `run.json` sidecars present for every promoted run
- [ ] `LOG.md` global index row is updated
- [ ] PAPER.md Results section is current
- [ ] PAPER.md Ablation Summary row exists for this variant
- [ ] README.md "Current Results" row is current
