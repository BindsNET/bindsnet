---
name: harvest-slurm
description: Extract durable compute provenance (node, GPU/CPU specs, wall time, job ids, fitness/result lines) from SLURM stdout/stderr logs into research/results/compute_provenance.md, so the raw per-task logs can be deleted without losing reproducibility metadata. Use when a run's SLURM logs pile up and you want the proof-of-run kept in the research record, not as thousands of scheduler files.
user-invocable: true
disable-model-invocation: false
argument-hint: [glob of slurm logs] [run-id]
---

# Harvest SLURM logs → compute provenance

SLURM job logs (`*.out`/`*.err`, `slurm-*.out`, array logs) are scheduler noise, not
research records — but they carry real proof-of-run: node, GPU model + memory, gpu/cpu
counts, start/wall time, job ids, and any fitness/result lines the job printed. This skill
harvests those facts into `research/results/compute_provenance.md` (keyed by LOG run-id),
after which the raw logs are safe to delete.

## Procedure

1. **Identify the log group** and the LOG run-id it belongs to (map job-name → run via
   `research/LOG.md`). Group by job family if a run produced several (train + eval array, ES
   coord + workers, etc.).
2. **Harvest** (prints a Markdown block; add `--json out.json` for machine-readable):
   ```
   python .claude/skills/harvest-slurm/harvest_slurm.py --md \
     --label "<run name>" --run-id <YYYY-MM-DD-rNN> --glob '<path/to/*.out>'
   ```
   It extracts: node(s), GPU model+MiB, gpu/cpu per job, start times, per-cell wall seconds
   (summed), ES per-gen fitness trajectory, aggregate fitness units (n/mean/range), job ids,
   and a count of files with error/traceback markers. Absent fields are simply omitted.
3. **Append the block** to `research/results/compute_provenance.md` under the right run-id.
   Note in it that the raw logs were harvested then deleted.
4. **Delete the raw logs** once the provenance is captured. Do not add them to `.gitignore`
   — the workflow no longer dumps logs into the repo; if any reappear, harvest + delete again.
5. **Sanity-check** the harvested numbers against the run's own result JSON / LOG entry
   (e.g. ES fitness trajectory should match the recorded verdict). Flag discrepancies.

The extractor is pure-stdlib regex over heterogeneous formats; if a new log family prints
specs differently, extend the patterns in `harvest_slurm.py`.
