---
name: hpc-submit
description: Guides creation of a compliant SLURM submitter + runner script pair for this project. Use when preparing a new experiment for cluster submission, writing new HPC scripts, or auditing existing scripts against project policy. Also use when the user says "create a submitter", "write SLURM scripts", or "submit to cluster".
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# HPC Submit — New Experiment

Full policy: [docs/policy/HPC_RESULT_INTAKE_POLICY.md](docs/policy/HPC_RESULT_INTAKE_POLICY.md).

**Cluster facts + resilience templates live in the global `pax-hpc` skill**
(`~/.claude/skills/pax-hpc/`): cluster choice (NEW=login-p02/conda/2d cap;
OLD=login-prod/Singularity-always/6d cap), VRAM-precise GPU labels
(`references/new-cluster.md`), and the copy-paste requeue/resume/VRAM-guard
scaffolding (`references/resilience.md`). This project skill is the *house style*
on top of that. Verified on both clusters, re-confirmed 2026-07-17: `preempt`
`GraceTime=0` (no window on a preempt kill — only the last periodic checkpoint
survives; the 30 s `KillWait` grace is the *walltime*/`scancel` path, not
preemption), `PreemptMode=REQUEUE` **does not auto-requeue** — so the resilience
contract below is mandatory, not optional. **Preempt jobs also require
`--qos=preempt`** (not just `-p preempt`): without it the job silently runs under
the `normal` QOS (cpu=250) and gets none of the higher preempt ceiling. The
`normal` and `preempt` GPU caps are separate 10-GPU pools, so preempt spillover
buys up to 20 concurrent GPUs (verified 2026-07-17).

## Script pair overview

Every experiment needs exactly two scripts:

| Script | Runs on | Does |
|---|---|---|
| `scripts/<task>/submit_<task>_<variant>.sh` | Gateway/login node | `sbatch` orchestration only |
| `scripts/<task>/run_<task>_<variant>.sh` (or `.py`) | Compute node via SLURM | Training, logging, packaging |

## Submitter checklist (`bash` on gateway)

- [ ] `set -euo pipefail` at top
- [ ] Verifies `sbatch` is available
- [ ] Validates required files/paths before submission
- [ ] Prints usage and key env overrides
- [ ] GPU constraint default uses **VRAM-precise labels**, not bare type names:
      `a100-80G|h100-80G|h200-141G|l40s-48G` (bare `a100` silently admits the
      40 GB card → OOM). Use the `constraint_for_tier` recipe from the global
      `pax-hpc` skill's `references/new-cluster.md`.
- [ ] Supports override via `GPU_CONSTRAINT` or `SBATCH_CONSTRAINT` env vars
- [ ] Constraint fallback: try full VRAM-precise alternation → retry token-by-token → retry without `--constraint`
- [ ] **Resilience SBATCH directives on EVERY job** (walltime trap is universal —
      see contract below): `--requeue`, `--signal=B:USR1@30`, `--open-mode=append`
- [ ] **`preempt` jobs pass BOTH `--partition=preempt` and `--qos=preempt`** — the
      QOS flag is required or you silently fall back to the `normal` ceiling (cpu=250)
- [ ] `--time` within the cap of the target cluster (**2d new / 6d old**); never rely on the 15-min default
- [ ] **No hardcoded `--exclude`** — bad nodes are learned at runtime by the runner
- [ ] Token rotation: when splitting across tokens, rotate preference across jobs (not all on first accepted token)
- [ ] One SLURM job per independent sweep point (mode × seed grid) — no bundling unless true dependency
- [ ] Big / multi-stage work uses `--dependency` chains, not one mega-job
- [ ] Does NOT run Python, training, plotting, or `singularity exec` on gateway

## Runner checklist (compute node)

- [ ] Writes only inside one unique package root per task (no out-of-package writes)
- [ ] Emits all four required files inside package root:
  - `status.json` with `task_id` and `state` (`finished`/`partial`/`failed`)
  - `manifest.json` with `required_files` and `promote_to`
  - `README.md` (human-readable run context)
  - `INTAKE.md` (local post-transfer commands)
- [ ] Uses fail-fast validation for required runtime inputs
- [ ] Verifies required Python modules before training
- [ ] Records bootstrap/failure outcome in package metadata
- [ ] Logs node/hardware info at startup (see **Hardware & timing log block** below)
- [ ] Logs job start time and wall-clock end time (success and failure paths)
- [ ] Registers EXIT trap that logs reason + elapsed time on any exit (OOM, kill, error)
- [ ] **VRAM guard**: read `nvidia-smi` at startup; if VRAM `< MIN_GPU_MEM_MB`,
      append the node to `ExcNodeList` and `scontrol requeue` (don't just requeue —
      exclude, or it loops onto the same small card)
- [ ] **CUDA-init guard**: `torch.zeros(1, device="cuda")` probe AFTER env activate;
      on failure, exclude-node + requeue (a healthy `nvidia-smi` can still have a dead CUDA ctx)
- [ ] **Signal traps** (`SIGUSR1` + `SIGTERM`) on EVERY job: requeue on the walltime
      warning (universal — a job that overruns `--time` must requeue, never die) AND on
      preemption; on preemption requeue *first* (GraceTime=0 → no window). Gate the
      requeue loop on a `DONE` sentinel so it stops once the work is complete. See contract below.
- [ ] **Robust resume**: resolve the resume dir by probing for the **highest-step
      checkpoint** across `run_*/`, not a single sentinel file; honor `RESUME_RUN_DIR`
- [ ] **Deliverable checkpoints off by default** (see **Checkpoint policy** below):
  - Training code does not call `torch.save` / `save_pretrained` / Keras `save_best=True`
    for *deliverable weights* unless gated by `--save-checkpoint` (or `KEEP_CHECKPOINT=1`), default **off**
  - `required_files` in manifest.json omits any `*.pt`/`*.pth`/`*.safetensors`/`*.bin` entries
  - **EXCEPTION — resume checkpoints are mandatory for preempt/long jobs**: a single
    latest, atomically-written (`*.tmp` → `mv`) checkpoint every ≤~30 min of compute,
    **deleted on clean finish**. This is transient resume state, not a deliverable, so
    it stays out of `required_files` but is required for the job to survive `GraceTime=0`.

## Walltime policy (user directive 2026-07-11 — HARD RULE)

- **`--time=23:55:00` is the MAXIMUM walltime for any job.** Do not request more.
- **Every task MUST be able to requeue itself** on timeout OR on an HPC kill, and
  resume from its last checkpoint — never from step 0. A job that cannot save and
  resume cannot be submitted.
- Rationale (measured): a tied-LoRA run at ~270M on FineWeb-Edu took **22h31m**;
  a 14h walltime would have failed it. Long jobs are normal, so the 23:55 cap +
  self-requeue is what makes them finish. Jobs longer than 23:55 complete by
  requeueing across multiple windows, resuming from the periodic checkpoint.
- This means the periodic atomic checkpoint (≤ ~30 min) is **mandatory for every
  job**, not just preempt/near-cap ones — it is what a requeue resumes from.

## Resilience contract (walltime trap MANDATORY for ALL jobs)

> **User directive (2026-06-10): every HPC task traps the walltime SIGTERM and
> auto-requeues until completed — no exceptions.** Walltime estimates are
> routinely wrong; an overrun must resume, never silently die. The periodic
> *checkpoint* layer below stays scoped to preempt / near-cap jobs (it's what
> lets a requeue resume near where it died); the *trap + `--requeue`* layer is
> universal. The requeue loop is gated on a `DONE` sentinel — requeue while
> unfinished, stop once the run completes.


Copy-paste bash/Python templates: global `pax-hpc` skill →
`references/resilience.md`. The canonical in-repo example is
[scripts/wikitext/run_wikitext_v6_slurm.sh](scripts/wikitext/run_wikitext_v6_slurm.sh)
(already implements `--requeue`, USR1/TERM handlers, VRAM guard, cuda-exclude).

Why mandatory: `preempt` has `GraceTime=0` on both clusters — a preempted job is
SIGKILLed with **no checkpoint window**, and `PreemptMode=REQUEUE` does **not**
auto-requeue unless the job is `--requeue`-eligible. Several tasks also exceed the
cap (AWD-LSTM ≈3d vs the new 2d cap), so they *cannot finish* without resume.

The contract has three layers — get all three or the job is not resilient:

1. **Eligibility (submitter):** `#SBATCH --requeue` + `--signal=B:USR1@30` +
   `--open-mode=append`, `--time` within the cluster cap. On `preempt`, also
   `#SBATCH --qos=preempt` (required for the higher ceiling; a bare `-p preempt`
   falls back to the `normal` QOS).
2. **Survival (runner, bash):** trap `SIGUSR1`/`SIGTERM` → on preemption requeue
   *first* then best-effort checkpoint, on walltime checkpoint-then-requeue;
   VRAM-guard + cuda-guard that exclude-node + requeue; resume by highest-step probe.
3. **State (trainer, Python):** trap `SIGUSR1` → write latest checkpoint → exit;
   **periodic atomic checkpoint every ≤~30 min** (the only thing that survives a
   0-grace preemption); resume from the highest-step checkpoint; delete on clean finish.

Requeue lives in the **bash** layer (survives a hung trainer); the Python layer only
checkpoints. The walltime trap + `--requeue` are NOT optional for any job —
CPU-only and sub-hour jobs on `batch`/`gpu` still carry them (a wrong runtime
estimate must requeue, not die); they may skip only the *periodic checkpoint*
layer if a from-scratch restart on requeue is acceptable.

## Checkpoint policy

> Full policy: [docs/policy/HPC_RESULT_INTAKE_POLICY.md](docs/policy/HPC_RESULT_INTAKE_POLICY.md)
> section **Checkpoint Persistence Policy**.

**Default**: evaluation-only experiments persist config + metrics + history,
not weights. `config + code commit + seed = same result`; checkpoints add no
reproducibility and a lot of disk.

**Opt in** (explicit flag, default off) only when:
1. The weight artifact itself is the deliverable (released model, teacher, warmstart)
2. Multi-stage training needs mid-run state (ReLoRA merge, warmstart handoff)
3. Active investigation needs post-hoc weight probing (eigenvalues, activations)

When opting in, record `extra.checkpoint_reason` in manifest.json and keep the
file optional (never in `required_files`) so evaluation reruns don't trip
`INCOMPLETE`.

**Cleanup of legacy dirs**: `python scripts/strip_checkpoints.py --scan experiments/<task>/`
removes `*.pt`/`*.pth`/`*.safetensors`/`*.bin` while preserving all metadata.

## Hardware & timing log block

Paste this block **once, near the top of every runner script**, immediately after the SLURM env debug section:

```bash
# =============================================================================
# HARDWARE & TIMING INFO
# =============================================================================
JOB_START_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
JOB_START_EPOCH=${SECONDS}
echo "===== JOB START: ${JOB_START_TIME} ====="
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "NodeList:   ${SLURM_JOB_NODELIST:-unknown}"
echo "JobID:      ${SLURM_JOB_ID:-unknown}"
echo "ArrayTask:  ${SLURM_ARRAY_TASK_ID:-none}"
echo "Cluster:    $(hostname -d 2>/dev/null || hostname)"
echo "SLURM job: ${SLURM_JOB_ID}"

# GPU info (model + VRAM)
if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "  GPU %s: %s  |  VRAM: %s MiB  |  Driver: %s\n",$1,$2,$3,$4}'
else
    echo "GPU: not available"
fi

# CPU memory
echo "--- CPU Memory ---"
free -h | grep -E '^Mem'

# Register EXIT trap: always log finish time + status + elapsed
_log_job_exit() {
    local exit_code=$?
    local elapsed=$(( SECONDS - JOB_START_EPOCH ))
    local end_time
    end_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo ""
    echo "===== JOB EXIT: ${end_time} ====="
    echo "Exit code:    ${exit_code}"
    echo "Elapsed:      ${elapsed}s  (~$(( elapsed/3600 ))h $(( (elapsed%3600)/60 ))m $(( elapsed%60 ))s)"
    if [ "${exit_code}" -eq 0 ]; then
        echo "Status:       SUCCESS"
    elif [ "${exit_code}" -eq 137 ]; then
        echo "Status:       KILLED (OOM or external kill signal — check memory limits)"
    else
        echo "Status:       FAILED (see Python traceback above)"
    fi
    echo "Start:        ${JOB_START_TIME}"
    echo "End:          ${end_time}"
}
trap '_log_job_exit' EXIT
# =============================================================================
```

**Debug notes captured by this block:**
- GPU model + VRAM → tells you if a different GPU type ran (A100 vs L40s affects numerics/OOM thresholds)
- Exit code 137 → OOM kill (not a Python crash); increase `--mem` or reduce batch size
- Elapsed time → compare against `--time` budget; if close, job was likely preempted or killed at limit
- Python loss/parameter logs → emitted to the SLURM `.out` file by the training script; grep `loss` or `param` in that file to trace divergence

## Policy alignment checklist

- [ ] Config/wrapper paths remain under task-scoped roots (not repo root)
- [ ] Script behavior aligns with `scripts/validate_experiment_policy.py`
- [ ] Docs synchronized: `README.md`, `PAPER.md`, `LOG.md`
- [ ] Aggregation of split-job outputs happens locally after intake

## Common failure patterns to avoid

1. `Invalid feature specification` — use bracket constraint with token fallback
2. Running `bash submitter.sh` instead of `sbatch submitter.sh` (or vice versa)
3. Python/container execution on gateway node
4. Shared HF cache across parallel jobs — use per-job `HF_HOME` or `TRANSFORMERS_CACHE`
5. Non-self-contained outputs (writes outside package root)
6. **Silent OOM from a bare `a100` constraint** landing on a 40 GB card — use a
   VRAM-precise label (`a100-80G`) + the runtime VRAM guard with `MIN_GPU_MEM_MB`
7. **Preempted job vanishes instead of requeuing** — missing `#SBATCH --requeue`
   (`PreemptMode=REQUEUE` alone does not requeue your job)
8. **Resume starts from zero after ≥2 requeues** — trusting a stale sentinel file
   instead of probing for the highest-step checkpoint

## After writing scripts

Run the policy validator:
```bash
python scripts/validate_experiment_policy.py
```

Then invoke `/log-entry` to record the new scripts.
