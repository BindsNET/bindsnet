---
name: pax-hpc
description: >-
  Prepare and launch compute work on the user's two Tufts "Pax" SLURM HPC
  clusters. Use this skill WHENEVER the user wants to run, batch, submit,
  array, schedule, profile, or GPU-accelerate any job, experiment, script, or
  training run that is meant for the cluster — even if they don't say the word
  "cluster". Trigger on mentions of sbatch, srun, salloc, squeue, slurm,
  "the cluster", "on pax", "submit a job", "run this on a GPU", singularity /
  apptainer containers, conda environments meant for the cluster, login-p02,
  login-prod-03, preempt / requeue / resume, or any request to turn local code
  into something that runs on HPC. There are two clusters: OLD (login-prod-03,
  RHEL7, runs programs via Singularity) and NEW (login-p02, Rocky9, runs
  programs via conda). Always ask which one first.
---

# Pax HPC (two clusters)

This skill captures how the user works with their two Tufts Pax SLURM clusters
so they never have to re-explain it. Both clusters share ONE filesystem, so
data, repos, conda envs, and Singularity images live in the same place on both.
The clusters differ in OS, software runtime, walltime limits, and GPU hardware.

| | OLD cluster | NEW cluster |
|---|---|---|
| Login host | `login-prod-0X.pax.tufts.edu` (01, 03, …) | `login-p02.pax.tufts.edu` |
| Compute nodes | legacy `cc1gpu / s1cmp / p1cmp / d1cmp###` | `pax001`–`pax120` |
| OS / Slurm | RHEL 7.5 / Slurm 23.02 | Rocky 9.6 / Slurm 23.11 |
| Runs programs via | **Singularity** (`module load singularity`) | **conda** (`conda activate`) |
| Max walltime | 6 days | **2 days** |
| GPUs | p100, v100, t4, a100(40G), + some newer | h100, h200, a100(40/80G), l40s, + more |
| Why not the other runtime | host glibc too old for modern conda binaries | conda is native; apptainer also available |

Username: `hhazan01`. Full hardware, partitions, QOS, and GPU **VRAM-tagged
feature labels** are in `references/old-cluster.md` and
`references/new-cluster.md` — read the relevant one when sizing a job.
Resilience mechanics (requeue, resume, preempt signal-trap, walltime
self-requeue, adaptive node-exclude, VRAM guard, dependency chains) live in
`references/resilience.md`. Bringing results back and classifying them lives in
`references/intake.md`.

## The non-negotiable rules

1. **Never run ANY task on a login node — under any circumstances** (OLD or NEW).
   This is absolute (user directive, 2026-06-07). No training, no data crunching,
   no `python long_thing.py`, no compiling, no `singularity exec` of a workload —
   **and also no "lightweight" inline work**: no `python scripts/intake.py`,
   `consolidate.py`, results-aggregation, jsonl parsing, or env-python one-liners
   on the login node, even for a quick verification. If you need to run code
   against cluster data, submit it (`sbatch`) or use an interactive compute
   allocation (`srun --pty`) — or pull the data local and run it here. The login
   node is ONLY for: editing files, submitting jobs (`sbatch`), launching
   allocations (`srun --pty`/`salloc`), monitoring (`squeue`, `sacct`,
   `scancel`), small `ls`/`tail`/`grep` on output files, and `rsync`. **This
   applies to `login-p02` even when it is used as a no-Duo ProxyJump gateway to
   the OLD cluster — gateway convenience is not a compute license.** When in
   doubt, it does not run on login: pull the data local and run it here, or
   `sbatch`/`srun`.

2. **Always ask which cluster first: OLD or NEW?** The answer changes the
   runtime wrapper (Singularity vs conda), the walltime ceiling (6d vs 2d), and
   which GPUs are valid. Do not guess. (If the user already said this session,
   don't re-ask.)

3. **Prepare everything locally.** Build the job script and any code/config in a
   local staging folder. Do not assume you can write on the cluster.

4. **The user does the rsync** unless they explicitly say otherwise. Give them
   the exact `rsync` command, but they run it.

5. **End every preparation with ONE line** the user pastes on the login node to
   launch — normally `sbatch <script>` (or an array/dependency line). That single
   line is the deliverable of local prep.

6. **The preempt contract — EVERY `preempt` job carries it, no exceptions**
   (user directive, 2026-06-20). A job on the `preempt` partition that is not
   wired for kill→requeue→resume is a bug, not a job. The full contract:
   - **`#SBATCH --requeue`** — makes the job requeue-eligible. Without it,
     preemption *cancels* the job (it does not come back). Mandatory.
   - **Trap the kill signals SLURM sends to stop the process** — the runner
     traps **`SIGUSR1` + `SIGTERM` + `SIGCONT`** (request the advance warning
     with `#SBATCH --signal=B:USR1@30`) and routes them to one handler that
     `scontrol requeue "$SLURM_JOB_ID"`s **first**, then forwards a checkpoint
     request to the trainer. Requeue first so a slow checkpoint can never cost
     you the restart. Do the `scontrol requeue` in bash, the checkpoint in
     Python (§0.1 of resilience.md).
   - **Auto-requeue until done** — the requeued job restarts the batch script
     from the top and keeps going until a completion (`DONE`) sentinel exists;
     `SLURM_RESTART_COUNT` caps runaway loops.
   - **Periodic checkpoint + resume-from-last is MANDATORY, not optional.**
     `GraceTime=0` on `preempt` (both clusters, verified) means preemption gives
     **no on-signal checkpoint window at all** — SIGTERM and SIGKILL arrive
     back-to-back. The *only* thing that survives is the last **periodic
     checkpoint** (write every K steps, ≤~30 min of compute at risk; write
     atomically — tmp + `os.replace`). On restart the job MUST detect the
     checkpoint and **resume from the last completed step**, never from step 0.
     A preempt job whose code cannot save and resume from a checkpoint cannot
     run on `preempt` — fix the resume path first or use a non-preempt partition.
   - Verified facts to size against: `PreemptMode=REQUEUE`, `GraceTime=0`,
     `KillWait=30`. Full copy-paste scaffolding in `references/resilience.md`
     §0–§2.

7. **EVERY task auto-requeues on walltime until it finishes — but ONLY on
   walltime, and NEVER a crash, a cancel, or a completed job** (user directive,
   2026-06-10/-12). Walltime estimates are routinely wrong; a job that overruns its
   `--time` must resume, not die. Every runner carries `#SBATCH --requeue` +
   `--signal=B:USR1@N`, **traps `SIGUSR1` + `SIGTERM` + `SIGCONT`**, forwards a
   checkpoint request to the child, and after the child is reaped requeues
   **iff: (a) the `SIGUSR1` walltime warning fired (`WALLTIME=1`) AND (b) no
   completion sentinel (unfinished)**. The requeue is **rc-independent**: a walltime
   stop may exit 0 (checkpointed in time) or non-zero (hard-killed) — `WALLTIME=1`
   already excludes crashes and `scancel` (neither sends `SIGUSR1`), and the sentinel
   excludes a clean finish. Trapping `TERM`/`CONT` only keeps the shell alive to make
   that decision; a `scancel` (also `TERM`) carries `WALLTIME=0`, so it is never
   requeued. `SLURM_RESTART_COUNT` caps the loop. Preempt-partition jobs differ (no
   warning) — see §1b. Refs: `references/resilience.md` §0.2, §0.3, §1, §1b.

## Standard workflow

1. **Ask: OLD or NEW cluster?** Then read that cluster's reference file.
2. **Clarify the job shape**: CPU or GPU, cores, memory, expected walltime,
   single job vs array vs dependency chain, which conda env (new) or `.sif`
   image (old). Size it against the reference file's limits.
3. **Pick the GPU robustly** (see *GPU requests* below) — by VRAM constraint,
   not by bare typed gres.
4. **Wire resilience — always.** Every job gets the walltime SIGTERM trap +
   `--requeue` + resume scaffold from `references/resilience.md` (rule 7).
   `preempt` jobs and jobs near the cap additionally need the periodic
   checkpoint (≤~30 min at risk) so they resume near where they died, not from
   step 0.
5. **Write the script(s) locally** in a staging dir (default `./<project>_job/`).
6. **Give the rsync command** to the shared project space. Default destination:
   `/cluster/tufts/levinlab/hhazan01/<project>/`.
7. **Give the one launch line**, e.g.
   `cd /cluster/tufts/levinlab/hhazan01/<project> && sbatch run.sbatch`.
8. **Tell them how to pull results back** (`references/intake.md`).

## Paths (shared across BOTH clusters)

- **Home:** `/cluster/home/hhazan01` — **small (~28 GB quota, usually near
  full)**. Do NOT stage data, datasets, package caches, or run output here. Use
  it only for dotfiles and small scripts.
- **Lab / project space:** `/cluster/tufts/levinlab/hhazan01` — main working
  area. Conda envs, Singularity images, git repos, datasets, and staged jobs all
  live here. `~/cluster` already symlinks to it.
- **Scratch:** `/cluster/scratch` — large, fast, for transient run output.
- **Conda envs (new):** `/cluster/tufts/levinlab/hhazan01/miniconda3/envs/*`
  (verified 2026-05-31: `betse`, `bindsnet`, `delayW`, `dt-d4rl`, `LoRa`).
- **Singularity images (old):** `/cluster/tufts/levinlab/hhazan01/singularity/*.sif`
- `umask 0007` is already in `.bashrc` (group-shared lab files).

Because home is tiny, always point conda package dirs, HuggingFace caches, torch
extensions, and dataset downloads into the lab/scratch space, not home. Do not
write large checkpoints unless the weight artifact is itself the deliverable —
config + code commit + seed reproduces a result; checkpoints just burn disk.
(Checkpoints written *for resume* are exempt — keep only the latest, delete on
clean finish.)

## Job script templates

Both clusters use the same SBATCH header style; only the run wrapper differs.
Keep `--time` within the ceiling (6d old, **2d new**).

### Shared SBATCH header

```bash
#!/usr/bin/env bash
#SBATCH --job-name=NAME
#SBATCH --output=./res_%j_%N.txt
#SBATCH --error=./res_%j_%N.err
#SBATCH --time=00-04:00:00          # DD-HH:MM:SS  (<=2d new, <=6d old)
#SBATCH --partition=PARTITION       # see reference file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
## GPU jobs: see "GPU requests" below — prefer --gres=gpu:1 + --constraint=<VRAM label>
## Arrays add, e.g.:    #SBATCH --array=1-20   (MaxArraySize=2000)
## Resilient jobs add:  #SBATCH --signal=B:USR1@30  --open-mode=append  --requeue
```

### NEW cluster (login-p02) — conda

```bash
source ~/.bashrc
conda activate ENV_NAME          # e.g. LoRa, delayW, betse, bindsnet, dt-d4rl
cd /cluster/tufts/levinlab/hhazan01/PROJECT
python my_script.py --args ...
```

### OLD cluster (login-prod-03) — Singularity

Conda will NOT run here (host glibc too old). Wrap the workload in the container:

```bash
module load singularity
SIF=/cluster/tufts/levinlab/hhazan01/singularity/IMAGE.sif
cd /cluster/tufts/levinlab/hhazan01/PROJECT
singularity exec --nv --bind /cluster:/cluster "$SIF" \
    python3 my_script.py --args ...
```

`--nv` enables GPUs (drop it for CPU-only). `--bind /cluster:/cluster` makes the
shared filesystem visible inside the container. `--cleanenv` helps when host env
vars leak in.

## GPU requests (the #1 source of silent OOMs)

**Typed gres does not pin VRAM.** `--gres=gpu:a100:1` can land you on a 40 GB
A100 *or* an 80 GB A100 — they share the type name `a100`. A 300M+ model that
needs 80 GB will then OOM at runtime with no obvious cause. **Request by VRAM
constraint instead:**

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G        # VRAM-precise; see reference file for labels
```

- Use the VRAM-tagged feature labels (`a100-40G`, `a100-80G`, `h200-141G`,
  `l40s-48G`, `h100-80G`, …) from the cluster reference file. For a "≥80 GB"
  job use the alternation `a100-80G|h100-80G|h200-141G`, never coarse
  `a100|h100|h200` (which silently admits 40 GB cards).
- **Belt-and-suspenders:** even with a constraint, every A100 (or mixed-VRAM)
  request must carry the runtime **VRAM guard** from `references/resilience.md`:
  the runner reads `nvidia-smi` at startup and, if it landed below
  `MIN_GPU_MEM_MB`, excludes that node and requeues automatically. Constraint +
  guard together make a wrong-card landing self-correcting.
- **Constraint fallback ladder** (avoids `Invalid feature specification` hard
  failures): submit with the full alternation → on reject, retry token-by-token
  → last resort, retry with no constraint. See `references/resilience.md`.

## Excludes are learned, not hardcoded

**Every submission starts with NO `--exclude`.** Do not bake a stale bad-node
list into submitters. Instead, the runner discovers a bad node at runtime (wrong
VRAM, `libtorch_global_deps.so` NFS error, CUDA init failure) and requeues
itself with that node added to a job-scoped `ExcNodeList`. This way each fresh
submission re-tests previously-bad nodes (they get fixed), and a still-broken
node is walked away from within seconds. Mechanism in `references/resilience.md`.

## Splitting big tasks — dependency chains

For work that exceeds one allocation (a multi-stage pipeline, or a run longer
than the 2d/6d cap split into segments), chain jobs with SLURM dependencies
rather than one mega-job:

```bash
jid1=$(sbatch --parsable stage1.sbatch)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 stage2.sbatch)
jid3=$(sbatch --parsable --dependency=afterok:$jid2 stage3.sbatch)
```

`afterok` = run only if the prior succeeded; `afterany` = run regardless (use
for a resume segment that should start even if the prior hit the walltime);
`singleton` = serialize same-name jobs. Details and resume-segment patterns in
`references/resilience.md`.

## Interactive sessions (debugging only, still NOT the login node)

`srun --pty` grabs a compute allocation for quick interactive testing — that is
a compute node, not the login node. Prefer batch for anything real. Example:
`srun -p preempt --time=2:00:00 -c 4 --mem=8g --pty bash`.

## Queue ceiling — fine to over-submit

`QOSMaxGRESPerUser` caps *simultaneously running* GPU jobs (~10). Submitting more
than that is fine — the extras simply sit `PENDING` with reason
`QOSMaxGRESPerUser` and start as running ones finish. Queue depth is not a
problem; just submit the whole sweep.

## Preflight checklist before handing over the launch line

- Chosen cluster confirmed (old/new) and matching runtime wrapper used.
- `--time` within ceiling (2d new / 6d old); default partition time is only
  15 min, so always set `--time`.
- Partition exists on that cluster and matches CPU/GPU need.
- GPU jobs: `--gres=gpu:1` + a **VRAM `--constraint`** (not typed gres) + the
  runtime VRAM guard for any A100/mixed-VRAM request.
- No hardcoded `--exclude`; bad nodes are learned at runtime.
- **Every job**: walltime SIGTERM trap + `--signal=B:USR1@30` + `--requeue` +
  `--open-mode=append`, auto-requeuing until a `DONE` sentinel (rule 7).
- **`preempt` jobs MUST carry the full preempt contract (rule 6)**: `--requeue`
  + `SIGUSR1`/`SIGTERM`/`SIGCONT` trap → `scontrol requeue` + periodic atomic
  checkpoint + resume-from-last-step. No preempt job without a working
  save/resume path. Near-cap jobs need the same periodic checkpoint
  (`references/resilience.md`).
- `--mem` / `--cpus-per-task` fit a real node in the reference file.
- Output/data paths point at lab or scratch space, not home.
- Big/multi-stage work uses dependency chains, not one mega-job.
- Final answer ends with the single `sbatch` (or dependency/array) line to run,
  and a note on how to pull results back (`references/intake.md`).
```
