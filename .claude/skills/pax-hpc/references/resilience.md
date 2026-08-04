# Resilience — requeue, resume, preemption, VRAM guard, dependencies

Any job that can be interrupted must survive the interruption: catch the kill,
save a checkpoint, requeue itself, and resume from that checkpoint on restart.

**The walltime trap is universal (user directive, 2026-06-10): EVERY job — not
just preempt or near-cap jobs — traps the SIGTERM the scheduler sends at the
`--time` limit and `scontrol requeue`s itself, auto-requeuing until the run
writes a `DONE` sentinel.** Walltime estimates are routinely wrong; a job that
overruns must resume, never silently die. The *periodic checkpoint* layer
(§0.2) stays scoped to **every `preempt` job** and **every job whose walltime
approaches the cluster cap** (2 days new / 6 days old) — that's what lets a
requeued job pick up near where it died instead of from step 0.

This file gives copy-paste scaffolding. Drop the bash blocks into the runner
(the script that executes on the compute node), not the submitter.

---

## 0. The signal facts (read first)

Verified on the NEW cluster 2026-05-31: `KillWait=30 sec`,
`PreemptMode=REQUEUE`, `PreemptType=preempt/partition_prio`,
`PreemptExemptTime=0`.

- **Signal 9 (SIGKILL) cannot be caught, trapped, or ignored.** Anything that
  says "catch kill -9" is impossible. You catch the signal SLURM sends *before*
  the SIGKILL, during a grace window.
- On **preemption** (`preempt` partition), SLURM sends `SIGCONT` + `SIGTERM`,
  waits the partition `GraceTime`, then `SIGKILL`. **`GraceTime=0` on the preempt
  partition of BOTH clusters (verified 2026-05-31)** — so on preemption there is
  **no on-signal window at all**; SIGTERM and SIGKILL arrive back-to-back and the
  handler cannot complete a checkpoint. For preempt jobs the **periodic
  checkpoint (every K steps) is the ONLY safety net** — this is mandatory, not
  optional (see §0.2). The trap is dead weight on preemption; it earns its keep
  only on the walltime path below.
- On **time limit**, SLURM sends `SIGTERM` at the limit, waits `KillWait` (=30 s
  here), then `SIGKILL`. To get advance warning *before* the limit, request a
  signal: `#SBATCH --signal=B:USR1@30` → SLURM delivers `SIGUSR1` to the batch
  shell ~30 s before the limit. (`B:` = batch shell, not the job steps. `@30`
  matches the 30 s `KillWait`; raise it only if a checkpoint write needs longer.)
- So the robust runner traps **SIGUSR1 + SIGTERM + SIGCONT** and routes
  all three to the same "checkpoint + requeue" handler. This covers preemption
  *and* walltime on both clusters with one code path.

Required SBATCH directives for any resilient job:

```bash
#SBATCH --signal=B:USR1@30     # advance warning before the time limit
#SBATCH --open-mode=append     # preserve logs across requeues (do NOT overwrite)
#SBATCH --requeue              # mark job requeue-ELIGIBLE (necessary, not sufficient — see §0.1)
```

## 0.1 Critical: `PreemptMode=REQUEUE` does NOT requeue your job by itself

Verified empirically on this cluster: jobs preempted on `preempt` are **not**
auto-requeued, despite `PreemptMode=REQUEUE`. Two things defeat the automatic
path, and you must assume both:

1. **The job must be requeue-eligible.** Without `#SBATCH --requeue` the job is
   not requeueable, so `PreemptMode=REQUEUE` has nothing to requeue and the job
   is **cancelled** on preemption. With `GraceTime=0` this is the most likely
   reason a preempted job vanishes instead of requeuing — it never got a chance
   to do anything. **`--requeue` is mandatory.**
2. **A clean exit cancels the requeue** (the walltime path). If your handler ends
   with `exit 0` *before* SLURM kills the job — which only happens when there IS
   a grace window, i.e. at the time limit (`KillWait=30`), not on `GraceTime=0`
   preemption — SLURM records COMPLETED and won't requeue. So at the walltime
   boundary you must call `scontrol requeue` *explicitly* in the handler.
3. **Auto-requeue only re-runs the batch script from the top** — it never
   restores program state. Resume is your code's job regardless (§2), and on
   preemption the periodic checkpoint (§0.2) is the only thing that survived.

**The fix (what to propose): requeue yourself, explicitly, from the batch
script — do not depend on `PreemptMode`.** In the trap handler call
`scontrol requeue "$SLURM_JOB_ID"` *before* the job can exit. Once that call is
accepted the job is marked for requeue and will restart even if SIGKILL hits a
moment later — which is why we requeue *first* and checkpoint *second* (a slow
checkpoint must never cost you the requeue). `#SBATCH --requeue` only makes the
job *eligible*; the explicit `scontrol requeue` is what actually does it.

**Requeue from bash, checkpoint from Python.** Put the `scontrol requeue` in the
SLURM batch (bash) layer, not in the training code — bash is the job's top
process, it can requeue even if Python is wedged, and it keeps the training
script SLURM-agnostic. Python's only job is: trap `SIGUSR1` → write the latest
checkpoint → exit. This division is the "whatever is safer" answer: the requeue
can't be lost to a hung trainer, and the trainer stays portable.

## 0.2 MANDATORY: periodic checkpointing for every preempt job

Because `GraceTime=0`, a preempted job is killed with no warning the handler can
act on. The **only** state that survives is the last checkpoint the training loop
wrote on its own schedule. Therefore, for any job on `preempt` (or any job that
could be requeued), periodic checkpointing is **mandatory**:

- Write a checkpoint every `K` steps such that **at most ~30 min of compute** is
  ever at risk (tune `K` to the model's step time). This bounds worst-case loss
  on an unannounced preemption.
- Overwrite a single latest checkpoint with an **atomic write** (write to
  `*.tmp`, then `mv` into place) so a kill mid-write can't corrupt the resume
  point.
- Delete the checkpoint on clean finish so a completed run leaves no resume bait.
- The `SIGUSR1` handler's checkpoint is a *best-effort bonus* for the walltime
  path only; never rely on it for preemption. A preempt job without periodic
  checkpointing is not resilient — it just hasn't been preempted yet.

Combined with `#SBATCH --requeue`, this is the whole preempt-survival contract:
**`--requeue` (so SLURM puts it back) + periodic checkpoint (so it resumes near
where it died)**. The trap adds the walltime case on top.

---

## 0.3 Sizing the grace window AND the walltime (the 2026-06-10 timeout incident)

Two sizing mistakes both end in a dead job; the trap only saves you from the first.

**Grace (`@N` in `--signal=B:USR1@N`) must cover the time to the NEXT checkpoint
opportunity, not just the checkpoint write.** If the loop can only checkpoint at a
step/epoch boundary, and a boundary is preceded by a long *uninterruptible* section
— an eval pass, a validation epoch, a big batch — then `@30` is too short: the
handler sets the flag, but the flag is not read until that section finishes, and
SIGKILL lands first. Size `N` ≥ (longest uninterruptible section + checkpoint
write). Example: the DT-AntMaze runner evals between train steps, so it uses
`--signal=B:USR1@300`, not `@30`.

**Walltime must be sized to the SLOWEST cell of a sweep — never inherited from a
faster one.** Incident (2026-06-10): a Meta-LoRA DT sweep set `--time` per *cadence*
(mild 3:00, medium 3:30, aggressive 5:00) assuming more-frequent resampling = slower.
But runtime was dominated by the *dataset*: the `umaze-diverse` runs took ~3.0–4.5 h
regardless of cadence. The 3:00 / 3:30 limits — fine for the ~2× faster non-diverse
`umaze` grid they were copied from — were ~5 min too short, and **44 of 90 jobs hit
TIMEOUT** while the 5:00 aggressive cells (same dataset) all finished. The kicker:
the trainer already had a full SIGUSR1 → checkpoint → resume path, but the SLURM
template never set `--signal` / `--requeue`, so the trap never fired and the work
was lost instead of resumed. That is exactly the failure rules 6–7 exist to prevent
— wire the trap and a too-short walltime becomes a free requeue, not 44 dead jobs.

---

## 1. The trap + checkpoint + requeue handler (CRASH-SAFE walltime default)

**Requeue IF AND ONLY IF: (a) the walltime warning fired AND (b) the run did not
finish.** Never on a crash, never on a `scancel`, never on a job that completed
before walltime. User directive (2026-06-10/-12): auto-requeue an overrunning job
until it finishes, but never resurrect a job that died for another reason — that
loops a crash forever or fights an operator's `scancel`.

The two discriminators (note: **the exit code is NOT one** — see below):

- **`SIGUSR1` = the walltime advance-warning** from `--signal=B:USR1@N`. ONLY a
  walltime stop produces it; a crash, a `scancel`, and a `GraceTime=0` preemption do
  not. So `WALLTIME=1` (set by the USR1 trap) is the authoritative "this is a
  walltime stop" flag, and it is what excludes crash and cancel.
- **A completion sentinel** written ONLY on true finish (`metrics.json`, `DONE`, …):
  present ⇒ finished ⇒ no requeue. This excludes "completed before walltime."
- **Trap `USR1` + `TERM` + `CONT`.** USR1 sets `WALLTIME=1` and forwards a checkpoint
  request to the child. TERM/CONT also forward a checkpoint and — by being trapped —
  keep the shell alive through the hard-kill window so the post-`wait` decision runs
  instead of the shell dying instantly. A `scancel` is also `TERM`, but it arrives
  with `WALLTIME=0`, so it is never requeued.
- **Do NOT gate the requeue on the exit code.** A walltime stop can exit cleanly
  (the trainer checkpointed and exited 0) *or* be hard-killed non-zero (its checkpoint
  ran past the grace). Both must requeue. `WALLTIME=1` already excludes crash and
  cancel, so testing `rc` adds nothing and would wrongly drop the hard-killed case.
  (For a *generic* job that does NOT self-handle USR1, the equivalent test is "exited
  non-zero after the walltime warline" — i.e. requeue iff `WALLTIME=1 && rc!=0` and
  there is no sentinel; a clean `rc==0` then means it completed. Our trainers write a
  sentinel, so we key on the sentinel, which is unambiguous either way.)
- `SLURM_RESTART_COUNT` hard-caps the loop (a backstop, not the stop condition).

```bash
# ---- resilience: crash-safe walltime requeue -----------------------------
REQUEUE_CAP=8
WALLTIME=0
CHILD_PID=""
SENTINEL="${RUN_DIR}/metrics.json"          # written ONLY on true completion

_ckpt_child() { [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null && kill -USR1 "$CHILD_PID" 2>/dev/null || true; }
_on_usr1() { WALLTIME=1; echo "[resilience] SIGUSR1 (walltime) $(date -Is); child -> checkpoint+exit"; _ckpt_child; }
_on_term() { echo "[resilience] TERM/CONT (walltime=$WALLTIME); forward checkpoint, defer requeue decision"; _ckpt_child; }
trap _on_usr1 USR1
trap _on_term TERM CONT                      # keep the shell alive; scancel(WALLTIME=0) won't requeue

python my_script.py --resume-dir "$RUN_DIR" --args ... &
CHILD_PID=$!
# wait returns >128 either because a trap interrupted it (child alive — re-wait) OR
# because the child itself died from a signal (already reaped). Re-wait ONLY while the
# child is still alive, so a signal-death (SIGKILL/SIGSEGV) cannot spin the loop.
rc=0; wait "$CHILD_PID" || rc=$?
while [ "$rc" -gt 128 ] && kill -0 "$CHILD_PID" 2>/dev/null; do rc=0; wait "$CHILD_PID" || rc=$?; done

if [ -f "$SENTINEL" ]; then
    echo "[resilience] finished before walltime (sentinel, rc=$rc) — no requeue"
elif [ "$WALLTIME" -eq 1 ]; then            # walltime + unfinished — rc-independent
    n="${SLURM_RESTART_COUNT:-0}"
    if [ "$n" -lt "$REQUEUE_CAP" ]; then
        echo "[resilience] walltime + unfinished (rc=$rc, $n/$REQUEUE_CAP) — requeue"
        scontrol requeue "$SLURM_JOB_ID"; exit 0
    fi
    echo "[resilience] requeue cap reached — FAIL" >&2; exit 1
else
    echo "[resilience] rc=$rc walltime=0 no sentinel — crash/cancel/preempt, NO requeue" >&2
    exit "$([ "$rc" -eq 0 ] && echo 1 || echo "$rc")"
fi
# --------------------------------------------------------------------------
```

The requeue decision is made *after* the child is reaped (not "requeue first") so it
can see the sentinel and the `WALLTIME` flag together. The slow-checkpoint risk that
motivates "requeue first" is covered instead by a generous grace (`@300`, §0.3) plus
trapping `TERM` (which holds the shell open if the hard kill arrives before the child
finishes). For `GraceTime=0` **preemption** there is no warning at all — see §1b.

## 1b. Preemption addendum (only for `preempt`-partition jobs)

Preemption delivers `SIGTERM`+`SIGKILL` with `GraceTime=0` — no warning, and the
`SIGTERM` is indistinguishable from a `scancel`. So the §1 walltime handler cannot
safely requeue a preemption. For preempt jobs the survival contract is instead
**`#SBATCH --requeue` (eligibility) + periodic checkpoint (§0.2) + resume (§2)**:
SLURM's `PreemptMode=REQUEUE` puts the eligible job back, and it resumes from the last
periodic checkpoint. Do not add a `SIGTERM` requeue trap to "improve" this — it buys
nothing on `GraceTime=0` and reintroduces the scancel-requeue bug.

**Python side** (the child) traps `SIGUSR1`, writes the latest checkpoint, and
exits non-zero-but-clean (or 0) so the bash handler proceeds:

```python
import signal, sys
_should_stop = False
def _on_usr1(signum, frame):
    global _should_stop
    _should_stop = True
signal.signal(signal.SIGUSR1, _on_usr1)
# ... in the training loop, after each step/epoch:
if _should_stop:
    save_checkpoint(run_dir)   # latest only; overwrite previous
    sys.exit(0)
```

If your training framework can't be interrupted mid-step, the simplest robust
form is: checkpoint every K steps unconditionally, and on `SIGUSR1` just set the
flag and let the next step boundary save+exit. The grace window only needs to
cover one checkpoint write.

---

## 2. Resume on restart (robust, sentinel-independent)

When the requeued job starts, it must find the latest checkpoint and resume.
**Do not rely on a single `last_run_dir.txt` sentinel** — that has silently
gone stale after multiple requeues on this cluster, causing a fresh-from-zero
restart that abandons a near-finished run. Probe the filesystem directly for the
highest-step checkpoint:

```bash
# Prefer an explicit override; else find the newest checkpoint by step number.
RUN_DIR="${RESUME_RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then
    latest_ckpt=$(ls -1 "$OUT_ROOT"/run_*/checkpoints*/checkpoint_*.pt 2>/dev/null \
        | sed -E 's/.*checkpoint_[a-z]*([0-9]+)\.pt/\1 &/' \
        | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$latest_ckpt" ] && RUN_DIR=$(dirname "$(dirname "$latest_ckpt")")
fi
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
    echo "[resilience] resuming from $RUN_DIR"
else
    RUN_DIR="$OUT_ROOT/run_$(date +%s)"; mkdir -p "$RUN_DIR"
    echo "[resilience] no checkpoint found; starting fresh in $RUN_DIR"
fi
```

- Always honor a manual `RESUME_RUN_DIR=<dir>` env override — it's the escape
  hatch when automatic detection picks wrong.
- Keep only the latest checkpoint (overwrite), and **delete it on clean finish**
  so a finished run doesn't leave resume bait or waste disk.
- `--open-mode=append` (set in the submitter) keeps every requeue's log in one
  file so the failure trail survives.

---

## 3. VRAM guard — auto-requeue off a too-small GPU

Even with a VRAM `--constraint`, add this backstop for any A100 / mixed-VRAM
request. At job start the runner checks the card it actually got; if it's below
the requirement it excludes that node and requeues. Over repeated requeues the
job walks itself onto a correct card.

```bash
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-80000}"   # set to the model's real need
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$gpu_mem" ] && [ "$gpu_mem" -lt "$MIN_GPU_MEM_MB" ]; then
    echo "[vram-guard] landed on ${gpu_mem} MiB < required ${MIN_GPU_MEM_MB} on ${SLURMD_NODENAME}; excluding + requeuing"
    _exclude_node_and_requeue "$SLURMD_NODENAME"   # see §4
    exit 0
fi
echo "[vram-guard] GPU OK: ${gpu_mem} MiB on ${SLURMD_NODENAME}"
```

Set `MIN_GPU_MEM_MB` to match the constraint — don't claim `a100-80G` but set
`MIN_GPU_MEM_MB=40000`, or the guard goes slack and you get confusing late OOMs.

---

## 4. Adaptive exclude — start empty, learn at runtime

**Submitters set NO `--exclude`.** Each submission re-tests every node (broken
nodes do get fixed). The runner discovers a bad node — wrong VRAM (§3), a
`libtorch_global_deps.so: cannot open shared object file` NFS error, a CUDA init
failure — and requeues itself with that node appended to a **job-scoped**
exclude list via `ExcNodeList`:

```bash
_exclude_node_and_requeue() {
    local bad="$1"
    # Accumulate across this job's requeues in a small per-job file.
    local exfile="${EXCLUDE_FILE:-$OUT_ROOT/.exclude_${SLURM_JOB_ID}}"
    echo "$bad" >> "$exfile"
    local list; list=$(sort -u "$exfile" | paste -sd, -)
    echo "[exclude] requeuing ${SLURM_JOB_ID} with ExcNodeList=${list}"
    scontrol requeue "$SLURM_JOB_ID"
    scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$list"
}
```

This is strictly better than a hardcoded list: it's current (re-tests on every
fresh submission), self-healing (walks off a broken node in seconds), and leaves
no stale exclusions to silently shrink the pool months later.

Guard the runner's startup with a quick sanity import so a node-level breakage is
caught *before* training starts:

```bash
python -c "import torch; assert torch.cuda.is_available()" \
    || _exclude_node_and_requeue "$SLURMD_NODENAME"
```

---

## 5. Walltime self-requeue (universal — until the job is DONE)

**This is mandatory on every job, regardless of expected runtime** (rule 7).
The same trap handles the cap: `--signal=B:USR1@30` fires ~30 s before the
`--time` limit, the handler checkpoints + `scontrol requeue`s, and the requeued
job resumes (§2). So a run that overruns one allocation just needs `--time` set,
the trap wired, and a working resume — it will tick across as many requeues as
needed.

**"Until completed" needs a stop condition.** Auto-requeue must end when the
work is finished, not loop forever after a clean finish or a non-walltime crash.
Gate it on a `DONE` sentinel the training code writes on successful completion:
resume while `DONE` is absent, and skip the trap/requeue once it exists.

```bash
# at the top of the runner, before launching training:
if [ -f "$RUN_DIR/DONE" ]; then
    echo "[resilience] $RUN_DIR already DONE; nothing to requeue"; exit 0
fi
# ... trap _on_signal USR1 TERM CONT ; run training ...
# the trainer writes "$RUN_DIR/DONE" (atomically) only on successful completion.
# the _on_signal handler should no-op if DONE already exists:
#   _on_signal() { [ -f "$RUN_DIR/DONE" ] && exit 0; ... ; scontrol requeue ...; }
```

Verify the checkpoint cadence is short enough that losing the work since the
last checkpoint is acceptable (e.g. checkpoint every ≤30 min of compute).

For the **new cluster's 2-day cap specifically**: set `--time=2-00:00:00`, wire
the trap, and the job auto-requeues + resumes at the boundary. (There is a
`normal-7days` QOS on the new cluster but it has **no GPU** — it can't substitute
for resume on GPU work.)

---

## 6. Splitting big tasks — dependency chains

When a task is naturally staged, or you'd rather split a long run into bounded
segments than rely on in-place requeue, chain jobs with `--dependency`:

```bash
jid1=$(sbatch --parsable stage1.sbatch)                                  # e.g. preprocess
jid2=$(sbatch --parsable --dependency=afterok:$jid1   stage2.sbatch)     # train, runs iff stage1 ok
jid3=$(sbatch --parsable --dependency=afterok:$jid2   stage3.sbatch)     # eval/package
```

Dependency types:
- `afterok:JID` — start only if JID succeeded (exit 0).
- `afterany:JID` — start regardless of how JID ended. Use for a **resume
  segment**: segment N+1 should run even if segment N hit the walltime (non-zero
  exit), picking up from the checkpoint.
- `afternotok:JID` — start only if JID failed (cleanup / fallback paths).
- `singleton` — serialize all jobs sharing the same `--job-name` + user; a clean
  way to make a self-chaining resume job never run two copies at once.

Self-chaining segment pattern (each segment trains for ~1 cap, then submits its
own successor before exiting, conditioned on "not yet converged"):

```bash
# near the end of the runner, if training is incomplete:
if [ ! -f "$RUN_DIR/DONE" ]; then
    sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$0"   # queue the next segment
fi
```

Prefer in-place requeue (§1–§2) for a single long run; prefer dependency chains
when stages have distinct resource shapes (CPU preprocess → GPU train → CPU
package) or when you want each segment's logs and accounting separated.

---

## 7. Putting it together — minimal resilient GPU runner skeleton

```bash
#!/usr/bin/env bash
#SBATCH --job-name=myrun
#SBATCH --output=./res_%j_%N.txt
#SBATCH --error=./res_%j_%N.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append
#SBATCH --requeue

set -uo pipefail
OUT_ROOT=/cluster/tufts/levinlab/hhazan01/PROJECT/out
MIN_GPU_MEM_MB=80000

source ~/.bashrc && conda activate LoRa          # NEW cluster
# (OLD cluster: module load singularity; wrap the python line in singularity exec)

# --- instrumentation: which node/GPU did we actually get? (see ../SKILL.md) ---
echo "===== START $(date -Is)  node=${SLURMD_NODENAME} job=${SLURM_JOB_ID} ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- §4 startup sanity + §3 VRAM guard ---
# ( _exclude_node_and_requeue + _on_signal defined as above )
python -c "import torch; assert torch.cuda.is_available()" || _exclude_node_and_requeue "$SLURMD_NODENAME"
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${gpu_mem:-0}" -lt "$MIN_GPU_MEM_MB" ] && { _exclude_node_and_requeue "$SLURMD_NODENAME"; exit 0; }

# --- §2 resolve resume dir ---
# ( RUN_DIR resolution block as above )

# --- §1 trap + launch ---
trap _on_signal USR1 TERM CONT
python my_script.py --resume-dir "$RUN_DIR" --args ... &
CHILD_PID=$!
wait "$CHILD_PID"
echo "===== END $(date -Is)  exit=$? ====="
```

This single runner is preempt-safe, walltime-safe, lands on a correct-VRAM GPU
or requeues, learns bad nodes with no hardcoded exclude, and resumes from its own
checkpoint — the full set of guarantees the user asked for.
