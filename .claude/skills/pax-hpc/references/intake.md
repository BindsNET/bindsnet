# Closing the loop — pulling results back and classifying them

The skill's main job ends at a launch line, but a run isn't done until its
results are back on the user's machine, validated, and filed. This is the return
half. It has two parts: (1) jobs write **self-describing packages** on the
cluster, and (2) after the user rsyncs them back, you **classify and
disseminate** them.

The user runs the rsync. You give the command and do the classification.

---

## 1. On the cluster: write a self-describing package per run

A bare directory of `.csv`/`.pt` files is not intake-able — you can't tell a
finished run from one the walltime killed. Each run's output dir should contain:

- `status.json` — the single source of truth for "did this finish":
  ```json
  {"run_id": "myrun_s42", "state": "finished",
   "exit_code": 0, "started": "...", "ended": "...",
   "config": {"model": "...", "seed": 42, "lr": 3e-4},
   "metrics": {"test_ppl": 25.82, "val_ppl": 25.1},
   "git_commit": "abc1234"}
  ```
  `state ∈ {finished, partial, failed}`. Write `partial` from the resilience
  trap when requeuing, `finished` only on clean completion, `failed` from the
  EXIT trap on error.
- `manifest.json` — `{"required_files": ["results.json","history.csv"], "promote_to": "<task>/<variant>"}`.
  List only what a downstream reader needs; **do not** put resume checkpoints in
  `required_files` (they're deleted on clean finish).
- `README.md` — human context: what this run was, how to reproduce
  (config + commit + seed), node/GPU it ran on.

Have the EXIT trap stamp `status.json` on *every* exit path (success, error,
OOM-137, requeue) so a killed run is still self-describing.

---

## 2. The user rsyncs results back

Pull only the small metadata/metrics, not multi-GB checkpoints:

```bash
# from the user's local machine — they run this
rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='status.json' --include='manifest.json' \
  --include='README.md'   --include='results.json' --include='*.csv' \
  --exclude='*' \
  hhazan01@login-p02.pax.tufts.edu:/cluster/tufts/levinlab/hhazan01/PROJECT/out/ \
  ./from_hpc/PROJECT/
```

(Use `login-prod-03` for the old cluster.) The `--include/--exclude` filter
keeps weights on the cluster. If a checkpoint genuinely *is* the deliverable,
add `--include='*.pt'` deliberately.

---

## 3. Classify and disseminate (you do this)

Walk each package under `./from_hpc/` and sort it by `status.json`:

| Classification | Condition | Action |
|---|---|---|
| **COMPLETE** | `status.json` valid, `state=finished`, all `required_files` present | Promote to its `promote_to` destination |
| **INCOMPLETE** | `state=partial`/`failed`, or a required file missing | Offer: rerun missing, rerun all, accept partial, or skip — never silently promote |
| **UNKNOWN** | `status.json` missing or unparseable | Investigate first; quarantine, do not promote |

- **Promote**: move the package to its final task/variant location; record one
  log line (what ran, key metric, commit, seed, node/GPU).
- **Quarantine**: move UNKNOWN/failed-but-kept packages to a
  `_quarantine/<name>_<timestamp>/` dir so they don't pollute the promoted tree
  but aren't lost.
- **Dedup requeued runs**: a resilient job can leave several `run_*` dirs from
  successive requeues. Keep the one with the highest step / `state=finished`;
  quarantine the rest. Classify by `(run_id, seed)`, not by directory count, or
  you double-count requeues.
- **Audit trail**: append one line per package to an `intake_events.jsonl`
  (`{ts, package, classification, action, dest}`) so a re-run of intake is
  idempotent and you can answer "did everything land?" without re-walking.

---

## 4. If this is a project with its own intake pipeline

Some projects ship a dedicated intake (validator + promoter + log/STATUS
regeneration). **Defer to the project's pipeline when one exists** — this file is
the generic fallback for projects that don't. The contract is the same either
way: self-describing packages on the cluster → rsync the metadata back →
classify by `status.json` → promote / quarantine → record what landed.
