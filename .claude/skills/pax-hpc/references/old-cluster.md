# OLD cluster — login-prod-03.pax.tufts.edu

Legacy cluster. **Use Singularity to run programs — always, by user policy.**
conda fails on the cluster's `rhel7` nodes (glibc < 2.28 → `GLIBC_2.28 not
found`), which are the majority of the pool and **include every A100**. The
`rhel8` nodes (l40 / h100 / l40s / rtx_a5000, on `preempt`) *could* in principle
run conda via `-C rhel8`, but that would (a) lose all A100 access and (b) split
the pool into two runtimes. **The user deliberately uses Singularity for ALL
old-cluster jobs to avoid that complication — do not propose conda here.**
Containerize the workload.

- OS: RHEL 7.5 (Maipo) · Slurm 23.02.7
- Login: `login-prod-0X.pax.tufts.edu` (there are several — e.g. `login-prod-01`,
  `login-prod-03`).
- Compute nodes keep the **legacy** naming `cc1gpu###`, `s1cmp###`, `p1cmp###`,
  `d1cmp###` — this cluster did **not** migrate to `pax###` (only the NEW cluster
  did). Don't expect `pax` names here.
- Runtime: `module load singularity` then `singularity exec ...`
  (apptainer/singularity is a module, not on the default PATH)
- Images live at `/cluster/tufts/levinlab/hhazan01/singularity/*.sif`
  (e.g. `mettagrid.sif`, `delayW.sif`, `LoRa.sif`, `Heb.sif` — confirm with `ls`)

## Run wrapper

```bash
module load singularity
SIF=/cluster/tufts/levinlab/hhazan01/singularity/IMAGE.sif
singularity exec --nv --bind /cluster:/cluster "$SIF" python3 script.py --args
```

- `--nv` = expose GPUs (omit for CPU-only jobs)
- `--bind /cluster:/cluster` = make the shared filesystem visible in-container
- `--cleanenv` is useful when host env vars leak into the container

## Scheduler limits (config)

- Max walltime: **6 days** (`6-00:00:00`) — longer than the new cluster.
- Default memory per CPU: 2000 MB. `cons_tres`, `CR_CPU_MEMORY`.
- Max array size: 2000. Max job count: 100000.

## Partitions

| Partition | Use | Walltime | Notes |
|---|---|---|---|
| `batch` | default CPU jobs | 6d | rhel7/rhel8 nodes |
| `gpu` | GPU jobs | 6d | a100/p100 reserved GPU nodes |
| `mpi` | multi-node MPI | 6d | large core counts |
| `largemem` | big-memory jobs | 6d | ~1 TB nodes |
| `interactive` | quick interactive | 4h | for `srun --pty` |
| `preempt` | opportunistic, largest/most varied pool | 6d | **requeued when preempted**; widest GPU selection. **Jobs here MUST be requeue-safe — see `resilience.md`.** |

QOS ceilings: `preempt` cpu≤4000 / gpu≤20 / mem≤10000G; `public` cpu≤2000 /
gpu≤10 / mem≤5000G; `expanded` cpu≤2000 / gpu≤24; `normal` default.

## CPU node types

- rhel8 nodes: 128 cores, ~510 GB (newest of the old pool)
- broadwell rhel7: 36+ cores, 120–248 GB; large-mem up to ~1 TB / 1.5 TB
- Constrain OS generation if a container needs it: `--constraint=rhel8` or
  `--constraint=rhel7`.

## GPUs

Request generic gres + a VRAM/type constraint (same rationale as the new
cluster — typed `--gres=gpu:a100:1` does not pin VRAM):

```bash
#SBATCH --partition=gpu        # or preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100      # tighten to a VRAM label if exposed (sinfo -o '%N %f %G')
```

Verified `sinfo` 2026-05-31. **VRAM feature labels exist only for A100 here** —
everything else must be requested by GRES type, and a few labels are inconsistent
(noted below):

| GPU | VRAM | Feature label | GRES type | Nodes (count) |
|---|---|---|---|---|
| A100 | 40 GB | `a100-40G` ✓ | `a100` | cc1gpu001–005 ·8; p1cmp110–111 ·2 |
| A100 | 80 GB | `a100-80G` ✓ | `a100` | s1cmp001–005 ·8; s1cmp006–007 ·2 |
| H100 | 80 GB | **none** ⚠️ | `h100` | s1cmp010 ·3 — **no feature label; request by GRES only** |
| RTX A6000 | 48 GB | `rtx_a6000` | `rtx_a6000` | s1cmp008 ·8 |
| RTX 6000 Ada | 48 GB | `rtx_a6000ada` ⚠️ | `rtx_6000ada` ⚠️ | s1cmp009 ·4 — **feature/GRES names disagree** |
| RTX A5000 | 24 GB | `rtx_a5000` | `rtx_a5000` | s1cmp012–015 ·8 |
| L40 | 48 GB | `l40` | `l40` | d1cmp042–050 ·4 |
| L40S | 48 GB | `l40` ⚠️ | `l40s` | s1cmp011 ·4 — **feature says `l40` but GRES is `l40s`** |
| RTX 6000 | — | `rtx_6000` | `rtx_6000` | p1cmp077 ·2 |
| V100 | — | `v100` | `v100` | p1cmp071–076 (·2–4) |
| P100 | — | `p100` | `p100` | p1cmp073 ·4, p1cmp075 ·6 |
| T4 | 16 GB | `t4` (no VRAM tag) | `t4` | p1cmp090–109 ·4 (≈20 nodes) |

Practical rules for the old cluster:
- **A100**: use `--constraint=a100-80G` / `a100-40G` (labels are reliable). Pair
  with the runtime VRAM guard (`resilience.md` §3).
- **H100**: there is no feature label — request `--gres=gpu:h100:1` directly.
- **L40S vs L40**: don't trust the `l40` feature to mean 48 GB L40 specifically;
  if you need L40S, request `--gres=gpu:l40s:1`. If you need L40, `--gres=gpu:l40:1`.
- **RTX 6000 Ada**: feature is `rtx_a6000ada` but GRES is `rtx_6000ada` — use the
  **GRES** form `--gres=gpu:rtx_6000ada:1`.
- Always confirm live: `sinfo -N -p gpu,preempt -o '%N %f %G'`. Node names and
  labels here are messier than the new cluster — never hardcode them.

The user's `.bashrc` interactive GPU aliases (`sbash_gpu` → `gpu:p100:1`,
`sbash_gpu_k20_x` → `gpu:k20xm:1` on partition `gpu`) were written for this
cluster.
