# NEW cluster — login-p02.pax.tufts.edu

Modern cluster. Use **conda** to run programs. Apptainer/Singularity 1.4.2 is
also available at `/usr/bin` if you need a container.

- OS: Rocky Linux 9.6 · Slurm 23.11.11
- Login: `login-p02.pax.tufts.edu` (the OLD cluster is `login-prod-03`)
- Compute nodes are named `pax001`–`pax120` (growing; verify the top end with
  `sinfo`).
- Runtime: `source ~/.bashrc && conda activate <env>`
- Envs live in `/cluster/tufts/levinlab/hhazan01/miniconda3/envs/`
  (verified 2026-05-31): `betse`, `bindsnet`, `delayW`, `dt-d4rl`, `LoRa`.
  Confirm current list with `conda env list`.

## Scheduler limits (config)

- Max walltime: **2 days** (`2-00:00:00`). Default time if unset: only 15 min —
  always set `--time`. For runs longer than 2d, use in-place requeue+resume or a
  dependency chain (see `resilience.md`).
- Default memory per CPU: 2000 MB. `SelectType=cons_tres`, `CR_CPU_MEMORY`.
- Max array size: 2000. Max job count: 100000.
- Your QOS: `interactive`, `normal` (default).

## Partitions

| Partition | Use | Priority | Notes |
|---|---|---|---|
| `batch` | default CPU jobs | normal | QoS `normal` (cpu≤250, gpu≤10, mem≤5000G) |
| `gpu` | lab-allocated GPU jobs | normal | reserved GPU nodes (h100/h200/a100/l40s) — preferred for training |
| `preempt` | opportunistic, largest pool | low, **requeued when preempted** | QoS `preempt` (cpu≤1000, gpu≤10, mem≤10000G); most nodes incl. many GPUs. **Jobs here MUST be requeue-safe — see `resilience.md`.** |

QOS ceilings: `normal` 2d / cpu=250 / gpu=10 / mem=5000G; `preempt` 2d /
cpu=1000 / gpu=10 / mem=10000G; `normal-7days` 7d but **no GPU**; `interactive`
4h / cpu=16 / gpu=1 / mem=64G. `QOSMaxGRESPerUser` caps simultaneously-*running*
GPU jobs (~10) — over-submitting is fine, extras just queue.

## CPU node types (cores / memory)

- sapphirerapids: 64 cores, ~510 GB (the bulk of `batch`/`preempt`)
- broadwell: 36–40 cores, 128–257 GB; some 515 GB and 1 TB large-mem
- Constrain arch with `--constraint=sapphirerapids` or `broadwell` if needed.

## GPUs — request by VRAM label, not bare type

**Do not use typed gres (`--gres=gpu:a100:1`) for memory-sensitive jobs** — the
type `a100` exists at both 40 GB and 80 GB, so a typed request can land on a
40 GB card and OOM a 300M+ model. Request generic gres + a **VRAM-precise
constraint**:

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G       # or an alternation; see tiers below
```

VRAM-tagged feature labels, GRES type, and where they live (verified
`sinfo` 2026-05-31):

| GPU | VRAM | Feature label | GRES type | Partition (nodes) |
|---|---|---|---|---|
| H200 | 141 GB | `h200-141G` | `h200` | gpu: pax008–011 ·8; preempt: pax047 ·2 (lulab) |
| H100 | 80 GB | `h100-80G` | `h100` | gpu: pax063 ·3 |
| A100 | 80 GB | `a100-80G` | `a100` | gpu: pax007, pax049–050, pax105–106 ·8 (icelake) |
| A100 | 40 GB | `a100-40G` | `a100` | gpu: pax003 ·2, pax051–052 ·8 (cascadelake) |
| L40S | 48 GB | `l40s-48G` | `l40s` | gpu: pax020–026 ·4; preempt: pax064 ·4 (laolab) |
| L40 | 48 GB | `l40-48G` | `l40` | preempt: pax048 ·4 (pettilab), pax110–113 ·4 (linlab) |
| RTX 6000 Ada | 48 GB | `rtx_6000_ada-48G` | `rtx_6000_ada` | preempt: pax062 ·4 (heldweinlab) |
| RTX A6000 | 48 GB | `rtx_a6000-48G` | `rtx_a6000` | preempt: pax119 ·8 (hugheslab) |
| RTX A5000 | 24 GB | `rtx_a5000-24G` | `rtx_a5000` | preempt: pax065–068 ·8 (dinglab) |
| RTX 6000 | 24 GB | `rtx_6000-24G` | `rtx_6000` | preempt: pax118 ·8 (hugheslab) |
| T4 | 16 GB | `t4-16G` | `t4` | preempt: pax077, pax104 ·4 (linlab) |

- **The type `a100` spans both 40 GB and 80 GB** (different nodes) — this is the
  concrete reason `--gres=gpu:a100:1` can OOM. Pin with `--constraint=a100-80G`.
- The `gpu` partition holds the lab/RT-allocated cards (feature `rtgpu`);
  everything else is lab-private and surfaces only on `preempt` (so it's
  backfill, and must be requeue-safe — `resilience.md`).
- Node↔GPU assignments drift and the node count grows; the **labels** are durable,
  the node names are not. **Never hardcode node names** — query live with
  `sinfo -N -p gpu,preempt -o '%N %f %G'`.
- The libtorch-NFS breakage that once hit pax009/010/011/024/025/026/063 is
  cleared as of 2026-05-31 (all healthy with GRES) — exactly why excludes are
  learned at runtime (`resilience.md` §4), never baked in.

### Tier helper (drop into a submitter)

```bash
constraint_for_tier() {
    case "$1" in
        flex)      echo "" ;;                                                  # any GPU
        ge_24g)    echo "rtx_a5000-24G|l40-48G|l40s-48G|rtx_6000_ada-48G|a100-40G|a100-80G|h100-80G|h200-141G" ;;
        ge_40g)    echo "a100-40G|a100-80G|h100-80G|h200-141G|l40-48G|l40s-48G|rtx_6000_ada-48G" ;;
        ge_48g)    echo "l40-48G|l40s-48G|rtx_6000_ada-48G|a100-80G|h100-80G|h200-141G" ;;
        ge_80g)    echo "a100-80G|h100-80G|h200-141G" ;;                       # ≥80 GB, no 40G slip
        a100_80)   echo "a100-80G" ;;                                          # avoid H100/H200 contention
        h200)      echo "h200-141G" ;;
        h100_h200) echo "h100-80G|h200-141G" ;;
        *)         echo "" ;;
    esac
}
```

Tier guidance:
- 300M body, dense or LoRA r≥256 → **`ge_80g`** (the 40 GB card OOMs).
- 600M+ FFN-trainable / dense ablations → **`h200`** (80 GB cards peaked ~76 GB
  and OOMed in past runs).
- 2B body → **`h200`**, no exceptions.
- Tiny pilots (≤10M) → **`flex`**, or `ge_24g` if the runner needs ≥24 GB.

Always pair the constraint with the runtime **VRAM guard** (`resilience.md` §3)
and set `MIN_GPU_MEM_MB` to match — the guard requeues off a wrong-VRAM landing
instead of silently OOMing.

Example GPU header:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01-00:00:00
```

> The user's old `.bashrc` GPU aliases reference `p100`/`k20xm` — those are
> OLD-cluster GPUs and do not exist here. Pick from the table above.
