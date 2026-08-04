#!/usr/bin/env python3
"""Harvest compute provenance from SLURM stdout/stderr logs, then they can be deleted.

Scans a set of SLURM log files and extracts the durable "proof of run" facts —
node(s), GPU model + memory, gpu/cpu counts, start time, wall time, job ids, and a
sample of result/fitness lines — into a compact JSON + Markdown summary. The summary
is meant to live in the research record (e.g. research/results/compute_provenance.md)
so the raw per-task logs can be removed without losing reproducibility metadata.

Pure stdlib. Robust to heterogeneous formats (training logs, ES coord/eval, eval
arrays, probe units) via regex signal extraction — absent fields are simply omitted.

Usage:
    python harvest_slurm.py --label "ES run (E009)" --run-id 2026-06-03-r01 \
        --glob 'scripts/stage_b/slurm_logs/*.out' [--json out.json] [--md]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from collections import Counter

# --- signal patterns (lenient; match whatever a given log family happens to print)
RE_GPU = re.compile(r"(NVIDIA[\w .\-]*?,\s*\d+\s*MiB|NVIDIA[\w .\-]+-\d+GB)")
RE_NODE = re.compile(
    r"\bnode:\s*([\w\-]+)|on\s+([\w\-]+)\s*=*$|coordinator:.*node:\s*([\w\-]+)"
)
RE_NODE2 = re.compile(r"\bnode:\s*([\w\-]+)")
RE_GPUS = re.compile(r"\bgpus:\s*(\d+)")
RE_CPUS = re.compile(r"\bcpus:\s*(\d+)")
RE_STARTED = re.compile(r"\bstarted:\s*(.+)$")
RE_JOB = re.compile(r"\bjob:\s*(\d+)")
RE_CELL_SECS = re.compile(r"\((\d+)s\)")  # eval array: "(887s)"
RE_FIT_COORD = re.compile(r"coord gen (\d+):\s*fit_mean=([-\d.]+)\s*fit_max=([-\d.]+)")
RE_FITNESS = re.compile(r"FITNESS\s+mean_per_cog=([-\d.]+)")
RE_ERROR = re.compile(
    r"\b(Traceback|Error|CUDA out of memory|Killed|oom-kill|FAILED)\b", re.I
)


def harvest(paths: list[str]) -> dict:
    nodes, gpus_models, jobs = Counter(), Counter(), set()
    gpu_count = cpu_count = None
    starts: list[str] = []
    cell_secs: list[int] = []
    coord_series: list[tuple[int, float, float]] = []
    fitness_vals: list[float] = []
    errors = 0
    n = 0
    for p in paths:
        n += 1
        try:
            text = open(p, errors="replace").read()
        except OSError:
            continue
        for m in RE_GPU.finditer(text):
            gpus_models[m.group(1).strip()] += 1
        for m in RE_NODE2.finditer(text):
            nodes[m.group(1)] += 1
        if g := RE_GPUS.search(text):
            gpu_count = g.group(1)
        if c := RE_CPUS.search(text):
            cpu_count = c.group(1)
        for m in RE_STARTED.finditer(text):
            starts.append(m.group(1).strip())
        for m in RE_JOB.finditer(text):
            jobs.add(m.group(1))
        cell_secs += [int(s) for s in RE_CELL_SECS.findall(text)]
        for m in RE_FIT_COORD.finditer(text):
            coord_series.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        fitness_vals += [float(v) for v in RE_FITNESS.findall(text)]
        if RE_ERROR.search(text):
            errors += 1

    out: dict = {"n_files": n}
    if nodes:
        out["nodes"] = sorted(nodes)
    if gpus_models:
        out["gpu_models"] = sorted(gpus_models)
    if gpu_count:
        out["gpus_per_job"] = gpu_count
    if cpu_count:
        out["cpus_per_job"] = cpu_count
    if starts:
        out["started"] = sorted(set(starts))[:4]
    if jobs:
        out["job_ids_sample"] = sorted(jobs)[:8]
        out["n_job_ids"] = len(jobs)
    if cell_secs:
        out["cell_wall_seconds"] = {
            "n": len(cell_secs),
            "sum_s": sum(cell_secs),
            "sum_h": round(sum(cell_secs) / 3600, 2),
            "max_s": max(cell_secs),
        }
    if coord_series:
        coord_series.sort()
        out["es_fitness_by_gen"] = [
            {"gen": g, "fit_mean": fm, "fit_max": fx} for g, fm, fx in coord_series
        ]
    if fitness_vals:
        out["fitness_units"] = {
            "n": len(fitness_vals),
            "mean": round(sum(fitness_vals) / len(fitness_vals), 4),
            "min": min(fitness_vals),
            "max": max(fitness_vals),
        }
    if errors:
        out["files_with_error_markers"] = errors
    return out


def to_md(label: str, run_id: str, src: str, h: dict) -> str:
    lines = [
        f"### {label}" + (f"  (run {run_id})" if run_id else ""),
        f"- Source logs: `{src}` — {h.get('n_files', 0)} files (harvested, then deleted).",
    ]
    if "gpu_models" in h:
        lines.append(
            f"- GPU: {', '.join(h['gpu_models'])}"
            + (
                f"; {h['gpus_per_job']} gpu / {h.get('cpus_per_job','?')} cpu per job"
                if "gpus_per_job" in h
                else ""
            )
        )
    if "nodes" in h:
        ns = h["nodes"]
        lines.append(f"- Node(s): {', '.join(ns[:8])}" + (" …" if len(ns) > 8 else ""))
    if "started" in h:
        lines.append(f"- Started: {'; '.join(h['started'])}")
    if "cell_wall_seconds" in h:
        cw = h["cell_wall_seconds"]
        lines.append(
            f"- Eval wall: {cw['n']} cells, Σ {cw['sum_h']} h (max cell {cw['max_s']} s)."
        )
    if "fitness_units" in h:
        fu = h["fitness_units"]
        lines.append(
            f"- Fitness units: n={fu['n']}, mean={fu['mean']}, range [{fu['min']}, {fu['max']}]."
        )
    if "es_fitness_by_gen" in h:
        s = h["es_fitness_by_gen"]
        traj = ", ".join(f"g{d['gen']}:{d['fit_mean']:.3f}" for d in s)
        lines.append(f"- ES fitness/gen ({len(s)} gens): {traj}")
    if "n_job_ids" in h:
        lines.append(
            f"- SLURM job ids: {h['n_job_ids']} (e.g. {', '.join(h['job_ids_sample'])})."
        )
    if "files_with_error_markers" in h:
        lines.append(
            f"- ⚠ {h['files_with_error_markers']} file(s) contained error/traceback markers."
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", required=True, help="glob for the log files")
    ap.add_argument("--label", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--md", action="store_true", help="print a Markdown block")
    args = ap.parse_args(argv)

    paths = sorted(globmod.glob(args.glob))
    if not paths:
        print(f"no files matched: {args.glob}", file=sys.stderr)
        return 1
    h = harvest(paths)
    if args.json:
        json.dump(
            {"label": args.label, "run_id": args.run_id, "glob": args.glob, **h},
            open(args.json, "w"),
            indent=2,
        )
    print(
        to_md(args.label, args.run_id, args.glob, h)
        if args.md
        else json.dumps(h, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
