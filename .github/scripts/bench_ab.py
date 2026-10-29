#!/usr/bin/env python3
"""Alternating timing comparison of a pull request's base and head (informational).

    python .github/scripts/bench_ab.py --base <git ref> --head <git ref> --out DIR
    python .github/scripts/bench_ab.py --base HEAD --head-tree . --out DIR   # local

``--base`` and ``--head`` are checked out with ``git worktree add`` into
DIR/base and DIR/head (removed again at the end); ``--head-tree`` uses an
existing folder instead (for example the working tree, to test uncommitted
changes). Nothing is installed: both trees use the current Python environment.
Each measurement is a fresh process with the tree first on PYTHONPATH, and the
process checks that ``bindsnet.__file__`` lies inside that tree.

The workloads are those of examples/benchmark/hot_path_bench.py, same sizes,
copied here so that editing the example cannot change what is measured. Each
process fixes torch to --threads threads (default 2; also OMP_NUM_THREADS and
MKL_NUM_THREADS, and pins itself to the first --threads CPUs it may use),
builds every workload, does one warm-up run, then repeats run-and-reset until
at least --min-seconds have passed and records the mean time per run.
Processes alternate base, head, head, base, ... for --rounds rounds, so slow
drift of the machine affects both sides alike.

Report per workload: median seconds per run on each side, the median of the
per-round ratios head/base, its 95% bootstrap interval (2000 resamples of the
round ratios, fixed seed 0), and the lowest and highest round ratio. Flag
SLOWER when the median ratio is above --threshold (default 1.25) and the
interval's lower end is above 1; "possible slowdown" when the median is above
the threshold but the interval reaches 1; "faster" and "possible speed-up" the
same way below 1/threshold; otherwise "no clear change".

The report (Markdown) is written to DIR/bench_ab.md, appended to --summary if
given (GitHub's $GITHUB_STEP_SUMMARY), and the raw numbers to DIR/bench_ab.json.
Without --fail-above the exit status is always 0 and a human decides; the
verdict, or a TOOL ERROR line, is in the report. With --fail-above X the exit
status is 1 when any workload's median ratio is above X and its interval's lower
end is above 1.5 (a large, clear slowdown, far outside the no-change spread of
0.90 to 1.12 per round measured on GitHub runners), or when the tool failed; CI
uses X = 2.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORKLOADS = [
    "dc2015_b1",
    "dc2015_b16",
    "conn_postpre_b1",
    "conn_postpre_b16",
    "local2d_postpre",
    "conn_mstdp",
]


def build(name, T):
    """(network, inputs, run kwargs) for one workload; mirrors hot_path_bench.py."""
    import torch

    from bindsnet.encoding import poisson
    from bindsnet.learning import MSTDP, PostPre
    from bindsnet.models import DiehlAndCook2015
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection, LocalConnection2D

    torch.manual_seed(0)
    if name.startswith("dc2015"):
        b = int(name.split("_b")[1])
        net = DiehlAndCook2015(
            n_inpt=784, n_neurons=400, batch_size=b, inpt_shape=(1, 28, 28)
        )
        return net, {"X": poisson(torch.rand(b, 1, 28, 28) * 128, time=T)}, {}
    if name.startswith("conn_postpre"):
        b = int(name.split("_b")[1])
        net = Network(dt=1.0, batch_size=b)
        net.add_layer(Input(n=784, traces=True), "in")
        net.add_layer(LIFNodes(n=1000, traces=True), "out")
        conn = Connection(
            net.layers["in"],
            net.layers["out"],
            nu=(1e-4, 1e-2),
            update_rule=PostPre,
            wmin=0,
            wmax=1,
            norm=78.4,
        )
        net.add_connection(conn, "in", "out")
        return net, {"in": poisson(torch.rand(b, 784) * 128, time=T)}, {}
    if name == "local2d_postpre":
        net = Network(dt=1.0)
        net.add_layer(Input(shape=[1, 28, 28], traces=True), "in")
        net.add_layer(LIFNodes(shape=[16, 6, 6], traces=True), "out")
        conn = LocalConnection2D(
            net.layers["in"],
            net.layers["out"],
            kernel_size=8,
            stride=4,
            n_filters=16,
            nu=(1e-4, 1e-2),
            update_rule=PostPre,
            wmin=0,
            wmax=1,
        )
        net.add_connection(conn, "in", "out")
        return net, {"in": poisson(torch.rand(1, 1, 28, 28) * 128, time=T)}, {}
    if name == "conn_mstdp":
        net = Network(dt=1.0)
        net.add_layer(Input(n=784), "in")
        net.add_layer(LIFNodes(n=500), "out")
        conn = Connection(
            net.layers["in"],
            net.layers["out"],
            nu=1e-3,
            update_rule=MSTDP,
            wmin=0,
            wmax=1,
        )
        net.add_connection(conn, "in", "out")
        return net, {"in": poisson(torch.rand(1, 784) * 128, time=T)}, {"reward": 1.0}
    raise ValueError(f"unknown workload {name}")


def worker(args) -> int:
    """One measurement process: time every workload once, print JSON."""
    tree = args.tree.resolve()
    import torch

    import bindsnet

    where = Path(bindsnet.__file__).resolve()
    if tree not in where.parents:
        print(f"TOOL ERROR: bindsnet imported from {where}, not from {tree}")
        return 2
    torch.set_num_threads(args.threads)
    try:  # the same CPUs for every process, base and head alike
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.threads])
    except (AttributeError, OSError):
        pass
    result = {"bindsnet_file": str(where)}
    for name in args.only.split(","):
        net, inputs, kw = build(name, args.time)
        net.run(inputs=inputs, time=args.time, **kw)  # warm-up
        net.reset_state_variables()
        runs, t = 0, time.perf_counter()
        while runs == 0 or time.perf_counter() - t < args.min_seconds:
            net.run(inputs=inputs, time=args.time, **kw)
            net.reset_state_variables()
            runs += 1
        result[name] = {"wall": (time.perf_counter() - t) / runs, "runs": runs}
    print(json.dumps(result))
    return 0


def median_interval(ratios, n=2000, seed=0):
    """95% bootstrap interval of the median of the round ratios (fixed seed)."""
    rng = random.Random(seed)
    meds = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(n)
    )
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def flag(median, lo, hi, threshold):
    if median > threshold:
        return "SLOWER" if lo > 1.0 else "possible slowdown"
    if median < 1 / threshold:
        return "faster" if hi < 1.0 else "possible speed-up"
    return "no clear change"


def machine():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info["cpu_count"] = os.cpu_count()
    code = "import torch; print(torch.__version__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    info["torch"] = out.stdout.strip() or "unknown"
    return info


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def measure(trees, args, out, log):
    """Alternate base/head processes; return {side: [per-round result dicts]}."""
    times = {side: [] for side in trees}
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
    )
    for i in range(args.rounds):
        for side in ("base", "head") if i % 2 == 0 else ("head", "base"):
            cmd = [
                sys.executable,
                "-s",
                str(Path(__file__).resolve()),
                "--worker",
                "--tree",
                str(trees[side]),
                "--time",
                str(args.time),
                "--threads",
                str(args.threads),
                "--only",
                args.only,
                "--min-seconds",
                str(args.min_seconds),
            ]
            proc = subprocess.run(
                cmd,
                env=dict(env, PYTHONPATH=str(trees[side])),
                cwd=out,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            lines = [x for x in proc.stdout.splitlines() if x.startswith("{")]
            if proc.returncode != 0 or not lines:
                raise RuntimeError(
                    f"{side} worker exited {proc.returncode}: "
                    f"{(proc.stdout + proc.stderr)[-1500:]}"
                )
            times[side].append(json.loads(lines[-1]))
            log(f"round {i + 1}/{args.rounds} {side} done")
    return times


def summarise(times, args):
    rows = []
    for name in args.only.split(","):
        b = [r[name]["wall"] for r in times["base"]]
        h = [r[name]["wall"] for r in times["head"]]
        ratios = [hh / bb for hh, bb in zip(h, b)]
        lo, hi = median_interval(ratios)
        med = statistics.median(ratios)
        rows.append(
            {
                "workload": name,
                "base_median_s": statistics.median(b),
                "head_median_s": statistics.median(h),
                "ratio_median": med,
                "ratio_ci95": [lo, hi],
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
                "ratios": ratios,
                "runs_per_sample": min(
                    r[name]["runs"] for r in times["base"] + times["head"]
                ),
                "flag": flag(med, lo, hi, args.threshold),
            }
        )
    return rows


def markdown(rows, args, info, refs, error=None):
    lines = ["## Speed comparison, base versus head (informational)", ""]
    lines.append(
        f"Base `{refs['base']}`, head `{refs['head']}`. CPU, {args.threads} threads, "
        f"{args.time} time steps per run, {args.rounds} alternating rounds, at least "
        f"{args.min_seconds} s per measurement, SLOWER threshold {args.threshold}."
    )
    lines.append(
        f"Machine: {info.get('cpu', 'unknown CPU')}, {info.get('cpu_count')} CPUs, "
        f"Python {info['python']}, torch {info['torch']}."
    )
    lines.append("")
    if error:
        lines += ["**TOOL ERROR** (no timing result):", "", "```", error, "```"]
        return "\n".join(lines) + "\n"
    lines.append(
        "| workload | base s/run | head s/run | head/base median | 95% interval "
        "| lowest..highest round | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["ratio_ci95"]
        lines.append(
            f"| {r['workload']} | {r['base_median_s']:.4f} | {r['head_median_s']:.4f} "
            f"| {r['ratio_median']:.3f} | {lo:.3f}..{hi:.3f} "
            f"| {r['ratio_min']:.3f}..{r['ratio_max']:.3f} | {r['flag']} |"
        )
    slower = [r["workload"] for r in rows if r["flag"] == "SLOWER"]
    lines.append("")
    lines.append(
        f"**Verdict: SLOWER in {', '.join(slower)}.** A ratio of 1.30 means the head "
        "takes 30% longer. Shared CI machines are noisy; a maintainer decides."
        if slower
        else "**Verdict: no workload flagged SLOWER.**"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="git ref of the base (checked out as a worktree)")
    ap.add_argument("--head", help="git ref of the head (checked out as a worktree)")
    ap.add_argument("--head-tree", type=Path, help="existing folder used as the head")
    ap.add_argument("--out", type=Path, help="output folder")
    ap.add_argument("--summary", type=Path, help="also append the report here")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--time", type=int, default=100, help="time steps per run")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=1.25)
    ap.add_argument(
        "--fail-above",
        type=float,
        default=None,
        help="exit 1 when a median ratio is above this and the interval's lower end above 1.5",
    )
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--only", default=",".join(WORKLOADS))
    ap.add_argument("--timeout", type=int, default=900, help="seconds per process")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    if not args.base or not args.out or bool(args.head) == bool(args.head_tree):
        ap.error("need --base, --out, and exactly one of --head or --head-tree")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(git("rev-parse", "--show-toplevel"))
    refs = {"base": git("rev-parse", "--short", args.base, cwd=repo)}
    refs["head"] = (
        git("rev-parse", "--short", args.head, cwd=repo)
        if args.head
        else f"{args.head_tree.resolve()} (working tree)"
    )
    worktrees, rows, times, error = [], [], {}, None

    def log(msg):
        print(msg, flush=True)

    try:
        trees = {}
        for side, ref in (("base", args.base), ("head", args.head)):
            if ref is None:
                trees[side] = args.head_tree.resolve()
                continue
            path = out / side
            git("worktree", "add", "--detach", "--force", str(path), ref, cwd=repo)
            worktrees.append(path)
            trees[side] = path
        times = measure(trees, args, out, log)
        rows = summarise(times, args)
    except Exception:  # report every failure; the check is informational
        error = traceback.format_exc()[-3000:]
    finally:
        for path in worktrees:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=repo,
                capture_output=True,
            )

    info = machine()
    report = markdown(rows, args, info, refs, error)
    print(report)
    (out / "bench_ab.md").write_text(report)
    settings = {k: str(v) for k, v in vars(args).items()}
    (out / "bench_ab.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "machine": info,
                "refs": refs,
                "settings": settings,
                "error": error,
                "raw": times,
            },
            indent=1,
        )
    )
    failing = []
    if args.fail_above is not None:
        failing = [
            r["workload"]
            for r in rows
            if r["ratio_median"] > args.fail_above and r["ratio_ci95"][0] > 1.5
        ]
        if error:
            failing.append("tool error")
        report += (
            f"\n**Failed:** {', '.join(failing)} (median above {args.fail_above}x "
            "the base, or the tool failed).\n"
            if failing
            else f"\nRequired limit: no workload above {args.fail_above}x the base.\n"
        )
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(report)
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
