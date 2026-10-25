"""
Regenerate test/regression/op_counts.json, the stored operation counts.

Run from the repository root:

    python test/regression/regen_op_counts.py --reason "why the counts changed"
    python test/regression/regen_op_counts.py --check   # compare only, write nothing

Writing is a statement that a hot path now does a different amount of work on
purpose, so a reason is required; it is stored in the file's ``meta`` and
appended to its ``history``. Every workload is measured twice, each time on a
freshly built object; if the two counts differ the script writes nothing and
exits with status 1 (an unstable count cannot guard anything).
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_op_counts as T  # noqa: E402
import torch  # noqa: E402


def _commit():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=HERE.parent.parent, capture_output=True, text=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--", "bindsnet/*.py")
    return head + ("+uncommitted bindsnet code changes" if dirty else "")


def measure():
    """Workload -> per-operation counts; raises if two measurements differ."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    counts, unstable = {}, []
    for name in T.COUNTED:
        first, second = T.count_ops(name), T.count_ops(name)
        if first != second:
            unstable.append(f"{name}: {sum(first.values())} vs {sum(second.values())}")
        counts[name] = first
    if unstable:
        raise RuntimeError("unstable counts (nothing written): " + "; ".join(unstable))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--reason", help="why the stored counts change (required to write)")
    ap.add_argument(
        "--check", action="store_true", help="compare with the file, write nothing"
    )
    args = ap.parse_args(argv)
    if not args.check and not (args.reason or "").strip():
        ap.error("--reason is required (and must not be empty) to write the file")

    try:
        counts = measure()
    except RuntimeError as e:
        print(e)
        return 1
    totals = {k: sum(v.values()) for k, v in counts.items()}

    if args.check:
        old = T.load_counts()["counts"]
        changed = 0
        for name, total in totals.items():
            before = sum(old[name].values()) if name in old else None
            mark = "" if old.get(name) == counts[name] else "  CHANGED"
            changed += bool(mark)
            print(f"{name:<40s} {total:6d}  stored {before}{mark}")
        print(f"{changed} workload(s) differ from {T.COUNTS_FILE.name}")
        return 1 if changed else 0

    history = []
    if T.COUNTS_FILE.exists():
        old = T.load_counts()
        history = old.get("history", [])
    meta = {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "date": datetime.date.today().isoformat(),
        "commit": _commit(),
        "reason": args.reason.strip(),
    }
    history.append(meta)
    data = {
        "meta": meta,
        "excluded": T.EXCLUDED,
        "totals": totals,
        "counts": counts,
        "history": history,
    }
    T.COUNTS_FILE.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    for name, total in totals.items():
        print(f"{name:<40s} {total:6d}")
    print(f"wrote {T.COUNTS_FILE} ({len(counts)} workloads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
