"""
Regenerate the golden files of the regression tests.

Run from the repository root:

    python test/regression/regen.py --reason "why the results changed" [--only a,b]

Regenerating is a statement that the simulated results changed on purpose, so a
reason is required; it is written into each file's ``meta``. A torch upgrade
must regenerate in its own commit. Before writing, every network must pass the
threshold-margin check (voltage at least ``MARGIN`` away from the threshold at
every recorded spike check), which keeps exact spike comparison safe on other
CPUs; a network that fails is not written and the script exits with status 1.
"""

import argparse
import datetime
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cpu_pins  # noqa: E402

cpu_pins.apply()  # before torch is imported

import networks as N  # noqa: E402
import torch  # noqa: E402


def _commit():
    root = HERE.parent.parent

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--", "bindsnet")
    return head + ("+uncommitted bindsnet changes" if dirty else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--reason", required=True, help="why the golden files change")
    ap.add_argument("--only", default="", help="comma-separated network names")
    args = ap.parse_args(argv)
    if not args.reason.strip():
        ap.error("--reason must not be empty")

    names = [n for n in args.only.split(",") if n] or list(N.SPECS)
    unknown = sorted(set(names) - set(N.SPECS))
    if unknown:
        ap.error(f"unknown networks {unknown}; known: {list(N.SPECS)}")

    N.deterministic()
    base_meta = {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "date": datetime.date.today().isoformat(),
        "commit": _commit(),
        "reason": args.reason,
    }
    failed = []
    for name in names:
        tensors, errors, margins = N.run_standard(name)
        low = {k: m for k, m in margins.items() if not m > N.MARGIN}
        line = ", ".join(f"{k} {m:.2e}" for k, m in margins.items()) or "none"
        status = "FAIL margin" if low else "ok"
        print(f"{name:<20s} {status:<12s} margins: {line}  errors: {errors or 'none'}")
        if low:
            failed.append(name)
            continue
        meta = dict(base_meta, margins=margins, margin_required=N.MARGIN)
        N.save_golden(name, {"meta": meta, "errors": errors, "tensors": tensors})
    if failed:
        print(
            f"not written (threshold margin <= {N.MARGIN}): {failed}. Change the "
            "network's seed or input rate in networks.py and record the change."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
