"""Regenerate test/api_surface.json, the public API snapshot.

Usage (from the repository root):

    python test/regen_api_surface.py --reason "removed the unused foo argument"

The reason is required and is stored in the file, together with the date, the
commit and the Python and torch versions, so the pull request diff says why the
public API changed. Earlier reasons are kept in ``history``.
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_api_surface import SNAPSHOT, collect_surface  # noqa: E402


def _commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reason",
        required=True,
        help="why the public API changed (stored in api_surface.json)",
    )
    args = parser.parse_args(argv)
    reason = args.reason.strip()
    if not reason:
        parser.error("--reason must not be empty")

    history = []
    if SNAPSHOT.exists():
        old = json.loads(SNAPSHOT.read_text())
        history = old.get("history", [])
    meta = {
        "reason": reason,
        "date": datetime.date.today().isoformat(),
        "commit": _commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    history.append({"date": meta["date"], "commit": meta["commit"], "reason": reason})
    surface = collect_surface()
    data = {"meta": meta, "history": history, "surface": surface}
    SNAPSHOT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(surface)} signatures to {SNAPSHOT}")


if __name__ == "__main__":
    main()
