#!/usr/bin/env python3
"""Fail when line coverage of a package directory drops below its floor.

    python .github/scripts/coverage_floor.py coverage.xml

Reads the Cobertura XML that ``pytest --cov=bindsnet --cov-report=xml`` writes and
checks each directory in FLOORS. The floors are one point under the value measured
when they were set (see the comment on each). They are floors, not targets: raise one
when coverage has clearly moved up, never lower one to make a pull request pass.
Prints one line per directory and exits 1 if any is below its floor.
"""

import sys
import xml.etree.ElementTree as ET

# One point under the coverage measured on GitHub CI (build job, Python 3.13, CPU
# torch): network and learning in run 40798879593, the others in run 40804291700
# (master 33913c7), both 2026-10-28.
FLOORS = {
    "network": 89.6,  # 90.6% measured
    "learning": 94.2,  # 95.2% measured
    "models": 98.3,  # 99.3% measured
    "encoding": 73.6,  # 74.6% measured
    "conversion": 58.8,  # 59.8% measured
    "analysis": 32.8,  # 33.8% measured
    "environment": 26.3,  # 27.3% measured
    "pipeline": 19.8,  # 20.8% measured
    "datasets": 20.6,  # 21.6% measured
}
# Cobertura filenames are relative to the <source> element, which pytest-cov sets to
# the bindsnet package directory, so "network/nodes.py", not "bindsnet/network/...".


def main(path: str) -> int:
    root = ET.parse(path).getroot()
    lines = {d: [0, 0] for d in FLOORS}  # dir -> [covered, valid]
    for cls in root.iter("class"):
        filename = cls.get("filename", "")
        for d in FLOORS:
            if filename.startswith(d + "/"):
                for line in cls.iter("line"):
                    lines[d][1] += 1
                    if int(line.get("hits", "0")) > 0:
                        lines[d][0] += 1
    bad = 0
    for d, floor in FLOORS.items():
        covered, valid = lines[d]
        pct = 100.0 * covered / valid if valid else 0.0
        if floor is None:
            print(f"bindsnet/{d}: {pct:.1f}% (no floor set yet)")
            continue
        status = "ok" if pct >= floor else "BELOW FLOOR"
        print(f"bindsnet/{d}: {pct:.1f}% (floor {floor}%) {status}")
        bad += pct < floor
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"))
