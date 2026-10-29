"""Check that a pull request which changes library code shows a test that fails
without the change and passes with it.

A bug fix should come with the test that exposes the bug. This script runs the
test files the pull request adds or changes twice:

- on the base commit's ``bindsnet/`` with the pull request's ``test/`` folder
  copied over it (``git worktree`` of the base, then ``git checkout <head> -- test``),
- on the head commit as checked out.

It passes when at least one test fails (or cannot be collected) on the base and
passes on the head. Only a test that ran on the base and failed there counts
(any exception in its body, including an assertion or "DID NOT RAISE"). A test
that cannot be imported or set up on the base (for example, it imports a
function the pull request adds) shows no bug and does not count; a pull request
that adds a feature needs the maintainer label. The summary shows each counted
test's failure on the base, because a test can also fail for reasons unrelated
to the bug, and a maintainer must read it.

It also passes, with a note, when the pull request does not touch any
``bindsnet/*.py`` file (documentation, dependencies, workflows), or when the
pull request carries the label given by ``--skip-label``. Only maintainers can
add labels, so contributors cannot skip the check themselves. Use the label for
changes that are not bug fixes, such as a refactor or a speed-up, which must
instead keep every existing test passing.

Usage (in CI, from the head checkout):

    python .github/scripts/fails_before_fix.py --base <sha> --head <sha> \
        --labels "<comma-separated labels>" --summary "$GITHUB_STEP_SUMMARY"

Exit code 0 when the check passes, 1 when it fails, 2 on a tool error.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SKIP_LABEL = "no-bug-test"


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def changed(base, head, path, filters="ACMR"):
    out = git("diff", "--name-only", f"--diff-filter={filters}", base, head, "--", path)
    return [p for p in out.splitlines() if p.endswith(".py")]


def run_pytest(tree, files, xml):
    """Run pytest on ``files`` inside ``tree`` with ``tree`` first on the path.

    Returns {test id: ("pass" | "fail" | "error", message)} and the modules that
    could not be collected. "fail" means the test ran and failed (any exception
    in its body); "error" means set-up or collection failed. Skipped and xfailed
    tests are left out. The settings come from the tree's pyproject.toml only, so
    an added pytest.ini or setup.cfg cannot change what runs.
    """
    env = dict(os.environ, PYTHONPATH=str(tree))
    where = subprocess.run(
        [sys.executable, "-c", "import bindsnet; print(bindsnet.__file__)"],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not Path(where).resolve().is_relative_to(Path(tree).resolve()):
        raise RuntimeError(f"bindsnet imported from {where}, not from {tree}")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + ["-c", "pyproject.toml", "--continue-on-collection-errors"]
        + [f"--junitxml={xml}", *files],
        cwd=tree,
        env=env,
    )
    results, broken = {}, set()
    for case in ET.parse(xml).iter("testcase"):
        tags = {child.tag for child in case}
        if not case.get("classname"):  # a file that could not be collected
            broken.add(case.get("name"))
            continue
        test_id = f"{case.get('classname')}::{case.get('name')}"
        if "skipped" in tags:
            continue
        for kind, tag in (("fail", "failure"), ("error", "error"), ("pass", None)):
            child = case.find(tag) if tag else None
            if tag is None or child is not None:
                message = (child.get("message") or "") if child is not None else ""
                results[test_id] = (kind, message.splitlines()[0][:200] if message else "")
                break
    return results, broken


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--skip-label", default=SKIP_LABEL)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    lines = ["## Test that fails before the fix", ""]

    def finish(code, message):
        lines.append(message)
        text = "\n".join(lines) + "\n"
        print(text)
        if args.summary:
            with open(args.summary, "a") as f:
                f.write(text)
        return code

    # Commit ids, so "master" or "HEAD" mean the same inside the base worktree.
    args.base = git("rev-parse", args.base).strip()
    args.head = git("rev-parse", args.head).strip()

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    if args.skip_label in labels:
        return finish(0, f"Skipped: a maintainer added the label `{args.skip_label}`.")

    source = changed(args.base, args.head, "bindsnet/", filters="ACDMR")
    if not source:
        return finish(0, "Passed: no file under `bindsnet/` changed.")
    lines += ["Library files changed: " + ", ".join(f"`{p}`" for p in source), ""]

    tests = [p for p in changed(args.base, args.head, "test/") if "/test_" in p]
    if not tests:
        return finish(
            1,
            "**Failed:** library code changed, but no test file was added or "
            "changed. Add a test that fails on `master` and passes with this "
            "change. If this is not a bug fix (a refactor or a speed-up), a "
            f"maintainer can add the label `{args.skip_label}`.",
        )
    lines += ["Test files run: " + ", ".join(f"`{p}`" for p in tests), ""]

    work = Path(tempfile.mkdtemp(prefix="fails_before_fix_"))
    base_tree = work / "base"
    try:
        git("worktree", "add", "--detach", str(base_tree), args.base)
        git("checkout", args.head, "--", "test", cwd=base_tree)
        on_base, broken = run_pytest(base_tree, tests, work / "base.xml")
        on_head, _ = run_pytest(Path.cwd(), tests, work / "head.xml")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_tree)])
        shutil.rmtree(work, ignore_errors=True)

    def collection_broken(test_id):
        module = test_id.split("::")[0]
        return any(module == m or module.startswith(m + ".") for m in broken)

    passing = [t for t, (kind, _) in on_head.items() if kind == "pass"]
    # Evidence: the test ran on the base and failed there. A test that could not
    # be imported or set up on the base (for example, it imports a name the PR
    # adds) shows nothing about a bug; it is listed for the maintainer instead.
    shown = [t for t in passing if on_base.get(t, ("", ""))[0] == "fail"]
    unshown = [
        t
        for t in passing
        if t not in shown and (collection_broken(t) or on_base.get(t, ("",))[0] == "error")
    ]
    if unshown:
        lines.append(
            "Not counted (could not be collected or set up on the base, so they "
            "show no bug): " + ", ".join(f"`{t}`" for t in sorted(unshown)[:20])
        )
        lines.append("")
    if shown:
        lines.append(
            "Tests that fail on the base and pass with this change, with the "
            "base failure (a maintainer checks that it is the bug):"
        )
        lines += [f"- `{t}`: {on_base[t][1]}" for t in sorted(shown)[:50]]
        if len(shown) > 50:
            lines.append(f"- ... and {len(shown) - 50} more")
        return finish(0, "\n**Passed.**")
    return finish(
        1,
        "**Failed:** every test in the changed test files gives the same result "
        "with and without the library change, so none of them shows the bug "
        "this pull request fixes. Add a test that fails on `master` and passes "
        "with this change. If this is not a bug fix (a refactor or a speed-up), "
        f"a maintainer can add the label `{args.skip_label}`.",
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # report tool errors without hiding them
        print(f"fails_before_fix.py: tool error: {exc!r}")
        sys.exit(2)
