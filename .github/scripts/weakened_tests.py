"""Flag pull requests that weaken the test suite.

A change can make the tests pass by changing the tests instead of the code.
This script compares the base and head commits and lists every place where
that may have happened:

- a test file or a test function was removed;
- a function in a test file (helpers included) has fewer checks (``assert``
  statements and calls whose name contains ``assert``, ``allclose``,
  ``equal`` or ``raises``);
- a test file has more skip or xfail markers (``skip``, ``skipif``, ``xfail``,
  ``importorskip``);
- a tolerance in a test file now allows a larger value than before: a keyword
  argument ``atol``, ``rtol``, ``abs``, ``rel``, ``tol`` or ``delta``, or an
  assigned name containing ``tol``;
- a recorded-result file changed: golden outputs, operation counts, the API
  snapshot, or the coverage floor;
- a ``conftest.py`` or the pytest settings in ``pyproject.toml`` changed (they
  decide which tests run).

Each finding needs a maintainer's look, not a fix: removing a wrong test or
regenerating goldens after a real fix is fine when the pull request says why.
The check fails while there are findings, unless a maintainer adds the label
given by ``--accept-label``. Contributors cannot add labels.

Usage (in CI): python .github/scripts/weakened_tests.py --base <sha> \
    --head <sha> --labels "<comma-separated labels>" --summary "$GITHUB_STEP_SUMMARY"

Exit code 0 when nothing is found or the label is present, 1 otherwise, 2 on a
tool error.
"""

import argparse
import ast
import subprocess
import sys

ACCEPT_LABEL = "tests-changed-ok"
CHECK_WORDS = ("assert", "allclose", "equal", "raises")
SKIP_WORDS = {"skip", "skipif", "xfail", "importorskip"}
TOLERANCE_KEYWORDS = {"atol", "rtol", "abs", "rel", "tol", "delta"}
RECORDED = (
    "test/regression/golden/",
    "test/regression/op_counts.json",
    "test/api_surface.json",
    ".github/scripts/coverage_floor.py",
)


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def source_at(commit, path):
    try:
        return git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def number(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    return value if isinstance(value, (int, float)) else None


def tolerances_in(nodes):
    """Largest value of each tolerance keyword argument, and of each assigned
    name containing "tol" (``ATOL = 1e-6``), among ``nodes``."""
    found = {}
    for node in nodes:
        pairs = []
        if isinstance(node, ast.Call):
            pairs = [
                (k.arg, k.value) for k in node.keywords if k.arg in TOLERANCE_KEYWORDS
            ]
        elif isinstance(node, ast.Assign):
            pairs = [
                (t.id, node.value)
                for t in node.targets
                if isinstance(t, ast.Name) and "tol" in t.id.lower()
            ]
        for key, value_node in pairs:
            value = number(value_node)
            if value is not None:
                found[key] = max(found.get(key, value), value)
    return found


def describe(source):
    """Per function (helpers included, so a check moved into a helper still
    counts): number of checks and the largest value of each tolerance. The
    module level is the entry "<module>". Plus the file's count of skip and
    xfail markers."""
    tree = ast.parse(source)
    skips = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            skips += name in SKIP_WORDS
    functions = {"<module>": (0, tolerances_in(tree.body))}
    stack = [(node, "") for node in tree.body]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, ast.ClassDef):
            stack += [(child, f"{prefix}{node.name}.") for child in node.body]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = list(ast.walk(node))
            checks = sum(isinstance(n, ast.Assert) for n in inner) + sum(
                any(word in call_name(n).lower() for word in CHECK_WORDS)
                for n in inner
                if isinstance(n, ast.Call)
            )
            functions[prefix + node.name] = (checks, tolerances_in(inner))
    return functions, skips


def compare_test_file(path, before, after):
    if after is None:
        return [f"`{path}`: test file removed"]
    if before is None:
        return []
    old_functions, old_skips = describe(before)
    new_functions, new_skips = describe(after)
    found = []
    for name, (old_checks, old_tols) in old_functions.items():
        if name not in new_functions:
            if name.rsplit(".", 1)[-1].startswith("test"):
                found.append(f"`{path}::{name}`: test removed")
            continue
        new_checks, new_tols = new_functions[name]
        if new_checks < old_checks:
            found.append(
                f"`{path}::{name}`: checks went from {old_checks} to {new_checks}"
            )
        for keyword, value in new_tols.items():
            if keyword in old_tols and value > old_tols[keyword]:
                found.append(
                    f"`{path}::{name}`: `{keyword}` widened from "
                    f"{old_tols[keyword]:g} to {value:g}"
                )
    if new_skips > old_skips:
        found.append(
            f"`{path}`: skip/xfail markers went from {old_skips} to {new_skips}"
        )
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--accept-label", default=ACCEPT_LABEL)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    paths = git("diff", "--name-only", args.base, args.head).splitlines()
    found = []
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if path.startswith(RECORDED):
            found.append(f"`{path}`: recorded results changed")
        elif name == "conftest.py":
            found.append(f"`{path}`: changed (it decides how tests run)")
        elif path == "pyproject.toml":
            before = source_at(args.base, path) or ""
            after = source_at(args.head, path) or ""
            section = "[tool.pytest"
            if before.partition(section)[2] != after.partition(section)[2]:
                found.append("`pyproject.toml`: pytest settings changed")
        elif (
            path.startswith("test/")
            and name.startswith("test_")
            and name.endswith(".py")
        ):
            found += compare_test_file(
                path, source_at(args.base, path), source_at(args.head, path)
            )

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    lines = ["## Tests weakened or recorded results changed", ""]
    if not found:
        lines.append("**Passed:** nothing found.")
        code = 0
    else:
        lines += [f"- {item}" for item in found] + [""]
        if args.accept_label in labels:
            lines.append(
                f"**Passed:** a maintainer reviewed these and added `{args.accept_label}`."
            )
            code = 0
        else:
            lines.append(
                "**Failed:** a maintainer must review each item above. The pull "
                "request should say why each one is right. A maintainer then "
                f"adds the label `{args.accept_label}`."
            )
            code = 1
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # report tool errors without hiding them
        print(f"weakened_tests.py: tool error: {exc!r}")
        sys.exit(2)
