"""Flag pull requests that weaken the test suite.

A change can make the tests pass by changing the tests instead of the code.
This script compares the base and head commits and lists every place where
that may have happened:

- a test file, test function or collected test id disappeared (``--collect``
  compares what pytest collects on both commits, which also catches
  ``del test_x``, ``__test__ = False``, shorter parameter lists and tests
  deselected by a conftest); renamed files count as removed plus added;
- a function in a ``test/`` file (test helpers and ``networks.py`` included)
  has fewer checks or fewer exact comparisons (``torch.equal`` and the like),
  more checks that cannot fail (``assert x or True``, an ``except
  AssertionError``, a ``return`` before the first check), or more skip/xfail
  markers (counted per function and for the module, e.g. ``pytestmark``);
- a tolerance is new or larger: keyword ``atol``/``rtol``/``abs``/``rel``/
  ``tol``/``delta``, the positional tolerance of ``allclose``/``isclose``/
  ``approx``, a parameter default, or an assigned name containing ``tol``
  (constant arithmetic is evaluated);
- a recorded-result file changed: golden outputs, operation counts, the API
  snapshot, or the coverage floor;
- a file that steers the tests changed: any ``conftest.py``, ``pyproject.toml``,
  ``pytest.ini``, ``tox.ini``, ``setup.cfg``, ``.coveragerc``, or the torch
  versions in ``poetry.lock`` (goldens are tied to them).

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
import os
import shutil
import subprocess
import sys
import tempfile

ACCEPT_LABEL = "tests-changed-ok"
CHECK_WORDS = ("assert", "allclose", "isclose", "equal", "raises", "approx")
EXACT_WORDS = ("equal",)  # torch.equal, array_equal, assert_equal
SKIP_WORDS = {"skip", "skipif", "xfail", "importorskip"}
TOLERANCE_KEYWORDS = {"atol", "rtol", "abs", "rel", "tol", "delta"}
# Positional tolerance arguments: function name -> {position: keyword}.
POSITIONAL_TOLERANCE = {
    "allclose": {2: "rtol", 3: "atol"},
    "isclose": {2: "rtol", 3: "atol"},
    "approx": {1: "rel", 2: "abs"},
}
RECORDED = (
    "test/regression/golden/",
    "test/regression/op_counts.json",
    "test/api_surface.json",
    ".github/scripts/coverage_floor.py",
    # How goldens are compared and computed (FLOAT_ATOL, the CPU pins).
    "test/regression/networks.py",
    "test/regression/cpu_pins.py",
)
# Files that decide which tests run, how, or how coverage is measured.
STEERING = {
    "pyproject.toml",
    "pytest.ini",
    ".pytest.ini",
    "tox.ini",
    "setup.cfg",
    ".coveragerc",
}


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
    """Value of a numeric constant expression (``1e-6``, ``10 * 1e-6``), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return None if isinstance(node.value, bool) else node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = number(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp):
        left, right = number(node.left), number(node.right)
        if left is None or right is None:
            return None
        ops = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.Div: lambda x, y: x / y if y else None,
            ast.Pow: lambda x, y: x**y if abs(y) < 100 else None,
        }
        op = ops.get(type(node.op))
        return op(left, right) if op else None
    return None


def tolerances_in(nodes):
    """Largest value of each tolerance among ``nodes``: keyword arguments,
    positional tolerance arguments of allclose/isclose/approx, names containing
    "tol" that are assigned (also with a type annotation), and parameter
    defaults named like a tolerance."""
    found = {}
    for node in nodes:
        pairs = []
        if isinstance(node, ast.Call):
            pairs = [
                (k.arg, k.value) for k in node.keywords if k.arg in TOLERANCE_KEYWORDS
            ]
            for pos, key in POSITIONAL_TOLERANCE.get(call_name(node), {}).items():
                if len(node.args) > pos:
                    pairs.append((key, node.args[pos]))
        elif isinstance(node, ast.Assign):
            pairs = [
                (t.id, node.value)
                for t in node.targets
                if isinstance(t, ast.Name) and "tol" in t.id.lower()
            ]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            t = node.target
            if isinstance(t, ast.Name) and "tol" in t.id.lower():
                pairs = [(t.id, node.value)]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults))
            defaults += node.args.defaults + node.args.kw_defaults
            pairs = [
                (arg.arg, d)
                for arg, d in zip(args, defaults)
                if d is not None
                and (arg.arg in TOLERANCE_KEYWORDS or "tol" in arg.arg.lower())
            ]
        for key, value_node in pairs:
            value = number(value_node)
            if value is not None:
                found[key] = max(found.get(key, value), value)
    return found


def _always_true(test):
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        return any(_always_true(v) for v in test.values)
    return False


def _swallows_assertions(node):
    if not isinstance(node, ast.Try):
        return False
    for handler in node.handlers:
        caught = handler.type
        names = []
        if caught is None:
            return True
        for c in caught.elts if isinstance(caught, ast.Tuple) else [caught]:
            names.append(c.id if isinstance(c, ast.Name) else getattr(c, "attr", ""))
        if any(n in ("AssertionError", "Exception", "BaseException") for n in names):
            return True
    return False


def _early_return(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    return bool(body) and isinstance(body[0], ast.Return)


def describe(source):
    """Per function (helpers included, so a check moved into a helper still
    counts) and for the module level ("<module>"): number of checks, of exact
    comparisons, of checks that cannot fail, of skip/xfail markers, and the
    largest value of each tolerance."""
    tree = ast.parse(source)

    def summary(nodes, own=None):
        nodes = list(nodes)
        calls = [n for n in nodes if isinstance(n, ast.Call)]
        names = [
            n.id if isinstance(n, ast.Name) else n.attr
            for n in nodes
            if isinstance(n, (ast.Name, ast.Attribute))
        ]
        return {
            "checks": sum(isinstance(n, ast.Assert) for n in nodes)
            + sum(any(w in call_name(c).lower() for w in CHECK_WORDS) for c in calls),
            "exact": sum(any(w in call_name(c).lower() for w in EXACT_WORDS) for c in calls),
            "cannot fail": sum(
                isinstance(n, ast.Assert) and _always_true(n.test) for n in nodes
            )
            + sum(_swallows_assertions(n) for n in nodes)
            + int(own is not None and _early_return(own)),
            "skip/xfail markers": sum(n in SKIP_WORDS for n in names),
            "tolerances": tolerances_in(nodes),
        }

    module_nodes = [
        n
        for top in tree.body
        if not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for n in ast.walk(top)
    ]
    functions = {"<module>": summary(module_nodes)}
    stack = [(node, "") for node in tree.body]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, ast.ClassDef):
            stack += [(child, f"{prefix}{node.name}.") for child in node.body]
            functions[prefix + node.name] = summary(
                [n for d in node.decorator_list for n in ast.walk(d)]
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[prefix + node.name] = summary(ast.walk(node), own=node)
    return functions


def compare_source(path, before, after):
    """Weakenings in one test or test-helper file."""
    if after is None:
        return [f"`{path}`: file removed"]
    if before is None:
        return []
    old, new = describe(before), describe(after)
    found = []
    for name, o in old.items():
        where = f"`{path}::{name}`"
        if name not in new:
            if name.rsplit(".", 1)[-1].startswith("test"):
                found.append(f"{where}: test removed")
            continue
        n = new[name]
        for key in ("checks", "exact"):
            if n[key] < o[key]:
                found.append(f"{where}: {key} went from {o[key]} to {n[key]}")
        for key in ("cannot fail", "skip/xfail markers"):
            if n[key] > o[key]:
                found.append(f"{where}: {key} went from {o[key]} to {n[key]}")
        for keyword, value in n["tolerances"].items():
            if keyword not in o["tolerances"]:
                found.append(f"{where}: tolerance `{keyword}` added ({value:g})")
            elif value > o["tolerances"][keyword]:
                found.append(
                    f"{where}: `{keyword}` widened from "
                    f"{o['tolerances'][keyword]:g} to {value:g}"
                )
    for name, n in new.items():  # new functions or classes that start skipped
        if name not in old and n["skip/xfail markers"]:
            found.append(f"`{path}::{name}`: new, with skip/xfail markers")
    return found


def torch_pins(lock):
    """Locked versions of torch and torchvision in a poetry.lock text."""
    pins, name = {}, None
    for line in (lock or "").splitlines():
        if line.startswith("name = "):
            name = line.split('"')[1]
        elif line.startswith("version = ") and name in ("torch", "torchvision"):
            pins[name] = line.split('"')[1]
    return pins


def collected_ids(commit):
    """Test ids pytest collects at ``commit`` (its own code and pyproject.toml)."""
    work = tempfile.mkdtemp(prefix="weakened_tests_")
    tree = os.path.join(work, "tree")
    try:
        git("worktree", "add", "--detach", tree, commit)
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"]
            + ["-p", "no:cacheprovider", "-c", "pyproject.toml"],
            cwd=tree,
            env=dict(os.environ, PYTHONPATH=tree),
            capture_output=True,
            text=True,
        ).stdout
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", tree])
        shutil.rmtree(work, ignore_errors=True)
    return {line.strip() for line in out.splitlines() if "::" in line}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--accept-label", default=ACCEPT_LABEL)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="also compare the test ids pytest collects (needs the test environment)",
    )
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    args.base = git("rev-parse", args.base).strip()
    args.head = git("rev-parse", args.head).strip()

    # --no-renames: a renamed file is a removal plus an addition, so changes
    # inside a renamed test file are compared with the old file.
    paths = git("diff", "--name-only", "--no-renames", args.base, args.head).splitlines()
    found = []
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if path.startswith(RECORDED):
            found.append(f"`{path}`: recorded results, or how they are compared, changed")
        elif name == "conftest.py":
            found.append(f"`{path}`: changed (it decides how tests run)")
        elif path in STEERING:
            found.append(f"`{path}`: changed (it can decide which tests run and how)")
        elif path == "poetry.lock":
            old, new = torch_pins(source_at(args.base, path)), torch_pins(
                source_at(args.head, path)
            )
            if old != new:
                found.append(
                    f"`poetry.lock`: torch versions changed {old} -> {new}; the golden "
                    "and operation-count files must be regenerated in this PR"
                )
        elif path.startswith("test/") and name.endswith(".py"):
            found += compare_source(
                path, source_at(args.base, path), source_at(args.head, path)
            )

    if args.collect:
        gone = sorted(collected_ids(args.base) - collected_ids(args.head))
        found += [f"`{t}`: no longer collected" for t in gone[:50]]
        if len(gone) > 50:
            found.append(f"... and {len(gone) - 50} more tests no longer collected")

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
