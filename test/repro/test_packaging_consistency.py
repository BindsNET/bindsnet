"""Static checks that the reproducibility artifacts agree with each other.

Each check here guards a mismatch that once shipped: the Docker image on a
Python that pyproject.toml does not support (#810), and examples whose
``--seed`` left the CPU generator unseeded on GPU runs.
"""

import ast
import re
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[2]


def _requires_python():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return SpecifierSet(tomllib.load(f)["project"]["requires-python"])


def _supported_minors():
    spec = _requires_python()
    return {f"3.{m}" for m in range(0, 30) if f"3.{m}.0" in spec}


def test_docker_python_is_supported():
    dockerfile = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^FROM python:(\d+\.\d+)", dockerfile, re.M)
    assert match, "Dockerfile base image is not python:X.Y"
    assert f"{match.group(1)}.0" in _requires_python(), (
        f"Dockerfile uses Python {match.group(1)}, outside requires-python "
        f"{_requires_python()}"
    )


def test_ci_covers_every_supported_python():
    workflow = (ROOT / ".github" / "workflows" / "python-app.yml").read_text()
    tested = set(re.findall(r'"(3\.\d+)"', workflow))
    assert tested == _supported_minors(), (
        f"CI tests {sorted(tested)}, requires-python allows "
        f"{sorted(_supported_minors())}"
    )


def test_citation_version_matches_package():
    with open(ROOT / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(
        rf"^version: {re.escape(version)}$", citation, re.M
    ), f"CITATION.cff version does not match pyproject.toml ({version})"


def _seeds_unconditionally(tree):
    # True if some torch.manual_seed / torch.random.manual_seed call is not
    # inside an if/else, a try, or a loop.
    def visit(node, guarded):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if (
                name in ("torch.manual_seed", "torch.random.manual_seed")
                and not guarded
            ):
                return True
        guards = (ast.If, ast.Try, ast.For, ast.While, ast.IfExp)
        child_guarded = guarded or isinstance(node, guards)
        return any(visit(c, child_guarded) for c in ast.iter_child_nodes(node))

    return visit(tree, False)


def test_examples_seed_cpu_generator_unconditionally():
    # torch.manual_seed seeds the CPU generator and every GPU. Called only in a
    # CPU branch, GPU runs draw weights, Poisson spikes and shuffling unseeded.
    missing = []
    for path in sorted((ROOT / "examples").rglob("*.py")):
        source = path.read_text()
        if '"--seed"' not in source:
            continue
        if not _seeds_unconditionally(ast.parse(source)):
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, f"--seed does not seed torch unconditionally in: {missing}"
