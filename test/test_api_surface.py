"""Public API snapshot: every public class, method and function in ``bindsnet``.

``test/api_surface.json`` stores the call signature of each public name. When a
parameter is added, removed, renamed, reordered or gets a new default, this test
fails, and the fix is to regenerate the file in the same pull request, so the
change shows up in the diff for the reviewer:

    python test/regen_api_surface.py --reason "why the public API changed"

Regeneration refuses to run without ``--reason`` and stores the reason in the file.

Only names, parameter kinds and defaults are recorded. Type annotations are left out
on purpose: Python 3.11, 3.12 and 3.13 print them differently, and changing an
annotation does not break callers.
"""

import importlib
import importlib.util
import inspect
import json
import pkgutil
import re
from pathlib import Path

import pytest
import torch

import bindsnet

SNAPSHOT = Path(__file__).with_name("api_surface.json")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _default_text(value):
    """A default value as text that is the same on every machine and run."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if inspect.isclass(value) or inspect.isfunction(value):
        return f"{value.__module__}.{value.__qualname__}"
    return _ADDRESS.sub("", repr(value))


def _signature_text(obj):
    """``(a, b=1, *args, c=None, **kwargs)`` without annotations."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "<no signature>"
    parts = []
    seen_kw_only = False
    params = list(sig.parameters.values())
    for i, p in enumerate(params):
        if p.kind is p.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        if p.kind is p.VAR_POSITIONAL:
            seen_kw_only = True
            parts.append(f"*{p.name}")
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        text = p.name
        if p.default is not p.empty:
            text += f"={_default_text(p.default)}"
        parts.append(text)
        is_last_pos_only = p.kind is p.POSITIONAL_ONLY and (
            i + 1 == len(params) or params[i + 1].kind is not p.POSITIONAL_ONLY
        )
        if is_last_pos_only:
            parts.append("/")
    return "(" + ", ".join(parts) + ")"


def collect_surface():
    """Map ``module.Name[.method]`` to its signature text, for all of bindsnet."""
    surface = {}
    modules = [bindsnet.__name__] + [
        m.name for m in pkgutil.walk_packages(bindsnet.__path__, "bindsnet.")
    ]
    for modname in sorted(modules):
        module = importlib.import_module(modname)
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != modname:
                continue
            key = f"{modname}.{name}"
            if inspect.isclass(obj):
                surface[key] = _signature_text(obj)
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        surface[f"{key}.{attr}"] = _signature_text(member)
            elif inspect.isfunction(obj):
                surface[key] = _signature_text(obj)
    return dict(sorted(surface.items()))


def test_public_api_matches_snapshot():
    """Every public signature in bindsnet equals the one in test/api_surface.json.

    Caught: renaming ``tc_trace`` to ``tc_traces`` in ``Nodes.__init__``.
    """
    stored = json.loads(SNAPSHOT.read_text())["surface"]
    live = collect_surface()
    removed = sorted(set(stored) - set(live))
    added = sorted(set(live) - set(stored))
    changed = sorted(k for k in set(stored) & set(live) if stored[k] != live[k])
    lines = [f"removed: {k}" for k in removed]
    lines += [f"added:   {k} {live[k]}" for k in added]
    for k in changed:
        lines += [f"changed: {k}", f"   was  {stored[k]}", f"   now  {live[k]}"]
    assert not lines, (
        "The public API differs from test/api_surface.json. If this is intended, "
        'run: python test/regen_api_surface.py --reason "<why>"\n' + "\n".join(lines)
    )


def test_regen_refuses_without_reason(tmp_path, monkeypatch):
    """regen_api_surface.py exits without writing when --reason is missing or blank.

    Caught: ``required=True`` removed from the ``--reason`` argument.
    """
    spec = importlib.util.spec_from_file_location(
        "regen_api_surface", Path(__file__).with_name("regen_api_surface.py")
    )
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    target = tmp_path / "api_surface.json"
    monkeypatch.setattr(regen, "SNAPSHOT", target)
    for argv in ([], ["--reason", "  "]):
        with pytest.raises(SystemExit):
            regen.main(argv)
        assert not target.exists()
    regen.main(["--reason", "test run"])
    data = json.loads(target.read_text())
    assert data["meta"]["reason"] == "test run"
    assert data["history"][-1]["reason"] == "test run"
