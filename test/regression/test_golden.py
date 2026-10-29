"""
Golden-output regression tests: every simulated spike, voltage, trace and weight
of sixteen small seeded networks must match the files in ``golden/``.

Caught on 2026-10-24 (each deliberate wrong change was applied, the tests run,
then reverted; the decision log in logs/test_hardening_2026-10/PLAN.md lists the
networks that failed for each):

* ``MSTDPET`` without the ``dt`` factor of Florian eq. 2.7;
* ``LIFNodes.forward`` updating traces before the spike check and reset;
* ``PostPre`` switching its layers to additive (all-to-all) traces.

Spikes and other non-float tensors must be identical; floats must agree within
``atol=1e-6``. If a change to the results is intended, regenerate with
``python test/regression/regen.py --reason "<why>"`` and commit the new files.
"""

import os

import networks as N
import pytest
import torch


def _torch_minor(version):
    return ".".join(str(version).split("+")[0].split(".")[:2])


@pytest.mark.parametrize("name", list(N.SPECS))
def test_golden(name):
    """Compare everything network ``name`` records with its golden file.
    Caught: MSTDPET without dt, LIF traces before reset, PostPre additive traces."""
    path = N.golden_path(name)
    assert path.exists(), f"no golden file {path}; run regen.py --reason"
    golden = N.load_golden(name)
    made_with = golden["meta"]["torch"]
    if _torch_minor(made_with) != _torch_minor(torch.__version__):
        # On CI a torch upgrade must regenerate the goldens in the same PR,
        # instead of silently skipping every golden test after it merges.
        (pytest.fail if os.environ.get("CI") else pytest.skip)(
            f"golden files were made with torch {made_with}; "
            "regenerate with regen --reason"
        )

    tensors, errors, _ = N.run_standard(name)

    assert errors == golden["errors"], (
        f"{name}: errors now {errors}, golden {golden['errors']}. If this is "
        "intended, regenerate with regen.py --reason."
    )
    problems = N.compare(golden["tensors"], tensors)
    assert not problems, (
        f"{name}: {len(problems)} recorded tensors differ from the golden file "
        f"(made {golden['meta']['date']}, commit {golden['meta']['commit']}):\n  "
        + "\n  ".join(problems[:20])
    )
