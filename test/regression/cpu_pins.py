"""Pin the CPU maths libraries to one code path on every processor.

The golden files store float32 voltages, traces and weights to within 1e-6.
Without these pins the same code gives different rounding on different CPUs,
because each library picks its fastest code path for the processor it runs on.
On GitHub's runners this moved voltages by up to 1.5e-5 (and an Izhikevich
voltage by 0.23 mV) against files made on an AVX2-only CPU (CI run
40800432540, 2026-10-28). Measured on that AVX2 CPU, changing only MKL's code
path changed 1 to 4 of the 16 goldens.

- ``ATEN_CPU_CAPABILITY=avx2``: PyTorch's own kernels use AVX2 even on
  AVX-512 processors.
- ``MKL_CBWR=COMPATIBLE``: MKL (matrix products) uses its code path that gives
  the same results on all x86 processors.
- oneDNN (convolutions) is switched off in ``networks.deterministic()``; its
  blocking depends on cache sizes, and it does not follow ``MKL_CBWR``.

The variables must be set before torch runs its first operation, so
``test/conftest.py`` and ``regen.py`` call ``apply()`` before importing torch.
"""

import os

PINS = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "COMPATIBLE"}


def apply():
    os.environ.update(PINS)
