# Regression tests: golden outputs and invariants

These tests catch a change in simulation results that nobody declared. They need no
dataset, no network access and no GPU, and they run on every pull request.

## Files

- `networks.py`: sixteen small seeded networks (one per neuron type, connection type
  and learning rule that BindsNET ships, some at `dt = 0.5` so a dropped time-step
  factor shows). Also the code that runs a network and records every layer's spikes,
  voltages and traces at every step, plus all weights.
- `golden/*.pt.xz`: the recorded outputs of each network, compressed. Each file also
  stores the torch version, Python version, machine, date, commit, the reason it was
  made, and how close every neuron came to its firing threshold.
- `test_golden.py`: runs each network again and compares with its golden file. Spikes
  must match exactly. Decimal values must match within 0.000001. The test is skipped
  when the installed torch differs from the one in the file in its first two version
  numbers, because torch does not promise identical random numbers across versions.
- `test_invariants.py`: checks that need no stored file, so they hold on any
  computer. Running for T steps equals running twice for T/2. A reset network equals a
  fresh one. Saving and loading the state gives the same run. Learning off leaves the
  weights untouched. A batch of identical samples equals one sample. Pair STDP has no
  time-step factor and the reward-modulated rule MSTDPET does.
- `regen.py`: rewrites the golden files.
- `conftest.py`: one thread and deterministic algorithms for every test here.

## When a golden test fails

Your change altered a result. Either that is the point of the change, or it is a bug.

- If it is a bug, fix the code until the test passes again.
- If it is intended, regenerate the files in their own commit and say why:

```shell
python test/regression/regen.py --reason "why the results changed"
```

The script refuses to run without a reason and stores it in every file. Results-changing
fixes are released in version 0.4.0, not in a 0.3.x patch release, so the pull request
will wait for that release. See `CONTRIBUTING.md`, "Tests that guard results".

## Expected failures

Some tests are marked as expected failures (`xfail`, strict). Each names a known bug
in the code it exercises. When the bug is fixed the test fails with "DID NOT RAISE" or
by passing; remove the mark in the same pull request as the fix.

## Wrong-change checks

Every test's docstring names one deliberate wrong change that made it fail when the
test was written. A test that cannot fail guards nothing.
