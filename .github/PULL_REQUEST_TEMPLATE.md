## What this changes and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] I ran `poetry run pytest` locally and it passed.
- [ ] If I fixed a bug, this test fails on `master` and passes with my change
      (checked by the `bug-test` check): `test/...::test_...`
- [ ] I did not remove tests, remove checks, add skips or xfails, or widen
      tolerances. If I did, I explain each one below (the `weakened-tests`
      check lists them).
- [ ] If I changed a formula (a neuron update, a learning rule, a connection's
      compute), this test pins the new formula: `test/...::test_...`
- [ ] Does this change simulated results (spikes, voltages, traces, weights)?
      - [ ] No.
      - [ ] Yes. I regenerated the golden files in `test/regression/` with
            `--reason`, and explain below why the results changed. (Results-changing
            fixes are released in 0.4.0, not 0.3.x; see `CONTRIBUTING.md`.)
- [ ] Did any of these files change? If yes, say which and why:
      - golden files under `test/regression/`
      - operation-count file under `test/regression/` (if present)
      - `test/api_surface.json` (public API)
      - `FIXED_BUGS` in `test/network/test_all_classes.py`
- [ ] I added an entry to the `Unreleased` section of `CHANGELOG.md`.
- [ ] If I used an AI tool to write or find this change, I name it here:

## Why the results or the API changed (if they did)

<!-- Which file changed, which numbers moved, and why the new numbers are right. -->
