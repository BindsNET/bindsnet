# BindsNET project notes

## Python environment

This project runs in the conda environment named `bindsNET`.

- Interpreter: `/home/hananel/miniconda3/envs/bindsNET/bin/python`
- Activate: `conda activate bindsNET`
- The package is installed in editable mode from this directory, so edits to
  `bindsnet/` are picked up without reinstalling.
- Torch 2.14 with CUDA 13.0 is installed there. Do not use the `base` env.

Run tests with:

```shell
conda run -n bindsNET python -m pytest -q
```

Run a single file with:

```shell
conda run -n bindsNET python -m pytest -q test/network/test_learning.py
```

## Committing

Use `sc "message"` instead of `git commit` (see the user's global instructions).
