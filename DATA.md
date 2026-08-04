# Datasets & Stimuli used in BindsNET

BindsNET ships **no** third-party datasets. Its dataset loaders fetch data from the
upstream sources declared below; all licenses are the upstream providers' and BindsNET
does not redistribute the data. This file declares every dataset and synthetic stimulus
referenced by the shipped examples and benchmarks, plus the additional dataset loaders
the library provides.

> Licenses below are pointers to the upstream source, not assertions by BindsNET.
> Confirm the current license at the source before using a dataset in your own work.

---

## 1. Datasets used by the shipped examples

### MNIST
- **Loader:** `from bindsnet.datasets import MNIST` — a thin wrapper over
  `torchvision.datasets.MNIST` (`bindsnet/datasets/torchvision_wrapper.py`).
- **Upstream source:** torchvision → http://yann.lecun.com/exdb/mnist/
- **Version/snapshot:** whatever the installed `torchvision` resolves (mirror-hosted).
- **Obtained by:** automatic download on first run (`download=True` in the examples).
- **License:** as published by the upstream/torchvision mirror (verify upstream).
- **Used in:** `examples/mnist/*.py`
  (e.g. `eth_mnist.py`, `batch_eth_mnist.py`, `supervised_mnist.py`, `conv_mnist.py`,
  `reservoir.py`, `MCC_reservoir.py`, `conv1d_MNIST.py`, `conv3d_MNIST.py`,
  `loc1d_mnist.py`, `loc2d_mnist.py`, `loc3d_mnist.py`, `SOM_LM-SNNs.py`).
- **Preprocessing → spikes:** `transforms.ToTensor()` then scaling by `--intensity`
  (default 128 in `eth_mnist.py`), then rate coding via
  `bindsnet.encoding.PoissonEncoder(time, dt)` — pixel intensities become Poisson
  spike trains over `time` ms at step `dt`.

### Atari — Breakout (and Space Invaders)
- **Loader:** `bindsnet.environment.GymEnvironment("BreakoutDeterministic-v4")`
  (see `examples/breakout/*.py`).
- **Upstream source:** Arcade Learning Environment via `gymnasium[atari]` + `ale-py`
  (declared in `pyproject.toml`). ROMs are provided through the ALE/AutoROM tooling.
- **Obtained by:** the Gymnasium/ALE runtime; not stored in this repo.
- **License:** ALE/ROM licensing applies (verify via ale-py / AutoROM).
- **Used in:** `examples/breakout/breakout.py`, `breakout_stdp.py`,
  `play_breakout_from_ANN.py`, `random_baseline.py`, `random_network_baseline.py`.
- **Preprocessing → spikes:** Atari observations are converted to network input by the
  example pipelines (see each script and `bindsnet/encoding/`).
- **Pretrained artifact:** `examples/breakout/trained_shallow_ANN.pt` (a Breakout
  Q-network transplanted into an SNN) — provenance in
  [examples/breakout/README.md](examples/breakout/README.md).

---

## 2. Synthetic stimuli (no external dataset)

### Scaling-benchmark Poisson drive
Used by `examples/benchmark/benchmark.py` and reported in the README "Benchmarking"
section and Hazan et al. 2018:
- Population of **n** Poisson input neurons, firing rates drawn from **U(0, 100) Hz**.
- Connected all-to-all to an equally sized population of LIF neurons; connection
  weights sampled from **N(0, 1)**.
- **n** varied 250 → 10,000 in steps of 250; each run simulated **1,000 ms** at
  **dt = 1.0 ms**.

This stimulus is generated programmatically; there is no dataset to download.

---

## 3. Additional dataset loaders provided by the library

These loaders are part of `bindsnet.datasets` and are available to users, though not
every one is exercised by a shipped example. Sources are taken directly from the loader
modules.

| Dataset | Loader | Upstream source | Notes |
|---------|--------|-----------------|-------|
| Spoken MNIST (Free Spoken Digit Dataset) | `bindsnet.datasets.SpokenMNIST` (`spoken_mnist.py`) | https://github.com/Jakobovski/free-spoken-digit-dataset (downloads `master.zip`) | License per upstream repo |
| ALOV300++ | `bindsnet.datasets.ALOV300` (`alov300.py`) | frames `http://isis-data.science.uva.nl/alov/alov300++_frames.zip`, GT text `http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip`; info `http://alov300pp.joomlafree.it/dataset-resources.html` | Visual-tracking dataset |
| DAVIS 2017 | `bindsnet.datasets.Davis` (`davis.py`) | https://davischallenge.org/davis2017/code.html | Video object segmentation |
| Other torchvision datasets | `create_torchvision_dataset_wrapper(...)` (`torchvision_wrapper.py`) | torchvision | Wrappers exported for CIFAR10/100, FashionMNIST, EMNIST, KMNIST, SVHN, STL10, Omniglot, VOC*, COCO*, etc. — each downloads from its torchvision-declared source |

---

## Data handling notes
- Datasets download to a user-specified `root` directory (the examples typically use a
  local `data/` path); they are **not** committed to this repository.
- BindsNET does not modify or redistribute upstream data; it applies encodings
  (`bindsnet/encoding/`) to turn inputs into spike trains at simulation time.
- If a download URL has moved, consult the loader module in `bindsnet/datasets/` and the
  upstream project page listed above.
