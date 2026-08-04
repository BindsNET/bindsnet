# Breakout examples

Scripts demonstrating reinforcement-learning-style use of BindsNET on the Atari
**Breakout** environment (`BreakoutDeterministic-v4`, via `gymnasium[atari]` + `ale-py`;
see [../../DATA.md](../../DATA.md)).

- `breakout.py`, `breakout_stdp.py` — run an SNN on Breakout (with/without STDP).
- `random_baseline.py`, `random_network_baseline.py` — random-action / random-network baselines.
- `play_breakout_from_ANN.py` — convert a pretrained ANN into an SNN and play (see below).

## Pretrained artifact: `trained_shallow_ANN.pt`

| Property | Value |
|----------|-------|
| File | `trained_shallow_ANN.pt` (~25 MB, tracked in git) |
| What it is | A pretrained **shallow ANN** (Q-network) for Atari Breakout |
| Architecture | `nn.Linear(6400, 1000)` → `ReLU` → `nn.Linear(1000, 4)` (class `Net` in `play_breakout_from_ANN.py`) |
| Input | 6400 features = a flattened 80×80 preprocessed Breakout frame |
| Output | 4 units = the Breakout discrete action space |
| Consumed by | `play_breakout_from_ANN.py:55` (`torch.load("trained_shallow_ANN.pt")`) |
| How it is used | Its `fc1`/`fc2` weights are transposed, scaled (`layer1scale=57.68`, `layer2scale=77.48`), and transplanted into a spiking network `Input(6400) → LIFNodes(1000) → LIFNodes(4)`, which is then run on Breakout through an `EnvironmentPipeline` with Poisson encoding — an ANN→SNN conversion demo. |

### Regeneration

**The training script that produced `trained_shallow_ANN.pt` is not included in this
repository.** The file is shipped as a pretrained weight blob. To regenerate it you would
need to train a network with the `Net` architecture above (input 6400, hidden 1000,
output 4) as a Breakout Q-network and save it with `torch.save`. If you reproduce or
replace this artifact, please document the training data, hyperparameters, and seed here.
