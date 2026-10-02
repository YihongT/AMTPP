# AMTPP

PyTorch implementation of **Attentive Marked Temporal Point Processes** for predicting the next trip's inter-trip time, origin, and destination.

The model combines causal attention, daily and weekly positional embeddings, an asymmetric log-Laplace time mixture, and a low-rank origin–destination head. An optional transportation network adapter propagates station embeddings through an aligned adjacency matrix. See [model calculations](docs/model.md) for causal alignment, density definitions, and destination objectives.

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/YihongT/AMTPP.git
cd AMTPP
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use a PyTorch build appropriate for your CPU or CUDA environment.

## Quick start

Generate artificial records and run a small CPU example:

```bash
python scripts/make_example.py
amtpp-run --config configs/example.json --data data/example.pkl --output outputs/example
amtpp-run --config configs/example.json --data data/example.pkl --output outputs/example --evaluate
```

Training saves `best.pt` and `selection.json`. Evaluation reloads that checkpoint and writes `test.json` and `events.npz`. Test metrics are computed after checkpoint selection.

## Your data

Supply a local pandas pickle containing one row per trip:

| Column | Meaning |
| --- | --- |
| `userID` | Consistent user identifier |
| `startTime` | Trip departure timestamp |
| `origin` | Origin station identifier |
| `destination` | Destination station identifier |

Station IDs use the same encoding for origins and destinations. Timestamps use a consistent local timezone; inter-trip time is measured in **hours**. See [data requirements](docs/data.md).

## Train and evaluate

```bash
amtpp-run --config configs/hangzhou.json --data data/hangzhou.pkl --output outputs/hangzhou --device cuda:0
amtpp-run --config configs/hangzhou.json --data data/hangzhou.pkl --output outputs/hangzhou --device cuda:0 --evaluate
```

Use `configs/guangzhou.json` for Guangzhou. To enable the network adapter, add `--topology data/topology.npz` to **both** commands. `python -m amtpp.train --help` lists the underlying model and training options.

The default configurations use user-disjoint splits, training-history station vocabulary, and training targets before the temporal cutoff. The original primary-table protocol is retained under `configs/legacy_*.json`; it uses full sequences and different metric aggregation. **Use the configuration matching the experiment being reproduced.** See [reproduction notes](docs/reproduction.md).

Only AMTPP code is included. Transit records, fitted checkpoints, and individual prediction artifacts are excluded; the example generator creates artificial data locally.

## Citation and license

**AMTPP: Learning to Predict Individual Mobility Trips with Attentive Marked Temporal Point Processes** — Yihong Tang, Yuankai Wu, Zhanhong Cheng, Hamzeh Alizadeh, and Lijun Sun. See [CITATION.cff](CITATION.cff). Code is distributed under the [MIT license](LICENSE).
