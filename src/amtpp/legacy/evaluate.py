"""Evaluate the original AMTPP checkpoint using the retained metric definitions."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.legacy import MetroConfig, MetroTripDataset, collate_metro, split_by_users
from ..models.prediction import AMTPPPred, AMTPPPredConfig
from ..utils.common import set_seed, json_ready
from .metrics import eval_predictions
from .train import data_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_arguments(parser)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tau-samples", type=int, default=20)
    parser.add_argument("--tau-cap", type=float, default=24.0)
    args = parser.parse_args()
    set_seed(args.seed)
    data_config = MetroConfig(min_trips=args.min_trips, history_end=args.history_end, future_start=args.future_start)
    dataset = MetroTripDataset(pd.read_pickle(args.data), data_config)
    _, _, test_indices = split_by_users(len(dataset), seed=args.seed)
    if not test_indices:
        parser.error("No test users remain after filtering")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_config = AMTPPPredConfig(**state["model_cfg"])
    if state.get("metro_cfg") != dataclasses.asdict(data_config):
        parser.error("Data configuration differs from the checkpoint")
    if state.get("seed", args.seed) != args.seed:
        parser.error("Split seed differs from the checkpoint")
    model = AMTPPPred(n_locs=dataset.S, cond_vocab_sizes=[dataset.num_users], allowed_od_mask=dataset.allowed_od_mask, cfg=model_config).to(args.device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    missing = set(incompatible.missing_keys) - {"allowed_dest_mask"}
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible checkpoint: {incompatible}")
    loader = DataLoader([dataset[i] for i in test_indices], batch_size=args.batch_size, collate_fn=collate_metro)
    with torch.no_grad():
        metrics = eval_predictions(model, loader, args.device, dataset.S, args.tau_samples, tau_cap=args.tau_cap)
    payload = {"protocol": "legacy-full-sequence", "metrics": metrics, "metro_cfg": dataclasses.asdict(data_config), "model_cfg": dataclasses.asdict(model_config), "seed": args.seed, "tau_samples": args.tau_samples, "tau_cap_hours": args.tau_cap, "checkpoint": str(args.checkpoint)}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False) + "\n")
    primary = {key: metrics[key] for key in ("acc_origin", "acc_dest", "f1_origin", "f1_dest", "mae_tau", "rmse_tau", "nll_tau", "nll_o", "nll_d")}
    print(json.dumps(json_ready(primary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
