"""Train AMTPP with the original full-sequence protocol."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.legacy import MetroConfig, MetroTripDataset, collate_metro, split_by_users
from ..models.prediction import AMTPPPred, AMTPPPredConfig
from ..utils.common import set_seed
from ..utils.provenance import file_sha256
from .epochs import train_one_epoch, eval_one_epoch


def data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--history-end", required=True)
    parser.add_argument("--future-start", required=True)
    parser.add_argument("--min-trips", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--rank", type=int, default=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    set_seed(args.seed)
    data_sha256 = file_sha256(args.data)
    data_config = MetroConfig(min_trips=args.min_trips, history_end=args.history_end, future_start=args.future_start)
    dataset = MetroTripDataset(pd.read_pickle(args.data), data_config)
    train_indices, validation_indices, _ = split_by_users(len(dataset), seed=args.seed)
    if not train_indices or not validation_indices:
        parser.error("No training or validation users remain after filtering")
    train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=args.batch_size, shuffle=True, collate_fn=collate_metro)
    validation_loader = DataLoader([dataset[i] for i in validation_indices], batch_size=args.batch_size, collate_fn=collate_metro)
    model_config = AMTPPPredConfig(K=args.K, r=args.rank, w_eos=0.0)
    model = AMTPPPred(n_locs=dataset.S, cond_vocab_sizes=[dataset.num_users], allowed_od_mask=dataset.allowed_od_mask, cfg=model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, optimizer, args.device)
        validation = eval_one_epoch(model, validation_loader, args.device)
        if not torch.isfinite(torch.tensor(validation["loss"])):
            raise FloatingPointError("Non-finite validation loss")
        print(f"epoch={epoch} train_nll={train['loss']:.6f} validation_nll={validation['loss']:.6f}", flush=True)
        if validation["loss"] < best:
            best = validation["loss"]
            stale = 0
            torch.save({"model": model.state_dict(), "metro_cfg": dataclasses.asdict(data_config), "data_sha256": data_sha256, "model_cfg": dataclasses.asdict(model_config), "evaluation_batch_size": args.batch_size, "epoch": epoch, "best_val": best, "model_type": "amtpp", "seed": args.seed}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break


if __name__ == "__main__":
    main()
