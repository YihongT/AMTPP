#!/usr/bin/env python3
"""Leakage-controlled metro data construction for AMTPP Revision 2.

The legacy data class remains untouched for diagnostic reproduction.  This
module freezes a user-disjoint, temporally strict protocol:

* eligibility is determined from history/future counts;
* the split is fixed independently of model initialization;
* station vocabulary and learned support use train users' history only;
* train targets stop at the history cutoff;
* validation/test targets are future-window events, with earlier trips used
  only as causal context;
* all users share one condition token (no free user-ID parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..utils.common import _hour_of_day_from_hour


PROTOCOL_VERSION = "amtpp-r2-strict-1.0.0"
USER_HASH_NAMESPACE = "amtpp-r2-protocol-audit-v1"


@dataclass
class StrictMetroConfig:
    city: str
    min_trips: int
    history_end: str
    future_start: str
    split_seed: int = 42
    frac_train: float = 0.8
    frac_val: float = 0.1
    support_mode: str = "full_nonself"
    eps: float = 1e-3
    user_col: str = "userID"
    time_col: str = "startTime"
    origin_col: str = "origin"
    dest_col: str = "destination"


def private_user_hash(city: str, value: object) -> str:
    payload = f"{USER_HASH_NAMESPACE}|{city}|{value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _split_users(
    users: List[Any], frac_train: float, frac_val: float, seed: int
) -> Tuple[List[Any], List[Any], List[Any]]:
    indices = list(range(len(users)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * frac_train)
    n_val = int(len(indices) * frac_val)
    return (
        [users[index] for index in indices[:n_train]],
        [users[index] for index in indices[n_train : n_train + n_val]],
        [users[index] for index in indices[n_train + n_val :]],
    )


class StrictMetroCorpus:
    """Constructs one immutable corpus and three split-specific datasets."""

    def __init__(self, trip_df: pd.DataFrame, cfg: StrictMetroConfig):
        self.cfg = cfg
        required = [cfg.user_col, cfg.time_col, cfg.origin_col, cfg.dest_col]
        missing = set(required) - set(trip_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if cfg.support_mode not in {"full_nonself", "train_observed"}:
            raise ValueError(f"Unsupported support_mode: {cfg.support_mode}")

        frame = trip_df.loc[:, required].copy()
        frame[cfg.time_col] = pd.to_datetime(frame[cfg.time_col])
        history_end = pd.Timestamp(cfg.history_end)
        future_start = pd.Timestamp(cfg.future_start)
        if history_end >= future_start:
            raise ValueError("History and future windows overlap")
        history = frame[cfg.time_col] <= history_end
        future = frame[cfg.time_col] >= future_start

        history_counts = frame.loc[history].groupby(cfg.user_col, sort=False).size()
        future_counts = frame.loc[future].groupby(cfg.user_col, sort=False).size()
        eligible_users = sorted(
            user
            for user, count in history_counts.items()
            if int(count) >= cfg.min_trips and int(future_counts.get(user, 0)) >= 1
        )
        split_values = _split_users(
            eligible_users, cfg.frac_train, cfg.frac_val, cfg.split_seed
        )
        self.split_users = dict(zip(("train", "validation", "test"), split_values))
        self.split_user_sets = {
            name: set(values) for name, values in self.split_users.items()
        }

        retained = frame.loc[frame[cfg.user_col].isin(set(eligible_users))].copy()
        retained = retained.sort_values(
            [cfg.user_col, cfg.time_col], kind="mergesort"
        ).reset_index(drop=True)
        retained_history = retained[cfg.time_col] <= history_end
        train_history = retained.loc[
            retained[cfg.user_col].isin(self.split_user_sets["train"])
            & retained_history
        ]

        station_values = sorted(
            set(train_history[cfg.origin_col]) | set(train_history[cfg.dest_col])
        )
        if not station_values:
            raise ValueError("No train-history stations")
        self.station_values = station_values
        self.station_to_index = {
            value: index + 1 for index, value in enumerate(station_values)
        }
        self.index_to_station = {
            index: value for value, index in self.station_to_index.items()
        }
        all_retained_stations = set(retained[cfg.origin_col]) | set(
            retained[cfg.dest_col]
        )
        missing_stations = all_retained_stations - set(station_values)
        if missing_stations:
            raise ValueError(
                "Validation/test station labels absent from train-history vocabulary: "
                f"{sorted(missing_stations)}"
            )

        self.train_history_pairs = set(
            zip(train_history[cfg.origin_col], train_history[cfg.dest_col])
        )
        self.allowed_od_mask = self._build_support(cfg.support_mode)
        self.retained = retained
        self.samples = {
            name: self._build_samples(name) for name in ("train", "validation", "test")
        }
        self._validate()

    @property
    def S(self) -> int:
        return len(self.station_values) + 1

    def _build_support(self, support_mode: str) -> np.ndarray:
        allowed = np.zeros((self.S, self.S), dtype=np.bool_)
        if support_mode == "full_nonself":
            allowed[1:, 1:] = True
            np.fill_diagonal(allowed, False)
        else:
            for origin, destination in self.train_history_pairs:
                oi = self.station_to_index[origin]
                di = self.station_to_index[destination]
                if oi != di:
                    allowed[di, oi] = True
        allowed[0, :] = False
        allowed[:, 0] = False
        if not allowed[:, 1:].any(axis=0).all():
            bad = np.flatnonzero(~allowed[:, 1:].any(axis=0)) + 1
            raise ValueError(f"Support has origin columns with no candidates: {bad.tolist()}")
        return allowed

    def _build_samples(self, split: str) -> List[Dict[str, Any]]:
        cfg = self.cfg
        history_end = pd.Timestamp(cfg.history_end)
        future_start = pd.Timestamp(cfg.future_start)
        users = self.split_user_sets[split]
        split_frame = self.retained.loc[self.retained[cfg.user_col].isin(users)]
        samples: List[Dict[str, Any]] = []
        for user, group in split_frame.groupby(cfg.user_col, sort=False):
            group = group.sort_values(cfg.time_col, kind="mergesort")
            if split == "train":
                group = group.loc[group[cfg.time_col] <= history_end]
            else:
                # There is normally no gap between these cutoffs.  If a caller
                # defines one, gap events are neither context nor targets.
                group = group.loc[
                    (group[cfg.time_col] <= history_end)
                    | (group[cfg.time_col] >= future_start)
                ]
            times = pd.to_datetime(group[cfg.time_col]).reset_index(drop=True)
            length = len(times)
            if length < 2:
                continue
            tau = np.empty(length, dtype=np.float32)
            tau[0] = cfg.eps
            delta = times.diff().dt.total_seconds().to_numpy(dtype=np.float64)[1:] / 3600.0
            tau[1:] = np.maximum(delta, cfg.eps).astype(np.float32)
            decimal_hour = (
                times.dt.hour
                + times.dt.minute / 60.0
                + times.dt.second / 3600.0
            )
            hour = np.asarray(
                [_hour_of_day_from_hour(float(value)) for value in decimal_hour],
                dtype=np.int64,
            )
            dow = times.dt.dayofweek.to_numpy(dtype=np.int64)
            origin = group[cfg.origin_col].map(self.station_to_index).to_numpy(dtype=np.int64)
            destination = group[cfg.dest_col].map(self.station_to_index).to_numpy(dtype=np.int64)
            revisit = np.zeros(length, dtype=np.int64)
            seen_destinations: set[int] = set()
            for index, destination_index in enumerate(destination.tolist()):
                if destination_index in seen_destinations:
                    revisit[index] = 1
                seen_destinations.add(destination_index)
            future_mask = (times >= future_start).to_numpy(dtype=np.bool_)
            target_mask = np.zeros(length, dtype=np.bool_)
            if split == "train":
                target_mask[1:] = True
            else:
                target_mask = future_mask.copy()
            samples.append(
                {
                    "private_user_hash": private_user_hash(cfg.city, user),
                    "cond": np.zeros(1, dtype=np.int64),
                    "tau": tau,
                    "hour": hour,
                    "dow": dow,
                    "origin": origin,
                    "dest": destination,
                    "revisit": revisit,
                    "event_time_ns": times.astype("int64").to_numpy(dtype=np.int64),
                    "future_mask": future_mask,
                    "target_mask": target_mask,
                    "history_trip_count": int((times <= history_end).sum()),
                    "length": length,
                }
            )
        return samples

    def _validate(self) -> None:
        split_hashes = {
            name: {sample["private_user_hash"] for sample in values}
            for name, values in self.samples.items()
        }
        if any(
            split_hashes[left] & split_hashes[right]
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise RuntimeError("User split overlap detected")
        for name, samples in self.samples.items():
            if len(samples) != len(self.split_users[name]):
                raise RuntimeError(
                    f"{name} lost users: {len(samples)} != {len(self.split_users[name])}"
                )
            for sample in samples:
                if not np.all(np.diff(sample["event_time_ns"]) > 0):
                    raise RuntimeError("Event times are not strictly increasing")
                if name == "train" and sample["future_mask"].any():
                    raise RuntimeError("Future event entered the train split")
                if name != "train" and not sample["target_mask"].any():
                    raise RuntimeError(f"{name} user has no future target")
                if np.any(sample["origin"] == sample["dest"]):
                    raise RuntimeError("Self-loop target conflicts with fixed non-self support")

    def dataset(self, split: str) -> "StrictMetroDataset":
        return StrictMetroDataset(self.samples[split])

    def summary(self) -> Dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "city": self.cfg.city,
            "retained_users": sum(len(values) for values in self.samples.values()),
            "station_classes_including_pad": self.S,
            "condition_vocab_size": 1,
            "support_mode": self.cfg.support_mode,
            "allowed_directed_od_pairs": int(self.allowed_od_mask.sum()),
            "splits": {
                name: {
                    "users": len(values),
                    "events": int(sum(sample["length"] for sample in values)),
                    "targets": int(
                        sum(sample["target_mask"].sum() for sample in values)
                    ),
                }
                for name, values in self.samples.items()
            },
        }


class StrictMetroDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def collate_strict_metro(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch

    batch_size = len(batch)
    max_length = max(sample["length"] for sample in batch)
    cond = torch.zeros((batch_size, 1), dtype=torch.long)
    tau = torch.zeros((batch_size, max_length), dtype=torch.float32)
    hour = torch.zeros((batch_size, max_length), dtype=torch.long)
    dow = torch.zeros((batch_size, max_length), dtype=torch.long)
    origin = torch.zeros((batch_size, max_length), dtype=torch.long)
    destination = torch.zeros((batch_size, max_length), dtype=torch.long)
    revisit = torch.zeros((batch_size, max_length), dtype=torch.long)
    event_time_ns = torch.zeros((batch_size, max_length), dtype=torch.long)
    original_sequence_index = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    future_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, sample in enumerate(batch):
        length = sample["length"]
        cond[row] = torch.from_numpy(sample["cond"])
        tau[row, :length] = torch.from_numpy(sample["tau"])
        hour[row, :length] = torch.from_numpy(sample["hour"])
        dow[row, :length] = torch.from_numpy(sample["dow"])
        origin[row, :length] = torch.from_numpy(sample["origin"])
        destination[row, :length] = torch.from_numpy(sample["dest"])
        revisit[row, :length] = torch.from_numpy(sample["revisit"])
        event_time_ns[row, :length] = torch.from_numpy(sample["event_time_ns"])
        original_sequence_index[row, :length] = torch.from_numpy(
            np.asarray(
                sample.get("original_sequence_index", np.arange(length)),
                dtype=np.int64,
            )
        )
        mask[row, :length] = True
        target_mask[row, :length] = torch.from_numpy(sample["target_mask"])
        future_mask[row, :length] = torch.from_numpy(sample["future_mask"])
    return {
        "cond": cond,
        "tau": tau,
        "hour": hour,
        "dow": dow,
        "origin": origin,
        "dest": destination,
        "revisit": revisit,
        "event_time_ns": event_time_ns,
        "original_sequence_index": original_sequence_index,
        "mask": mask,
        "target_mask": target_mask,
        "future_mask": future_mask,
        "lengths": torch.tensor([sample["length"] for sample in batch], dtype=torch.long),
        "private_user_hash": [sample["private_user_hash"] for sample in batch],
        "history_trip_count": torch.tensor(
            [sample["history_trip_count"] for sample in batch], dtype=torch.long
        ),
    }

