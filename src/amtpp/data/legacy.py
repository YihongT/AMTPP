#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset + collate for Hangzhou metro AMTPP prediction task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..utils.common import _hour_of_day_from_hour


@dataclass
class MetroConfig:
    user_col: str = "userID"
    time_col: str = "startTime"
    origin_col: str = "origin"
    dest_col: str = "destination"
    min_trips: int = 40
    history_end: str = "2019-01-20 23:59:59"
    future_start: str = "2019-01-21 00:00:00"
    eps: float = 1e-3


def _build_vocab(values: pd.Series) -> Dict:
    uniq = sorted(pd.unique(values))
    vocab = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved
    vocab["__PAD__"] = 0
    return vocab


class MetroTripDataset(Dataset):
    """
    One sequence per user, sorted by startTime.
    Returns fields compatible with AMTPP prediction training.
    """
    def __init__(self, trip_df: pd.DataFrame, cfg: MetroConfig):
        super().__init__()
        self.cfg = cfg
        df = trip_df.copy()
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])

        hist_end = pd.Timestamp(cfg.history_end)
        fut_start = pd.Timestamp(cfg.future_start)
        hist_mask = df[cfg.time_col] <= hist_end
        fut_mask = df[cfg.time_col] >= fut_start

        hist_counts = df[hist_mask].groupby(cfg.user_col).size()
        fut_counts = df[fut_mask].groupby(cfg.user_col).size()
        keep_users = set(
            u for u in df[cfg.user_col].unique()
            if hist_counts.get(u, 0) >= cfg.min_trips and fut_counts.get(u, 0) >= 1
        )
        df = df[df[cfg.user_col].isin(keep_users)].copy()

        self.user_ids = sorted(df[cfg.user_col].unique())
        self.user2idx = {u: i + 1 for i, u in enumerate(self.user_ids)}  # 0 reserved

        stations = sorted(set(df[cfg.origin_col].unique()) | set(df[cfg.dest_col].unique()))
        self.origin_vocab = {s: i + 1 for i, s in enumerate(stations)}
        self.origin_vocab["__PAD__"] = 0
        self.dest_vocab = dict(self.origin_vocab)

        self.df = df.sort_values([cfg.user_col, cfg.time_col], kind="mergesort").reset_index(drop=True)
        self.samples = self._build_samples()

    @property
    def S(self) -> int:
        return max(len(self.origin_vocab), len(self.dest_vocab))

    @property
    def num_users(self) -> int:
        return len(self.user2idx) + 1

    def _build_allowed_od_mask(self) -> np.ndarray:
        S = self.S
        allowed = np.zeros((S, S), dtype=np.bool_)
        o_idx = self.df[self.cfg.origin_col].map(self.origin_vocab).fillna(0).astype(int).values
        d_idx = self.df[self.cfg.dest_col].map(self.dest_vocab).fillna(0).astype(int).values
        for oi, di in zip(o_idx, d_idx):
            if oi == 0 or di == 0:
                continue
            allowed[di, oi] = True
        allowed[0, :] = False
        allowed[:, 0] = False
        return allowed

    @property
    def allowed_od_mask(self) -> np.ndarray:
        if not hasattr(self, "_allowed_od_mask"):
            self._allowed_od_mask = self._build_allowed_od_mask()
        return self._allowed_od_mask

    def _build_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        cfg = self.cfg

        for uid, g in self.df.groupby(cfg.user_col, sort=False):
            g = g.sort_values(cfg.time_col)
            times = pd.to_datetime(g[cfg.time_col]).reset_index(drop=True)
            if len(times) == 0:
                continue

            t_hours = times.dt.hour + times.dt.minute / 60.0 + times.dt.second / 3600.0
            inter_times = np.empty(len(times), dtype=np.float32)
            inter_times[0] = cfg.eps
            for i in range(1, len(times)):
                dt = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600.0
                inter_times[i] = max(float(dt), cfg.eps)

            hour = np.array([_hour_of_day_from_hour(float(h)) for h in t_hours], dtype=np.int64)
            dow = np.array([int(ts.dayofweek) for ts in times], dtype=np.int64)
            fut_start = pd.Timestamp(cfg.future_start)
            future_mask = (times >= fut_start).to_numpy(dtype=bool)

            origin = g[cfg.origin_col].map(self.origin_vocab).fillna(0).astype(int).values
            dest = g[cfg.dest_col].map(self.dest_vocab).fillna(0).astype(int).values
            revisit = np.zeros(len(dest), dtype=np.int64)
            seen = set()
            for i, di in enumerate(dest.tolist()):
                if di in seen and di != 0:
                    revisit[i] = 1
                seen.add(di)

            samples.append({
                "uid": uid,
                "uid_idx": int(self.user2idx.get(uid, 0)),
                "cond": np.array([self.user2idx.get(uid, 0)], dtype=np.int64),
                "tau": inter_times,
                "hour": hour,
                "dow": dow,
                "origin": origin,
                "dest": dest,
                "revisit": revisit,
                "future_mask": future_mask,
                "length": int(len(times)),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_metro(batch: List[Dict]) -> Dict:
    import torch

    B = len(batch)
    max_len = max(x["length"] for x in batch)

    cond = torch.tensor(np.stack([x["cond"] for x in batch]), dtype=torch.long)
    tau = torch.zeros((B, max_len), dtype=torch.float32)
    hour = torch.zeros((B, max_len), dtype=torch.long)
    dow = torch.zeros((B, max_len), dtype=torch.long)
    origin = torch.zeros((B, max_len), dtype=torch.long)
    dest = torch.zeros((B, max_len), dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    future_mask = torch.zeros((B, max_len), dtype=torch.bool)
    future_mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, x in enumerate(batch):
        L = x["length"]
        tau[i, :L] = torch.tensor(x["tau"], dtype=torch.float32)
        hour[i, :L] = torch.tensor(x["hour"], dtype=torch.long)
        dow[i, :L] = torch.tensor(x["dow"], dtype=torch.long)
        origin[i, :L] = torch.tensor(x["origin"], dtype=torch.long)
        dest[i, :L] = torch.tensor(x["dest"], dtype=torch.long)
        mask[i, :L] = True
        future_mask[i, :L] = torch.tensor(x["future_mask"], dtype=torch.bool)

    return {
        "cond": cond,
        "tau": tau,
        "hour": hour,
        "dow": dow,
        "origin": origin,
        "dest": dest,
        "mask": mask,
        "future_mask": future_mask,
        "lengths": torch.tensor([x["length"] for x in batch], dtype=torch.long),
    }




def split_by_users(
    n_users: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    import random
    idx = list(range(n_users))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx
