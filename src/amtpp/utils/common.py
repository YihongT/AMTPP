#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for metro prediction workflows.
"""

from __future__ import annotations

import math
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def _hour_of_day_from_hour(hour_since_start: float) -> int:
    # hour-of-day in {0..23}, wrapping around next day
    return int(math.floor(hour_since_start) % 24)


def json_ready(value):
    """Represent undefined statistics as JSON null, preserving finite values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
