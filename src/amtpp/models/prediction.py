#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMTPP prediction model wrapper for user-conditioned metro trips.
"""

from __future__ import annotations

from dataclasses import dataclass

from .amtpp import AMTPP, AMTPPConfig


@dataclass
class AMTPPPredConfig(AMTPPConfig):
    pass


class AMTPPPred(AMTPP):
    pass
