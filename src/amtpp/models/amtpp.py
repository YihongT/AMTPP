#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMTPP model and related components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.common import _hour_of_day_from_hour, safe_log


class LearnableSinusoidalPosEnc(nn.Module):
    """
    Sinusoidal positional encoding with learnable scaling factor Lpos.
    Implements Eq. (4) style:
      pe(pos, 2i)   = sin( pos / Lpos^(2i/d) )
      pe(pos, 2i+1) = cos( pos / Lpos^(2i/d) )
    where Lpos is learnable > 0.
    """
    def __init__(self, d: int, init_Lpos: float = 10000.0):
        super().__init__()
        self.d = int(d)
        # Learn Lpos in log-space to keep it positive
        self.log_Lpos = nn.Parameter(torch.tensor(math.log(init_Lpos), dtype=torch.float32))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: [B, T] or [B] int/float
        returns: [B, T, d] or [B, d]
        """
        if pos.dim() == 1:
            pos = pos.unsqueeze(1)  # [B,1]
            squeeze_back = True
        else:
            squeeze_back = False

        B, T = pos.shape
        device = pos.device
        d = self.d

        Lpos = torch.exp(self.log_Lpos)  # positive scalar
        i = torch.arange(0, d, 2, device=device, dtype=torch.float32)  # even indices
        # denom: Lpos^(2i/d)
        denom = torch.pow(Lpos, i / d)  # [d/2]
        angle = pos.float().unsqueeze(-1) / denom  # [B,T,d/2]

        pe_even = torch.sin(angle)
        pe_odd = torch.cos(angle)

        pe = torch.zeros((B, T, d), device=device, dtype=torch.float32)
        pe[:, :, 0::2] = pe_even
        pe[:, :, 1::2] = pe_odd

        if squeeze_back:
            return pe[:, 0, :]
        return pe


class TimeEmbedding(nn.Module):
    """
    embt_n = concat(pew_n, peh_n, tau_n)  (Eq. 5)
    """
    def __init__(self, d_dow: int = 64, d_hour: int = 64, disable_periodic_pe: bool = False):
        super().__init__()
        self.pe_dow = LearnableSinusoidalPosEnc(d_dow)
        self.pe_hour = LearnableSinusoidalPosEnc(d_hour)
        self.disable_periodic_pe = bool(disable_periodic_pe)

    @property
    def out_dim(self) -> int:
        return self.pe_dow.d + self.pe_hour.d + 1

    def forward(self, dow: torch.Tensor, hour: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        dow/hour: [B,T] ints
        tau:      [B,T] float (hours)
        returns:  [B,T,d_dow+d_hour+1]
        """
        pew = self.pe_dow(dow)     # [B,T,64]
        peh = self.pe_hour(hour)   # [B,T,64]
        if self.disable_periodic_pe:
            pew = torch.zeros_like(pew)
            peh = torch.zeros_like(peh)
        return torch.cat([pew, peh, tau.unsqueeze(-1)], dim=-1)


class LocationEmbedding(nn.Module):
    """
    embo_n = concat(Wo_em o_hat + b_o_em, p_o_n)  (Eq. 6)
    embd_n = concat(Wd_em d_hat + b_d_em, p_d_n)  (Eq. 6)
    Here we implement as nn.Embedding (learnable) and no POI features by default.
    """
    def __init__(self, n_locs: int, d_loc: int = 64, extra_dim: int = 0):
        super().__init__()
        self.emb = nn.Embedding(n_locs, d_loc)
        self.extra_dim = int(extra_dim)
        if self.extra_dim > 0:
            self.proj_extra = nn.Linear(self.extra_dim, self.extra_dim)
        else:
            self.proj_extra = None

    @property
    def out_dim(self) -> int:
        return self.emb.embedding_dim + self.extra_dim

    def forward(self, loc_idx: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        loc_idx: [B,T]
        extra:   [B,T,extra_dim] or None
        returns: [B,T,out_dim]
        """
        x = self.emb(loc_idx)
        if self.extra_dim > 0:
            if extra is None:
                extra = torch.zeros((x.shape[0], x.shape[1], self.extra_dim), device=x.device, dtype=x.dtype)
            extra = self.proj_extra(extra)
            x = torch.cat([x, extra], dim=-1)
        return x


class ConditionTokenEncoder(nn.Module):
    """
    Converts individual-level categorical features into a token embedding.
    This is the only "non-paper" component; it enables cold-start synthesis.
    """
    def __init__(self, cond_vocab_sizes: List[int], d_each: int = 16, out_dim: int = 193):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d_each) for v in cond_vocab_sizes])
        self.proj = nn.Linear(len(cond_vocab_sizes) * d_each, out_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: [B,F] categorical indices
        returns token: [B, out_dim]
        """
        parts = []
        for j, emb in enumerate(self.embs):
            parts.append(emb(cond[:, j]))
        x = torch.cat(parts, dim=-1)
        return self.proj(x)


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Implements Eq. (7)-(8) as a single-layer causal multi-head self-attention:
      head_l = softmax(QK^T/sqrt(dk) + mask) V
      H      = gelu(concat(heads) W_O)
    """
    def __init__(self, d_in: int, n_heads: int = 4, c_model: int = 100):
        super().__init__()
        assert c_model % n_heads == 0, "c_model must be divisible by n_heads"
        self.d_in = d_in
        self.n_heads = n_heads
        self.c_model = c_model
        self.dk = c_model // n_heads
        self.dv = c_model // n_heads

        # Separate projection matrices per head (as in the paper notation)
        self.Wq = nn.Parameter(torch.randn(n_heads, d_in, self.dk) * (1.0 / math.sqrt(d_in)))
        self.Wk = nn.Parameter(torch.randn(n_heads, d_in, self.dk) * (1.0 / math.sqrt(d_in)))
        self.Wv = nn.Parameter(torch.randn(n_heads, d_in, self.dv) * (1.0 / math.sqrt(d_in)))
        self.Wo = nn.Parameter(torch.randn(n_heads * self.dv, c_model) * (1.0 / math.sqrt(n_heads * self.dv)))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,d_in]
        attn_mask: [T,T] additive mask (0 allowed, -inf disallowed) or None
        returns: [B,T,c_model]
        """
        B, T, _ = x.shape
        device = x.device

        if attn_mask is None:
            # causal mask: disallow attending to future positions j>i
            attn_mask = torch.full((T, T), float("-inf"), device=device)
            attn_mask = torch.triu(attn_mask, diagonal=1)  # upper triangle is -inf, else 0

        heads = []
        # loop over heads (small n_heads, so this is fine)
        for h in range(self.n_heads):
            Q = x @ self.Wq[h]  # [B,T,dk]
            K = x @ self.Wk[h]  # [B,T,dk]
            V = x @ self.Wv[h]  # [B,T,dv]

            # scores: [B,T,T]
            scores = (Q @ K.transpose(1, 2)) / math.sqrt(self.dk)
            scores = scores + attn_mask  # broadcast [T,T] over batch

            A = torch.softmax(scores, dim=-1)
            head = A @ V  # [B,T,dv]
            heads.append(head)

        H = torch.cat(heads, dim=-1)  # [B,T,n_heads*dv]
        out = H @ self.Wo  # [B,T,c_model]
        return F.gelu(out)


class CausalGRUEncoder(nn.Module):
    """Drop-in causal recurrent encoder for controlled attention-vs-GRU ablations."""
    def __init__(self, d_in: int, c_model: int = 100):
        super().__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=c_model, batch_first=True)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


def al_component_logpdf(y: torch.Tensor, beta_hat: torch.Tensor, lambda_hat: torch.Tensor, gamma_hat: torch.Tensor) -> torch.Tensor:
    """
    Log pdf of the Asymmetric Laplace (AL) component in Eq. (10), parameterized by:
      beta_hat:  mode/location in log-space
      lambda_hat>0, gamma_hat>0

    Using the commonly used two-sided exponential form:
      a = lambda_hat / gamma_hat        (left rate)
      b = lambda_hat * gamma_hat        (right rate)
      C = a*b/(a+b)
    """
    # rates
    a = lambda_hat / torch.clamp(gamma_hat, min=1e-6)
    b = lambda_hat * torch.clamp(gamma_hat, min=1e-6)

    # logC = log(a*b/(a+b)) = log(a)+log(b)-log(a+b)
    logC = safe_log(a) + safe_log(b) - safe_log(a + b)

    left = logC + a * (y - beta_hat)
    right = logC - b * (y - beta_hat)

    return torch.where(y < beta_hat, left, right)


def almixture_log_prob_y(
    y: torch.Tensor,  # [B,T]
    w: torch.Tensor,  # [B,T,K] already softmax
    beta_hat: torch.Tensor,  # [B,T,K] > 0 in paper (exp), but can be any real; we keep >0 as in Eq. (11)
    lambda_hat: torch.Tensor,  # [B,T,K] > 0
    gamma_hat: torch.Tensor,   # [B,T,K] > 0
) -> torch.Tensor:
    """
    log p(y) for ALMixture (Eq. 10-11).
    """
    # Expand y to [B,T,1]
    y3 = y.unsqueeze(-1)
    log_comp = al_component_logpdf(y3, beta_hat, lambda_hat, gamma_hat)  # [B,T,K]
    log_w = safe_log(w)
    return torch.logsumexp(log_w + log_comp, dim=-1)  # [B,T]


def almixture_log_prob_tau(
    tau: torch.Tensor,  # [B,T] positive
    w: torch.Tensor,
    beta_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    gamma_hat: torch.Tensor,
) -> torch.Tensor:
    """
    log p(tau) = log p(y) - log(tau), y = log(tau).  (Eq. 15)
    """
    tau = torch.clamp(tau, min=1e-6)
    y = safe_log(tau)
    logpy = almixture_log_prob_y(y, w, beta_hat, lambda_hat, gamma_hat)
    return logpy - safe_log(tau)


def almixture_quantile_tau(
    w: torch.Tensor,
    beta_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    gamma_hat: torch.Tensor,
    quantile: float = 0.5,
    iterations: int = 48,
) -> torch.Tensor:
    """Deterministic finite quantile of the asymmetric-Laplace mixture in tau space."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    a = (lambda_hat / gamma_hat.clamp_min(1e-6)).clamp_min(1e-6)
    b = (lambda_hat * gamma_hat).clamp_min(1e-6)
    lower = (beta_hat - 30.0 / a).amin(dim=-1).clamp(-50.0, 50.0)
    upper = (beta_hat + 30.0 / b).amax(dim=-1).clamp(-50.0, 50.0)
    target = torch.as_tensor(quantile, dtype=w.dtype, device=w.device)
    for _ in range(iterations):
        midpoint = (lower + upper) * 0.5
        y = midpoint.unsqueeze(-1)
        left_mass = b / (a + b)
        component_cdf = torch.where(
            y < beta_hat,
            left_mass * torch.exp(a * (y - beta_hat)),
            1.0 - (a / (a + b)) * torch.exp(-b * (y - beta_hat)),
        )
        mixture_cdf = (w * component_cdf).sum(dim=-1)
        lower = torch.where(mixture_cdf < target, midpoint, lower)
        upper = torch.where(mixture_cdf < target, upper, midpoint)
    return torch.exp((lower + upper) * 0.5)


def lognormal_log_prob_tau(tau: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    tau = torch.clamp(tau, min=1e-6)
    log_tau = safe_log(tau)
    scale = torch.exp(log_scale)
    z = (log_tau - loc) / scale
    log_pdf = -0.5 * z ** 2 - log_scale - 0.5 * math.log(2 * math.pi)
    return log_pdf - log_tau


def lognormal_mean_tau(loc: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(loc + 0.5 * torch.exp(2.0 * log_scale))


def sample_lognormal_tau(loc: torch.Tensor, log_scale: torch.Tensor, n_samples: int) -> Optional[torch.Tensor]:
    if n_samples <= 0:
        return None
    eps = torch.randn((n_samples,) + loc.shape, device=loc.device, dtype=loc.dtype)
    scale = torch.exp(log_scale)
    y = loc.unsqueeze(0) + scale.unsqueeze(0) * eps
    return torch.exp(y)


def exponential_log_prob_tau(tau: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    tau = torch.clamp(tau, min=1e-6)
    rate = torch.clamp(rate, min=1e-6)
    return safe_log(rate) - rate * tau


def exponential_mean_tau(rate: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.clamp(rate, min=1e-6)


def sample_exponential_tau(rate: torch.Tensor, n_samples: int) -> Optional[torch.Tensor]:
    if n_samples <= 0:
        return None
    rate = torch.clamp(rate, min=1e-6)
    dist = torch.distributions.Exponential(rate)
    return dist.sample((n_samples,))


def sample_asymmetric_laplace(
    beta_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    gamma_hat: torch.Tensor,
    n_samples: int = 1,
) -> torch.Tensor:
    """
    Sample y from one AL component (vectorized).
    Inputs can be broadcastable to [*, K] or [*].
    Returns samples with the same broadcasted shape (and optionally an extra sample dim).
    """
    # rates
    a = lambda_hat / torch.clamp(gamma_hat, min=1e-6)
    b = lambda_hat * torch.clamp(gamma_hat, min=1e-6)

    p_left = b / (a + b)   # P(y < beta)
    # sample u in [0,1)
    shape = beta_hat.shape
    if n_samples > 1:
        u = torch.rand((n_samples,) + shape, device=beta_hat.device)
        beta = beta_hat.unsqueeze(0)
        a2 = a.unsqueeze(0)
        b2 = b.unsqueeze(0)
        pL = p_left.unsqueeze(0)
    else:
        u = torch.rand(shape, device=beta_hat.device)
        beta = beta_hat
        a2 = a
        b2 = b
        pL = p_left

    # left side: y = beta + (1/a) * log(u / p_left)
    # right side: y = beta - (1/b) * log((1-u) / (1-p_left))
    u_left = torch.clamp(u / torch.clamp(pL, min=1e-6), min=1e-12)
    y_left = beta + (1.0 / torch.clamp(a2, min=1e-6)) * torch.log(u_left)

    u_right = torch.clamp((1.0 - u) / torch.clamp(1.0 - pL, min=1e-6), min=1e-12)
    y_right = beta - (1.0 / torch.clamp(b2, min=1e-6)) * torch.log(u_right)

    y = torch.where(u < pL, y_left, y_right)
    return y


def sample_almixture_tau(
    w: torch.Tensor, beta_hat: torch.Tensor, lambda_hat: torch.Tensor, gamma_hat: torch.Tensor
) -> torch.Tensor:
    """
    Sample tau from ALMixture (via sampling component then sampling y then exp).
    Inputs are [K] or [B,K].
    Returns tau with shape [] or [B].
    """
    # choose component
    if w.dim() == 1:
        cat = torch.distributions.Categorical(probs=w)
        k = cat.sample()
        y = sample_asymmetric_laplace(beta_hat[k], lambda_hat[k], gamma_hat[k])
        tau = torch.exp(y)
        return tau
    if w.dim() == 2:
        B, K = w.shape
        cat = torch.distributions.Categorical(probs=w)
        k = cat.sample()  # [B]
        # gather params
        idx = k.unsqueeze(-1)
        beta = torch.gather(beta_hat, 1, idx).squeeze(-1)
        lam = torch.gather(lambda_hat, 1, idx).squeeze(-1)
        gam = torch.gather(gamma_hat, 1, idx).squeeze(-1)
        y = sample_asymmetric_laplace(beta, lam, gam)
        tau = torch.exp(y)
        return tau
    raise ValueError("w must be 1D or 2D")


@dataclass
class AMTPPConfig:
    # Dimensions (paper setup: embeddings 64, heads 4, c_model 100)
    d_loc: int = 64
    d_dow: int = 64
    d_hour: int = 64
    n_heads: int = 4
    c_model: int = 100

    # Time mixture
    K: int = 16
    time_dist: str = "almixture"

    # OD matrix low-rank
    r: int = 3
    od_head_type: str = "low_rank"
    destination_objective: str = "marginal"
    learn_topology_bias: bool = False
    # Lightweight network-aware extensions. ``learn_topology_bias`` is kept as
    # a backward-compatible alias for ``topology_mode="learned_global"``.
    topology_mode: str = "none"
    fixed_topology_strength: float = 0.0

    # Enable/disable likelihood terms (useful for TOD-only training)
    w_time: float = 1.0
    w_origin: float = 1.0
    w_dest: float = 1.0
    w_eos: float = 1.0

    # Ablation: predict destination with an independent head (no OD matrix learning)
    independent_dest_head: bool = False

    # Controlled revision ablations.
    disable_periodic_pe: bool = False
    encoder_type: str = "attention"
    spatial_uses_time_params: bool = True
    # R2 pathway audit.  ``legacy`` preserves checkpoint/config compatibility
    # and resolves through ``spatial_uses_time_params``.
    time_to_spatial_mode: str = "legacy"
    time_to_spatial_shuffle_seed: int = 0


class AMTPP(nn.Module):
    def __init__(
        self,
        n_locs: int,
        cond_vocab_sizes: List[int],
        allowed_od_mask: np.ndarray,
        cfg: AMTPPConfig,
        od_logit_bias: Optional[np.ndarray] = None,
        topology_distance: Optional[np.ndarray] = None,
        topology_adjacency: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_locs = int(n_locs)
        self.K = int(cfg.K)
        self.r = int(cfg.r)
        self.time_dist = cfg.time_dist
        
        
        self.phi_start_time = nn.Linear(cfg.c_model, 7)

        # Embeddings
        self.time_emb = TimeEmbedding(
            d_dow=cfg.d_dow,
            d_hour=cfg.d_hour,
            disable_periodic_pe=cfg.disable_periodic_pe,
        )
        self.orig_emb = LocationEmbedding(n_locs=n_locs, d_loc=cfg.d_loc, extra_dim=0)
        self.dest_emb = LocationEmbedding(n_locs=n_locs, d_loc=cfg.d_loc, extra_dim=0)

        self.event_dim = self.time_emb.out_dim + self.orig_emb.out_dim + self.dest_emb.out_dim

        # Condition token -> same event_dim
        self.cond_enc = ConditionTokenEncoder(cond_vocab_sizes, d_each=16, out_dim=self.event_dim)

        # History encoder (self-attention)
        if cfg.encoder_type == "attention":
            self.encoder = CausalMultiHeadSelfAttention(d_in=self.event_dim, n_heads=cfg.n_heads, c_model=cfg.c_model)
        elif cfg.encoder_type == "gru":
            self.encoder = CausalGRUEncoder(d_in=self.event_dim, c_model=cfg.c_model)
        else:
            raise ValueError(f"Unsupported encoder_type: {cfg.encoder_type}")

        # Time head (distribution-specific)
        if self.time_dist == "almixture":
            self.time_param_dim = 4 * self.K
            self.phi_w = nn.Linear(cfg.c_model, self.K)
            self.phi_beta = nn.Linear(cfg.c_model, self.K)
            self.phi_lambda = nn.Linear(cfg.c_model, self.K)
            self.phi_gamma = nn.Linear(cfg.c_model, self.K)
        elif self.time_dist == "lognormal":
            self.time_param_dim = 2
            self.phi_lognorm = nn.Linear(cfg.c_model, 2)
        elif self.time_dist == "exponential":
            self.time_param_dim = 1
            self.phi_exp_rate = nn.Linear(cfg.c_model, 1)
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")

        # Origin head (Eq. 12) under the controlled R2 time->space pathways.
        valid_pathways = {"legacy", "none", "raw", "interpretable", "shuffled", "oracle"}
        if cfg.time_to_spatial_mode not in valid_pathways:
            raise ValueError(
                f"Unsupported time_to_spatial_mode: {cfg.time_to_spatial_mode}"
            )
        if cfg.time_to_spatial_mode == "legacy":
            self.time_to_spatial_mode = (
                "raw" if cfg.spatial_uses_time_params else "none"
            )
        else:
            self.time_to_spatial_mode = cfg.time_to_spatial_mode
        pathway_dim = {
            "none": 0,
            "raw": self.time_param_dim,
            "shuffled": self.time_param_dim,
            # log q25/q50/q75/IQR and predicted clock/week cyclic features
            "interpretable": 8,
            # log true interval and true clock/week cyclic features
            "oracle": 5,
        }[self.time_to_spatial_mode]
        self.origin_in_dim = cfg.c_model + pathway_dim
        self.phi_o = nn.Linear(self.origin_in_dim, n_locs)

        # Destination head
        if self.cfg.independent_dest_head:
            self.phi_d = nn.Linear(self.origin_in_dim, n_locs)
        else:
            if self.cfg.od_head_type == "low_rank":
                # OD matrix low-rank factors (Eq. 13)
                self.phi_m1 = nn.Linear(self.origin_in_dim, n_locs * self.r)
                self.phi_m2 = nn.Linear(self.origin_in_dim, n_locs * self.r)
            elif self.cfg.od_head_type == "dense":
                # Controlled capacity comparison at the actual station scale.
                self.phi_m_dense = nn.Linear(self.origin_in_dim, n_locs * n_locs)
            else:
                raise ValueError(f"Unsupported od_head_type: {self.cfg.od_head_type}")

        self.phi_eos = nn.Linear(self.origin_in_dim, 1)

        # Register OD feasibility mask (dest x origin)
        allowed = torch.tensor(allowed_od_mask.astype(np.bool_), dtype=torch.bool)
        # print(f'allowed: {allowed.shape}')
        self.register_buffer("allowed_od_mask", allowed)
        self.register_buffer("allowed_dest_mask", allowed.any(dim=1))
        self.topology_mode = (
            "learned_global"
            if cfg.learn_topology_bias and cfg.topology_mode == "none"
            else cfg.topology_mode
        )
        valid_topology_modes = {
            "none",
            "fixed",
            "learned_global",
            "hop_bins",
            "conditional",
            "graph",
            "combo",
        }
        if self.topology_mode not in valid_topology_modes:
            raise ValueError(f"Unsupported topology_mode: {self.topology_mode}")
        if cfg.learn_topology_bias and cfg.topology_mode not in {
            "none",
            "learned_global",
        }:
            raise ValueError(
                "learn_topology_bias is a legacy alias and cannot be combined "
                f"with topology_mode={cfg.topology_mode!r}"
            )
        if cfg.fixed_topology_strength < 0:
            raise ValueError("fixed_topology_strength must be nonnegative")
        topology_required = self.topology_mode != "none"
        if topology_required and od_logit_bias is None:
            raise ValueError(f"topology_mode={self.topology_mode!r} requires topology data")

        if od_logit_bias is None:
            bias = torch.zeros((n_locs, n_locs), dtype=torch.float32)
        else:
            bias_array = np.asarray(od_logit_bias, dtype=np.float32)
            if bias_array.shape != (n_locs, n_locs):
                raise ValueError(
                    f"od_logit_bias shape {bias_array.shape} does not match {(n_locs, n_locs)}"
                )
            if not np.isfinite(bias_array).all():
                raise ValueError("od_logit_bias contains non-finite values")
            bias = torch.tensor(bias_array, dtype=torch.float32)
        # Derived from a separately hashed topology artifact, not learned state.
        self.register_buffer("od_logit_bias", bias, persistent=False)
        # A neutral-near initialization (effective strength 0.1) makes a learned
        # extension start close to base AMTPP instead of imposing a large bias.
        raw_strength_init = math.log(math.expm1(0.1))
        if self.topology_mode in {"learned_global", "combo"}:
            self.raw_topology_strength = nn.Parameter(
                torch.tensor(raw_strength_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "raw_topology_strength", torch.tensor(float("-inf")), persistent=False
            )

        if topology_distance is None:
            distance = np.zeros((n_locs, n_locs), dtype=np.float32)
        else:
            distance = np.asarray(topology_distance, dtype=np.float32)
            if distance.shape != (n_locs, n_locs):
                raise ValueError(
                    f"topology_distance shape {distance.shape} does not match "
                    f"{(n_locs, n_locs)}"
                )
            if not np.isfinite(distance).all() or np.any(distance < 0):
                raise ValueError("topology_distance must be finite and nonnegative")
        self.register_buffer(
            "topology_distance", torch.tensor(distance, dtype=torch.float32), persistent=False
        )
        # Nine bins: 0, 1, 2, 3, 4, 5, 6--8, 9--12, and 13+ hops.
        bucket_boundaries = torch.tensor(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.5, 12.5],
            dtype=torch.float32,
        )
        self.register_buffer("topology_bucket_boundaries", bucket_boundaries, persistent=False)
        bucket = torch.bucketize(
            torch.tensor(distance, dtype=torch.float32), bucket_boundaries
        )
        self.register_buffer("topology_bucket", bucket, persistent=False)
        if self.topology_mode == "hop_bins":
            self.hop_bin_bias = nn.Parameter(torch.zeros(9, dtype=torch.float32))
        else:
            self.register_buffer("hop_bin_bias", torch.zeros(9), persistent=False)

        if self.topology_mode == "conditional":
            self.phi_topology_strength = nn.Linear(self.origin_in_dim, 1)
            nn.init.zeros_(self.phi_topology_strength.weight)
            nn.init.constant_(self.phi_topology_strength.bias, raw_strength_init)
        else:
            self.phi_topology_strength = None

        if topology_adjacency is None:
            adjacency = np.zeros((n_locs, n_locs), dtype=np.float32)
        else:
            adjacency = np.asarray(topology_adjacency, dtype=np.float32)
            if adjacency.shape != (n_locs, n_locs):
                raise ValueError(
                    f"topology_adjacency shape {adjacency.shape} does not match "
                    f"{(n_locs, n_locs)}"
                )
            if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
                raise ValueError("topology_adjacency must be finite and nonnegative")
            if not np.allclose(adjacency, adjacency.T):
                raise ValueError("topology_adjacency must be symmetric")
        degree = adjacency.sum(axis=1)
        inv_sqrt_degree = np.zeros_like(degree)
        positive_degree = degree > 0
        inv_sqrt_degree[positive_degree] = 1.0 / np.sqrt(degree[positive_degree])
        normalized_adjacency = (
            inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        )
        self.register_buffer(
            "normalized_topology_adjacency",
            torch.tensor(normalized_adjacency, dtype=torch.float32),
            persistent=False,
        )
        if self.topology_mode in {"graph", "combo"}:
            if self.cfg.independent_dest_head or self.cfg.od_head_type != "low_rank":
                raise ValueError("graph topology modes require the low-rank OD head")
            self.raw_graph_strength = nn.Parameter(
                torch.tensor(raw_strength_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "raw_graph_strength", torch.tensor(float("-inf")), persistent=False
            )

    def _network_location_embedding(
        self, module: LocationEmbedding, location_index: torch.Tensor
    ) -> torch.Tensor:
        """Use one normalized-neighbor propagation step on station embeddings."""
        if self.topology_mode not in {"graph", "combo"}:
            return module(location_index)
        strength = F.softplus(self.raw_graph_strength)
        base_weight = module.emb.weight
        propagated_weight = torch.matmul(
            self.normalized_topology_adjacency, base_weight
        )
        return F.embedding(
            location_index, base_weight + strength * propagated_weight
        )

    def _apply_od_constraints(
        self,
        od_logits: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fixed support and optional topology bias, then normalize by origin."""
        if self.topology_mode == "fixed":
            od_logits = (
                od_logits
                + float(self.cfg.fixed_topology_strength) * self.od_logit_bias
            )
        elif self.topology_mode in {"learned_global", "combo"}:
            strength = F.softplus(self.raw_topology_strength)
            od_logits = od_logits + strength * self.od_logit_bias
        elif self.topology_mode == "hop_bins":
            od_logits = od_logits + self.hop_bin_bias[self.topology_bucket]
        elif self.topology_mode == "conditional":
            if context is None or self.phi_topology_strength is None:
                raise ValueError("conditional topology mode requires spatial context")
            strength = F.softplus(self.phi_topology_strength(context)).unsqueeze(-1)
            od_logits = od_logits + strength * self.od_logit_bias
        allowed = self.allowed_od_mask
        leading = [1] * (od_logits.dim() - 2)
        od_logits = od_logits.masked_fill(
            ~allowed.view(*leading, self.n_locs, self.n_locs), float("-inf")
        )
        od_prob = torch.zeros_like(od_logits)
        valid_cols = allowed.any(dim=0)
        od_prob[..., :, valid_cols] = torch.softmax(od_logits[..., :, valid_cols], dim=-2)
        return od_logits, od_prob

    @torch.no_grad()
    def topology_diagnostics(self) -> Dict[str, object]:
        """Small auditable summary of how a fitted network extension is used."""
        result: Dict[str, object] = {"mode": self.topology_mode}
        if self.topology_mode == "fixed":
            result["global_strength"] = float(self.cfg.fixed_topology_strength)
        elif self.topology_mode in {"learned_global", "combo"}:
            result["global_strength"] = float(
                F.softplus(self.raw_topology_strength).detach().cpu().item()
            )
        if self.topology_mode == "hop_bins":
            result["hop_bin_labels"] = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6--8",
                "9--12",
                "13+",
            ]
            result["hop_bin_bias"] = self.hop_bin_bias.detach().cpu().tolist()
        if self.topology_mode in {"graph", "combo"}:
            result["graph_strength"] = float(
                F.softplus(self.raw_graph_strength).detach().cpu().item()
            )
        return result

    def build_event_embedding(
        self,
        tau: torch.Tensor, hour: torch.Tensor, dow: torch.Tensor,
        origin: torch.Tensor, dest: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build e_n = concat(embt_n, embo_n, embd_n). (Eq. 3, 5, 6)
        Inputs: each [B,T]
        Returns: [B,T,event_dim]
        """
        t_emb = self.time_emb(dow=dow, hour=hour, tau=tau)
        o_emb = self._network_location_embedding(self.orig_emb, origin)
        d_emb = self._network_location_embedding(self.dest_emb, dest)
        return torch.cat([t_emb, o_emb, d_emb], dim=-1)

    def _time_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.time_dist == "almixture":
            w = torch.softmax(self.phi_w(h), dim=-1)
            beta_hat = torch.exp(self.phi_beta(h))
            lambda_hat = torch.exp(self.phi_lambda(h))
            gamma_hat = torch.exp(self.phi_gamma(h))
            time_feat = torch.cat([w, beta_hat, lambda_hat, gamma_hat], dim=-1)
            return time_feat, {
                "w": w,
                "beta_hat": beta_hat,
                "lambda_hat": lambda_hat,
                "gamma_hat": gamma_hat,
            }
        if self.time_dist == "lognormal":
            raw = self.phi_lognorm(h)
            loc = raw[..., 0]
            log_scale = raw[..., 1].clamp(-5.0, 3.0)
            time_feat = torch.stack([loc, log_scale], dim=-1)
            return time_feat, {"ln_loc": loc, "ln_log_scale": log_scale}
        if self.time_dist == "exponential":
            rate = F.softplus(self.phi_exp_rate(h)).squeeze(-1) + 1e-6
            time_feat = rate.unsqueeze(-1)
            return time_feat, {"exp_rate": rate}
        raise ValueError(f"Unsupported time_dist: {self.time_dist}")

    def _time_log_prob(self, tau: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.time_dist == "almixture":
            return almixture_log_prob_tau(tau, outputs["w"], outputs["beta_hat"], outputs["lambda_hat"], outputs["gamma_hat"])
        if self.time_dist == "lognormal":
            return lognormal_log_prob_tau(tau, outputs["ln_loc"], outputs["ln_log_scale"])
        if self.time_dist == "exponential":
            return exponential_log_prob_tau(tau, outputs["exp_rate"])
        raise ValueError(f"Unsupported time_dist: {self.time_dist}")

    def _time_quantile_tau(
        self, outputs: Dict[str, torch.Tensor], quantile: float
    ) -> torch.Tensor:
        """Distribution-specific deterministic time quantile in hours."""
        if self.time_dist == "almixture":
            return almixture_quantile_tau(
                outputs["w"],
                outputs["beta_hat"],
                outputs["lambda_hat"],
                outputs["gamma_hat"],
                quantile=quantile,
                iterations=32,
            )
        if self.time_dist == "lognormal":
            normal_quantile = {
                0.25: -0.6744897501960817,
                0.5: 0.0,
                0.75: 0.6744897501960817,
            }.get(float(quantile))
            if normal_quantile is None:
                raise ValueError("lognormal pathway supports q25/q50/q75")
            scale = torch.exp(outputs["ln_log_scale"])
            return torch.exp(outputs["ln_loc"] + normal_quantile * scale)
        if self.time_dist == "exponential":
            return -math.log1p(-float(quantile)) / outputs["exp_rate"].clamp_min(1e-6)
        raise ValueError(f"Unsupported time_dist: {self.time_dist}")

    @staticmethod
    def _cyclic_clock_features(
        hour_value: torch.Tensor, dow_value: torch.Tensor
    ) -> torch.Tensor:
        """Continuous cyclic encodings for hour of day and day of week."""
        hour_angle = 2.0 * math.pi * hour_value.float() / 24.0
        dow_angle = 2.0 * math.pi * dow_value.float() / 7.0
        return torch.stack(
            [
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(dow_angle),
                torch.cos(dow_angle),
            ],
            dim=-1,
        )

    def _shuffle_valid_time_features(
        self, time_feat: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply a reproducible permutation to valid event-level time features."""
        flat = time_feat.reshape(-1, time_feat.shape[-1])
        valid = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() <= 1:
            return time_feat
        generator = torch.Generator(device="cpu")
        # Include shape so differently padded batches do not reuse a prefix map.
        shape_offset = int(time_feat.shape[0] * 1_000_003 + time_feat.shape[1])
        generator.manual_seed(
            int(self.cfg.time_to_spatial_shuffle_seed) + shape_offset
        )
        permutation = torch.randperm(valid.numel(), generator=generator).to(valid.device)
        shuffled = flat.clone()
        shuffled[valid] = flat[valid[permutation]]
        return shuffled.view_as(time_feat)

    def _spatial_time_features(
        self,
        time_feat: torch.Tensor,
        time_params: Dict[str, torch.Tensor],
        tau: Optional[torch.Tensor],
        hour: Optional[torch.Tensor],
        dow: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        previous_hour: Optional[torch.Tensor] = None,
        previous_dow: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Construct the explicitly frozen R2 time-to-spatial pathway."""
        mode = self.time_to_spatial_mode
        if mode == "none":
            return None
        if mode == "raw":
            return time_feat
        if mode == "shuffled":
            if mask is None:
                # predict_next has one candidate per batch row.
                mask = torch.ones(time_feat.shape[:-1], dtype=torch.bool, device=time_feat.device)
            return self._shuffle_valid_time_features(time_feat, mask)
        if mode == "oracle":
            if tau is None or hour is None or dow is None:
                raise ValueError("oracle pathway requires the true target tau/hour/dow")
            interval = torch.log1p(tau.clamp(min=0.0, max=168.0)).unsqueeze(-1)
            return torch.cat(
                [interval, self._cyclic_clock_features(hour, dow)], dim=-1
            )
        if mode == "interpretable":
            if previous_hour is None or previous_dow is None:
                raise ValueError(
                    "interpretable pathway requires the preceding event clock"
                )
            q25 = self._time_quantile_tau(time_params, 0.25).clamp(0.0, 168.0)
            q50 = self._time_quantile_tau(time_params, 0.5).clamp(0.0, 168.0)
            q75 = self._time_quantile_tau(time_params, 0.75).clamp(0.0, 168.0)
            interval_summary = torch.stack(
                [
                    torch.log1p(q25),
                    torch.log1p(q50),
                    torch.log1p(q75),
                    torch.log1p((q75 - q25).clamp_min(0.0)),
                ],
                dim=-1,
            )
            predicted_clock = previous_hour.float() + q50
            predicted_hour = torch.remainder(predicted_clock, 24.0)
            predicted_dow = torch.remainder(
                previous_dow.float() + torch.floor(predicted_clock / 24.0), 7.0
            )
            return torch.cat(
                [
                    interval_summary,
                    self._cyclic_clock_features(predicted_hour, predicted_dow),
                ],
                dim=-1,
            )
        raise AssertionError(f"Unhandled pathway: {mode}")
    
    def predict_start_time_logits(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: [B, F]
        return logits: [B, 7]
        """
        cond_tok = self.cond_enc(cond).unsqueeze(1)   # [B,1,event_dim]
        h0 = self.encoder(cond_tok)[:, 0, :]          # [B,c_model]
        return self.phi_start_time(h0)

    def predict_start_time_probs(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.predict_start_time_logits(cond), dim=-1)
    
    def forward(
        self,
        cond: torch.Tensor,        # [B,F]
        tau: torch.Tensor,         # [B,T]
        hour: torch.Tensor,        # [B,T]
        dow: torch.Tensor,         # [B,T]
        origin: torch.Tensor,      # [B,T]
        dest: torch.Tensor,        # [B,T]
        mask: torch.Tensor,        # [B,T] valid positions
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forcing likelihood for all trips in a chain.

        We prepend a condition token at position 0.
        For predicting the i-th trip (1-indexed), we use h_{i-1}.
        Concretely, with T trips:
          input tokens length = T+1 (cond + trips)
          history states used for prediction = H[:, :T, :]   (positions 0..T-1)
          targets = trip attributes at 1..T

        Returns dict with params and probabilities aligned to targets [B,T,*].
        """
        B, T = tau.shape

        # Build token embeddings
        cond_tok = self.cond_enc(cond).unsqueeze(1)  # [B,1,event_dim]
        trip_emb = self.build_event_embedding(tau=tau, hour=hour, dow=dow, origin=origin, dest=dest)  # [B,T,event_dim]
        x = torch.cat([cond_tok, trip_emb], dim=1)  # [B,T+1,event_dim]

        # Causal attention
        h_all = self.encoder(x)  # [B,T+1,c_model]

        # Use states 0..T-1 to predict trips 1..T (aligned with targets)
        h = h_all[:, :T, :]  # [B,T,c_model]

        # Time params
        time_feat, time_params = self._time_params(h)

        previous_hour = torch.cat([torch.zeros_like(hour[:, :1]), hour[:, :-1]], dim=1)
        previous_dow = torch.cat([torch.zeros_like(dow[:, :1]), dow[:, :-1]], dim=1)
        spatial_time_feat = self._spatial_time_features(
            time_feat,
            time_params,
            tau,
            hour,
            dow,
            mask,
            previous_hour,
            previous_dow,
        )
        h_hat = (
            h
            if spatial_time_feat is None
            else torch.cat([h, spatial_time_feat], dim=-1)
        )

        # Origin distribution
        o_logits = self.phi_o(h_hat)  # [B,T,S]
        o_logits[..., 0] = float("-inf")   # disallow PAD origin
        o_prob = torch.softmax(o_logits, dim=-1)

        if self.cfg.independent_dest_head:
            # Independent destination scores are shared across origin columns;
            # the fixed non-self support is still applied and each column is
            # normalized, giving q(d|o,H) dependence only through feasibility.
            d_logits = self.phi_d(h_hat)  # [B,T,S]
            OD = d_logits.unsqueeze(-1).expand(-1, -1, -1, self.n_locs)
            _, OD_prob = self._apply_od_constraints(OD, h_hat)
            d_prob = torch.matmul(OD_prob, o_prob.unsqueeze(-1)).squeeze(-1)
        else:
            if self.cfg.od_head_type == "low_rank":
                D1 = self.phi_m1(h_hat).view(B, T, self.n_locs, self.r)
                D2 = self.phi_m2(h_hat).view(B, T, self.n_locs, self.r)
                OD = torch.matmul(D1, D2.transpose(-1, -2))
            else:
                OD = self.phi_m_dense(h_hat).view(B, T, self.n_locs, self.n_locs)
            _, OD_prob = self._apply_od_constraints(OD, h_hat)

            # Destination distribution d_hat = OD * o_hat
            d_prob = torch.matmul(OD_prob, o_prob.unsqueeze(-1)).squeeze(-1)  # [B,T,S]

        eos_logit = self.phi_eos(h_hat).squeeze(-1)  # [B,T]
        eos_prob = torch.sigmoid(eos_logit)

        out = {
            "o_prob": o_prob,
            "d_prob": d_prob,
            "mask": mask,
            "eos_logit": eos_logit,
            "eos_prob": eos_prob,
        }
        if self.topology_mode == "conditional":
            out["topology_strength"] = F.softplus(
                self.phi_topology_strength(h_hat)
            ).squeeze(-1)
        out["od_prob"] = OD_prob
        out.update(time_params)
        return out

    def nll(
        self,
        outputs: Dict[str, torch.Tensor],
        tau_tgt: torch.Tensor,
        origin_tgt: torch.Tensor,
        dest_tgt: torch.Tensor,
        eos_tgt: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NLL components and total, masked by valid positions.
        """
        o_prob = outputs["o_prob"]
        d_prob = outputs["d_prob"]
        eos_logit = outputs["eos_logit"]  # [B,T]

        # time log-prob
        logp_tau = self._time_log_prob(tau_tgt, outputs)  # [B,T]

        # origin log-prob
        logp_o = safe_log(torch.gather(o_prob, dim=-1, index=origin_tgt.unsqueeze(-1)).squeeze(-1))

        # J0 uses the marginal p(d|H); J1 uses q(d|o,H) at the true origin.
        if self.cfg.destination_objective == "marginal":
            destination_target_prob = torch.gather(
                d_prob, dim=-1, index=dest_tgt.unsqueeze(-1)
            ).squeeze(-1)
        elif self.cfg.destination_objective == "conditional":
            if "od_prob" not in outputs:
                raise ValueError("conditional destination objective requires an OD head")
            od_prob = outputs["od_prob"]
            origin_column = torch.gather(
                od_prob,
                dim=-1,
                index=origin_tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_locs, 1),
            ).squeeze(-1)
            destination_target_prob = torch.gather(
                origin_column, dim=-1, index=dest_tgt.unsqueeze(-1)
            ).squeeze(-1)
        else:
            raise ValueError(
                f"Unsupported destination_objective: {self.cfg.destination_objective}"
            )
        logp_d = safe_log(destination_target_prob)

        # apply weights
        if sample_weight is None:
            sw = torch.ones_like(logp_tau)
        else:
            sw = sample_weight

        m = mask.float() * sw

        # avoid NaNs from -inf if OD disallowed: clamp probabilities
        logp_tau = torch.nan_to_num(logp_tau, neginf=-50.0, posinf=50.0)
        logp_o = torch.nan_to_num(logp_o, neginf=-50.0, posinf=50.0)
        logp_d = torch.nan_to_num(logp_d, neginf=-50.0, posinf=50.0)

        # ---- NEW: exclude first trip (index 0) from time loss ----
        # We model first-trip time separately (start_time head), so tau[0] is just an eps placeholder.
        m_time = m.clone()
        if m_time.shape[1] > 0:
            m_time[:, 0] = 0.0

        nll_tau = -(logp_tau * m_time).sum() / (m_time.sum() + 1e-6)
        nll_o = -(logp_o * m).sum() / (m.sum() + 1e-6)
        nll_d = -(logp_d * m).sum() / (m.sum() + 1e-6)

        bce = F.binary_cross_entropy_with_logits(eos_logit, eos_tgt, reduction="none")  # [B,T]
        nll_eos = (bce * m).sum() / (m.sum() + 1e-6)

        total = (
            self.cfg.w_time * nll_tau
            + self.cfg.w_origin * nll_o
            + self.cfg.w_dest * nll_d
            + self.cfg.w_eos * nll_eos
        )
        return {"total": total, "nll_tau": nll_tau, "nll_o": nll_o, "nll_d": nll_d, "nll_eos": nll_eos}

    @torch.no_grad()
    def predict_next(
        self,
        cond: torch.Tensor,        # [B,F]
        tau_ctx: torch.Tensor,     # [B,T]  generated/observed tau
        hour_ctx: torch.Tensor,    # [B,T]
        dow_ctx: torch.Tensor,     # [B,T]
        origin_ctx: torch.Tensor,  # [B,T]
        dest_ctx: torch.Tensor,    # [B,T]
    ):
        """
        Return distribution parameters for the next event (T+1),
        conditioned on [cond_token + context events 1..T].
        """
        B = cond.shape[0]

        # tokens = [cond_token] + [events 1..T]
        cond_tok = self.cond_enc(cond).unsqueeze(1)  # [B,1,event_dim]

        if tau_ctx.shape[1] == 0:
            x = cond_tok  # [B,1,event_dim]
        else:
            trip_emb = self.build_event_embedding(
                tau=tau_ctx, hour=hour_ctx, dow=dow_ctx, origin=origin_ctx, dest=dest_ctx
            )  # [B,T,event_dim]
            x = torch.cat([cond_tok, trip_emb], dim=1)  # [B,T+1,event_dim]

        # causal attention
        h_all = self.encoder(x)  # [B,T+1,c_model]

        # next-step uses the last state (after seeing all available tokens)
        h_next = h_all[:, -1, :]  # [B,c_model]

        # time params for next
        time_feat, time_params = self._time_params(h_next)

        if self.time_to_spatial_mode == "oracle":
            raise ValueError(
                "oracle time-to-spatial pathway cannot predict an unknown next event"
            )
        if tau_ctx.shape[1] == 0:
            previous_hour = torch.zeros(B, dtype=torch.long, device=h_next.device)
            previous_dow = torch.zeros(B, dtype=torch.long, device=h_next.device)
        else:
            previous_hour = hour_ctx[:, -1]
            previous_dow = dow_ctx[:, -1]
        spatial_time_feat = self._spatial_time_features(
            time_feat,
            time_params,
            None,
            None,
            None,
            None,
            previous_hour,
            previous_dow,
        )
        h_hat = (
            h_next
            if spatial_time_feat is None
            else torch.cat([h_next, spatial_time_feat], dim=-1)
        )

        # origin prob
        o_logits = self.phi_o(h_hat)            # [B,S]
        o_logits[:, 0] = float("-inf")   # disallow PAD origin
        o_prob = torch.softmax(o_logits, dim=-1)

        if self.cfg.independent_dest_head:
            d_logits = self.phi_d(h_hat)  # [B,S]
            OD = d_logits.unsqueeze(-1).expand(-1, -1, self.n_locs)
            _, OD_prob = self._apply_od_constraints(OD, h_hat)
            d_prob = torch.matmul(OD_prob, o_prob.unsqueeze(-1)).squeeze(-1)
        else:
            if self.cfg.od_head_type == "low_rank":
                D1 = self.phi_m1(h_hat).view(B, self.n_locs, self.r)
                D2 = self.phi_m2(h_hat).view(B, self.n_locs, self.r)
                OD = torch.matmul(D1, D2.transpose(-1, -2))
            else:
                OD = self.phi_m_dense(h_hat).view(B, self.n_locs, self.n_locs)
            _, OD_prob = self._apply_od_constraints(OD, h_hat)

            d_prob = torch.matmul(OD_prob, o_prob.unsqueeze(-1)).squeeze(-1)  # [B,S]

        eos_logit = self.phi_eos(h_hat).squeeze(-1)  # [B]
        eos_prob = torch.sigmoid(eos_logit)

        out = {
            "o_prob": o_prob,
            "d_prob": d_prob,
            "eos_prob": eos_prob,
        }
        if self.topology_mode == "conditional":
            out["topology_strength"] = F.softplus(
                self.phi_topology_strength(h_hat)
            ).squeeze(-1)
        out["od_prob"] = OD_prob
        out.update(time_params)
        return out

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,  # [1,F]
        dow_value: Optional[int] = None,
        max_trips: int = 25,
        end_hour: float = 28.0,
        sample_origin_dest: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generative simulation with an explicit start-time model.

        Assumptions:
        - You have added `predict_start_time_probs(cond)` (or equivalent) that returns [B,7] probs
            for the first departure time bin (PM23 d_hrede_gr: 1..7 mapped to 0..6).
        - You have changed training data so tau[0] is NOT (t1-4.0), but a small eps, and tau[k]=t[k]-t[k-1] for k>=1.
            (Otherwise the tau head will still try to model "start waiting time".)

        Returns:
        t_hours: absolute event times (hours since day start, in [4,28])
        tod:     floor(t) % 24
        tau:     inter-event times (tau[0]=eps, tau[k]=t[k]-t[k-1])
        origin/dest: sampled or argmax
        eos_prob: model eos probability at each generated step
        start_time_bin: sampled start bin (0..6)
        """
        device = next(self.parameters()).device
        assert cond.shape[0] == 1

        if dow_value is None:
            dow_value = 2  # e.g. Wednesday

        # ---- helper: start bin (0..6) -> time window [a,b) in hours since day start ----
        def _time_window_from_start_bin(bin0_6: int) -> Tuple[float, float]:
            code = int(bin0_6) + 1  # back to 1..7
            if code == 1: return 4.0, 6.0
            if code == 2: return 6.0, 9.0
            if code == 3: return 9.0, 12.0
            if code == 4: return 12.0, 15.5
            if code == 5: return 15.5, 18.5
            if code == 6: return 18.5, 24.0
            if code == 7: return 24.0, 28.0
            return 12.0, 15.5

        eps = 1e-3
        tau_cap = 22.0  # safety cap to avoid rare heavy-tail samples exploding the day

        tau_list, t_list, o_list, d_list, eos_p_list = [], [], [], [], []

        # context buffers
        tau_ctx = torch.zeros((1, 0), dtype=torch.float32, device=device)
        hour_ctx = torch.zeros((1, 0), dtype=torch.long, device=device)
        dow_ctx = torch.zeros((1, 0), dtype=torch.long, device=device)
        o_ctx = torch.zeros((1, 0), dtype=torch.long, device=device)
        d_ctx = torch.zeros((1, 0), dtype=torch.long, device=device)

        # =========================
        # 1) Sample FIRST trip time from start-time head
        # =========================
        # Use your start-time head (probs over 7 bins)
        if hasattr(self, "predict_start_time_probs"):
            p_start = self.predict_start_time_probs(cond)[0]  # [7]
        elif hasattr(self, "predict_start_time"):
            # fallback if you kept old name returning probs
            p_start = self.predict_start_time(cond)[0]
        else:
            raise AttributeError("AMTPP must implement predict_start_time_probs(cond)->[B,7] (or predict_start_time).")

        start_bin = int(torch.distributions.Categorical(probs=p_start).sample().item())  # 0..6
        a, b = _time_window_from_start_bin(start_bin)
        cur_time = float(np.random.uniform(a, b))
        cur_time = min(max(cur_time, 4.0), end_hour)

        hour_first = _hour_of_day_from_hour(cur_time)
        dow_first = int(dow_value)

        # =========================
        # 2) Generate FIRST trip marks (OD/EOS) conditioned on cond only
        #    and push a first event token into the context with tau=eps.
        # =========================
        params0 = self.predict_next(
            cond=cond,
            tau_ctx=tau_ctx,
            hour_ctx=hour_ctx,
            dow_ctx=dow_ctx,
            origin_ctx=o_ctx,
            dest_ctx=d_ctx,
        )

        o_prob0 = params0["o_prob"][0]  # [S]
        d_prob0 = params0["d_prob"][0]  # [S]
        p_end0 = float(params0["eos_prob"][0].item())
        eos_p_list.append(p_end0)

        if sample_origin_dest:
            o1 = int(torch.distributions.Categorical(probs=o_prob0).sample().item())
            d1 = int(torch.distributions.Categorical(probs=d_prob0).sample().item())
        else:
            # IMPORTANT: do NOT feed PAD=0 as a real location; use argmax instead.
            o1 = int(torch.argmax(o_prob0).item())
            d1 = int(torch.argmax(d_prob0).item())

        # record first event
        tau_list.append(eps)
        t_list.append(cur_time)
        o_list.append(o1)
        d_list.append(d1)

        # update context with first event token
        tau_ctx = torch.cat([tau_ctx, torch.tensor([[eps]], dtype=torch.float32, device=device)], dim=1)
        hour_ctx = torch.cat([hour_ctx, torch.tensor([[hour_first]], dtype=torch.long, device=device)], dim=1)
        dow_ctx = torch.cat([dow_ctx, torch.tensor([[dow_first]], dtype=torch.long, device=device)], dim=1)
        o_ctx = torch.cat([o_ctx, torch.tensor([[o1]], dtype=torch.long, device=device)], dim=1)
        d_ctx = torch.cat([d_ctx, torch.tensor([[d1]], dtype=torch.long, device=device)], dim=1)

        # Optional: if EOS says "stop immediately"
        if random.random() < p_end0:
            t_hours = np.array(t_list, dtype=np.float32)
            tod = np.array([int(math.floor(t) % 24) for t in t_hours], dtype=np.int64)
            return {
                "t_hours": t_hours,
                "tod": tod,
                "tau": np.array(tau_list, dtype=np.float32),
                "origin": np.array(o_list, dtype=np.int64),
                "dest": np.array(d_list, dtype=np.int64),
                "eos_prob": np.array(eos_p_list, dtype=np.float32),
                "start_time_bin": np.array([start_bin], dtype=np.int64),
            }

        # =========================
        # 3) Generate subsequent trips (2..)
        # =========================
        for _ in range(max_trips - 1):
            params = self.predict_next(
                cond=cond,
                tau_ctx=tau_ctx,
                hour_ctx=hour_ctx,
                dow_ctx=dow_ctx,
                origin_ctx=o_ctx,
                dest_ctx=d_ctx,
            )

            o_prob = params["o_prob"][0]       # [S]
            d_prob = params["d_prob"][0]       # [S]
            p_end = float(params["eos_prob"][0].item())
            eos_p_list.append(p_end)

            # sample tau_{next} (inter-trip gap), with a safety cap
            if self.time_dist == "almixture":
                w = params["w"][0]                 # [K]
                beta_hat = params["beta_hat"][0]
                lambda_hat = params["lambda_hat"][0]
                gamma_hat = params["gamma_hat"][0]
                tau_next = float(sample_almixture_tau(w, beta_hat, lambda_hat, gamma_hat).item())
            elif self.time_dist == "lognormal":
                loc = params["ln_loc"][0]
                log_scale = params["ln_log_scale"][0]
                scale = torch.exp(log_scale)
                tau_next = float(torch.exp(torch.normal(loc, scale)).item())
            elif self.time_dist == "exponential":
                rate = params["exp_rate"][0]
                tau_next = float(torch.distributions.Exponential(rate).sample().item())
            else:
                raise ValueError(f"Unsupported time_dist: {self.time_dist}")
            tau_next = min(max(tau_next, eps), tau_cap)

            next_time = cur_time + tau_next
            if next_time > end_hour:
                break

            hour_next = _hour_of_day_from_hour(next_time)
            dow_next = int(dow_value)

            if sample_origin_dest:
                o_next = int(torch.distributions.Categorical(probs=o_prob).sample().item())
                d_next = int(torch.distributions.Categorical(probs=d_prob).sample().item())
            else:
                o_next = int(torch.argmax(o_prob).item())
                d_next = int(torch.argmax(d_prob).item())

            # record
            tau_list.append(tau_next)
            t_list.append(next_time)
            o_list.append(o_next)
            d_list.append(d_next)

            # update context
            tau_ctx = torch.cat([tau_ctx, torch.tensor([[tau_next]], dtype=torch.float32, device=device)], dim=1)
            hour_ctx = torch.cat([hour_ctx, torch.tensor([[hour_next]], dtype=torch.long, device=device)], dim=1)
            dow_ctx = torch.cat([dow_ctx, torch.tensor([[dow_next]], dtype=torch.long, device=device)], dim=1)
            o_ctx = torch.cat([o_ctx, torch.tensor([[o_next]], dtype=torch.long, device=device)], dim=1)
            d_ctx = torch.cat([d_ctx, torch.tensor([[d_next]], dtype=torch.long, device=device)], dim=1)

            cur_time = next_time

            # explicit EOS stop (after generating this trip)
            if random.random() < p_end:
                break

        t_hours = np.array(t_list, dtype=np.float32)
        tod = np.array([int(math.floor(t) % 24) for t in t_hours], dtype=np.int64)

        return {
            "t_hours": t_hours,
            "tod": tod,
            "tau": np.array(tau_list, dtype=np.float32),
            "origin": np.array(o_list, dtype=np.int64),
            "dest": np.array(d_list, dtype=np.int64),
            "eos_prob": np.array(eos_p_list, dtype=np.float32),
            "start_time_bin": np.array([start_bin], dtype=np.int64),
        }
