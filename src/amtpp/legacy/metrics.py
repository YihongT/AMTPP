"""Original AMTPP metric definitions retained for primary-table reproduction."""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from ..models.amtpp import (
    almixture_log_prob_tau, exponential_log_prob_tau, exponential_mean_tau,
    lognormal_log_prob_tau, lognormal_mean_tau, sample_almixture_tau,
    sample_exponential_tau, sample_lognormal_tau,
)
from ..utils.common import safe_log
@torch.no_grad()
def _init_f1_counts(num_classes: int) -> Dict[str, np.ndarray]:
    return {
        "tp": np.zeros(num_classes, dtype=np.int64),
        "fp": np.zeros(num_classes, dtype=np.int64),
        "fn": np.zeros(num_classes, dtype=np.int64),
    }


def _update_f1_counts(
    counts: Dict[str, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_index: int = 0,
    mask: Optional[np.ndarray] = None,
) -> None:
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    for c in range(len(counts["tp"])):
        if c == ignore_index:
            continue
        tp = np.logical_and(y_true == c, y_pred == c).sum()
        fp = np.logical_and(y_true != c, y_pred == c).sum()
        fn = np.logical_and(y_true == c, y_pred != c).sum()
        counts["tp"][c] += tp
        counts["fp"][c] += fp
        counts["fn"][c] += fn


def _macro_f1(counts: Dict[str, np.ndarray], ignore_index: int = 0) -> float:
    f1s = []
    for c in range(len(counts["tp"])):
        if c == ignore_index:
            continue
        tp = counts["tp"][c]
        fp = counts["fp"][c]
        fn = counts["fn"][c]
        denom = (2 * tp + fp + fn)
        if denom == 0:
            continue
        f1s.append((2 * tp) / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps=eps) + 0.5 * _kl_divergence(q, m, eps=eps)


def _w1_distance(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    diff = np.cumsum(p) - np.cumsum(q)
    return float(np.sum(np.abs(diff)))


def _ks_distance(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    diff = np.cumsum(p) - np.cumsum(q)
    return float(np.max(np.abs(diff)))


def _bin_log_tau(tau: np.ndarray, log_edges: np.ndarray, eps: float) -> np.ndarray:
    if tau.size == 0:
        return np.zeros(len(log_edges) - 1, dtype=np.float64)
    log_tau = np.log(np.clip(tau, eps, None))
    idx = np.searchsorted(log_edges, log_tau, side="right") - 1
    idx = np.clip(idx, 0, len(log_edges) - 2)
    return np.bincount(idx, minlength=len(log_edges) - 1).astype(np.float64)


def _compute_tau_cap(
    samples: List[Dict],
    quantile: Optional[float],
    default_cap: Optional[float],
    eps: float,
) -> Optional[float]:
    if quantile is None:
        return default_cap
    tau_vals = []
    for s in samples:
        tau = np.asarray(s["tau"], dtype=np.float32)
        if tau.size == 0:
            continue
        mask = np.asarray(s.get("future_mask", None), dtype=bool) if "future_mask" in s else None
        if mask is None:
            mask = np.ones_like(tau, dtype=bool)
        if mask.size > 0:
            mask = mask.copy()
            mask[0] = False
        tau = tau[mask]
        if tau.size > 0:
            tau_vals.append(tau)
    if not tau_vals:
        return default_cap
    tau_all = np.concatenate(tau_vals)
    q = float(np.quantile(np.clip(tau_all, eps, None), quantile))
    return q if q > 0 else default_cap


def _accumulate_kl_from_probs(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    prob = prob.detach().cpu()
    y_true = y_true.detach().cpu()
    mask = mask.detach().cpu().numpy().astype(bool)
    for i in range(prob.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        p_true = np.bincount(y_true[i][m].numpy(), minlength=num_classes)
        q_pred = prob[i][m].sum(dim=0).numpy()
        kl = _kl_divergence(p_true[1:], q_pred[1:])
        if not math.isnan(kl):
            total += kl
            count += 1
    return total, count


def _accumulate_js_from_probs(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    prob = prob.detach().cpu()
    y_true = y_true.detach().cpu()
    mask = mask.detach().cpu().numpy().astype(bool)
    for i in range(prob.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        p_true = np.bincount(y_true[i][m].numpy(), minlength=num_classes)
        q_pred = prob[i][m].sum(dim=0).numpy()
        js = _js_divergence(p_true[1:], q_pred[1:])
        if not math.isnan(js):
            total += js
            count += 1
    return total, count


def _masked_nll_from_logp(
    logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.float()
    return -(logp * m).sum() / (m.sum() + 1e-6)


def _masked_nll_from_probs(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logp = safe_log(torch.gather(prob, dim=-1, index=target.unsqueeze(-1)).squeeze(-1))
    return _masked_nll_from_logp(logp, mask)


def _accumulate_kl_argmax(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> Tuple[float, int]:
    y_pred = torch.argmax(prob, dim=-1)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return _accumulate_kl_from_preds(y_pred, y_true, mask_np, num_classes)


def _accumulate_mean_true_prob(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, int]:
    prob = prob.detach()
    y_true = y_true.detach()
    mask = mask.detach()
    idx = y_true.unsqueeze(-1)
    p_true = torch.gather(prob, dim=-1, index=idx).squeeze(-1)
    p_true = p_true * mask.float()
    return float(p_true.sum().item()), int(mask.sum().item())


def _accumulate_kl_from_preds(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
) -> Tuple[float, int]:
    total = 0.0
    count = 0
    for i in range(y_true.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        p_true = np.bincount(y_true[i][m], minlength=num_classes)
        q_pred = np.bincount(y_pred[i][m], minlength=num_classes)
        kl = _kl_divergence(p_true[1:], q_pred[1:])
        if not math.isnan(kl):
            total += kl
            count += 1
    return total, count


def _accumulate_kl_t_from_samples(
    tau_true: torch.Tensor,
    mask: torch.Tensor,
    tau_samples: Optional[torch.Tensor],
    log_edges: Optional[np.ndarray],
    eps: float,
) -> Tuple[float, int]:
    if tau_samples is None or log_edges is None:
        return 0.0, 0
    total = 0.0
    count = 0
    tau_true = tau_true.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    tau_samples = tau_samples.detach().cpu()
    for i in range(tau_true.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        p_counts = _bin_log_tau(tau_true[i][m], log_edges, eps)
        tau_s = tau_samples[:, i, m].reshape(-1).numpy()
        q_counts = _bin_log_tau(tau_s, log_edges, eps)
        kl = _kl_divergence(p_counts, q_counts)
        if not math.isnan(kl):
            total += kl
            count += 1
    return total, count


def _accumulate_js_w1_ks_t_from_samples(
    tau_true: torch.Tensor,
    mask: torch.Tensor,
    tau_samples: Optional[torch.Tensor],
    log_edges: Optional[np.ndarray],
    eps: float,
) -> Tuple[float, int, float, int, float, int]:
    if tau_samples is None or log_edges is None:
        return 0.0, 0, 0.0, 0, 0.0, 0
    js_sum = 0.0
    js_n = 0
    w1_sum = 0.0
    w1_n = 0
    ks_sum = 0.0
    ks_n = 0
    tau_true = tau_true.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    tau_samples = tau_samples.detach().cpu()
    for i in range(tau_true.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        p_counts = _bin_log_tau(tau_true[i][m], log_edges, eps)
        tau_s = tau_samples[:, i, m].reshape(-1).numpy()
        q_counts = _bin_log_tau(tau_s, log_edges, eps)
        js = _js_divergence(p_counts, q_counts)
        w1 = _w1_distance(p_counts, q_counts)
        ks = _ks_distance(p_counts, q_counts)
        if not math.isnan(js):
            js_sum += js
            js_n += 1
        if not math.isnan(w1):
            w1_sum += w1
            w1_n += 1
        if not math.isnan(ks):
            ks_sum += ks
            ks_n += 1
    return js_sum, js_n, w1_sum, w1_n, ks_sum, ks_n


def _masked_error_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    m = mask.float()
    diff = (pred - target) * m
    return {
        "abs_sum": float(diff.abs().sum().item()),
        "sq_sum": float((diff ** 2).sum().item()),
        "count": float(m.sum().item()),
    }


TAU_BIN_EDGES = [0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, float("inf")]


TAU_BIN_LABELS = ["0-1", "1-3", "3-6", "6-12", "12-24", "24-48", "48+"]


TAU_RANGE_SPECS = [
    ("0-24", 0.0, 24.0),
    ("0-12", 0.0, 12.0),
    ("1-12", 1.0, 12.0),
    ("3-12", 3.0, 12.0),
    ("1-24", 1.0, 24.0),
    ("3-24", 3.0, 24.0),
]


def _init_tau_bin_sums() -> Dict[str, List[float]]:
    n = len(TAU_BIN_LABELS)
    return {
        "tau_bin_abs_sum": [0.0] * n,
        "tau_bin_sq_sum": [0.0] * n,
        "tau_bin_count": [0.0] * n,
    }


def _init_tau_range_sums() -> Dict[str, List[float]]:
    n = len(TAU_RANGE_SPECS)
    return {
        "tau_range_abs_sum": [0.0] * n,
        "tau_range_sq_sum": [0.0] * n,
        "tau_range_count": [0.0] * n,
    }


def _accumulate_tau_bins(
    totals: Dict[str, List[float]],
    tau_pred: torch.Tensor,
    tau_true: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if mask is None:
        mask = torch.ones_like(tau_true, dtype=torch.bool)
    m = mask.bool()
    if m.sum().item() == 0:
        return
    tau_t = tau_true[m]
    tau_p = tau_pred[m]
    diff = tau_p - tau_t
    abs_err = diff.abs()
    sq_err = diff * diff
    for i, (lo, hi) in enumerate(zip(TAU_BIN_EDGES[:-1], TAU_BIN_EDGES[1:])):
        in_bin = (tau_t >= lo) & (tau_t < hi)
        if in_bin.any():
            totals["tau_bin_abs_sum"][i] += float(abs_err[in_bin].sum().item())
            totals["tau_bin_sq_sum"][i] += float(sq_err[in_bin].sum().item())
            totals["tau_bin_count"][i] += float(in_bin.sum().item())


def _accumulate_tau_ranges(
    totals: Dict[str, List[float]],
    tau_pred: torch.Tensor,
    tau_true: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if mask is None:
        mask = torch.ones_like(tau_true, dtype=torch.bool)
    m = mask.bool()
    if m.sum().item() == 0:
        return
    tau_t = tau_true[m]
    tau_p = tau_pred[m]
    diff = tau_p - tau_t
    abs_err = diff.abs()
    sq_err = diff * diff
    for i, (_, lo, hi) in enumerate(TAU_RANGE_SPECS):
        in_range = (tau_t >= lo) & (tau_t < hi)
        if in_range.any():
            totals["tau_range_abs_sum"][i] += float(abs_err[in_range].sum().item())
            totals["tau_range_sq_sum"][i] += float(sq_err[in_range].sum().item())
            totals["tau_range_count"][i] += float(in_range.sum().item())


def _finalize_tau_bins(totals: Dict) -> None:
    mae_bins = {}
    rmse_bins = {}
    for i, label in enumerate(TAU_BIN_LABELS):
        count = totals["tau_bin_count"][i]
        if count > 0:
            mae = totals["tau_bin_abs_sum"][i] / count
            rmse = math.sqrt(totals["tau_bin_sq_sum"][i] / count)
        else:
            mae = float("nan")
            rmse = float("nan")
        mae_bins[label] = mae
        rmse_bins[label] = rmse
    totals["mae_tau_bins"] = mae_bins
    totals["rmse_tau_bins"] = rmse_bins


def _finalize_tau_ranges(totals: Dict) -> None:
    mae_ranges = {}
    rmse_ranges = {}
    for i, (label, _, _) in enumerate(TAU_RANGE_SPECS):
        count = totals["tau_range_count"][i]
        if count > 0:
            mae = totals["tau_range_abs_sum"][i] / count
            rmse = math.sqrt(totals["tau_range_sq_sum"][i] / count)
        else:
            mae = float("nan")
            rmse = float("nan")
        mae_ranges[label] = mae
        rmse_ranges[label] = rmse
    totals["mae_tau_ranges"] = mae_ranges
    totals["rmse_tau_ranges"] = rmse_ranges


def _almixture_mean_tau(
    w: torch.Tensor,
    beta_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    gamma_hat: torch.Tensor,
) -> torch.Tensor:
    # E[tau] = E[exp(y)] for AL mixture, y ~ AL(beta, a, b).
    a = lambda_hat / gamma_hat.clamp_min(1e-6)
    b = (lambda_hat * gamma_hat).clamp_min(1.0 + 1e-6)
    c = (a * b) / (a + b)
    mean_comp = torch.exp(beta_hat) * c * (1.0 / (a + 1.0) + 1.0 / (b - 1.0))
    return (w * mean_comp).sum(dim=-1)


def _sample_almixture_tau(
    w: torch.Tensor,
    beta_hat: torch.Tensor,
    lambda_hat: torch.Tensor,
    gamma_hat: torch.Tensor,
    n_samples: int,
) -> Optional[torch.Tensor]:
    if n_samples <= 0:
        return None
    b, t, k = w.shape
    w_flat = w.reshape(-1, k)
    beta_flat = beta_hat.reshape(-1, k)
    lambda_flat = lambda_hat.reshape(-1, k)
    gamma_flat = gamma_hat.reshape(-1, k)
    samples = []
    for _ in range(n_samples):
        samples.append(sample_almixture_tau(w_flat, beta_flat, lambda_flat, gamma_flat))
    tau = torch.stack(samples, dim=0)
    return tau.reshape(n_samples, b, t)


def eval_predictions(
    model,
    loader,
    device: str,
    num_classes: int,
    tau_samples: int,
    gt_dists: Optional[Dict[str, Optional[np.ndarray]]] = None,
    tbin_eps: float = 1e-6,
    tau_cap: Optional[float] = None,
):
    model.eval()
    time_dist = getattr(model, "time_dist", None)
    if time_dist is None and hasattr(model, "cfg"):
        time_dist = getattr(model.cfg, "time_dist", "almixture")
    if time_dist is None:
        time_dist = "almixture"
    totals = {
        "loss": 0.0,
        "nll_tau": 0.0,
        "nll_o": 0.0,
        "nll_d": 0.0,
        "acc_origin": 0.0,
        "acc_dest": 0.0,
        "f1_origin": 0.0,
        "f1_dest": 0.0,
        "tau_abs_sum": 0.0,
        "tau_sq_sum": 0.0,
        "tau_count": 0.0,
        "n": 0,
    }
    totals.update(_init_tau_bin_sums())
    totals.update(_init_tau_range_sums())
    if gt_dists is not None:
        totals["kl_o_sum"] = 0.0
        totals["kl_d_sum"] = 0.0
        totals["kl_t_sum"] = 0.0
        totals["kl_o_n"] = 0
        totals["kl_d_n"] = 0
        totals["kl_t_n"] = 0
        totals["js_o_sum"] = 0.0
        totals["js_d_sum"] = 0.0
        totals["js_t_sum"] = 0.0
        totals["js_o_n"] = 0
        totals["js_d_n"] = 0
        totals["js_t_n"] = 0
        totals["w1_t_sum"] = 0.0
        totals["w1_t_n"] = 0
        totals["ks_t_sum"] = 0.0
        totals["ks_t_n"] = 0
        totals["kl_o_arg_sum"] = 0.0
        totals["kl_d_arg_sum"] = 0.0
        totals["kl_o_arg_n"] = 0
        totals["kl_d_arg_n"] = 0
        totals["mean_p_o_sum"] = 0.0
        totals["mean_p_d_sum"] = 0.0
        totals["mean_p_o_n"] = 0
        totals["mean_p_d_n"] = 0
    f1_o = _init_f1_counts(num_classes)
    f1_d = _init_f1_counts(num_classes)

    for batch in tqdm(loader, desc="Eval AMTPP", unit="batch"):
        cond = batch["cond"].to(device)
        tau = batch["tau"].to(device)
        hour = batch["hour"].to(device)
        dow = batch["dow"].to(device)
        origin = batch["origin"].to(device)
        dest = batch["dest"].to(device)
        mask = batch["mask"].to(device)
        eval_mask = batch["future_mask"].to(device) if "future_mask" in batch else mask
        out = model(cond=cond, tau=tau, hour=hour, dow=dow, origin=origin, dest=dest, mask=mask)
        o_prob = out["o_prob"][:, 1:, :]
        d_prob = out["d_prob"][:, 1:, :]
        tau_tgt = tau[:, 1:]
        mask_tgt = eval_mask[:, 1:]

        if time_dist == "almixture":
            w = out["w"][:, 1:, :]
            beta_hat = out["beta_hat"][:, 1:, :]
            lambda_hat = out["lambda_hat"][:, 1:, :]
            gamma_hat = out["gamma_hat"][:, 1:, :]
            logp_tau = almixture_log_prob_tau(tau_tgt, w, beta_hat, lambda_hat, gamma_hat)
        elif time_dist == "lognormal":
            loc = out["ln_loc"][:, 1:]
            log_scale = out["ln_log_scale"][:, 1:]
            logp_tau = lognormal_log_prob_tau(tau_tgt, loc, log_scale)
        elif time_dist == "exponential":
            rate = out["exp_rate"][:, 1:]
            logp_tau = exponential_log_prob_tau(tau_tgt, rate)
        else:
            raise ValueError(f"Unsupported time_dist: {time_dist}")
        nll_tau = _masked_nll_from_logp(logp_tau, mask_tgt)
        nll_o = _masked_nll_from_probs(o_prob, origin[:, 1:], mask_tgt)
        nll_d = _masked_nll_from_probs(d_prob, dest[:, 1:], mask_tgt)
        losses = {
            "total": nll_tau + nll_o + nll_d,
            "nll_tau": nll_tau,
            "nll_o": nll_o,
            "nll_d": nll_d,
            "o_prob": o_prob,
            "d_prob": d_prob,
        }

        o_pred = torch.argmax(out["o_prob"], dim=-1)
        d_pred = torch.argmax(out["d_prob"], dim=-1)
        o_tgt = origin[:, 1:]
        d_tgt = dest[:, 1:]
        o_pred_tgt = o_pred[:, 1:]
        d_pred_tgt = d_pred[:, 1:]
        m = eval_mask[:, 1:].float()
        acc_o = ((o_pred_tgt == o_tgt).float() * m).sum() / (m.sum() + 1e-6)
        acc_d = ((d_pred_tgt == d_tgt).float() * m).sum() / (m.sum() + 1e-6)

        totals["loss"] += float(losses["total"].item())
        totals["nll_tau"] += float(losses["nll_tau"].item())
        totals["nll_o"] += float(losses["nll_o"].item())
        totals["nll_d"] += float(losses["nll_d"].item())
        totals["acc_origin"] += float(acc_o.item())
        totals["acc_dest"] += float(acc_d.item())
        time_mask = eval_mask.clone()
        if time_mask.shape[1] > 0:
            time_mask[:, 0] = 0
        if time_dist == "almixture":
            tau_samples_tensor = _sample_almixture_tau(
                out["w"],
                out["beta_hat"],
                out["lambda_hat"],
                out["gamma_hat"],
                tau_samples,
            )
            if tau_samples_tensor is None:
                tau_pred = _almixture_mean_tau(
                    out["w"],
                    out["beta_hat"],
                    out["lambda_hat"],
                    out["gamma_hat"],
                )
            else:
                if tau_cap is not None:
                    tau_samples_tensor = tau_samples_tensor.clamp_max(tau_cap)
                tau_pred = tau_samples_tensor.mean(dim=0)
        elif time_dist == "lognormal":
            tau_samples_tensor = sample_lognormal_tau(
                out["ln_loc"],
                out["ln_log_scale"],
                tau_samples,
            )
            if tau_samples_tensor is None:
                tau_pred = lognormal_mean_tau(out["ln_loc"], out["ln_log_scale"])
            else:
                if tau_cap is not None:
                    tau_samples_tensor = tau_samples_tensor.clamp_max(tau_cap)
                tau_pred = tau_samples_tensor.mean(dim=0)
        elif time_dist == "exponential":
            tau_samples_tensor = sample_exponential_tau(out["exp_rate"], tau_samples)
            if tau_samples_tensor is None:
                tau_pred = exponential_mean_tau(out["exp_rate"])
            else:
                if tau_cap is not None:
                    tau_samples_tensor = tau_samples_tensor.clamp_max(tau_cap)
                tau_pred = tau_samples_tensor.mean(dim=0)
        else:
            raise ValueError(f"Unsupported time_dist: {time_dist}")
        err = _masked_error_sums(tau_pred, tau, time_mask)
        totals["tau_abs_sum"] += err["abs_sum"]
        totals["tau_sq_sum"] += err["sq_sum"]
        totals["tau_count"] += err["count"]
        _accumulate_tau_bins(totals, tau_pred, tau, time_mask)
        _accumulate_tau_ranges(totals, tau_pred, tau, time_mask)
        if gt_dists is not None:
            o_prob = out["o_prob"][:, 1:, :]
            d_prob = out["d_prob"][:, 1:, :]
            mask_tgt = eval_mask[:, 1:]
            kl_o_sum, kl_o_n = _accumulate_kl_from_probs(o_prob, o_tgt, mask_tgt, num_classes)
            kl_d_sum, kl_d_n = _accumulate_kl_from_probs(d_prob, d_tgt, mask_tgt, num_classes)
            totals["kl_o_sum"] += kl_o_sum
            totals["kl_o_n"] += kl_o_n
            totals["kl_d_sum"] += kl_d_sum
            totals["kl_d_n"] += kl_d_n
            js_o_sum, js_o_n = _accumulate_js_from_probs(o_prob, o_tgt, mask_tgt, num_classes)
            js_d_sum, js_d_n = _accumulate_js_from_probs(d_prob, d_tgt, mask_tgt, num_classes)
            totals["js_o_sum"] += js_o_sum
            totals["js_o_n"] += js_o_n
            totals["js_d_sum"] += js_d_sum
            totals["js_d_n"] += js_d_n
            kl_o_arg_sum, kl_o_arg_n = _accumulate_kl_argmax(o_prob, o_tgt, mask_tgt, num_classes)
            kl_d_arg_sum, kl_d_arg_n = _accumulate_kl_argmax(d_prob, d_tgt, mask_tgt, num_classes)
            totals["kl_o_arg_sum"] += kl_o_arg_sum
            totals["kl_o_arg_n"] += kl_o_arg_n
            totals["kl_d_arg_sum"] += kl_d_arg_sum
            totals["kl_d_arg_n"] += kl_d_arg_n
            mp_o_sum, mp_o_n = _accumulate_mean_true_prob(o_prob, o_tgt, mask_tgt)
            mp_d_sum, mp_d_n = _accumulate_mean_true_prob(d_prob, d_tgt, mask_tgt)
            totals["mean_p_o_sum"] += mp_o_sum
            totals["mean_p_o_n"] += mp_o_n
            totals["mean_p_d_sum"] += mp_d_sum
            totals["mean_p_d_n"] += mp_d_n
            mask_t = eval_mask.clone()
            if mask_t.shape[1] > 0:
                mask_t[:, 0] = False
            kl_t_sum, kl_t_n = _accumulate_kl_t_from_samples(
                tau,
                mask_t,
                tau_samples_tensor,
                gt_dists.get("t_edges"),
                tbin_eps,
            )
            totals["kl_t_sum"] += kl_t_sum
            totals["kl_t_n"] += kl_t_n
            js_t_sum, js_t_n, w1_t_sum, w1_t_n, ks_t_sum, ks_t_n = _accumulate_js_w1_ks_t_from_samples(
                tau,
                mask_t,
                tau_samples_tensor,
                gt_dists.get("t_edges"),
                tbin_eps,
            )
            totals["js_t_sum"] += js_t_sum
            totals["js_t_n"] += js_t_n
            totals["w1_t_sum"] += w1_t_sum
            totals["w1_t_n"] += w1_t_n
            totals["ks_t_sum"] += ks_t_sum
            totals["ks_t_n"] += ks_t_n
        _update_f1_counts(
            f1_o,
            y_true=o_tgt.cpu().numpy(),
            y_pred=o_pred_tgt.cpu().numpy(),
            ignore_index=0,
            mask=eval_mask[:, 1:].cpu().numpy(),
        )
        _update_f1_counts(
            f1_d,
            y_true=d_tgt.cpu().numpy(),
            y_pred=d_pred_tgt.cpu().numpy(),
            ignore_index=0,
            mask=eval_mask[:, 1:].cpu().numpy(),
        )
        totals["n"] += 1

    for k in ["loss", "nll_tau", "nll_o", "nll_d", "acc_origin", "acc_dest"]:
        totals[k] /= max(totals["n"], 1)
    totals["f1_origin"] = _macro_f1(f1_o, ignore_index=0)
    totals["f1_dest"] = _macro_f1(f1_d, ignore_index=0)
    if totals["tau_count"] > 0:
        totals["mae_tau"] = totals["tau_abs_sum"] / totals["tau_count"]
        totals["rmse_tau"] = math.sqrt(totals["tau_sq_sum"] / totals["tau_count"])
    else:
        totals["mae_tau"] = float("nan")
        totals["rmse_tau"] = float("nan")
    _finalize_tau_bins(totals)
    _finalize_tau_ranges(totals)
    if gt_dists is not None:
        totals["kl_o"] = totals["kl_o_sum"] / totals["kl_o_n"] if totals["kl_o_n"] > 0 else float("nan")
        totals["kl_d"] = totals["kl_d_sum"] / totals["kl_d_n"] if totals["kl_d_n"] > 0 else float("nan")
        totals["kl_t"] = totals["kl_t_sum"] / totals["kl_t_n"] if totals["kl_t_n"] > 0 else float("nan")
        totals["js_o"] = totals["js_o_sum"] / totals["js_o_n"] if totals["js_o_n"] > 0 else float("nan")
        totals["js_d"] = totals["js_d_sum"] / totals["js_d_n"] if totals["js_d_n"] > 0 else float("nan")
        totals["js_t"] = totals["js_t_sum"] / totals["js_t_n"] if totals["js_t_n"] > 0 else float("nan")
        totals["w1_t"] = totals["w1_t_sum"] / totals["w1_t_n"] if totals["w1_t_n"] > 0 else float("nan")
        totals["ks_t"] = totals["ks_t_sum"] / totals["ks_t_n"] if totals["ks_t_n"] > 0 else float("nan")
        totals["kl_o_argmax"] = totals["kl_o_arg_sum"] / totals["kl_o_arg_n"] if totals["kl_o_arg_n"] > 0 else float("nan")
        totals["kl_d_argmax"] = totals["kl_d_arg_sum"] / totals["kl_d_arg_n"] if totals["kl_d_arg_n"] > 0 else float("nan")
        totals["mean_p_o"] = totals["mean_p_o_sum"] / totals["mean_p_o_n"] if totals["mean_p_o_n"] > 0 else float("nan")
        totals["mean_p_d"] = totals["mean_p_d_sum"] / totals["mean_p_d_n"] if totals["mean_p_d_n"] > 0 else float("nan")
    for k in (
        "tau_abs_sum",
        "tau_sq_sum",
        "tau_count",
        "tau_bin_abs_sum",
        "tau_bin_sq_sum",
        "tau_bin_count",
        "tau_range_abs_sum",
        "tau_range_sq_sum",
        "tau_range_count",
    ):
        totals.pop(k, None)
    for k in (
        "kl_o_sum", "kl_o_n", "kl_d_sum", "kl_d_n", "kl_t_sum", "kl_t_n",
        "js_o_sum", "js_o_n", "js_d_sum", "js_d_n", "js_t_sum", "js_t_n",
        "w1_t_sum", "w1_t_n", "ks_t_sum", "ks_t_n",
        "kl_o_arg_sum", "kl_o_arg_n", "kl_d_arg_sum", "kl_d_arg_n",
        "mean_p_o_sum", "mean_p_o_n", "mean_p_d_sum", "mean_p_d_n",
    ):
        totals.pop(k, None)
    return totals

