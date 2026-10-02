#!/usr/bin/env python3
"""Train and evaluate AMTPP under the frozen Revision-2 strict protocol."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .metrics import (
    metrics_for,
    stable_hash,
    top_label_ece,
    user_weighted_top1,
)
from .models.prediction import AMTPPPred, AMTPPPredConfig
from .models.amtpp import al_component_logpdf, almixture_quantile_tau
from .data.strict import (
    PROTOCOL_VERSION,
    StrictMetroConfig,
    StrictMetroCorpus,
    collate_strict_metro,
)
from .utils.common import set_seed, json_ready


TRAINER_VERSION = "amtpp-r2-network-extension-trainer-1.2.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


HISTORY_SAMPLING_BINS = (
    ("5--9", 5, 9),
    ("10--19", 10, 19),
    ("20--39", 20, 39),
    ("40--79", 40, 79),
    ("80+", 80, None),
)


def history_sampling_bin(count: int) -> str:
    for label, lower, upper in HISTORY_SAMPLING_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    return "below-5"


def training_subset_indices(
    samples: list[dict[str, Any]],
    count: int,
    seed: int,
    mode: str,
) -> tuple[list[int], dict[str, Any]]:
    """Select an auditable user subset, optionally proportional by history bin."""
    total = len(samples)
    if not 0 < count <= total:
        raise ValueError(f"Training subset count {count} is outside 1..{total}")
    rng = np.random.default_rng(seed)
    labels = [history_sampling_bin(int(sample["history_trip_count"])) for sample in samples]
    population = {
        label: np.flatnonzero(np.asarray(labels) == label)
        for label in [item[0] for item in HISTORY_SAMPLING_BINS] + ["below-5"]
    }
    population = {label: value for label, value in population.items() if len(value)}
    if mode == "simple_random":
        selected = np.sort(rng.choice(total, size=count, replace=False))
        allocation = {
            label: int(np.isin(selected, indices).sum())
            for label, indices in population.items()
        }
    elif mode == "history_stratified_proportional":
        ordered_labels = list(population)
        ideal = {
            label: count * len(population[label]) / total for label in ordered_labels
        }
        allocation = {label: int(np.floor(ideal[label])) for label in ordered_labels}
        remaining = count - sum(allocation.values())
        remainder_order = sorted(
            ordered_labels,
            key=lambda label: (-(ideal[label] - allocation[label]), label),
        )
        for label in remainder_order[:remaining]:
            allocation[label] += 1
        if sum(allocation.values()) != count:
            raise RuntimeError("Proportional history-bin allocation did not reach target")
        chosen = [
            rng.choice(population[label], size=allocation[label], replace=False)
            for label in ordered_labels
            if allocation[label]
        ]
        selected = np.sort(np.concatenate(chosen).astype(np.int64))
    else:
        raise ValueError(f"Unknown training subset mode: {mode}")
    if len(selected) != count or len(np.unique(selected)) != count:
        raise RuntimeError("Training user subset is not exact and unique")
    return selected.tolist(), {
        "mode": mode,
        "requested_users": count,
        "population_users": total,
        "population_by_history_bin": {
            label: int(len(indices)) for label, indices in population.items()
        },
        "selected_by_history_bin": allocation,
        "selected_index_sha256": hashlib.sha256(
            selected.astype(np.int64).tobytes()
        ).hexdigest(),
        "selection_seed": seed,
    }


def topology_arrays(
    path: Optional[Path], corpus: StrictMetroCorpus
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict[str, Any]]]:
    if path is None:
        return None, None, None
    source = path.resolve()
    loaded = np.load(source)
    required = {"raw_station_id", "shortest_hops"}
    if not required.issubset(loaded.files):
        raise ValueError(f"Topology artifact missing {sorted(required - set(loaded.files))}")
    raw_values = loaded["raw_station_id"].tolist()
    source_index = {str(value): index + 1 for index, value in enumerate(raw_values)}
    source_hops = loaded["shortest_hops"].astype(np.float32)
    hops = np.full((corpus.S, corpus.S), np.inf, dtype=np.float32)
    hops[0, 0] = 0.0
    for raw_left, model_left in corpus.station_to_index.items():
        if str(raw_left) not in source_index:
            raise ValueError(f"Station {raw_left!r} absent from topology artifact")
        for raw_right, model_right in corpus.station_to_index.items():
            hops[model_left, model_right] = source_hops[
                source_index[str(raw_left)], source_index[str(raw_right)]
            ]
    finite = hops[1:, 1:][np.isfinite(hops[1:, 1:])]
    if finite.size != (corpus.S - 1) ** 2:
        raise ValueError("Aligned topology contains unreachable observed station pairs")
    diameter = float(finite.max())
    bias = np.zeros_like(hops)
    bias[1:, 1:] = -hops[1:, 1:] / max(diameter, 1.0)
    metadata = {
        "path": str(source),
        "sha256": sha256(source),
        "bias_definition": "negative undirected shortest-hop distance divided by graph diameter",
        "diameter_hops": diameter,
        "all_model_stations_aligned": True,
    }
    return bias, hops, metadata


def move_tensors(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: AMTPPPred,
    loader: DataLoader,
    device: str,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    max_batches: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = ("total", "nll_tau", "nll_o", "nll_d", "nll_eos")
    sums = {name: 0.0 for name in names}
    event_count = 0
    context_event_count = 0
    processed_batches = 0
    for batch_number, host_batch in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break
        processed_batches += 1
        batch = move_tensors(host_batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(
                cond=batch["cond"],
                tau=batch["tau"],
                hour=batch["hour"],
                dow=batch["dow"],
                origin=batch["origin"],
                dest=batch["dest"],
                mask=batch["mask"],
            )
            losses = model.nll(
                outputs,
                batch["tau"],
                batch["origin"],
                batch["dest"],
                torch.zeros_like(batch["tau"]),
                batch["target_mask"],
            )
            count = int(batch["target_mask"].sum().item())
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss in batch {batch_number}: {losses}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        for name in names:
            sums[name] += float(losses[name].detach().item()) * count
        event_count += count
        context_event_count += int(batch["mask"].sum().item())
    if event_count == 0:
        raise RuntimeError("Epoch produced no target events")
    result = {name: sums[name] / event_count for name in names}
    result.update(
        {
            "target_events": event_count,
            "context_events": context_event_count,
            "batches": processed_batches,
        }
    )
    return result


def _concat(collected: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    middle = 0.5 * (left + right)
    left_term = np.zeros_like(left)
    right_term = np.zeros_like(right)
    left_positive = left > 0
    right_positive = right > 0
    left_term[left_positive] = left[left_positive] * np.log(
        left[left_positive] / np.clip(middle[left_positive], 1e-12, None)
    )
    right_term[right_positive] = right[right_positive] * np.log(
        right[right_positive] / np.clip(middle[right_positive], 1e-12, None)
    )
    return float(0.5 * (left_term.sum() + right_term.sum()))


def aggregate_od_metrics(
    predicted: np.ndarray, observed: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    predicted_valid = predicted[valid].astype(np.float64)
    observed_valid = observed[valid].astype(np.float64)
    cosine = float(
        np.dot(predicted_valid, observed_valid)
        / max(
            np.linalg.norm(predicted_valid) * np.linalg.norm(observed_valid),
            1e-12,
        )
    )
    rmse = float(np.sqrt(np.mean(np.square(predicted_valid - observed_valid))))
    top_count = min(20, predicted_valid.size)
    predicted_top = set(np.argpartition(predicted_valid, -top_count)[-top_count:].tolist())
    observed_top = set(np.argpartition(observed_valid, -top_count)[-top_count:].tolist())
    origin_js: list[float] = []
    origin_weights: list[float] = []
    for origin in range(1, observed.shape[1]):
        column_valid = valid[:, origin]
        observed_column = observed[column_valid, origin]
        if observed_column.sum() <= 0:
            continue
        predicted_column = predicted[column_valid, origin]
        origin_js.append(_jensen_shannon(predicted_column, observed_column))
        origin_weights.append(float(observed_column.sum()))
    return {
        "pearson": _safe_pearson(predicted_valid, observed_valid),
        "cosine": cosine,
        "jensen_shannon": _jensen_shannon(predicted_valid, observed_valid),
        "rmse_event_count": rmse,
        "nrmse_by_mean_cell_count": float(
            rmse / max(float(observed_valid.mean()), 1e-12)
        ),
        "top20_flow_overlap": float(len(predicted_top & observed_top) / top_count),
        "origin_weighted_destination_js": float(
            np.average(origin_js, weights=origin_weights)
        ),
        "destination_inflow_pearson": _safe_pearson(
            predicted.sum(axis=1)[1:], observed.sum(axis=1)[1:]
        ),
        "origin_outflow_pearson": _safe_pearson(
            predicted.sum(axis=0)[1:], observed.sum(axis=0)[1:]
        ),
    }


def hop_distribution_metrics(
    predicted_od: np.ndarray,
    observed_od: np.ndarray,
    topology_hops: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    diameter = int(np.max(topology_hops[1:, 1:]))
    predicted_hist = np.zeros(diameter + 1, dtype=np.float64)
    observed_hist = np.zeros(diameter + 1, dtype=np.float64)
    integer_hops = np.zeros_like(topology_hops, dtype=np.int64)
    integer_hops[1:, 1:] = topology_hops[1:, 1:].astype(np.int64)
    for hop in range(diameter + 1):
        mask = valid & (integer_hops == hop)
        predicted_hist[hop] = predicted_od[mask].sum()
        observed_hist[hop] = observed_od[mask].sum()
    predicted_probability = predicted_hist / max(predicted_hist.sum(), 1e-12)
    observed_probability = observed_hist / max(observed_hist.sum(), 1e-12)
    hop_values = np.arange(diameter + 1, dtype=np.float64)

    def group_probability(probability: np.ndarray, lower: int, upper: Optional[int]) -> float:
        if upper is None:
            return float(probability[lower:].sum())
        return float(probability[lower : upper + 1].sum())

    return {
        "diameter_hops": diameter,
        "predicted_mean_hops": float(np.dot(hop_values, predicted_probability)),
        "observed_mean_hops": float(np.dot(hop_values, observed_probability)),
        "mean_hop_delta": float(
            np.dot(hop_values, predicted_probability - observed_probability)
        ),
        "jensen_shannon": _jensen_shannon(predicted_hist, observed_hist),
        "groups": {
            "short_1_3": {
                "predicted": group_probability(predicted_probability, 1, 3),
                "observed": group_probability(observed_probability, 1, 3),
            },
            "medium_4_7": {
                "predicted": group_probability(predicted_probability, 4, 7),
                "observed": group_probability(observed_probability, 4, 7),
            },
            "long_8_plus": {
                "predicted": group_probability(predicted_probability, 8, None),
                "observed": group_probability(observed_probability, 8, None),
            },
        },
        "predicted_probability_by_hop": predicted_probability.tolist(),
        "observed_probability_by_hop": observed_probability.tolist(),
    }


@torch.no_grad()
def evaluate_events(
    model: AMTPPPred,
    loader: DataLoader,
    corpus: StrictMetroCorpus,
    split: str,
    device: str,
    topology_hops: Optional[np.ndarray],
    artifact_path: Optional[Path],
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "user_hash": [],
        "event_id": [],
        "sequence_index": [],
        "history_trip_count": [],
        "event_time_ns": [],
        "previous_event_time_ns": [],
        "true_tau_hours": [],
        "true_hour_bin": [],
        "true_day_of_week": [],
        "previous_origin": [],
        "previous_destination": [],
        "true_origin": [],
        "true_destination": [],
        "origin_probability": [],
        "destination_marginal_probability": [],
        "destination_conditional_probability": [],
        "log_probability_tau": [],
        "predicted_tau_median_hours": [],
        "true_joint_probability": [],
        "joint_rank": [],
        "predicted_joint_origin": [],
        "predicted_joint_destination": [],
        "train_history_od_seen": [],
        "train_history_od_count": [],
    }
    if model.topology_mode == "conditional":
        collected["topology_strength"] = []
    if artifact_path is not None and model.cfg.time_dist == "almixture":
        collected["time_mixture_responsibility"] = []
    train_od_count = np.zeros((corpus.S, corpus.S), dtype=np.int64)
    for sample in corpus.samples["train"]:
        np.add.at(
            train_od_count,
            (sample["dest"].astype(np.int64), sample["origin"].astype(np.int64)),
            1,
        )
    train_support = train_od_count > 0
    predicted_joint_od = np.zeros((corpus.S, corpus.S), dtype=np.float64)
    predicted_true_origin_od = np.zeros((corpus.S, corpus.S), dtype=np.float64)
    observed_od = np.zeros((corpus.S, corpus.S), dtype=np.float64)

    forward_seconds = 0.0
    forward_batches = 0
    for batch_number, host_batch in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break
        batch = move_tensors(host_batch, device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        forward_start = time.perf_counter()
        outputs = model(
            cond=batch["cond"],
            tau=batch["tau"],
            hour=batch["hour"],
            dow=batch["dow"],
            origin=batch["origin"],
            dest=batch["dest"],
            mask=batch["mask"],
        )
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        forward_seconds += time.perf_counter() - forward_start
        forward_batches += 1
        if "od_prob" not in outputs:
            raise ValueError("Strict AMTPP evaluation requires an OD head")
        logp_tau = model._time_log_prob(batch["tau"], outputs)
        component_responsibility = None
        if model.cfg.time_dist == "almixture":
            predicted_tau_median = almixture_quantile_tau(
                outputs["w"], outputs["beta_hat"], outputs["lambda_hat"], outputs["gamma_hat"]
            )
            if "time_mixture_responsibility" in collected:
                log_component = torch.log(outputs["w"].clamp_min(1e-12))
                log_component = log_component + al_component_logpdf(
                    torch.log(batch["tau"].clamp_min(1e-6)).unsqueeze(-1),
                    outputs["beta_hat"],
                    outputs["lambda_hat"],
                    outputs["gamma_hat"],
                )
                component_responsibility = torch.softmax(log_component, dim=-1)
        elif model.cfg.time_dist == "lognormal":
            predicted_tau_median = torch.exp(outputs["ln_loc"])
        elif model.cfg.time_dist == "exponential":
            predicted_tau_median = np.log(2.0) / outputs["exp_rate"]
        else:
            raise ValueError(f"Unsupported time distribution: {model.cfg.time_dist}")
        od_prob = outputs["od_prob"]
        conditional = torch.gather(
            od_prob,
            dim=-1,
            index=batch["origin"].unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, corpus.S, 1
            ),
        ).squeeze(-1)
        joint = od_prob * outputs["o_prob"].unsqueeze(-2)
        true_o_prob = torch.gather(
            outputs["o_prob"], -1, batch["origin"].unsqueeze(-1)
        ).squeeze(-1)
        true_d_cond = torch.gather(
            conditional, -1, batch["dest"].unsqueeze(-1)
        ).squeeze(-1)
        true_joint = true_o_prob * true_d_cond
        flat_joint = joint.flatten(-2)
        joint_rank = 1 + (flat_joint > true_joint.unsqueeze(-1)).sum(dim=-1)
        predicted_flat = flat_joint.argmax(dim=-1)
        predicted_joint_origin = predicted_flat % corpus.S
        predicted_joint_destination = predicted_flat // corpus.S

        target_mask = host_batch["target_mask"].numpy()
        for row, user_hash in enumerate(host_batch["private_user_hash"]):
            positions = np.flatnonzero(target_mask[row])
            if positions.size == 0:
                continue
            event_ids = np.asarray(
                [
                    stable_hash(
                        corpus.cfg.city,
                        split,
                        user_hash,
                        int(host_batch["original_sequence_index"][row, position]),
                        int(host_batch["event_time_ns"][row, position]),
                        length=40,
                    )
                    for position in positions
                ],
                dtype="U40",
            )
            origin_np = host_batch["origin"][row, positions].numpy()
            destination_np = host_batch["dest"][row, positions].numpy()
            origin_cpu = origin_np.astype(np.int64)
            destination_cpu = destination_np.astype(np.int64)
            joint_cpu = joint[row, positions].cpu().numpy().astype(np.float64)
            conditional_cpu = conditional[row, positions].cpu().numpy().astype(np.float64)
            predicted_joint_od += joint_cpu.sum(axis=0)
            for event_index, (origin_value, destination_value) in enumerate(
                zip(origin_cpu, destination_cpu)
            ):
                predicted_true_origin_od[:, origin_value] += conditional_cpu[event_index]
                observed_od[destination_value, origin_value] += 1.0
            collected["user_hash"].append(
                np.full(len(positions), user_hash, dtype="U32")
            )
            collected["event_id"].append(event_ids)
            collected["sequence_index"].append(
                host_batch["original_sequence_index"][row, positions]
                .numpy()
                .astype(np.int32)
            )
            collected["history_trip_count"].append(
                np.full(
                    len(positions),
                    int(host_batch["history_trip_count"][row]),
                    dtype=np.int32,
                )
            )
            for name, source, dtype in (
                ("event_time_ns", host_batch["event_time_ns"], np.int64),
                ("true_tau_hours", host_batch["tau"], np.float32),
                ("true_hour_bin", host_batch["hour"], np.int8),
                ("true_day_of_week", host_batch["dow"], np.int8),
                ("true_origin", host_batch["origin"], np.int16),
                ("true_destination", host_batch["dest"], np.int16),
            ):
                collected[name].append(source[row, positions].numpy().astype(dtype))
            previous = np.maximum(positions - 1, 0)
            collected["previous_origin"].append(
                host_batch["origin"][row, previous].numpy().astype(np.int16)
            )
            collected["previous_destination"].append(
                host_batch["dest"][row, previous].numpy().astype(np.int16)
            )
            collected["previous_event_time_ns"].append(
                host_batch["event_time_ns"][row, previous].numpy().astype(np.int64)
            )
            for name, source in (
                ("origin_probability", outputs["o_prob"]),
                ("destination_marginal_probability", outputs["d_prob"]),
                ("destination_conditional_probability", conditional),
            ):
                collected[name].append(
                    source[row, positions].cpu().numpy().astype(np.float32)
                )
            for name, source, dtype in (
                ("log_probability_tau", logp_tau, np.float32),
                ("predicted_tau_median_hours", predicted_tau_median, np.float32),
                ("true_joint_probability", true_joint, np.float32),
                ("joint_rank", joint_rank, np.int32),
                ("predicted_joint_origin", predicted_joint_origin, np.int16),
                ("predicted_joint_destination", predicted_joint_destination, np.int16),
            ):
                collected[name].append(
                    source[row, positions].cpu().numpy().astype(dtype)
                )
            collected["train_history_od_seen"].append(
                train_support[destination_np, origin_np]
            )
            collected["train_history_od_count"].append(
                train_od_count[destination_np, origin_np].astype(np.int32)
            )
            if "topology_strength" in collected:
                collected["topology_strength"].append(
                    outputs["topology_strength"][row, positions]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            if component_responsibility is not None:
                collected["time_mixture_responsibility"].append(
                    component_responsibility[row, positions]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

    artifact = _concat(collected)
    count = len(artifact["event_id"])
    if count == 0 or len(np.unique(artifact["event_id"])) != count:
        raise RuntimeError("Event artifact is empty or has duplicate identifiers")
    for name in (
        "origin_probability",
        "destination_marginal_probability",
        "destination_conditional_probability",
    ):
        prob = artifact[name]
        if not np.isfinite(prob).all() or not np.allclose(
            prob.sum(axis=1), 1.0, atol=2e-5, rtol=0.0
        ):
            raise RuntimeError(f"Invalid probability array: {name}")

    origin_metrics = metrics_for(
        artifact["origin_probability"], artifact["true_origin"], corpus.S
    )
    marginal_metrics = metrics_for(
        artifact["destination_marginal_probability"],
        artifact["true_destination"],
        corpus.S,
    )
    conditional_metrics = metrics_for(
        artifact["destination_conditional_probability"],
        artifact["true_destination"],
        corpus.S,
    )
    metrics: dict[str, Any] = {
        "events": count,
        "origin": origin_metrics,
        "destination_marginal": marginal_metrics,
        "destination_conditional_given_true_origin": conditional_metrics,
        "joint_od": {
            "top1_exact_pair": float(
                np.logical_and(
                    artifact["predicted_joint_origin"] == artifact["true_origin"],
                    artifact["predicted_joint_destination"]
                    == artifact["true_destination"],
                ).mean()
            ),
            "nll": float(
                -np.log(
                    np.clip(
                        artifact["true_joint_probability"].astype(np.float64),
                        1e-12,
                        1.0,
                    )
                ).mean()
            ),
            "mrr": float((1.0 / artifact["joint_rank"].astype(np.float64)).mean()),
        },
        "time": {
            "nll": float(-artifact["log_probability_tau"].astype(np.float64).mean()),
            "point_estimator": "conditional median in hours (deterministic; appropriate for absolute error)",
            "mae_median_hours": float(
                np.abs(
                    artifact["predicted_tau_median_hours"].astype(np.float64)
                    - artifact["true_tau_hours"].astype(np.float64)
                ).mean()
            ),
            "rmse_median_hours": float(
                np.sqrt(
                    np.square(
                        artifact["predicted_tau_median_hours"].astype(np.float64)
                        - artifact["true_tau_hours"].astype(np.float64)
                    ).mean()
                )
            ),
        },
        "secondary_equal_user_weight": {
            "origin_top1": user_weighted_top1(
                artifact["user_hash"],
                artifact["origin_probability"],
                artifact["true_origin"],
            ),
            "destination_marginal_top1": user_weighted_top1(
                artifact["user_hash"],
                artifact["destination_marginal_probability"],
                artifact["true_destination"],
            ),
        },
        "train_history_od_support": {
            "covered_events": int(artifact["train_history_od_seen"].sum()),
            "coverage": float(artifact["train_history_od_seen"].mean()),
        },
        "runtime": {
            "forward_seconds": forward_seconds,
            "forward_batches": forward_batches,
            "target_events_per_forward_second": count / max(forward_seconds, 1e-12),
            "milliseconds_per_target_event": 1000.0 * forward_seconds / count,
            "scope": "model forward only; CUDA synchronized per batch; excludes data loading and metric computation",
        },
    }
    valid_od = corpus.allowed_od_mask.astype(bool)
    metrics["aggregate_od_reproduction"] = {
        "deployable_joint_distribution": aggregate_od_metrics(
            predicted_joint_od, observed_od, valid_od
        ),
        "true_origin_conditional_diagnostic": aggregate_od_metrics(
            predicted_true_origin_od, observed_od, valid_od
        ),
    }
    if "time_mixture_responsibility" in artifact:
        responsibility = artifact["time_mixture_responsibility"].astype(np.float64)
        hard = responsibility.argmax(axis=1)
        mean = responsibility.mean(axis=0)
        metrics["time_mixture_components"] = {
            "components": int(responsibility.shape[1]),
            "mean_posterior_responsibility": mean.tolist(),
            "hard_assignment_fraction": (
                np.bincount(hard, minlength=responsibility.shape[1]) / len(hard)
            ).tolist(),
            "effective_components_from_mean_responsibility": float(
                np.exp(-np.sum(mean * np.log(np.clip(mean, 1e-12, 1.0))))
            ),
        }
    if topology_hops is not None:
        true_destination = artifact["true_destination"]
        pred_conditional = artifact["destination_conditional_probability"].argmax(axis=1)
        pred_marginal = artifact["destination_marginal_probability"].argmax(axis=1)
        conditional_error = topology_hops[pred_conditional, true_destination]
        marginal_error = topology_hops[pred_marginal, true_destination]
        true_trip_hops = topology_hops[
            artifact["true_origin"], artifact["true_destination"]
        ]
        predicted_trip_hops = topology_hops[
            artifact["true_origin"], pred_conditional
        ]
        metrics["network_hop_metrics"] = {
            "conditional_destination_error_mean_hops": float(conditional_error.mean()),
            "conditional_destination_within_1_hop": float((conditional_error <= 1).mean()),
            "marginal_destination_error_mean_hops": float(marginal_error.mean()),
            "marginal_destination_within_1_hop": float((marginal_error <= 1).mean()),
            "conditional_predicted_trip_length_mae_hops": float(
                np.abs(predicted_trip_hops - true_trip_hops).mean()
            ),
        }
        metrics["network_trip_length_distribution"] = {
            "deployable_joint_distribution": hop_distribution_metrics(
                predicted_joint_od, observed_od, topology_hops, valid_od
            ),
            "true_origin_conditional_diagnostic": hop_distribution_metrics(
                predicted_true_origin_od, observed_od, topology_hops, valid_od
            ),
        }
    if "topology_strength" in artifact:
        strength = artifact["topology_strength"].astype(np.float64)
        peak = np.isin(artifact["true_hour_bin"], [7, 8, 9, 17, 18, 19])
        metrics["conditional_topology_strength"] = {
            "mean": float(strength.mean()),
            "std": float(strength.std(ddof=1)),
            "median": float(np.median(strength)),
            "q05": float(np.quantile(strength, 0.05)),
            "q95": float(np.quantile(strength, 0.95)),
            "observed_peak_hour_mean": float(strength[peak].mean()),
            "observed_offpeak_hour_mean": float(strength[~peak].mean()),
        }

    artifact_metadata = None
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact_path, **artifact)
        artifact_metadata = {
            "path": str(artifact_path.resolve()),
            "sha256": sha256(artifact_path.resolve()),
            "contains_raw_user_ids": False,
            "arrays": {key: list(value.shape) for key, value in artifact.items()},
        }
    return {"metrics": metrics, "artifact": artifact_metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["hangzhou", "guangzhou"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--history-end", required=True)
    parser.add_argument("--future-start", required=True)
    parser.add_argument("--min-trips", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-user-fraction", type=float, default=1.0)
    parser.add_argument("--train-user-count", type=int, default=0)
    parser.add_argument(
        "--train-user-sampling",
        choices=["simple_random", "history_stratified_proportional"],
        default="simple_random",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--d-loc", type=int, default=64)
    parser.add_argument("--d-dow", type=int, default=64)
    parser.add_argument("--d-hour", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--c-model", type=int, default=100)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--time-dist", choices=["almixture", "lognormal", "exponential"], default="almixture")
    parser.add_argument("--destination-objective", choices=["marginal", "conditional"], default="marginal")
    parser.add_argument("--od-head", choices=["low_rank", "dense"], default="low_rank")
    parser.add_argument("--independent-dest-head", action="store_true")
    parser.add_argument("--support-mode", choices=["full_nonself", "train_observed"], default="full_nonself")
    parser.add_argument("--disable-periodic-pe", action="store_true")
    parser.add_argument("--encoder-type", choices=["attention", "gru"], default="attention")
    parser.add_argument("--no-time-to-spatial", action="store_true")
    parser.add_argument(
        "--time-to-spatial-mode",
        choices=["legacy", "none", "raw", "interpretable", "shuffled", "oracle"],
        default="legacy",
    )
    parser.add_argument("--time-to-spatial-shuffle-seed", type=int, default=0)
    parser.add_argument("--time-only", action="store_true")
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--learn-topology-bias", action="store_true")
    parser.add_argument(
        "--topology-mode",
        choices=[
            "none",
            "fixed",
            "learned_global",
            "hop_bins",
            "conditional",
            "graph",
            "combo",
        ],
        default="none",
    )
    parser.add_argument("--fixed-topology-strength", type=float, default=0.0)
    parser.add_argument(
        "--topology-distance-source",
        choices=["shortest_hops", "station_index"],
        default="shortest_hops",
    )
    parser.add_argument("--topology-permutation-seed", type=int)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()

    if args.learn_topology_bias:
        if args.topology_mode not in {"none", "learned_global"}:
            parser.error(
                "--learn-topology-bias is a legacy alias and conflicts with "
                f"--topology-mode {args.topology_mode}"
            )
        args.topology_mode = "learned_global"
    topology_enabled = args.topology_mode != "none"
    if topology_enabled and args.topology is None:
        parser.error("network-aware topology modes require --topology")
    if args.topology_permutation_seed is not None and not topology_enabled:
        parser.error("--topology-permutation-seed requires a network-aware topology mode")
    if args.fixed_topology_strength < 0:
        parser.error("--fixed-topology-strength must be nonnegative")
    if args.topology_mode != "fixed" and args.fixed_topology_strength != 0.0:
        parser.error("--fixed-topology-strength is only valid with --topology-mode fixed")
    if args.topology_distance_source == "station_index" and not topology_enabled:
        parser.error("--topology-distance-source station_index requires a topology mode")
    if args.event_artifact is not None and not args.final_test:
        parser.error("--event-artifact is only valid with --final-test")
    if args.evaluate_only and not args.final_test:
        parser.error("--evaluate-only requires --final-test")
    if not 0 < args.train_user_fraction <= 1:
        parser.error("--train-user-fraction must be in (0, 1]")
    if args.train_user_count < 0:
        parser.error("--train-user-count must be nonnegative")
    if args.train_user_count and args.train_user_fraction != 1.0:
        parser.error("Use either --train-user-count or --train-user-fraction, not both")
    if args.no_time_to_spatial and args.time_to_spatial_mode != "legacy":
        parser.error(
            "--no-time-to-spatial is a legacy alias and cannot be combined "
            "with --time-to-spatial-mode"
        )

    started = dt.datetime.now(tz=dt.timezone.utc)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data_path = args.data.resolve()
    trip_frame = pd.read_pickle(data_path)
    metro_cfg = StrictMetroConfig(
        city=args.city,
        min_trips=args.min_trips,
        history_end=args.history_end,
        future_start=args.future_start,
        split_seed=args.split_seed,
        support_mode=args.support_mode,
    )
    corpus = StrictMetroCorpus(trip_frame, metro_cfg)
    del trip_frame
    od_bias, hops, topology_metadata = topology_arrays(args.topology, corpus)
    network_hops = hops
    topology_distance = None
    topology_adjacency = None
    if topology_enabled:
        if hops is None or od_bias is None:
            raise RuntimeError("Network-aware run did not load aligned topology")
        topology_distance = np.zeros_like(hops, dtype=np.float32)
        if args.topology_distance_source == "shortest_hops":
            topology_distance[1:, 1:] = hops[1:, 1:]
            topology_adjacency = (topology_distance == 1).astype(np.float32)
        else:
            station_index = np.arange(corpus.S, dtype=np.float32)
            topology_distance = np.abs(
                station_index[:, None] - station_index[None, :]
            )
            topology_distance[0, :] = 0.0
            topology_distance[:, 0] = 0.0
            diameter = float(topology_distance[1:, 1:].max())
            od_bias = np.zeros_like(topology_distance)
            od_bias[1:, 1:] = (
                -topology_distance[1:, 1:] / max(diameter, 1.0)
            )
            topology_adjacency = (topology_distance == 1).astype(np.float32)
            topology_metadata = dict(topology_metadata or {})
            topology_metadata.update(
                {
                    "distance_source": "ordinal station-index surrogate",
                    "bias_definition": (
                        "negative absolute model-station-index difference "
                        "divided by its maximum"
                    ),
                    "surrogate_control_not_physical_topology": True,
                }
            )
        topology_adjacency[0, :] = 0.0
        topology_adjacency[:, 0] = 0.0
    else:
        od_bias = None

    if topology_enabled and args.topology_permutation_seed is not None:
        permutation = np.arange(corpus.S)
        permutation[1:] = np.random.default_rng(
            args.topology_permutation_seed
        ).permutation(permutation[1:])
        od_bias = od_bias[np.ix_(permutation, permutation)]
        topology_distance = topology_distance[np.ix_(permutation, permutation)]
        topology_adjacency = topology_adjacency[np.ix_(permutation, permutation)]
        topology_metadata = dict(topology_metadata or {})
        topology_metadata.update(
            {
                "placebo": "station-label permutation applied during both training and evaluation",
                "permutation_seed": args.topology_permutation_seed,
                "permutation_sha256": hashlib.sha256(
                    permutation.astype(np.int16).tobytes()
                ).hexdigest(),
                "network_metrics_still_use_true_unpermuted_topology": True,
            }
        )
    if topology_metadata is not None:
        topology_metadata = dict(topology_metadata)
        topology_metadata.update(
            {
                "mode": args.topology_mode,
                "fixed_strength": args.fixed_topology_strength,
                "distance_source": topology_metadata.get(
                    "distance_source", args.topology_distance_source
                ),
            }
        )

    full_train_dataset = corpus.dataset("train")
    subset_count = (
        args.train_user_count
        if args.train_user_count
        else max(1, int(np.ceil(len(full_train_dataset) * args.train_user_fraction)))
    )
    if subset_count > len(full_train_dataset):
        parser.error("Requested training-user subset exceeds the training population")
    if subset_count < len(full_train_dataset):
        subset_indices, training_sampling = training_subset_indices(
            full_train_dataset.samples,
            subset_count,
            args.seed,
            args.train_user_sampling,
        )
        train_dataset = Subset(full_train_dataset, subset_indices)
    else:
        subset_count = len(full_train_dataset)
        train_dataset = full_train_dataset
        all_indices = np.arange(subset_count, dtype=np.int64)
        training_sampling = {
            "mode": "all",
            "requested_users": subset_count,
            "population_users": subset_count,
            "selected_index_sha256": hashlib.sha256(all_indices.tobytes()).hexdigest(),
            "selection_seed": None,
        }
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 10_000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_strict_metro,
        num_workers=0,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        corpus.dataset("validation"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_strict_metro,
        num_workers=0,
    )
    model_cfg = AMTPPPredConfig(
        d_loc=args.d_loc,
        d_dow=args.d_dow,
        d_hour=args.d_hour,
        n_heads=args.n_heads,
        c_model=args.c_model,
        K=args.K,
        r=args.rank,
        time_dist=args.time_dist,
        w_time=1.0,
        w_origin=0.0 if args.time_only else 1.0,
        w_dest=0.0 if args.time_only else 1.0,
        w_eos=0.0,
        od_head_type=args.od_head,
        independent_dest_head=args.independent_dest_head,
        destination_objective=args.destination_objective,
        learn_topology_bias=False,
        topology_mode=args.topology_mode,
        fixed_topology_strength=args.fixed_topology_strength,
        disable_periodic_pe=args.disable_periodic_pe,
        encoder_type=args.encoder_type,
        spatial_uses_time_params=not args.no_time_to_spatial,
        time_to_spatial_mode=args.time_to_spatial_mode,
        time_to_spatial_shuffle_seed=args.time_to_spatial_shuffle_seed,
    )
    model = AMTPPPred(
        n_locs=corpus.S,
        cond_vocab_sizes=[1],
        allowed_od_mask=corpus.allowed_od_mask,
        cfg=model_cfg,
        od_logit_bias=od_bias,
        topology_distance=topology_distance,
        topology_adjacency=topology_adjacency,
    ).to(args.device)
    device_object = torch.device(args.device)
    if device_object.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_object)
    checkpoint_path = args.checkpoint.resolve()
    history: list[dict[str, Any]] = []
    if args.evaluate_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        expected_model_config = dataclasses.asdict(model_cfg)
        if state.get("model_cfg") != expected_model_config:
            raise RuntimeError(
                "Frozen checkpoint model configuration does not match the final-test "
                f"command: checkpoint={state.get('model_cfg')}, command={expected_model_config}"
            )
        if state.get("seed") != args.seed:
            raise RuntimeError(
                f"Frozen checkpoint seed {state.get('seed')} != command seed {args.seed}"
            )
        if state.get("train_user_fraction", 1.0) != args.train_user_fraction:
            raise RuntimeError(
                "Frozen checkpoint train-user fraction does not match the final-test command"
            )
        if state.get("train_user_count", 0) != args.train_user_count:
            raise RuntimeError(
                "Frozen checkpoint train-user count does not match the final-test command"
            )
        if "training_sampling" in state and state["training_sampling"] != training_sampling:
            raise RuntimeError("Frozen checkpoint training-user sample changed")
        best_epoch = int(state["best_epoch"])
        best_validation = float(state["best_validation_eventweighted_total_nll"])
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_validation = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model,
                train_loader,
                args.device,
                optimizer,
                args.grad_clip,
                args.max_train_batches,
            )
            validation_loss = run_epoch(
                model,
                validation_loader,
                args.device,
                None,
                args.grad_clip,
                args.max_eval_batches,
            )
            record = {
                "epoch": epoch,
                "seconds": time.time() - epoch_start,
                "train": train_metrics,
                "validation": validation_loss,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            selection_value = validation_loss["total"]
            if selection_value < best_validation:
                best_validation = selection_value
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_cfg": dataclasses.asdict(model_cfg),
                        "metro_cfg": dataclasses.asdict(metro_cfg),
                        "corpus_summary": corpus.summary(),
                        "run_id": args.run_id,
                        "seed": args.seed,
                        "train_user_fraction": args.train_user_fraction,
                        "train_user_count": args.train_user_count,
                        "train_users_used": subset_count,
                        "training_sampling": training_sampling,
                        "best_epoch": epoch,
                        "best_validation_eventweighted_total_nll": best_validation,
                        "topology": topology_metadata,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            if stale >= args.patience:
                break
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    validation_evaluation = evaluate_events(
        model,
        validation_loader,
        corpus,
        "validation",
        args.device,
        network_hops,
        None,
        args.max_eval_batches,
    )
    test_evaluation = None
    if args.final_test:
        test_loader = DataLoader(
            corpus.dataset("test"),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_strict_metro,
            num_workers=0,
        )
        test_evaluation = evaluate_events(
            model,
            test_loader,
            corpus,
            "test",
            args.device,
            network_hops,
            args.event_artifact.resolve() if args.event_artifact else None,
            args.max_eval_batches,
        )

    result_path = args.result.resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    finished = dt.datetime.now(tz=dt.timezone.utc)
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device_object))
        if device_object.type == "cuda"
        else None
    )
    result = {
        "schema_version": "1.0.0",
        "trainer_version": TRAINER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "final-test" if args.final_test else "selection-only-no-test-access",
        "run_id": args.run_id,
        "city": args.city,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "device": args.device,
        "data": {"path": str(data_path), "sha256": sha256(data_path)},
        "protocol": dataclasses.asdict(metro_cfg),
        "corpus": corpus.summary(),
        "model_config": dataclasses.asdict(model_cfg),
        "training_sample": {
            "train_user_fraction": args.train_user_fraction,
            "train_user_count": args.train_user_count,
            "train_users_used": subset_count,
            "full_train_users": len(full_train_dataset),
            "sampling": training_sampling,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_allocated_bytes": peak_gpu_memory_bytes,
        "topology": topology_metadata,
        "topology_diagnostics": model.topology_diagnostics(),
        "selection": {
            "metric": "validation global-event-weighted total NLL",
            "best_epoch": best_epoch,
            "best_value": best_validation,
            "patience": args.patience,
            "epochs_completed": len(history) if not args.evaluate_only else None,
            "frozen_checkpoint_evaluation_only": args.evaluate_only,
        },
        "history": history,
        "validation": validation_evaluation,
        "test": test_evaluation,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "test_access_guard": {
            "test_loader_constructed_during_training_or_model_selection": False,
            "test_evaluated_once_after_best_checkpoint_reload": bool(args.final_test),
        },
        "aggregation": {
            "primary": "global-event-weighted",
            "secondary": "equal-weight-per-user",
        },
    }
    result_path.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "best_epoch": best_epoch, "best_validation": best_validation}), flush=True)


if __name__ == "__main__":
    main()
