"""Event metrics for strict AMTPP evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np



SCHEMA_VERSION = "1.0.0"
EVALUATOR_VERSION = "amtpp-r2-eventwise-1.0.0"
HASH_NAMESPACE = "amtpp-r2-private-event-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object, length: int = 32) -> str:
    payload = "|".join([HASH_NAMESPACE, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for label in range(1, num_classes):
        true_label = y_true == label
        pred_label = y_pred == label
        tp = int(np.logical_and(true_label, pred_label).sum())
        fp = int(np.logical_and(~true_label, pred_label).sum())
        fn = int(np.logical_and(true_label, ~pred_label).sum())
        denominator = 2 * tp + fp + fn
        if denominator:
            scores.append(2 * tp / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def topk_accuracy(prob: np.ndarray, target: np.ndarray, k: int) -> float:
    effective_k = min(k, prob.shape[1])
    indices = np.argpartition(prob, -effective_k, axis=1)[:, -effective_k:]
    return float(np.any(indices == target[:, None], axis=1).mean())


def multiclass_brier(prob: np.ndarray, target: np.ndarray) -> float:
    rows = np.arange(len(target))
    return float(np.mean(np.square(prob).sum(axis=1) - 2.0 * prob[rows, target] + 1.0))


def top_label_ece(prob: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    predicted = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    correct = predicted == target
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = np.logical_and(confidence >= edges[index], confidence <= edges[index + 1])
        else:
            mask = np.logical_and(confidence >= edges[index], confidence < edges[index + 1])
        count = int(mask.sum())
        if count:
            ece += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def metrics_for(prob: np.ndarray, target: np.ndarray, num_classes: int) -> dict[str, float]:
    rows = np.arange(len(target))
    true_prob = np.clip(prob[rows, target].astype(np.float64), 1e-12, 1.0)
    predicted = prob.argmax(axis=1)
    rank = 1 + (prob > true_prob[:, None]).sum(axis=1)
    return {
        "top1": float((predicted == target).mean()),
        "top3": topk_accuracy(prob, target, 3),
        "top5": topk_accuracy(prob, target, 5),
        "macro_f1_union_support_excluding_pad": macro_f1(target, predicted, num_classes),
        "nll": float(-np.log(true_prob).mean()),
        "mrr": float((1.0 / rank).mean()),
        "multiclass_brier": multiclass_brier(prob, target),
        "top_label_ece_15_equal_width_bins": top_label_ece(prob, target, 15),
    }


def user_weighted_top1(user_hash: np.ndarray, prob: np.ndarray, target: np.ndarray) -> float:
    predicted = prob.argmax(axis=1)
    values = []
    for uid in np.unique(user_hash):
        mask = user_hash == uid
        values.append(float((predicted[mask] == target[mask]).mean()))
    return float(np.mean(values)) if values else float("nan")
