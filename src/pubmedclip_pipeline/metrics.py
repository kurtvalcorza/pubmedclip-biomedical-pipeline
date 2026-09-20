"""Corpus-level zero-shot classification and retrieval metrics with two non-neural baselines, in numpy.

`pipeline.py` keeps the per-grid plumbing checks (`top1_accuracy`, `recall_at_1`); this module scores a
labelled dataset the way a zero-shot classifier is read:

- **accuracy** and **macro F1** of the top-scoring class prompt per image, with a per-class breakdown;
- **text-to-image mean average precision**: each class prompt as a query over every image of the set,
  the average precision of its ranking (the class's own images are the relevant ones), averaged over
  classes — the retrieval view of the same score matrix.

Two baselines a fine-tuned model must beat: the **majority floor** (the most frequent training label for
every image; chance-level macro F1 on a balanced set) and a **colour nearest neighbour** (the label of
the training image whose 3x3 mean-colour grid is closest — a classifier that knows the image through
27 numbers; on grey-scale CT slices the three channels coincide and the grid is a 3x3 mean-intensity layout).
"""
# ruff: noqa: E501  -- fleet metrics module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

COLOUR_GRID = 3
METRIC_DEFINITIONS = {
    "accuracy": "fraction of images whose highest-scoring class prompt is the gold label; in 0..1",
    "macro_f1": "unweighted mean over classes of the F1 of predicting that class; in 0..1",
    "t2i_map": (
        "mean over class prompts of the average precision of ranking every image by the prompt's "
        "score (the class's own images are relevant); in 0..1"
    ),
}


def _per_class(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> dict[str, Any]:
    out = {}
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"n": int(np.sum(gold == c)), "recall": recall, "precision": precision, "f1": f1}
    return out


def average_precision(scores: Sequence[float], relevant: Sequence[bool]) -> float:
    """AP of ranking `scores` descending against binary relevance (0 when nothing is relevant)."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevant, dtype=bool)
    if scores_arr.shape != rel.shape or scores_arr.ndim != 1:
        raise ValueError("scores and relevant must be parallel 1-D sequences")
    if not rel.any():
        return 0.0
    order = np.argsort(-scores_arr, kind="stable")
    hits, total = 0, 0.0
    for rank, index in enumerate(order, start=1):
        if rel[index]:
            hits += 1
            total += hits / rank
    return total / int(rel.sum())


def classification_metrics(
    scores: Any, gold: Sequence[int], classes: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro F1, per-class breakdown and text-to-image mAP for an `[image][class]` score grid."""
    grid = np.asarray(scores, dtype=np.float64)
    gold_arr = np.asarray(gold, dtype=int)
    if grid.ndim != 2 or grid.shape != (len(gold_arr), len(classes)):
        raise ValueError(f"scores must be an [images x classes] grid, got {grid.shape}")
    if not len(gold_arr) or gold_arr.min() < 0 or gold_arr.max() >= len(classes):
        raise ValueError("gold must hold class indices for at least one image")
    if not np.all(np.isfinite(grid)):
        raise ValueError("scores must be finite")
    pred = grid.argmax(axis=1)
    per = _per_class(pred, gold_arr, len(classes))
    aps = [average_precision(grid[:, c], gold_arr == c) for c in range(len(classes))]
    return {
        "n": int(len(gold_arr)),
        "accuracy": float(np.mean(pred == gold_arr)),
        "macro_f1": float(np.mean([per[c]["f1"] for c in range(len(classes))])),
        "t2i_map": float(np.mean(aps)),
        "per_class": {classes[c]: {**per[c], "ap": aps[c]} for c in range(len(classes))},
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _scores_from_predictions(pred: Sequence[int], n_classes: int) -> np.ndarray:
    grid = np.zeros((len(pred), n_classes))
    for i, c in enumerate(pred):
        grid[i, c] = 1.0
    return grid


def majority_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    """The most frequent training label for every image."""
    if not train:
        raise ValueError("the majority baseline needs training records")
    counts = Counter(str(r["label"]) for r in train)
    label = max(sorted(counts), key=counts.get)
    index = list(classes).index(label)
    gold = [list(classes).index(str(r["label"])) for r in records]
    result = classification_metrics(
        _scores_from_predictions([index] * len(records), len(classes)), gold, classes
    )
    result["baseline"] = f"majority floor ({label!r} for every image)"
    return result


def colour_signature(image: str | Image.Image, *, grid: int = COLOUR_GRID) -> list[float]:
    """Mean RGB of each cell of a `grid` x `grid` partition of the image, in 0..1 (27 numbers by default)."""
    handle = image if isinstance(image, Image.Image) else Image.open(image)
    small = handle.convert("RGB").resize((grid * 8, grid * 8), Image.BILINEAR)
    pixels = list(small.getdata())
    out: list[float] = []
    for row in range(grid):
        for col in range(grid):
            cell = [
                pixels[(row * 8 + y) * grid * 8 + col * 8 + x] for y in range(8) for x in range(8)
            ]
            out.extend(sum(p[channel] for p in cell) / (64 * 255.0) for channel in range(3))
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def colour_neighbour_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    """Label every image with the label of the training image whose colour signature is closest;
    the score grid is the negative distance to the nearest training image of each class."""
    if not train:
        raise ValueError("the colour-neighbour baseline needs training records")
    class_list = list(classes)
    signatures = [(colour_signature(r["image"]), class_list.index(str(r["label"]))) for r in train]
    grid = np.zeros((len(records), len(class_list)))
    for i, record in enumerate(records):
        query = colour_signature(record["image"])
        best = [math.inf] * len(class_list)
        for sig, c in signatures:
            best[c] = min(best[c], _distance(sig, query))
        grid[i] = [-d for d in best]
    gold = [class_list.index(str(r["label"])) for r in records]
    result = classification_metrics(grid, gold, class_list)
    result["baseline"] = f"colour nearest neighbour ({len(train)} training images)"
    return result
