"""The tutorial's pre-registered evaluation protocol (docs/pubmedclip-preregistration.md, 2026-10-17).

A three-config grid is scored on validation only and a challenger replaces the default only by a margin; the fresh test
is scored only when every pre-registered setting holds (SCORE_ONCE_RULE). Any other setting is exploratory: it reads
validation and the labelled exposed development test, and writes a separate `*_exploratory_<n>.json`.
"""
# ruff: noqa: E501  -- written at the 110-column fleet width, like samples.py

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .samples import pin_manifest_digest

PREREGISTRATION = "docs/pubmedclip-preregistration.md (+ docs/pubmedclip-preregistration-amendment-1.md)"
CT_PROMPT = "An axial abdominal CT slice showing the {label}."
GRID_MARGIN = 0.02  # validation text-to-image mAP on 352 slices; a practical tolerance, not a significance threshold
DEFAULT_CONFIG = "C0"
GRID_CONFIGS: dict[str, dict[str, Any]] = {
    "C0": {"trainable_vision_layers": 2, "lr": 5e-5},
    "C1": {"trainable_vision_layers": 2, "lr": 1e-5},
    "C2": {"trainable_vision_layers": 4, "lr": 5e-5},
}
# Every setting the pre-registration fixes; `config` must also equal the grid's selection in the same run.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "pinned sample",
    "pin_manifest_sha256": pin_manifest_digest(),
    "split_seed": 42,
    "adapt_seed": 0,
    "epochs": 6,
    "batch_size": 16,
    "prompt_template": CT_PROMPT,
    "grid_run": True,
}
# SCORE_ONCE_RULE item 8: a runtime cannot know about other runtimes, so the labels say "(this runtime)"; which run is the
# evaluation of record (the first hosted pre-registered run) is stated only in docs/release-verification.md.
TEST_LOOKS = ("pre-registered evaluation (this runtime)", "reproduction (this runtime)", "not scored (exploratory)")
MODEL_STATE_MISMATCH = "model state is not the selected adapted model"
_ADAPTER_KEYS = ("epochs", "lr", "trainable_vision_layers", "seed", "best_epoch", "n_train", "n_val")


def _scope(name: str) -> tuple[int, float]:
    config = GRID_CONFIGS[name]
    return (int(config["trainable_vision_layers"]), float(config["lr"]))


def run_fingerprint(
    *,
    data_source: str,
    pin_manifest_sha256: str | None,
    split_seed: int,
    validation_digest: str,
    epochs: int,
    batch_size: int,
    seed: int,
    prompt_template: str,
) -> dict[str, Any]:
    """The inputs a grid selection is only valid for (amendment 1, section 2)."""
    return {
        "data_source": data_source,
        "pin_manifest_sha256": pin_manifest_sha256,
        "split_seed": split_seed,
        "validation_digest": validation_digest,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "prompt_template": prompt_template,
    }


def adapter_mismatch(adapter: Mapping[str, Any] | None, adapt_result: Mapping[str, Any] | None) -> list[str]:
    """Why the model in memory is not the one Section 7b trained (empty = it is)."""
    if adapter is None or adapt_result is None:
        return [MODEL_STATE_MISMATCH + " (no adapter on the pipeline, or no Section 7b result)"]
    bad = [k for k in _ADAPTER_KEYS if adapter.get(k) != adapt_result.get(k)]
    return [f"{MODEL_STATE_MISMATCH} (differs in {bad})"] if bad else []


def select_config(
    scores: Mapping[str, float],
    frozen_val_map: float,
    *,
    margin: float = GRID_MARGIN,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered rule to best-epoch validation mAP per config.

    C0 stays unless a challenger beats it by >= margin; if both challengers clear it, the higher wins unless they are
    within the margin of each other, when the smaller trainable scope (then the lower learning rate) wins. The selected
    config must beat the frozen model's validation mAP by the margin, else adaptation is not kept."""
    if set(scores) != set(GRID_CONFIGS):
        raise ValueError(f"grid scores must cover exactly {sorted(GRID_CONFIGS)}")
    base = float(scores[DEFAULT_CONFIG])
    clear = [n for n in GRID_CONFIGS if n != DEFAULT_CONFIG and float(scores[n]) >= base + margin - 1e-12]
    if not clear:
        selected, reason = DEFAULT_CONFIG, f"no challenger beat {DEFAULT_CONFIG} by {margin}"
    elif len(clear) == 1:
        selected, reason = clear[0], f"{clear[0]} beat {DEFAULT_CONFIG} by >= {margin}"
    else:
        a, b = sorted(clear, key=_scope)
        if abs(float(scores[a]) - float(scores[b])) < margin:
            selected, reason = a, f"{a} and {b} both beat {DEFAULT_CONFIG} and are within {margin}: smaller scope"
        else:
            selected = a if float(scores[a]) > float(scores[b]) else b
            reason = f"{a} and {b} both beat {DEFAULT_CONFIG}; {selected} is higher by >= {margin}"
    kept = float(scores[selected]) >= float(frozen_val_map) + margin - 1e-12
    return {
        "selected": selected,
        "config": dict(GRID_CONFIGS[selected]),
        "adaptation_kept": kept,
        "reason": reason
        + ("" if kept else f"; adaptation did not beat frozen on validation by {margin}: the frozen model is kept"),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "frozen_val_map": round(float(frozen_val_map), 4),
        "margin": margin,
        "fingerprint": dict(fingerprint) if fingerprint is not None else None,
    }


def settings_differing(
    settings: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    fingerprint: Mapping[str, Any] | None = None,
    model_state: list[str] | None = None,
) -> list[str]:
    """The settings that differ from the pre-registration (empty = pre-registered): every fixed setting, the grid's
    selection and the inputs it ran on (`fingerprint`), and the model state (`model_state`, from `adapter_mismatch`)."""
    differing = [k for k, v in PREREGISTERED_SETTINGS.items() if settings.get(k) != v]
    differing += list(model_state or [])
    if selection is None:
        differing.append("config (no grid selection in this run)")
    else:
        if fingerprint is None or selection.get("fingerprint") != dict(fingerprint):
            differing.append("grid selection (it ran on other data or settings than this run; re-run Section 7)")
        used = {"trainable_vision_layers": settings.get("trainable_vision_layers"), "lr": settings.get("lr")}
        if used != selection["config"]:
            differing.append(f"config (used {used}, grid selected {selection['selected']} {selection['config']})")
    return differing


def look_at_test(differing: list[str], prior: Mapping[str, Any] | None = None) -> str:
    """`not scored (exploratory)` when any setting differs; `reproduction (this runtime)` when an earlier report in this
    runtime carries a pre-registered scoring; else `pre-registered evaluation (this runtime)`."""
    if differing:
        return TEST_LOOKS[2]
    if (prior or {}).get("test_look") in TEST_LOOKS[:2]:
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def output_suffix(path: str | Path, test_look: str) -> str:
    """One suffix per run for every file it writes: '' for the first pre-registered scoring in this runtime,
    `_reproduction_<n>` or `_exploratory_<n>` (first free n beside `path`) otherwise, so no run overwrites another."""
    if test_look == TEST_LOOKS[0]:
        return ""
    kind = "reproduction" if test_look == TEST_LOOKS[1] else "exploratory"
    path = Path(path)
    n = 1
    while path.with_name(f"{path.stem}_{kind}_{n}{path.suffix}").exists():
        n += 1
    return f"_{kind}_{n}"


def exploratory_path(path: str | Path) -> Path:
    """`<stem>_exploratory_<n><suffix>` beside `path`, with the first free n: exploratory runs never overwrite the
    pre-registered result file."""
    path = Path(path)
    n = 1
    while (candidate := path.with_name(f"{path.stem}_exploratory_{n}{path.suffix}")).exists():
        n += 1
    return candidate
