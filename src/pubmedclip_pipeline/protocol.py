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

PREREGISTRATION = "docs/pubmedclip-preregistration.md"
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
TEST_LOOKS = ("pre-registered evaluation", "reproduction", "not scored (exploratory)")
# The release-verification row of the first recorded pre-registered hosted run, once there is one; every later run with
# identical settings is then a reproduction of it.
RECORDED_EVALUATION: str | None = None


def _scope(name: str) -> tuple[int, float]:
    config = GRID_CONFIGS[name]
    return (int(config["trainable_vision_layers"]), float(config["lr"]))


def select_config(
    scores: Mapping[str, float], frozen_val_map: float, *, margin: float = GRID_MARGIN
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
    }


def settings_differing(settings: Mapping[str, Any], selection: Mapping[str, Any] | None) -> list[str]:
    """The names of the settings that differ from the pre-registration (empty = pre-registered)."""
    differing = [k for k, v in PREREGISTERED_SETTINGS.items() if settings.get(k) != v]
    if selection is None:
        differing.append("config (no grid selection in this run)")
    else:
        used = {"trainable_vision_layers": settings.get("trainable_vision_layers"), "lr": settings.get("lr")}
        if used != selection["config"]:
            differing.append(f"config (used {used}, grid selected {selection['selected']} {selection['config']})")
    return differing


def look_at_test(differing: list[str], prior: Mapping[str, Any] | None = None) -> str:
    """`not scored (exploratory)` when any setting differs; `reproduction` when a pre-registered evaluation was already
    recorded (RECORDED_EVALUATION) or an earlier report in this runtime carries one; else `pre-registered evaluation`."""
    if differing:
        return TEST_LOOKS[2]
    if RECORDED_EVALUATION is not None or (prior or {}).get("test_look") in TEST_LOOKS[:2]:
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def exploratory_path(path: str | Path) -> Path:
    """`<stem>_exploratory_<n><suffix>` beside `path`, with the first free n: exploratory runs never overwrite the
    pre-registered result file."""
    path = Path(path)
    n = 1
    while (candidate := path.with_name(f"{path.stem}_exploratory_{n}{path.suffix}")).exists():
        n += 1
    return candidate
