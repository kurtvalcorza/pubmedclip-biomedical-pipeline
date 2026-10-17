"""Row 71 review fixes (2026-10-17): the pre-registered grid rule and the fresh-test score-once wiring (SCORE_ONCE_RULE).

Section 8 of the generated notebook is executed with stand-ins: a changed epochs / batch / seed / LR / data source never
scores the fresh test and writes an `_exploratory_<n>` file; an identical re-run is labelled a reproduction.
"""
# ruff: noqa: E501  -- written at the 110-column fleet width, like samples.py

from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import pytest

from pubmedclip_pipeline import (
    CT_PROMPT,
    GRID_CONFIGS,
    PREREGISTERED_SETTINGS,
    exploratory_path,
    look_at_test,
    pin_manifest_digest,
    select_config,
    settings_differing,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "pubmedclip_biomedical_colab.ipynb"
METRICS = ("accuracy", "macro_f1", "t2i_map")


# --- the pre-registered selection rule --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "frozen", "selected", "kept"),
    [
        ({"C0": 0.70, "C1": 0.715, "C2": 0.719}, 0.36, "C0", True),  # no challenger clears the 0.02 margin
        ({"C0": 0.70, "C1": 0.69, "C2": 0.73}, 0.36, "C2", True),  # one challenger clears it
        ({"C0": 0.70, "C1": 0.73, "C2": 0.745}, 0.36, "C1", True),  # both clear, within 0.02: smaller scope
        ({"C0": 0.70, "C1": 0.73, "C2": 0.76}, 0.36, "C2", True),  # both clear, 0.03 apart: the higher
        ({"C0": 0.37, "C1": 0.36, "C2": 0.37}, 0.36, "C0", False),  # adaptation does not beat frozen by 0.02
    ],
)
def test_select_config_applies_the_margin_tie_and_keep_frozen_rules(scores, frozen, selected, kept):
    result = select_config(scores, frozen)
    assert result["selected"] == selected and result["adaptation_kept"] is kept
    assert result["config"] == GRID_CONFIGS[selected] and result["margin"] == 0.02
    with pytest.raises(ValueError):
        select_config({"C0": 0.1}, 0.0)


def test_preregistered_settings_and_labels():
    assert GRID_CONFIGS == {
        "C0": {"trainable_vision_layers": 2, "lr": 5e-5},
        "C1": {"trainable_vision_layers": 2, "lr": 1e-5},
        "C2": {"trainable_vision_layers": 4, "lr": 5e-5},
    }
    assert PREREGISTERED_SETTINGS["pin_manifest_sha256"] == pin_manifest_digest()
    assert PREREGISTERED_SETTINGS["prompt_template"] == CT_PROMPT
    assert look_at_test(["epochs"]) == "not scored (exploratory)"
    assert look_at_test([]) == "pre-registered evaluation"
    assert look_at_test([], {"test_look": "pre-registered evaluation"}) == "reproduction"
    assert look_at_test([], {"test_look": "not scored (exploratory)"}) == "pre-registered evaluation"


def test_exploratory_path_never_reuses_a_file(tmp_path):
    base = tmp_path / "report.json"
    first = exploratory_path(base)
    assert first.name == "report_exploratory_1.json"
    first.write_text("{}")
    assert exploratory_path(base).name == "report_exploratory_2.json"


# --- Section 8 executed with stand-ins ---------------------------------------------------------------------------------


def _section8() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    found = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and "run_settings = " in "".join(c["source"])]
    assert len(found) == 1
    return found[0]


def _scores(t2i: float) -> dict:
    return {"accuracy": t2i, "macro_f1": t2i, "t2i_map": t2i, "n": 2, "per_class": {"a": {"n": 1, "recall": t2i, "ap": t2i}, "b": {"n": 1, "recall": t2i, "ap": t2i}}}


class _Pipe:
    def __init__(self):
        self.scored: list[str] = []
        self.frozen = False

    def evaluate(self, records, **_kw):
        self.scored.append(records[0]["id"] + (" frozen" if self.frozen else ""))
        return _scores(0.3 if self.frozen else 0.6)

    def save_artifact(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def restore_base(self):
        self.frozen = True

    def load_artifact(self, _path):
        self.frozen = False


SELECTION = select_config({"C0": 0.7, "C1": 0.6, "C2": 0.71}, 0.36)


def _namespace(pipe, **changes) -> dict:
    ns = {
        "json": json, "Path": Path, "shutil": shutil, "USE_BYOD": False, "pin_manifest_digest": pin_manifest_digest,
        "SPLIT_SEED": 42, "adapt_result": {"seed": 0, "history": [], "best_epoch": 1}, "EPOCHS": 6, "BATCH_SIZE": 16,
        "PROMPT_TEMPLATE": CT_PROMPT, "grid_selection": SELECTION, "layers_used": 2, "lr_used": 5e-5,
        "settings_differing": settings_differing, "look_at_test": look_at_test, "exploratory_path": exploratory_path,
        "PREREGISTRATION": "docs/pubmedclip-preregistration.md", "METRICS": METRICS, "classes": ["a", "b"], "pipe": pipe,
        "train_records": [{"id": "train"}], "val_records": [{"id": "val"}], "test_records": [{"id": "fresh"}],
        "dev_records": [{"id": "exposed"}], "dev_label": "exposed development test", "display_names": {}, "plain_names": {},
        "baseline_majority": _scores(0.1), "baseline_neighbour": _scores(0.5), "frozen_test": _scores(0.3), "frozen_plain": _scores(0.3),
        "majority_baseline": lambda tr, te, c: _scores(0.1), "colour_neighbour_baseline": lambda tr, te, c: _scores(0.5),
        "MODEL_ID": "m", "MODEL_REVISION": "r", "DEFAULT_MODEL_KEY": "k", "data_source": "pinned", "dataset_manifests": {},
        "disjoint": {}, "adapt_seconds": 0.0, "types": types,
    }
    ns.update(changes)
    return ns


def test_preregistered_run_scores_the_fresh_test_once_and_a_rerun_is_a_reproduction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe()
    ns = _namespace(pipe)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is True and ns["test_look"] == "pre-registered evaluation"
    assert "fresh" in pipe.scored and "fresh frozen" in pipe.scored and pipe.frozen is False  # adapted state put back
    report = json.loads((tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report.json").read_text())
    assert report["test_look"] == "pre-registered evaluation" and report["settings_differing"] == []
    assert report["fresh_test"]["verdict"]["adapted_vs_frozen_t2i_map"] == "improved"
    assert report["development_view"]["label"] == "exposed development test"
    again = _Pipe()
    ns = _namespace(again)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["test_look"] == "reproduction"
    assert json.loads((tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report.json").read_text())["test_look"] == "reproduction"


@pytest.mark.parametrize(
    "change",
    [
        {"EPOCHS": 3},
        {"BATCH_SIZE": 8},
        {"adapt_result": {"seed": 1, "history": [], "best_epoch": 1}},
        {"lr_used": 1e-5},
        {"SPLIT_SEED": 7},
        {"USE_BYOD": True, "dev_label": "your own test split"},
        {"grid_selection": None},
        {"PROMPT_TEMPLATE": "A medical image of {label}."},
    ],
)
def test_a_changed_setting_never_scores_the_fresh_test(tmp_path, monkeypatch, change):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    canonical = tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report.json"
    canonical.write_text(json.dumps({"test_look": "pre-registered evaluation", "marker": "kept"}))
    pipe = _Pipe()
    ns = _namespace(pipe, **change)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and ns["test_look"] == "not scored (exploratory)" and ns["differing"]
    assert not any(s.startswith("fresh") for s in pipe.scored)
    assert json.loads(canonical.read_text())["marker"] == "kept"  # never overwritten
    written = json.loads((tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report_exploratory_1.json").read_text())
    assert written["test_look"] == "not scored (exploratory)" and written["fresh_test"] is None


def test_frozen_is_kept_when_adaptation_does_not_beat_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe()
    frozen_wins = select_config({"C0": 0.37, "C1": 0.36, "C2": 0.37}, 0.36)
    ns = _namespace(pipe, grid_selection=frozen_wins)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is True and "fresh" not in pipe.scored and "fresh frozen" in pipe.scored
    assert "adapted" not in ns["fresh_test"]["t2i_map"]
    assert ns["fresh_test"]["verdict"]["adapted_vs_frozen_t2i_map"].startswith("adaptation not kept")
