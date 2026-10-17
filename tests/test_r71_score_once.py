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

import numpy as np
import pytest

from pubmedclip_pipeline import (
    CT_PROMPT,
    GRID_CONFIGS,
    PREREGISTERED_SETTINGS,
    adapter_mismatch,
    apply_embedding_probe,
    exploratory_path,
    fit_embedding_probe,
    look_at_test,
    output_suffix,
    pin_manifest_digest,
    run_fingerprint,
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
    # SCORE_ONCE_RULE item 8: a runtime only knows about itself
    assert look_at_test(["epochs"]) == "not scored (exploratory)"
    assert look_at_test([]) == "pre-registered evaluation (this runtime)"
    assert look_at_test([], {"test_look": "pre-registered evaluation (this runtime)"}) == "reproduction (this runtime)"
    assert look_at_test([], {"test_look": "not scored (exploratory)"}) == "pre-registered evaluation (this runtime)"


def test_output_suffix_and_exploratory_path_never_reuse_a_file(tmp_path):
    base = tmp_path / "report.json"
    first = exploratory_path(base)
    assert first.name == "report_exploratory_1.json"
    first.write_text("{}")
    assert exploratory_path(base).name == "report_exploratory_2.json"
    assert output_suffix(base, "pre-registered evaluation (this runtime)") == ""
    assert output_suffix(base, "not scored (exploratory)") == "_exploratory_2"
    assert output_suffix(base, "reproduction (this runtime)") == "_reproduction_1"


def test_frozen_embedding_probe_fits_on_train_and_picks_the_ridge_on_validation():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 8))
    labels = ["a", "b", "c"]

    def draw(n):
        y = [labels[i % 3] for i in range(n)]
        return centres[[i % 3 for i in range(n)]] + 0.3 * rng.normal(size=(n, 8)), y

    (xt, yt), (xv, yv), (xs, ys) = draw(60), draw(30), draw(30)
    probe = fit_embedding_probe(xt, yt, xv, yv, labels)
    assert probe["ridge"] in (10.0, 1.0, 0.1, 0.01, 0.001) and len(probe["ridges_tried"]) == 5
    assert probe["validation"]["t2i_map"] == max(probe["ridges_tried"].values()) or abs(probe["validation"]["t2i_map"] - max(probe["ridges_tried"].values())) < 1e-4
    scored = apply_embedding_probe(probe, xs, ys, labels)
    assert scored["accuracy"] > 0.9 and "ridge readout on frozen image embeddings" in scored["baseline"]


# --- Section 8 executed with stand-ins ---------------------------------------------------------------------------------


def _section8() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    found = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and "run_settings = " in "".join(c["source"])]
    assert len(found) == 1
    return found[0]


def _scores(t2i: float) -> dict:
    return {"accuracy": t2i, "macro_f1": t2i, "t2i_map": t2i, "n": 2, "per_class": {"a": {"n": 1, "recall": t2i, "ap": t2i}, "b": {"n": 1, "recall": t2i, "ap": t2i}}}


ADAPTED = {"epochs": 6, "lr": 5e-5, "trainable_vision_layers": 2, "seed": 0, "best_epoch": 1, "n_train": 1, "n_val": 1, "history": []}


class _Pipe:
    def __init__(self):
        self.scored: list[str] = []
        self.frozen = False
        self.adapter = dict(ADAPTED)

    def evaluate(self, records, **_kw):
        self.scored.append(records[0]["id"] + (" frozen" if self.frozen else ""))
        return _scores(0.3 if self.frozen else 0.6)

    def embed_image(self, images):
        self.scored.append("embedded" + (" frozen" if self.frozen else ""))
        return np.ones((len(images), 2))

    def save_artifact(self, path):
        if self.adapter is None:
            raise ValueError("nothing to save")
        Path(path).mkdir(parents=True, exist_ok=True)

    def restore_base(self):
        self.frozen = True
        self.adapter = None

    def load_artifact(self, _path):
        self.frozen = False
        self.adapter = dict(ADAPTED)


def _fingerprint(**changes):
    values = {"data_source": "pinned sample", "pin_manifest_sha256": pin_manifest_digest(), "split_seed": 42, "validation_digest": "val-digest", "epochs": 6, "batch_size": 16, "seed": 0, "prompt_template": CT_PROMPT}
    values.update(changes)
    return run_fingerprint(**values)


SELECTION = select_config({"C0": 0.7, "C1": 0.6, "C2": 0.71}, 0.36, fingerprint=_fingerprint())
PROBE = {"weights": None, "ridge": 1.0, "ridges_tried": {"1.0": 0.5}, "validation": _scores(0.5), "baseline": "probe"}


def _namespace(pipe, **changes) -> dict:
    ns = {
        "json": json, "Path": Path, "shutil": shutil, "USE_BYOD": False, "pin_manifest_digest": pin_manifest_digest,
        "SPLIT_SEED": 42, "adapt_result": dict(ADAPTED), "EPOCHS": 6, "BATCH_SIZE": 16,
        "PROMPT_TEMPLATE": CT_PROMPT, "grid_selection": SELECTION, "layers_used": 2, "lr_used": 5e-5,
        "settings_differing": settings_differing, "look_at_test": look_at_test, "exploratory_path": exploratory_path,
        "output_suffix": output_suffix, "run_fingerprint": run_fingerprint, "adapter_mismatch": adapter_mismatch,
        "PREREGISTERED_SETTINGS": PREREGISTERED_SETTINGS, "apply_embedding_probe": lambda probe, emb, labels, classes: _scores(0.5),
        "probe": PROBE, "probe_dev": _scores(0.5),
        "PREREGISTRATION": "docs/pubmedclip-preregistration.md", "METRICS": METRICS, "classes": ["a", "b"], "pipe": pipe,
        "train_records": [{"id": "train", "label": "a"}], "val_records": [{"id": "val", "label": "a"}], "test_records": [{"id": "fresh", "label": "a", "image": None}],
        "dev_records": [{"id": "exposed", "label": "a"}], "dev_label": "exposed development test", "display_names": {}, "plain_names": {},
        "baseline_majority": _scores(0.1), "baseline_neighbour": _scores(0.5), "frozen_test": _scores(0.3), "frozen_plain": _scores(0.3),
        "majority_baseline": lambda tr, te, c: _scores(0.1), "colour_neighbour_baseline": lambda tr, te, c: _scores(0.5),
        "MODEL_ID": "m", "MODEL_REVISION": "r", "DEFAULT_MODEL_KEY": "k", "data_source": "pinned", "dataset_manifests": {"validation": {"digest": "val-digest"}},
        "disjoint": {}, "adapt_seconds": 0.0, "types": types,
    }
    ns.update(changes)
    return ns


def _fresh_scored(pipe) -> bool:
    return any(s.startswith("fresh") for s in pipe.scored)


def test_preregistered_run_scores_the_fresh_test_and_a_rerun_is_a_reproduction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe()
    ns = _namespace(pipe)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is True and ns["test_look"] == "pre-registered evaluation (this runtime)"
    assert "fresh" in pipe.scored and "fresh frozen" in pipe.scored and pipe.frozen is False  # adapted state put back
    first = tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report.json"
    report = json.loads(first.read_text())
    assert report["test_look"] == "pre-registered evaluation (this runtime)" and report["settings_differing"] == []
    assert "release-verification" in report["evaluation_of_record"]
    assert report["fresh_test"]["verdict"]["adapted_vs_frozen_t2i_map"] == "improved"
    assert "frozen_embedding_probe" in report["fresh_test"]["t2i_map"]
    assert report["development_view"]["note"].startswith("reported in every run")
    again = _Pipe()
    ns = _namespace(again)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["test_look"] == "reproduction (this runtime)"
    assert json.loads(first.read_text())["test_look"] == "pre-registered evaluation (this runtime)"  # never overwritten
    second = tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report_reproduction_1.json"
    assert json.loads(second.read_text())["test_look"] == "reproduction (this runtime)"


@pytest.mark.parametrize(
    "change",
    [
        {"EPOCHS": 3},
        {"BATCH_SIZE": 8},
        {"adapt_result": {**ADAPTED, "seed": 1}},
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
    canonical.write_text(json.dumps({"test_look": "pre-registered evaluation (this runtime)", "marker": "kept"}))
    pipe = _Pipe()
    ns = _namespace(pipe, **change)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and ns["test_look"] == "not scored (exploratory)" and ns["differing"]
    assert not _fresh_scored(pipe)
    assert json.loads(canonical.read_text())["marker"] == "kept"  # never overwritten
    written = json.loads((tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report_exploratory_1.json").read_text())
    assert written["test_look"] == "not scored (exploratory)" and written["fresh_test"] is None


def test_rerun_path_a_a_stale_grid_selection_from_another_data_load_does_not_count(tmp_path, monkeypatch):
    """R71P-M1 path A: a BYOD run ran the grid; the learner switches back to the sample and re-runs Section 4, then 7b and 8."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    byod_grid = select_config({"C0": 0.7, "C1": 0.6, "C2": 0.71}, 0.36, fingerprint=_fingerprint(data_source="BYOD", pin_manifest_sha256=None, validation_digest="byod-val"))
    pipe = _Pipe()
    ns = _namespace(pipe, grid_selection=byod_grid)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and any("grid selection" in d for d in ns["differing"]) and not _fresh_scored(pipe)
    # ...and a grid run with another SPLIT_SEED, then a return to 42 without re-running the grid
    pipe = _Pipe()
    ns = _namespace(pipe, grid_selection=select_config({"C0": 0.7, "C1": 0.6, "C2": 0.71}, 0.36, fingerprint=_fingerprint(split_seed=7)))
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and not _fresh_scored(pipe)
    # Section 4 itself resets every selection / result from an earlier data load
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    section4 = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "USE_BYOD = False" in "".join(c["source"]))
    assert "grid_selection = adapt_result = fresh_test = probe = None" in section4
    assert section4.index("grid_selection = adapt_result") < section4.index("if USE_BYOD:")


def test_rerun_path_b_a_base_model_restored_by_section_6_or_7_never_scores_the_fresh_test(tmp_path, monkeypatch):
    """R71P-M1 path B: Section 6 (or the grid) is re-run after 7b, which restores the pinned base; then Section 8."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe()
    pipe.restore_base()  # what a re-run of Section 5, 6 or 7 does
    pipe.frozen = False
    ns = _namespace(pipe)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and not _fresh_scored(pipe)
    assert any("model state is not the selected adapted model" in d for d in ns["differing"])
    # an adapter from other settings (e.g. an exploratory 7b) does not count either
    pipe = _Pipe()
    pipe.adapter = {**ADAPTED, "best_epoch": 3}
    ns = _namespace(pipe)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is False and not _fresh_scored(pipe)


def test_frozen_is_kept_when_adaptation_does_not_beat_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe()
    frozen_wins = select_config({"C0": 0.37, "C1": 0.36, "C2": 0.37}, 0.36, fingerprint=_fingerprint())
    ns = _namespace(pipe, grid_selection=frozen_wins)
    exec(compile(_section8(), "<section 8>", "exec"), ns)
    assert ns["preregistered"] is True and "fresh" not in pipe.scored and "fresh frozen" in pipe.scored
    assert "adapted" not in ns["fresh_test"]["t2i_map"]
    assert ns["fresh_test"]["verdict"]["adapted_vs_frozen_t2i_map"].startswith("adaptation not kept")
