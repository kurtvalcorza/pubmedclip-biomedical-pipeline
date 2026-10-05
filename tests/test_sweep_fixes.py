"""Regression tests for the 2026-10-05 fleet-sweep fixes of pubmedclip_biomedical_colab.ipynb (SWP-R, SWP-G, SWP-A, SWP-F, SWP-B).

Every test needs only CI's dependencies (NumPy-level, no model, no torch): the notebook's own cell sources are executed
with stand-ins where a model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "pubmedclip_biomedical_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "pubmedclip_pipeline" / "pipeline.py"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in _code_cells(notebook) if marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart guard, environment reuse, idempotent Section 1 ----------------------


def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = _source(kernel[0])
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", "_isolated_environment_ready()", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source, needed


def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock([p for p in build._pins(ROOT) if "==" in p], lock_text)
    # the environment folder is keyed on the lock digest, so a second Run all reuses it
    assert "'dimer_isolated_env_' + LOCK_SHA256[:12]" in source


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker, with every variable later cells created, is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("import os; print('value', learner_value, os.environ.get('MPLBACKEND'), os.environ.get('PYTHONSTARTUP'))\n")
        assert "value 42 Agg None" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- SWP-G: the guided layer ----------------------------------------------------------------------------------------


def test_swp_g_guided_layer_is_present_and_infrastructure_is_collapsed(notebook):
    md = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**How to use this notebook.**", "**Roadmap:**", "**Input → Model → Output.**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert heading in md, heading
    assert md.count("**Predict") >= 6
    assert md.count("<details><summary>Check your reasoning</summary>") >= 6
    assert md.index("**How to use this notebook.**") < md.index("## 1. Install the pinned runtime")
    for leftover in ("{{", "}}", "{MODEL_ID}", "@P:", "TODO", "TBD"):
        assert leftover not in md, leftover
    infra = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c) or c["metadata"].get("dimer", {}).get("embedded_module") or "stage_missing_files(WEIGHTS_DIR, allow_download=True)" in _source(c)]
    assert infra and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert "> **Infrastructure.**" in md


def test_swp_g_checkpoint_answers_quote_the_recorded_run(notebook):
    md = _markdown(notebook)
    for value in ("0.182", "0.812", "0.511", "0.091", "0.284", "0.654", "0.815"):
        assert value in md, value


# --- SWP-A: quality comparisons are recorded verdicts, not asserts ---------------------------------------------------


def test_swp_a_no_model_quality_assert_remains(notebook):
    code = "\n".join(_source(c) for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = [line.strip() for line in code.splitlines() if line.strip().startswith("assert ")]
    assert asserts == ["assert parity['identical_rows'] == parity['of']"]


def _scores(acc: float, t2i: float) -> dict:
    return {"accuracy": acc, "macro_f1": acc, "t2i_map": t2i, "n": 4, "per_class": {"a": {"n": 2, "recall": acc, "ap": t2i}, "b": {"n": 2, "recall": acc, "ap": t2i}}}


def test_swp_a_section_6_records_frozen_vs_majority_without_stopping(notebook):
    source = _cell(notebook, "frozen_vs_majority = ")
    snippet = source[source.index("frozen_vs_majority = ") :]
    namespace = {"frozen_test": _scores(0.05, 0.1), "baseline_majority": _scores(0.09, 0.12)}
    exec(compile(snippet, "<section 6 verdict>", "exec"), namespace)
    assert namespace["frozen_vs_majority"] == {"accuracy": "not above", "t2i_map": "not above"}


@pytest.mark.parametrize(("adapted_map", "verdict"), [(0.2, "worse"), (0.3, "no change"), (0.6, "improved")])
def test_swp_a_section_8_records_the_verdict_and_writes_the_report(notebook, tmp_path, monkeypatch, adapted_map, verdict):
    source = _cell(notebook, "delta_map = ")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    frozen = _scores(0.2, 0.3)
    namespace = {
        "json": json, "pipe": types.SimpleNamespace(evaluate=lambda records, **kw: _scores(0.5, adapted_map)), "test_records": [], "val_records": [],
        "classes": ["a", "b"], "display_names": {}, "plain_names": {}, "CT_PROMPT": "p {label}", "METRICS": ("accuracy", "macro_f1", "t2i_map"),
        "baseline_majority": _scores(0.1, 0.1), "baseline_neighbour": _scores(0.5, 0.5), "frozen_test": frozen, "frozen_plain": frozen,
        "frozen_fields": {c: {"n": 2, "recall": 0.2, "ap": 0.3} for c in ("a", "b")}, "frozen_vs_majority": {"accuracy": "above", "t2i_map": "above"},
        "MODEL_ID": "m", "MODEL_REVISION": "r", "DEFAULT_MODEL_KEY": "k", "data_source": "stand-in", "dataset_manifests": {}, "disjoint": {},
        "adapt_result": {"history": [], "best_epoch": 0}, "adapt_seconds": 0.0,
    }
    exec(compile(source, "<section 8>", "exec"), namespace)
    assert namespace["comparison"]["verdict"]["adapted_vs_frozen_t2i_map"] == verdict
    report = json.loads((tmp_path / "outputs" / "pubmedclip_biomedical_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdict"]["adapted_vs_frozen_t2i_map"] == verdict
    assert "'verdict': comparison['verdict']," in _cell(notebook, "result_payload = {")


# --- SWP-F: every adaptation starts from the pinned base ---------------------------------------------------------------


def test_swp_f_adapt_and_load_artifact_restore_the_pinned_base_first():
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def restore_base(")]
    order = [adapt.index(m) for m in ("previous_state = {k: current[k].detach().clone() for k in self._base_state}", "restored = self.restore_base()", "_remember_base(self._base_state, model, names)", '"note": "frozen model"', "initial_state = {")]
    assert order == sorted(order)
    assert "restore.update(previous_state)" in adapt
    load = text[text.index("    def load_artifact(") :]
    assert load.index("self.restore_base()") < load.index("self.model.load_state_dict(merged, strict=True)")


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _FakeModel:
    def __init__(self):
        self.state = {"vision_model.w": _Tensor([1.0, 2.0]), "text_model.w": _Tensor([5.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state, strict=True):
        assert strict and set(state) == set(self.state)
        self.state = {k: _Tensor(v.value.copy()) for k, v in state.items()}


def test_swp_f_restore_helpers_put_trained_tensors_back():
    """The carried helpers, extracted from pipeline.py (which imports torch at module level) and run on NumPy stand-ins."""
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_remember_base", "_restore_base")]
    assert len(funcs) == 2
    namespace: dict = {"Any": object, "Mapping": dict, "Sequence": list}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "<helpers>", "exec"), namespace)
    model, base = _FakeModel(), {}
    assert namespace["_restore_base"](base, model) == []
    namespace["_remember_base"](base, model, ["vision_model.w"])
    model.state["vision_model.w"] = _Tensor([9.0, 9.0])
    namespace["_remember_base"](base, model, ["vision_model.w"])  # a second run must not overwrite the base copy
    assert namespace["_restore_base"](base, model) == ["vision_model.w"]
    assert model.state["vision_model.w"].value.tolist() == [1.0, 2.0]


# --- SWP-B: BYOD_PATH works off Colab; the upload fallback is guarded ------------------------------------------------


def _byod_block(notebook: dict) -> str:
    source = _cell(notebook, "if USE_BYOD:")
    return source[source.index("if USE_BYOD:") : source.index("else:\n    t_fetch")]


def _byod_namespace(path: str) -> dict:
    loaded = []
    return {
        "USE_BYOD": True, "BYOD_PATH": path, "Path": Path, "loaded": loaded, "SPLIT_SEED": 42,
        "load_byod_dataset": lambda p: loaded.append(Path(p)) or ["r"],
        "split_dataset": lambda records, seed: {"train": records, "validation": records, "test": records},
    }


def test_swp_b_byod_path_reads_a_folder_without_colab(notebook, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    folder = tmp_path / "my_scenes"
    folder.mkdir()
    namespace = _byod_namespace(str(folder))
    exec(compile(_byod_block(notebook), "<byod>", "exec"), namespace)
    assert namespace["loaded"] == [folder] and namespace["data_source"] == "BYOD (my_scenes)"


def test_swp_b_missing_path_and_off_colab_upload_give_clear_messages(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nope.zip' does not exist"):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace("nope.zip"))
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace(""))


@pytest.mark.parametrize(("uploaded", "message"), [(None, "Upload exactly one .zip file \\(received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2"), ({"scenes.tar": b""}, "scenes.tar: upload one .zip")])
def test_swp_b_cancelled_or_wrong_upload_is_refused(notebook, tmp_path, monkeypatch, uploaded, message):
    monkeypatch.chdir(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match=message):
        exec(compile(_byod_block(notebook), "<byod>", "exec"), _byod_namespace(""))
