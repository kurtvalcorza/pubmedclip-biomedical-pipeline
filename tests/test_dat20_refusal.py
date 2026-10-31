"""DAT19/DAT20: a refused BYOD input renders one `Refused: <file>: <reason>` line, no traceback.

Executes the committed Section 4 BYOD block with the real loader (or stubs). Covers the scripted
hosted refusal input (`BYOD_PATH='/etc/hostname'`, a file that is not a zip), a directory without
labels.csv, a missing path, an empty path outside Colab, a split refusal, the in-cell invalidation,
and that a genuine error keeps its own type and traceback.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pubmedclip_pipeline.samples import load_byod_dataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "pubmedclip_biomedical_colab.ipynb"
REFUSAL_CELL = "pubmedclip_biomedical-24"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _byod_cell() -> dict:
    found = [
        c for c in _cells() if c["cell_type"] == "code" and "if USE_BYOD:" in "".join(c["source"])
    ]
    assert len(found) == 1
    return found[0]


def _block() -> str:
    source = "".join(_byod_cell()["source"])
    return source[source.index("class BYODRefusal(") : source.index("else:\n    t_fetch")]


def _run(path: str, loader=load_byod_dataset, splitter=None):
    namespace = {
        "USE_BYOD": True,
        "BYOD_PATH": path,
        "Path": Path,
        "SPLIT_SEED": 42,
        "os": os,
        "json": json,
        "validate_dataset": lambda part, min_records=1: None,
        "load_byod_dataset": loader,
        "byod_split_mode": lambda records: "stratified shuffle",
        "split_dataset": splitter
        or (lambda records, seed: {"train": records, "validation": records, "test": records}),
        "splits": {"train": ["stale"]},
        "train_records": ["stale"],
    }
    exec(compile(_block(), "<byod>", "exec"), namespace)
    return namespace


def _refused(path: str, **kwargs):
    with pytest.raises(ValueError) as caught:
        _run(path, **kwargs)
    exc = caught.value
    assert getattr(exc, "dimer_refusal", False), exc
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "pubmedclip_biomedical_result.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    return tmp_path


def test_scripted_hosted_input_a_file_that_is_not_a_zip(workdir):
    hostname = workdir / "hostname"
    hostname.write_text("colab-vm\n")
    line = _refused(str(hostname))
    assert line.startswith("Refused: hostname: hostname is not a zip archive"), line
    assert "BYOD: " not in line


def test_a_directory_without_labels_csv(workdir):
    folder = workdir / "etc"
    folder.mkdir()
    assert _refused(str(folder)).startswith("Refused: etc: ")


def test_missing_path_and_empty_path_off_colab(workdir):
    assert _refused("nope.zip").startswith("Refused: nope.zip: BYOD_PATH does not exist")
    assert _refused("").startswith("Refused: (no file): USE_BYOD is True but BYOD_PATH is empty")


def test_split_refusal_names_the_zip(workdir):
    def too_small(records, seed):
        raise ValueError("split leaves 2 training records; at least 12 are required")

    folder = workdir / "tiny"
    folder.mkdir()
    line = _refused(str(folder), loader=lambda p: ["r"], splitter=too_small)
    assert line.startswith("Refused: tiny: split leaves 2 training records")


def test_refusal_invalidates_earlier_results_in_the_cell(workdir):
    with pytest.raises(ValueError):
        _run("nope.zip")
    marker = json.loads(
        (workdir / "outputs" / "pubmedclip_biomedical_run_invalidated.json").read_text()
    )
    assert marker["files_from_an_earlier_run"] == ["pubmedclip_biomedical_result.json"]
    assert (workdir / "outputs" / "pubmedclip_biomedical_result.json").exists()


def test_genuine_error_keeps_its_type(workdir):
    folder = workdir / "ok"
    folder.mkdir()

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(str(folder), loader=broken)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_refusal_cell_id():
    assert _byod_cell()["id"] == REFUSAL_CELL
