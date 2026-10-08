"""Regression tests for the Notebook Review Framework v1 findings on pubmedclip_biomedical_colab (PMC-M1..m6).

The review is docs/reviews/2026-10-02-notebook-review/pubmedclip_biomedical_colab_Review.md. M1 (isolated uv runtime)
and the guided layer of M4 came with the 2026-10-05 sweep commit and are tested in test_sweep_fixes.py; the worker's
google.colab stubs are tested in test_worker_colab_stubs.py.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from pubmedclip_pipeline import (
    BYOD_MIN_RECORDS,
    DEFAULT_PROMPT_TEMPLATE,
    byod_split_mode,
    load_byod_dataset,
    split_dataset,
)
from pubmedclip_pipeline.metrics import average_precision, majority_baseline

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "pubmedclip_biomedical_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_module")]


def _cell(notebook: dict, marker: str) -> str:
    found = [s for s in _code(notebook) if marker in s]
    assert len(found) == 1, marker
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _png(colour: tuple[int, int, int], mark: int) -> bytes:
    image = Image.new("RGB", (16, 16), colour)
    image.putpixel((mark % 16, mark // 16), (mark, 255 - mark, 7))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(path: Path, rows: list[dict], files: dict[str, bytes], *, table: str = "labels.csv") -> Path:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(table, out.getvalue())
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def _dataset(n: int, *, folder: str = "", groups: int = 0, split: bool = False) -> tuple[list[dict], dict[str, bytes]]:
    rows, files = [], {}
    for i in range(n):
        name = f"{folder}img{i:03d}.png"
        files[name] = _png((200, 20, 20) if i % 2 else (20, 20, 200), i)
        row = {"id": f"r{i:03d}", "file": name, "label": "left" if i % 2 else "right"}
        if groups:
            row["group"] = f"patient{i % groups}"
        if split:
            row["split"] = "test" if i < 4 else ("validation" if i < 6 else "train")
        rows.append(row)
    return rows, files


# --- PMC-m6: the majority floor's mAP is the prevalence, independent of record order -------------------------------


def test_pmc_m6_all_tie_column_scores_its_prevalence_whatever_the_order():
    assert average_precision([1.0] * 11, [True] + [False] * 10) == pytest.approx(1 / 11)
    assert average_precision([1.0] * 11, [False] * 10 + [True]) == pytest.approx(1 / 11)
    # without ties the value is the usual precision-at-each-hit mean
    assert average_precision([0.9, 0.1, 0.5], [True, False, True]) == pytest.approx(1.0)
    assert average_precision([0.1, 0.9, 0.5], [True, False, False]) == pytest.approx(1 / 3)
    # a partial tie is one threshold: ranks 1-2 tie (one hit), rank 3 hit -> (1 * 1/2 + 1 * 2/3) / 2
    assert average_precision([0.5, 0.5, 0.2], [False, True, True]) == pytest.approx((0.5 + 2 / 3) / 2)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_pmc_m6_majority_floor_map_equals_mean_prevalence(seed):
    import random

    classes = [f"c{k}" for k in range(11)]
    records = [{"label": c} for c in classes for _ in range(16)]
    random.Random(seed).shuffle(records)
    result = majority_baseline(records, records, classes)
    assert result["t2i_map"] == pytest.approx(1 / 11)


# --- PMC-m2: the stated minimum, nested paths, collisions and actionable errors -----------------------------------


def test_pmc_m2_byod_minimum_is_twelve_and_eight_is_refused_with_the_fix(tmp_path):
    assert BYOD_MIN_RECORDS == 12
    rows, files = _dataset(8)
    with pytest.raises(ValueError, match="at least 12"):
        split_dataset(load_byod_dataset(_zip(tmp_path / "eight.zip", rows, files)), seed=42)
    rows, files = _dataset(12)
    splits = split_dataset(load_byod_dataset(_zip(tmp_path / "twelve.zip", rows, files)), seed=42)
    assert len(splits["train"]) == 8


def test_pmc_m2_nested_paths_load_and_a_bare_unique_name_still_works(tmp_path):
    rows, files = _dataset(12, folder="images/")
    loaded = load_byod_dataset(_zip(tmp_path / "nested.zip", rows, files))
    assert len(loaded) == 12
    for row in rows:
        row["file"] = row["file"].split("/")[-1]  # bare name, unique in the zip
    assert len(load_byod_dataset(_zip(tmp_path / "bare.zip", rows, files))) == 12
    # labels.csv inside a top folder: paths are relative to it
    rows, files = _dataset(12, folder="images/")
    nested = {f"mydata/{k}": v for k, v in files.items()}
    assert len(load_byod_dataset(_zip(tmp_path / "top.zip", rows, nested, table="mydata/labels.csv"))) == 12


def test_pmc_m2_colliding_basenames_are_never_resolved_silently(tmp_path):
    rows, files = _dataset(12, folder="a/")
    files["b/img000.png"] = _png((0, 255, 0), 99)
    rows[0]["file"] = "img000.png"  # ambiguous: a/img000.png and b/img000.png
    with pytest.raises(ValueError, match="matches 2 files"):
        load_byod_dataset(_zip(tmp_path / "clash.zip", rows, files))
    rows[0]["file"] = "b/img000.png"  # the path picks the right one
    loaded = load_byod_dataset(_zip(tmp_path / "path.zip", rows, files))
    assert loaded[0]["image"].getpixel((99 % 16, 99 // 16)) == (99, 156, 7)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "is not in the dataset"),
        ("not_image", "not an image Pillow can decode"),
        ("escape", "must be a path inside the dataset"),
        ("no_labels", "exactly one labels.csv"),
    ],
)
def test_pmc_m2_loader_failures_are_value_errors_naming_file_and_fix(tmp_path, change, message):
    rows, files = _dataset(12)
    if change == "missing":
        rows[3]["file"] = "nope.png"
    elif change == "not_image":
        files[rows[3]["file"]] = b"not a png"
    elif change == "escape":
        rows[3]["file"] = "../img003.png"
    path = _zip(tmp_path / "x.zip", rows, files, table="other.csv" if change == "no_labels" else "labels.csv")
    with pytest.raises(ValueError, match=message) as info:
        load_byod_dataset(path)
    assert str(info.value).startswith("BYOD:")


def test_pmc_m2_a_non_zip_file_is_a_value_error(tmp_path):
    bogus = tmp_path / "upload.zip"
    bogus.write_bytes(b"plain text")
    with pytest.raises(ValueError, match="upload.zip is not a zip archive"):
        load_byod_dataset(bogus)


# --- PMC-m4: group and split columns -------------------------------------------------------------------------------


def test_pmc_m4_group_column_keeps_every_group_in_one_role(tmp_path):
    rows, files = _dataset(40, groups=8)
    records = load_byod_dataset(_zip(tmp_path / "groups.zip", rows, files))
    assert byod_split_mode(records) == "group column"
    splits = split_dataset(records, seed=42)
    roles = {name: {r["group"] for r in part} for name, part in splits.items()}
    assert all(roles.values())
    assert not (roles["train"] & roles["test"] or roles["train"] & roles["validation"] or roles["validation"] & roles["test"])
    assert sum(len(p) for p in splits.values()) == 40


def test_pmc_m4_split_column_is_used_as_given_and_plain_sets_stay_stratified(tmp_path):
    rows, files = _dataset(14, split=True)
    records = load_byod_dataset(_zip(tmp_path / "split.zip", rows, files))
    assert byod_split_mode(records) == "split column"
    splits = split_dataset(records, seed=42)
    assert sorted(r["id"] for r in splits["test"]) == ["r000", "r001", "r002", "r003"]
    assert len(splits["validation"]) == 2 and len(splits["train"]) == 8
    rows, files = _dataset(12)
    assert byod_split_mode(load_byod_dataset(_zip(tmp_path / "plain.zip", rows, files))) == "stratified shuffle"
    with pytest.raises(ValueError, match="at least 3"):
        rows, files = _dataset(12, groups=2)
        split_dataset(load_byod_dataset(_zip(tmp_path / "two.zip", rows, files)), seed=0)


def test_pmc_m4_section_4_prints_the_mode_and_asserts_group_disjointness(notebook):
    source = _cell(notebook, "split_mode = byod_split_mode(records)")
    assert "'split_mode': split_mode" in source and "a group is in two roles" in source


# --- PMC-M2: every frozen measurement starts from the pinned base --------------------------------------------------


def test_pmc_m2_sections_5_and_6_restore_the_pinned_base_before_scoring(notebook):
    s5 = _cell(notebook, "SAMPLE_DIGESTS = {")
    assert s5.index("pipe.restore_base()") < s5.index("pipe.zero_shot_classify(")
    s6 = _cell(notebook, "frozen_test = pipe.evaluate(")
    assert s6.index("restored = pipe.restore_base()") < s6.index("frozen_test = pipe.evaluate(")
    md = _markdown(notebook)
    assert "Scope of a re-run" in md and "run Sections 7, 8 and 9" in md


WEIGHTS = Path(os.environ.get("PUBMEDCLIP_WEIGHTS_DIR", str(ROOT / "weights" / "pubmed-clip-vit-base-patch32")))


@pytest.mark.skipif(not (WEIGHTS / "model.safetensors").is_file(), reason="pinned snapshot not staged")
def test_pmc_m2_rerun_after_adaptation_reproduces_the_frozen_scores(tmp_path):
    """Acceptance check of PMC-M2 with the real pinned weights on drawn stand-in records (CPU)."""
    from pubmedclip_pipeline import PubMedClipPipeline

    colours = [(220, 30, 30), (30, 30, 220)]
    records = []
    for i in range(16):
        image = Image.new("RGB", (64, 64), (240, 240, 240))
        image.paste(colours[i % 2], (8 + i % 4, 8, 40, 40))
        records.append({"id": f"s{i:02d}", "image": image, "label": "red square" if i % 2 == 0 else "blue square"})
    train, val, test = records[:8], records[8:12], records[12:]
    pipe = PubMedClipPipeline.from_pretrained(device="cpu", weights_dir=WEIGHTS)
    frozen = pipe.evaluate(test, prompt_template=DEFAULT_PROMPT_TEMPLATE)
    first = pipe.adapt(train, val, epochs=1, lr=1e-4, batch_size=4, trainable_vision_layers=1)
    assert pipe.evaluate(test)["adapted"] is True
    assert pipe.restore_base()  # what Sections 5 and 6 now do first
    again = pipe.evaluate(test, prompt_template=DEFAULT_PROMPT_TEMPLATE)
    assert again["adapted"] is False and again["t2i_map"] == frozen["t2i_map"] and again["accuracy"] == frozen["accuracy"]
    second = pipe.adapt(train, val, epochs=1, lr=1e-4, batch_size=4, trainable_vision_layers=2)
    assert second["history"][0]["note"] == "frozen model"
    assert second["history"][0]["val"]["t2i_map"] == first["history"][0]["val"]["t2i_map"]


# --- PMC-M3: the quality asserts guard the sample only -------------------------------------------------------------


def test_pmc_m3_byod_below_the_floor_continues_and_the_sample_still_asserts(notebook, capsys):
    source = _cell(notebook, "frozen_vs_majority = ")
    snippet = source[source.index("frozen_vs_majority = ") :]
    below = {"accuracy": 0.4, "t2i_map": 0.5}
    floor = {"accuracy": 0.5, "t2i_map": 0.5}
    exec(compile(snippet, "<section 6>", "exec"), {"frozen_test": below, "baseline_majority": floor, "USE_BYOD": True})
    assert "That is a result, not an error" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="pinned sample"):
        exec(compile(snippet, "<section 6>", "exec"), {"frozen_test": below, "baseline_majority": floor, "USE_BYOD": False})
    s8 = _cell(notebook, "delta_map = ")
    assert "if not USE_BYOD:" in s8 and "assert delta_map > 0" in s8


# --- PMC-m1: the BYOD prompt is the documented template ------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_byod", "form", "expected"),
    [
        (False, "", "An axial abdominal CT slice showing the {label}."),
        (True, "", DEFAULT_PROMPT_TEMPLATE),
        (True, "A chest X-ray showing {label}.", "A chest X-ray showing {label}."),
    ],
)
def test_pmc_m1_prompt_template_follows_the_data_source(notebook, use_byod, form, expected):
    source = _cell(notebook, "PROMPT_TEMPLATE = CT_PROMPT")
    head = source[: source.index("# PMC-M2")]
    namespace = {"USE_BYOD": use_byod, "BYOD_PROMPT_TEMPLATE": form, "DEFAULT_PROMPT_TEMPLATE": DEFAULT_PROMPT_TEMPLATE}
    exec(compile(head, "<section 6 prompt>", "exec"), namespace)
    assert namespace["PROMPT_TEMPLATE"] == expected
    with pytest.raises(ValueError, match="must contain"):
        exec(compile(head, "<section 6 prompt>", "exec"), {**namespace, "USE_BYOD": True, "BYOD_PROMPT_TEMPLATE": "no slot"})
    assert "'prompt_template': PROMPT_TEMPLATE," in _cell(notebook, "evaluation_report_payload = {")
    assert "BYOD_PROMPT_TEMPLATE = ''" in _cell(notebook, "USE_BYOD = False")


# --- PMC-m5: the right lung's AP drop is named and computed --------------------------------------------------------


def test_pmc_m5_section_8_names_the_right_lung_and_lists_organs_that_fell(notebook):
    md = _markdown(notebook)
    assert "right lung's AP fell from 0.79 to 0.34" in md and "`ap_fell`" in md
    source = _cell(notebook, "delta_map = ")
    assert "comparison['ap_fell'] = sorted(" in source and "comparison['recall_fell'] = sorted(" in source


def test_pmc_s1_declares_notebook_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
