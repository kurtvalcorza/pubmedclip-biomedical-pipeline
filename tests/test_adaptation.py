"""Offline tests for the labelled-image dataset contract, the pinned corpus readers, the classification /
retrieval metrics and baselines, BYOD loaders, CSV export, artifact-manifest rejections and adapt()
argument validation. Nothing here loads the checkpoint; photos come from injected fetchers and tiny PIL
drawings."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pubmedclip_pipeline import (
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    ORGANS,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    VISION_LAYERS,
    PubMedClipPipeline,
    average_precision,
    build_sample_dataset,
    check_split_disjoint,
    class_names,
    classification_metrics,
    colour_neighbour_baseline,
    colour_signature,
    dataset_digest,
    fetch_corpus,
    load_byod_dataset,
    majority_baseline,
    read_corpus,
    source_overlap,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from pubmedclip_pipeline import pipeline as pl
from pubmedclip_pipeline import samples as sm
from test_pipeline import FakeModel, FakeProcessor

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _picture(colour, size=(96, 64), mark=0):
    image = Image.new("RGB", size, "gray")
    ImageDraw.Draw(image).rectangle([16, 12, 80, 52], fill=colour)
    image.putpixel((mark % size[0], 0), (mark % 256, 0, 0))  # a per-record pixel so digests differ
    return image


def _records(n=12, prefix="r"):
    return [
        {
            "id": f"{prefix}{i:03d}",
            "image": _picture(COLOURS[i % 6], mark=i),
            "label": COLOURS[i % 6],
        }
        for i in range(n)
    ]


def _fake_pipeline():
    return PubMedClipPipeline(FakeModel(), FakeProcessor(), device="cpu")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert len(SAMPLE_RECORDS) == 660 and len(ORGANS) == 11
    labels = {r[1] for r in SAMPLE_RECORDS}
    assert labels == set(ORGANS) and all(
        sum(1 for r in SAMPLE_RECORDS if r[1] == k) == 60 for k in ORGANS
    )
    per_split = {"train": 36, "val": 8, "test": 16}
    for organ in ORGANS:
        for split, n in per_split.items():
            assert sum(1 for r in SAMPLE_RECORDS if r[1] == organ and r[2] == split) == n
    assert all(len(r[4]) == 64 and isinstance(r[3], int) and r[3] >= 0 for r in SAMPLE_RECORDS)
    assert len({(r[2], r[3]) for r in SAMPLE_RECORDS}) == 660  # distinct slices
    assert len({r[4] for r in SAMPLE_RECORDS}) == 660  # distinct pixel digests
    assert SAMPLE_SPLIT == {"train": 36, "validation": 8, "test": 16}
    assert sm.CORPUS_ARCHIVE_BYTES > 1_000_000_000 and len(sm.CORPUS_ARCHIVE_SHA256) == 64
    assert {ORGANS[k][1] for k in ORGANS} == set(range(11))


def _fake_archive(monkeypatch):
    """A small npz in the MedMNIST layout with three organs; SAMPLE_RECORDS re-pinned to it."""
    import io

    import numpy as np

    rng = np.random.default_rng(0)
    arrays, pins = {}, []
    for split, n_per in (("train", 3), ("val", 1), ("test", 2)):
        images, labels = [], []
        for organ_index, organ in enumerate(list(ORGANS)[:3]):
            for k in range(n_per):
                image = rng.integers(0, 255, size=(224, 224), dtype=np.uint8)
                index = len(images)
                images.append(image)
                labels.append([organ_index])
                pins.append(
                    (
                        f"{organ}-{split}-{k:02d}",
                        organ,
                        split,
                        index,
                        hashlib.sha256(image.tobytes()).hexdigest(),
                    )
                )
        arrays[f"{split}_images"] = np.stack(images)
        arrays[f"{split}_labels"] = np.array(labels, dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(pins))
    monkeypatch.setattr(sm, "CORPUS_ARCHIVE_BYTES", len(data))
    monkeypatch.setattr(sm, "CORPUS_ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    return data, pins


def test_fetch_corpus_pins_the_archive_and_every_slice_and_caches(tmp_path, monkeypatch):
    data, pins = _fake_archive(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        assert url == sm.CORPUS_URL
        return data

    files = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert sorted(files) == sorted(p[0] for p in pins) and len(calls) == 1
    assert all(len(v) == 224 * 224 for v in files.values())
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files and len(calls) == 1
    (tmp_path / "slices" / f"{pins[0][0]}.raw").write_bytes(b"drifted")
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files and len(calls) == 1  # archive cached
    (tmp_path / "organamnist_224.npz").write_bytes(b"drifted")
    (tmp_path / "slices" / f"{pins[1][0]}.raw").unlink()
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files and len(calls) == 2  # re-fetched
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "bad", fetcher=lambda url: b"tampered")
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple((*p[:4], "0" * 64) for p in pins))
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "bad-slice", fetcher=fetcher)
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(pins))
    records = read_corpus(files)
    assert [r["label"] for r in records] == [p[1] for p in pins]
    assert all(r["image"].mode == "RGB" and r["image"].size == (224, 224) for r in records)
    assert all("medmnist_split" in r and r["source"].startswith("organamnist_224/") for r in records)


def test_build_sample_dataset_keeps_the_official_roles(monkeypatch):
    data, pins = _fake_archive(monkeypatch)
    records = read_corpus(fetch_corpus(cache_dir=None, fetcher=lambda url: data)) if False else None
    # records straight from the pins (no archive needed for the role logic)
    records = [
        {
            "id": rid,
            "image": _picture(COLOURS[i % 6], mark=i),
            "label": organ,
            "medmnist_split": split,
            "medmnist_index": index,
        }
        for i, (rid, organ, split, index, _d) in enumerate(pins)
    ]
    sizes = {"train": 3, "validation": 1, "test": 2}
    splits = build_sample_dataset(records, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 9, "validation": 3, "test": 6}
    assert class_names(splits["train"]) == sorted(list(ORGANS)[:3])
    assert all(r["medmnist_split"] == "val" for r in splits["validation"])
    assert all(r["medmnist_split"] == "test" for r in splits["test"])
    assert splits["train"][0]["id"] == "train-000" and "source_id" in splits["train"][0]
    assert build_sample_dataset(records, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(records, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="expected 4"):
        build_sample_dataset(records, sizes={"train": 4, "validation": 1, "test": 2})
    with pytest.raises(ValueError, match="medmnist_split"):
        build_sample_dataset([{**records[0], "medmnist_split": "other"}], sizes=sizes)
    overlap = source_overlap(splits)
    assert overlap["roles_mixing_splits"] == 0
    assert overlap["medmnist_splits_per_role"] == {"train": ["train"], "validation": ["val"], "test": ["test"]}


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path):
    records = _records()
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["classes"] == sorted(COLOURS)
    assert report["label_counts"] == {c: 2 for c in COLOURS} and report["image_side"] == {
        "min": 96,
        "max": 96,
    }
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    path = tmp_path / "photo.jpg"
    _picture("red").save(path)
    assert validate_dataset([{**r, "image": str(path)} for r in records])["n_records"] == 12
    good = records
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "image": str(tmp_path / "missing.jpg")}, *good[1:]], "not found"),
        ([{**good[0], "image": Image.new("RGB", (5000, 10))}, *good[1:]], "MAX_IMAGE_SIDE"),
        ([{**good[0], "label": ""}, *good[1:]], "label must be"),
        ([{**r, "label": "same"} for r in good], "distinct labels"),
        ([{k: v for k, v in good[0].items() if k != "label"}, *good[1:]], "missing 'label'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_is_stratified_seeded_and_deduplicated():
    records = _records(n=30)
    records += [{**r, "id": r["id"] + "dup"} for r in records[:6]]  # pixel-identical duplicates
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 30 and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": [{**records[0], "id": "x"}]})


# --- metrics and baselines ----------------------------------------------------------------------------


def test_classification_metrics_and_average_precision():
    classes = ["a", "b", "c"]
    gold = [0, 1, 2, 0]
    perfect = np.eye(3)[gold]
    m = classification_metrics(perfect, gold, classes)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["t2i_map"] == 1.0 and m["n"] == 4
    assert m["per_class"]["a"]["n"] == 2 and m["per_class"]["a"]["ap"] == 1.0
    wrong = np.roll(perfect, 1, axis=1)
    w = classification_metrics(wrong, gold, classes)
    assert w["accuracy"] == 0.0 and w["macro_f1"] == 0.0
    assert average_precision([0.9, 0.1, 0.5], [True, False, True]) == pytest.approx(1.0)
    assert average_precision([0.1, 0.9, 0.5], [True, False, False]) == pytest.approx(1 / 3)
    assert average_precision([0.1, 0.9], [False, False]) == 0.0
    with pytest.raises(ValueError, match="grid"):
        classification_metrics(perfect[:, :2], gold, classes)
    with pytest.raises(ValueError, match="finite"):
        classification_metrics(np.full_like(perfect, np.nan), gold, classes)
    with pytest.raises(ValueError, match="parallel"):
        average_precision([0.1], [True, False])


def test_baselines_need_training_records_and_sit_near_chance():
    records = _records(n=24)
    classes = class_names(records)
    majority = majority_baseline(records[:12], records[12:], classes)
    assert majority["accuracy"] == pytest.approx(1 / 6) and "majority floor" in majority["baseline"]
    assert len(colour_signature(records[0]["image"])) == 27
    neighbour = colour_neighbour_baseline(records[:12], records[12:], classes)
    assert (
        neighbour["accuracy"] == 1.0
    )  # every test drawing has an identical-colour twin in training
    with pytest.raises(ValueError, match="training records"):
        majority_baseline([], records, classes)
    with pytest.raises(ValueError, match="training records"):
        colour_neighbour_baseline([], records, classes)


# --- BYOD loaders and CSV -----------------------------------------------------------------------------


def test_byod_directory_zip_round_trip_and_rejections(tmp_path):
    records = _records(n=8)
    for r in records:
        r["image"].save(
            tmp_path / r["id"], format="JPEG"
        )  # `write_dataset_csv` names BYOD files by id
    write_dataset_csv(records, tmp_path / "labels.csv")
    loaded = load_byod_dataset(tmp_path)
    assert [r["label"] for r in loaded] == [r["label"] for r in records]
    with zipfile.ZipFile(tmp_path / "byod.zip", "w") as archive:
        archive.write(tmp_path / "labels.csv", "labels.csv")
        for r in records:
            archive.write(tmp_path / r["id"], r["id"])
    assert [r["id"] for r in load_byod_dataset(tmp_path / "byod.zip")] == [r["id"] for r in records]
    with zipfile.ZipFile(tmp_path / "nolabels.zip", "w") as archive:
        archive.writestr("x.txt", "x")
    with pytest.raises(ValueError, match="labels.csv"):
        load_byod_dataset(tmp_path / "nolabels.zip")
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "labels.csv")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_validate_arguments_before_touching_the_model(tmp_path):
    pipe = _fake_pipeline()
    records = _records()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="trainable_vision_layers"):
        pipe.adapt(records, trainable_vision_layers=VISION_LAYERS + 1)
    with pytest.raises(ValueError, match="placeholder"):
        pipe.evaluate(records, prompt_template="no placeholder")
    with pytest.raises(ValueError, match="outside classes"):
        pipe.evaluate(records, classes=["red"])
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["vision_model.head.probe"],
        "adapter": {"trainable_vision_layers": 1},
    }

    def write(m):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(m))

    write({**manifest, "format": "other"})
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "base_model": {**manifest["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "files": manifest["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write(
        {**manifest, "files": [{**manifest["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]}
    )
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "adapter": {"trainable_vision_layers": VISION_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_vision_layers"):
        pipe.load_artifact(tmp_path)
    write(manifest)
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
