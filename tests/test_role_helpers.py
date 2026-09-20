"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import pytest
from PIL import Image

from pubmedclip_pipeline import (
    DEFAULT_PROMPT_TEMPLATE,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    ClassificationScore,
    PubMedClipPipeline,
    RetrievalHit,
    evaluation_report,
    recall_at_1,
    top1_accuracy,
    validate_inputs,
)
from pubmedclip_pipeline.config import TEXT_MAX_LENGTH
from test_pipeline import FakeModel, FakeProcessor


def _image(side: int = 8) -> Image.Image:
    return Image.new("RGB", (side, side), (120, 30, 10))


def _pipeline() -> PubMedClipPipeline:
    return PubMedClipPipeline(FakeModel(), FakeProcessor(), device="cpu")


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(
        [_image(), _image(16)], ["red square", "green circle"], top_k=2, names=["a", "b"]
    )
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_max_length"] == TEXT_MAX_LENGTH
    assert manifest["schema"]["prompt_template"] == DEFAULT_PROMPT_TEMPLATE
    assert manifest["inputs"] == [
        {"id": "a", "mode": "RGB", "size": [8, 8]},
        {"id": "b", "mode": "RGB", "size": [16, 16]},
    ]
    assert manifest["texts"] == ["red square", "green circle"]
    assert manifest["top_k"] == 2
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_ids_and_texts_only() -> None:
    assert [e["id"] for e in validate_inputs(_image())["inputs"]] == ["image-0"]
    texts_only = validate_inputs(None, ["a query"])
    assert texts_only["inputs"] == [] and texts_only["texts"] == ["a query"]


def test_validate_inputs_rejects_like_the_core_methods() -> None:
    pipe = _pipeline()
    with pytest.raises(ValueError, match="Remote image URLs are not accepted") as via_helper:
        validate_inputs("https://example.com/x.png", ["a"])
    with pytest.raises(ValueError) as via_core:
        pipe.zero_shot_classify("https://example.com/x.png", ["a"])
    assert str(via_helper.value) == str(via_core.value)

    with pytest.raises(
        ValueError, match="labels must contain at least one non-empty string"
    ) as via_helper:
        validate_inputs(_image(), [])
    with pytest.raises(ValueError) as via_core:
        pipe.zero_shot_classify(_image(), [])
    assert str(via_helper.value) == str(via_core.value)

    with pytest.raises(ValueError, match="prompt_template must contain the literal") as via_helper:
        validate_inputs(_image(), ["a"], prompt_template="no placeholder")
    with pytest.raises(ValueError) as via_core:
        pipe.zero_shot_classify(_image(), ["a"], prompt_template="no placeholder")
    assert str(via_helper.value) == str(via_core.value)

    with pytest.raises(ValueError, match="top_k must be >= 1") as via_helper:
        validate_inputs(_image(), ["a"], top_k=0)
    with pytest.raises(ValueError) as via_core:
        pipe.retrieve("a", [_image()], top_k=0)
    assert str(via_helper.value) == str(via_core.value)

    with pytest.raises(ValueError, match="images must contain at least one image") as via_helper:
        validate_inputs([])
    with pytest.raises(ValueError) as via_core:
        pipe.embed_image([])
    assert str(via_helper.value) == str(via_core.value)

    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs([_image()], names=["a", "b"])


def _scores(*labels: str) -> list[ClassificationScore]:
    return [ClassificationScore(label=label, score=1.0 - 0.1 * i) for i, label in enumerate(labels)]


def _hits(*indices: int) -> list[RetrievalHit]:
    return [RetrievalHit(index=i, score=1.0 - 0.1 * n) for n, i in enumerate(indices)]


def test_top1_accuracy_and_recall_at_1_are_the_tutorial_sanity_metrics() -> None:
    assert top1_accuracy([_scores("a", "b"), _scores("b", "a")], ["a", "a"]) == 0.5
    assert recall_at_1([_hits(0, 1), _hits(1, 0), _hits(2, 0)], [0, 1, 0]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="same length"):
        top1_accuracy([_scores("a")], ["a", "b"])
    with pytest.raises(ValueError, match="must not be empty"):
        recall_at_1([], [])


def test_evaluation_report_not_measurable_without_targets() -> None:
    report = evaluation_report(
        {"classifications": [_scores("a", "b")], "embedding_shapes": {"image": [1, 768]}}
    )
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert "top1_accuracy" in report["needs"] and "recall_at_1" in report["needs"]
    assert "representations, not predictions" in report["embeddings"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_labels_and_retrieval_indices() -> None:
    result = {
        "classifications": [
            _scores("red", "green", "blue", "other"),
            _scores("green", "red", "blue", "other"),
            _scores("red", "blue", "green", "other"),
        ],
        "retrievals": [_hits(0, 1, 2), _hits(1, 0, 2), _hits(2, 0, 1)],
        "gallery_ids": ["r", "g", "b"],
    }
    report = evaluation_report(
        result,
        {"labels": ["red", "green", "blue"], "retrieval_indices": [0, 1, 2]},
        sample_kind="synthetic",
    )
    assert report["verdict"] == "sample-sanity"
    by_id = {m["id"]: m for m in report["metrics"]}
    assert by_id["top1_accuracy"]["value"] == pytest.approx(2 / 3)
    assert by_id["top1_accuracy"]["n_candidate_labels"] == 4
    assert by_id["recall_at_1"]["value"] == 1.0
    assert by_id["recall_at_1"]["n_gallery_images"] == 3
    assert report["baselines"] == [
        {
            "id": "fixed_class_baseline",
            "metric": "top1_accuracy",
            "value": 0.25,
            "note": "always predicting one fixed candidate label",
        }
    ]
    assert all(m["estimation"] for m in report["metrics"])


def test_evaluation_report_labels_only_or_indices_only() -> None:
    only_labels = evaluation_report({"classifications": [_scores("a", "b")]}, {"labels": ["a"]})
    assert [m["id"] for m in only_labels["metrics"]] == ["top1_accuracy"]
    only_retrieval = evaluation_report({"retrievals": [_hits(0, 1)]}, {"retrieval_indices": [0]})
    assert [m["id"] for m in only_retrieval["metrics"]] == ["recall_at_1"]
    assert only_retrieval["baselines"] == []
