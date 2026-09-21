"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a zero-shot
evaluation with the per-class breakdown, a one-epoch adaptation of the last vision block on a dozen drawn
scenes, the artifact round trip, the loader's scope check, the transactional guarantee and — where CUDA is
visible — the same path on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from pubmedclip_pipeline import DEFAULT_WEIGHTS_DIR, PubMedClipPipeline
from pubmedclip_pipeline.config import MODEL_FILENAME

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / MODEL_FILENAME).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow"]


def _scene(colour, shape, mark):
    image = Image.new("RGB", (256, 192), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 130, 256, 192], fill=(60, 179, 75))
    if shape == "circle":
        draw.ellipse([80, 40, 176, 136], fill=colour)
    else:
        draw.rectangle([80, 40, 176, 136], fill=colour)
    image.putpixel((mark, 0), (mark, 0, 0))
    return image


@pytest.fixture(scope="module")
def records():
    return [
        {
            "id": f"s{i:02d}",
            "image": _scene(COLOURS[i % 4], "circle" if i % 2 else "square", i),
            "label": f"{COLOURS[i % 4]} {'circle' if i % 2 else 'square'}",
        }
        for i in range(16)
    ]


@pytest.fixture(scope="module")
def pipe():
    return PubMedClipPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_evaluate_scores_zero_shot_with_the_models_logits(pipe, records):
    metrics = pipe.evaluate(records[:8])
    assert metrics["n"] == 8 and 0.0 <= metrics["accuracy"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics["per_class"]) == set(metrics["classes"]) and len(metrics["classes"]) == 4
    assert metrics["verdict"] == "measured-small-sample" and "t2i_map" in metrics
    # the evaluate() score grid reproduces zero_shot_classify's per-image ranking
    scores = pipe.zero_shot_classify(records[0]["image"], metrics["classes"])
    assert scores[0].label == metrics["predictions"][0] and len(metrics["predictions"]) == 8


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(
        records[:12], records[12:], epochs=1, trainable_vision_layers=1, batch_size=4
    )
    assert (
        result["n_trainable"] == 7_482_624
    )  # last vision block + post-layernorm + visual projection
    assert (
        result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    )
    assert set(result["history"][1]["val"]) == {"accuracy", "macro_f1", "t2i_map", "n"}
    assert all(
        name.startswith(
            (
                "vision_model.encoder.layers.11.",
                "vision_model.post_layernorm.",
                "visual_projection.",
            )
        )
        for name in result["trainable_names"]
    )
    assert "logit_scale" not in result["trainable_names"] and not any(
        n.startswith("text_model.") for n in result["trainable_names"]
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert (
        len(manifest["tensors"]) == len(result["trainable_names"])
        and manifest["adapter"]["classes"] == result["classes"]
    )
    reloaded = PubMedClipPipeline.from_artifact(
        artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR
    )
    a = pipe.embed_image([r["image"] for r in records[:4]])
    b = reloaded.embed_image([r["image"] for r in records[:4]])
    assert np.abs(a - b).max() < 1e-6 and reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe.model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, trainable_vision_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith(
        "final epoch"
    )
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = PubMedClipPipeline.from_artifact(
        artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR
    )
    state, other = pipe.model.state_dict(), reloaded.model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["trainable_vision_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(
    pipe, records, tmp_path
):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, trainable_vision_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(
        json.dumps({**manifest, "tensors": manifest["tensors"][:-1]})
    )
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        PubMedClipPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {
            **manifest["files"][0],
            "bytes": (extra / "adapter.safetensors").stat().st_size,
            "sha256": digest,
        }
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        PubMedClipPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    (other_layers / "manifest.json").write_text(
        json.dumps({**manifest, "adapter": {**manifest["adapter"], "trainable_vision_layers": 2}})
    )
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        PubMedClipPipeline.from_artifact(other_layers, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(
            records[:12], None, epochs=2, trainable_vision_layers=1, batch_size=4, progress=boom
        )
    after = pipe.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the encoders and the trainer build must land on the model's device."""
    cuda = PubMedClipPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert str(cuda.device) == "cuda:0"
    metrics = cuda.evaluate(records[:8])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    result = cuda.adapt(
        records[:12], records[12:], epochs=1, trainable_vision_layers=1, batch_size=4
    )
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = PubMedClipPipeline.from_artifact(
        artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR
    )
    a = cuda.embed_image([r["image"] for r in records[:4]])
    b = reloaded.embed_image([r["image"] for r in records[:4]])
    assert np.abs(a - b).max() < 1e-5
