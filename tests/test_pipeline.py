from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image

from pubmedclip_pipeline import PubMedClipPipeline


class FakeProcessor:
    def __init__(self) -> None:
        self.last_text = None

    def __call__(self, *, text=None, images=None, **kwargs):
        self.last_text = text
        batch = {}
        if images is not None:
            batch["pixel_values"] = torch.ones((len(images), 1), dtype=torch.float32)
        if text is not None:
            batch["input_ids"] = torch.arange(len(text), dtype=torch.float32).unsqueeze(1)
        return batch


class FakeOutput:
    def __init__(self, logits_per_image: torch.Tensor) -> None:
        self.logits_per_image = logits_per_image


class FakeModel:
    def __call__(self, **batch):
        count = batch["input_ids"].shape[0]
        logits = torch.tensor([[0.0, 2.0, -2.0]], dtype=torch.float32)[:, :count]
        return FakeOutput(logits)

    def get_image_features(self, **batch):
        count = batch["pixel_values"].shape[0]
        bank = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
        return bank[:count]

    def get_text_features(self, **batch):
        count = batch["input_ids"].shape[0]
        bank = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
        return bank[:count]


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), (120, 30, 10))


def _pipeline() -> PubMedClipPipeline:
    return PubMedClipPipeline(FakeModel(), FakeProcessor(), device="cpu")


def test_zero_shot_classification_is_a_softmax_over_the_candidate_set():
    result = _pipeline().zero_shot_classify(_image(), ["cat", "dog", "road"])

    assert [item.label for item in result] == ["dog", "cat", "road"]
    expected = torch.softmax(torch.tensor([0.0, 2.0, -2.0]), dim=-1)
    assert result[0].score == pytest.approx(expected[1].item())
    assert math.isclose(sum(item.score for item in result), 1.0, abs_tol=1e-6)


def test_clip_text_is_lowercased_without_changing_returned_label():
    processor = FakeProcessor()
    pipe = PubMedClipPipeline(FakeModel(), processor, device="cpu")

    result = pipe.zero_shot_classify(_image(), ["Flooded Street"])
    assert processor.last_text == ["a medical image of flooded street."]
    assert result[0].label == "Flooded Street"

    pipe.embed_text(["Mixed CASE Query"])
    assert processor.last_text == ["mixed case query"]


def test_prompt_template_requires_label_placeholder():
    with pytest.raises(ValueError, match="literal .*label.* placeholder"):
        _pipeline().zero_shot_classify(_image(), ["cat"], prompt_template="A photo.")


def test_remote_url_is_rejected():
    with pytest.raises(ValueError, match="Remote image URLs"):
        _pipeline().embed_image(["https://example.test/image.jpg"])


def test_embeddings_are_l2_normalized():
    image_embeddings = _pipeline().embed_image([_image(), _image()])
    text_embeddings = _pipeline().embed_text(["alpha", "beta"])

    assert image_embeddings.dtype == np.float32
    assert text_embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(image_embeddings, axis=1), np.ones(2))
    np.testing.assert_allclose(np.linalg.norm(text_embeddings, axis=1), np.ones(2))


def test_similarity_and_retrieval_use_cosine_space():
    pipe = _pipeline()
    matrix = pipe.similarity([_image(), _image()], ["alpha", "beta"])

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)

    hits = pipe.retrieve("alpha", [_image(), _image()], top_k=2)
    assert [hit.index for hit in hits] == [0, 1]
    assert hits[0].score >= hits[1].score


def test_empty_inputs_are_rejected():
    pipe = _pipeline()
    with pytest.raises(ValueError):
        pipe.embed_image([])
    with pytest.raises(ValueError):
        pipe.embed_text([])
    with pytest.raises(ValueError):
        pipe.retrieve("query", [], top_k=1)
    with pytest.raises(ValueError):
        pipe.retrieve("query", [_image()], top_k=0)


def test_zero_shot_classification_preserves_label_exact_whitespace():
    processor = FakeProcessor()
    pipe = PubMedClipPipeline(FakeModel(), processor, device="cpu")

    result = pipe.zero_shot_classify(_image(), ["  Flooded Street  ", "\tRural Road\n"])
    assert processor.last_text == [
        "a medical image of flooded street.",
        "a medical image of rural road.",
    ]
    returned_labels = {item.label for item in result}
    assert returned_labels == {"  Flooded Street  ", "\tRural Road\n"}


def test_retrieval_tie_breaking_is_deterministic_and_stable():
    class EqualScoreModel(FakeModel):
        def get_image_features(self, **batch):
            count = batch["pixel_values"].shape[0]
            return torch.tensor([[1.0, 0.0]] * count, dtype=torch.float32)

        def get_text_features(self, **batch):
            count = batch["input_ids"].shape[0]
            return torch.tensor([[1.0, 0.0]] * count, dtype=torch.float32)

    pipe = PubMedClipPipeline(EqualScoreModel(), FakeProcessor(), device="cpu")
    images = [_image(), _image(), _image(), _image()]
    hits = pipe.retrieve("query", images, top_k=4)
    assert [hit.index for hit in hits] == [0, 1, 2, 3]
    assert all(hit.score == pytest.approx(1.0) for hit in hits)
