from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from pubmedclip_pipeline import load_pipeline

pytestmark = pytest.mark.integration


def test_real_checkpoint_smoke():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to download and run the pinned checkpoint")

    pipe = load_pipeline(device="cpu")
    image = Image.new("RGB", (256, 256), (180, 30, 30))

    scores = pipe.zero_shot_classify(image, ["a red image", "a blue image"])
    assert len(scores) == 2
    assert all(0.0 <= item.score <= 1.0 for item in scores)

    image_embedding = pipe.embed_image([image])
    text_embedding = pipe.embed_text(["a red image"])
    assert image_embedding.shape[0] == 1
    assert text_embedding.shape[0] == 1
    assert image_embedding.shape[1] == text_embedding.shape[1]
    np.testing.assert_allclose(np.linalg.norm(image_embedding, axis=1), [1.0], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(text_embedding, axis=1), [1.0], atol=1e-5)
