from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from pubmedclip_pipeline import DEFAULT_WEIGHTS_DIR, MODEL_SHA256, PubMedClipPipeline

pytestmark = pytest.mark.integration


def test_real_checkpoint_smoke():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to download and run the pinned checkpoint")

    # The pinned Hub revision ships only the audited pytorch_model.bin. The documented path
    # stages it into the committed snapshot directory and runs convert_source() (pickle audit,
    # one weights-only unpickle, safetensors export refused unless it matches the pinned
    # digest) before the strict load.
    pipe = PubMedClipPipeline.from_pretrained(
        weights_dir=DEFAULT_WEIGHTS_DIR, device="cpu", allow_download=True
    )
    assert pipe.weight_sha256 == MODEL_SHA256
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
