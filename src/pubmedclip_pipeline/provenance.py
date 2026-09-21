from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PROMPT_TEMPLATE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    TEXT_MAX_LENGTH,
)

_RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    include_runtime: bool = True,
) -> dict[str, Any]:
    checkpoint_source = None
    manifest_verified = False
    weight_file = MODEL_FILENAME
    weight_sha256 = MODEL_SHA256
    weight_size = MODEL_SIZE_BYTES
    device_str = None
    resolved_checkpoint_path = None

    if pipeline is not None:
        if hasattr(pipeline, "checkpoint_path") and pipeline.checkpoint_path is not None:
            resolved_checkpoint_path = str(pipeline.checkpoint_path)
        if hasattr(pipeline, "checkpoint_source"):
            checkpoint_source = pipeline.checkpoint_source
        if hasattr(pipeline, "manifest_verified"):
            manifest_verified = bool(pipeline.manifest_verified)
        if hasattr(pipeline, "weight_sha256") and pipeline.weight_sha256:
            weight_sha256 = pipeline.weight_sha256
        if hasattr(pipeline, "weight_size_bytes") and pipeline.weight_size_bytes:
            weight_size = pipeline.weight_size_bytes
        if hasattr(pipeline, "device"):
            device_str = str(pipeline.device)

    if checkpoint_path is not None:
        resolved_checkpoint_path = str(checkpoint_path)
        if checkpoint_source is None:
            checkpoint_source = "explicit_path"

    model_record: dict[str, Any] = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_file": weight_file,
        "weight_sha256": weight_sha256,
        "weight_size_bytes": weight_size,
    }
    if checkpoint_source is not None:
        model_record["checkpoint_source"] = checkpoint_source
    if resolved_checkpoint_path is not None:
        model_record["checkpoint_path"] = resolved_checkpoint_path
    if pipeline is not None or checkpoint_path is not None:
        model_record["manifest_verified"] = manifest_verified

    inference_record: dict[str, Any] = {
        "prompt_template": prompt_template,
        "zero_shot_score_semantics": "softmax_over_candidate_set_not_calibrated_probability",
        "embedding_normalization": "l2",
        "similarity": "cosine_via_normalized_dot_product",
    }
    if device_str is not None:
        inference_record["device"] = device_str

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model": model_record,
        "processor": {
            "input_resolution": [224, 224],
            "text_max_length": TEXT_MAX_LENGTH,
            "lowercase_model_bound_text": True,
        },
        "inference": inference_record,
    }
    if include_runtime:
        provenance["runtime"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "packages": {name: _package_version(name) for name in _RUNTIME_PACKAGES},
        }
    return provenance


def write_provenance(
    path: str | Path,
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    include_runtime: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            build_provenance(
                pipeline=pipeline,
                checkpoint_path=checkpoint_path,
                prompt_template=prompt_template,
                include_runtime=include_runtime,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
