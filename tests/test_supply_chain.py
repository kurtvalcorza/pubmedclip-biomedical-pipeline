from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import pubmedclip_pipeline.model as model_module
from pubmedclip_pipeline.config import (
    ALLOWED_CHECKPOINT_FILES,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
)


def test_model_identity_and_revision_are_explicit():
    assert MODEL_ID == "flaviagiammarino/pubmed-clip-vit-base-patch32"
    assert MODEL_REVISION == "26c0c67f6da303ad2a38909130bd35744ea93517"
    assert MODEL_FILENAME == "model.safetensors"
    # the one pinned pickle is the conversion source; no other .bin is allowed anywhere
    assert [p for p in ALLOWED_CHECKPOINT_FILES if p.endswith(".bin")] == ["pytorch_model.bin"]


def test_verify_checkpoint_checks_size_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"verified-pubmedclip-test-payload"
    weight = tmp_path / MODEL_FILENAME
    weight.write_bytes(payload)
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    assert model_module.verify_checkpoint(tmp_path) == tmp_path


def test_verify_checkpoint_rejects_pickle_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"verified-pubmedclip-test-payload"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    (tmp_path / "other_model.bin").write_bytes(b"pickle-style")
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(RuntimeError, match="Refusing unsafe weight files"):
        model_module.verify_checkpoint(tmp_path)
    # the pinned source is tolerated (it is digest-checked through the manifest, never loaded)
    (tmp_path / "other_model.bin").unlink()
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle-style")
    assert model_module.verify_checkpoint(tmp_path) == tmp_path


@pytest.mark.parametrize("bad_ext", [".pt", ".pth", ".ckpt", ".pkl", ".h5", ".msgpack"])
def test_verify_checkpoint_rejects_all_unsafe_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_ext: str,
):
    payload = b"verified-pubmedclip-test-payload"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    (tmp_path / f"weights{bad_ext}").write_bytes(b"unsafe-content")
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(RuntimeError, match="Refusing unsafe weight files"):
        model_module.verify_checkpoint(tmp_path)


def test_verify_checkpoint_validates_manifest_and_detects_tampered_preprocessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import json

    payload = b"verified-pubmedclip-test-payload"
    weight = tmp_path / MODEL_FILENAME
    weight.write_bytes(payload)
    preproc = tmp_path / "preprocessor_config.json"
    preproc.write_text('{"do_resize": true}', encoding="utf-8")

    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": "pubmed-clip-vit-base-patch32",
        "files": [
            {
                "path": MODEL_FILENAME,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            {
                "path": "preprocessor_config.json",
                "bytes": preproc.stat().st_size,
                "sha256": hashlib.sha256(preproc.read_bytes()).hexdigest(),
            },
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    path, verified = model_module.verify_checkpoint(tmp_path, return_manifest_verified=True)
    assert path == tmp_path
    assert verified is True

    # Mutate preprocessor_config.json to simulate tampering
    preproc.write_text('{"do_resize": false, "tampered": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch for preprocessor_config.json"):
        model_module.verify_checkpoint(tmp_path)


def test_resolve_weights_path_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    explicit = tmp_path / "explicit"
    resolved, source = model_module.resolve_weights_path(weights_path=explicit)
    assert resolved == explicit
    assert source == "explicit_path"

    # Env var precedence
    env_dir = tmp_path / "env_weights"
    monkeypatch.setenv("PUBMEDCLIP_WEIGHTS_DIR", str(env_dir))
    resolved, source = model_module.resolve_weights_path()
    assert resolved == env_dir
    assert source == "env_var"

    monkeypatch.delenv("PUBMEDCLIP_WEIGHTS_DIR", raising=False)

    # Hub fallback
    monkeypatch.setattr(
        model_module,
        "snapshot_download",
        lambda **kwargs: str(tmp_path / "hub_cache"),
    )
    # Monkeypatch __file__ to pretend we are installed in site-packages without pyproject.toml
    fake_pkg_file = tmp_path / "site-packages" / "pubmedclip_pipeline" / "model.py"
    fake_pkg_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(model_module, "__file__", str(fake_pkg_file))

    resolved, source = model_module.resolve_weights_path()
    assert resolved == tmp_path / "hub_cache"
    assert source == "hf_hub"
