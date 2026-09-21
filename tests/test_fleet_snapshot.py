"""Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): manifest-driven staging and verification.

Nothing here touches the network or the real weights: the snapshot directory is built from
stand-in bytes, the manifest is written by hand, and downloads go through an injected callable.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pubmedclip_pipeline import (
    DEFAULT_MODEL_KEY,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    PubMedClipPipeline,
    stage_missing_files,
    verify_snapshot,
)
from pubmedclip_pipeline import model as model_mod
from pubmedclip_pipeline.config import SOURCE_FILENAME, SOURCE_SHA256, SOURCE_SIZE_BYTES

ROOT = Path(__file__).resolve().parents[1]
OTHER_SHA = "0" * 40
SMALL = (
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "spiece.model",
    "tokenizer_config.json",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_manifest(root: Path, files: dict[str, bytes], **overrides) -> dict:
    entries = []
    for name, payload in files.items():
        (root / name).write_bytes(payload)
        entries.append({"path": name, "bytes": len(payload), "sha256": _sha(payload)})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
        **overrides,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def stand_in_snapshot(root: Path) -> dict:
    files = {name: b"{}" for name in SMALL}
    files["model.safetensors"] = b"safetensors-stand-in"
    return write_manifest(root, files)


@pytest.fixture
def pinned_to_stand_in(monkeypatch, tmp_path):
    """Point the pinned weight digest/size at the stand-in bytes so verify_checkpoint accepts it."""
    stand_in_snapshot(tmp_path)
    monkeypatch.setattr(model_mod, "MODEL_SHA256", _sha(b"safetensors-stand-in"))
    monkeypatch.setattr(model_mod, "MODEL_SIZE_BYTES", len(b"safetensors-stand-in"))
    return tmp_path


def test_default_weights_dir_and_committed_manifest_match_the_pins():
    assert model_mod.DEFAULT_WEIGHTS_DIR == ROOT / "weights" / DEFAULT_MODEL_KEY
    manifest = json.loads(
        (ROOT / "weights" / DEFAULT_MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (
        MODEL_ID,
        MODEL_REVISION,
        DEFAULT_MODEL_KEY,
    )
    by_path = {e["path"]: e for e in manifest["files"]}
    assert "model.safetensors" not in by_path  # converted locally, not a Hub file
    assert by_path[SOURCE_FILENAME]["sha256"] == SOURCE_SHA256
    assert by_path[SOURCE_FILENAME]["bytes"] == SOURCE_SIZE_BYTES
    assert set(model_mod.ALLOWED_CHECKPOINT_FILES) <= set(by_path)
    assert (MODEL_SHA256, MODEL_SIZE_BYTES) == (
        "81de21a0f1b6a3faaf1ea9e8fed4d570c671808ad72c948b4dac173eabcf676e",
        605_156_676,
    )
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])


def test_stage_fetches_only_the_absent_entries_through_the_injected_downloader(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "tokenizer.json").unlink()
    calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, root: Path) -> None:
        calls.append((relative_path, root))
        (root / relative_path).write_bytes(
            b"safetensors-stand-in" if relative_path.endswith(".safetensors") else b"{}"
        )

    fetched = stage_missing_files(tmp_path, allow_download=True, downloader=downloader)
    assert sorted(fetched) == ["model.safetensors", "tokenizer.json"]
    assert [c[0] for c in calls] == fetched and all(c[1] == tmp_path for c in calls)
    assert stage_missing_files(tmp_path, allow_download=False, downloader=downloader) == []


def test_stage_refuses_to_download_by_default(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)


@pytest.mark.parametrize("override", [{"modelId": "someone/else"}, {"revision": OTHER_SHA}])
def test_stage_and_verify_refuse_a_manifest_with_the_wrong_identity(tmp_path, override):
    write_manifest(tmp_path, {"model.safetensors": b"w", "config.json": b"{}"}, **override)
    with pytest.raises(ValueError, match="refusing"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
    with pytest.raises(ValueError, match="refusing"):
        verify_snapshot(tmp_path)


def test_stage_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        stage_missing_files(tmp_path)


def test_the_default_downloader_fetches_at_the_pinned_revision(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_hf_hub_download(repo_id, filename, *, revision, local_dir):
        calls.append(
            {"repo_id": repo_id, "filename": filename, "revision": revision, "local_dir": local_dir}
        )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    model_mod._hub_download("model.safetensors", tmp_path)
    assert calls == [
        {
            "repo_id": MODEL_ID,
            "filename": "model.safetensors",
            "revision": MODEL_REVISION,
            "local_dir": str(tmp_path),
        }
    ]


def test_verify_snapshot_returns_the_manifest_and_calls_the_existing_verifier(pinned_to_stand_in):
    result = verify_snapshot(pinned_to_stand_in)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(pinned_to_stand_in)
    assert [e["path"] for e in result["files"]] == [*SMALL, "model.safetensors"]


def test_verify_snapshot_refuses_a_tampered_manifest_digest(pinned_to_stand_in):
    manifest = json.loads((pinned_to_stand_in / MANIFEST_NAME).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == "tokenizer.json":
            entry["sha256"] = "f" * 64
    (pinned_to_stand_in / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch for tokenizer.json"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_tampered_weight_file(pinned_to_stand_in):
    (pinned_to_stand_in / "model.safetensors").write_bytes(b"Safetensors-stand-in")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch for model.safetensors"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_still_asserts_the_pinned_weight_digest(monkeypatch, pinned_to_stand_in):
    """A manifest that agrees with the files but not with MODEL_SHA256 is refused (constants win)."""
    monkeypatch.setattr(model_mod, "MODEL_SHA256", "e" * 64)
    with pytest.raises(RuntimeError, match="Unexpected model.safetensors SHA-256"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_missing_manifest_entry(pinned_to_stand_in):
    (pinned_to_stand_in / "spiece.model").unlink()
    with pytest.raises(RuntimeError, match="Manifest file missing: spiece.model"):
        verify_snapshot(pinned_to_stand_in)


def test_from_pretrained_weights_dir_stages_verifies_and_loads_the_explicit_path(
    monkeypatch, pinned_to_stand_in
):
    seen: dict = {}

    def fake_load_components(*, device, cache_dir, weights_path, return_metadata):
        seen["weights_path"] = weights_path
        return (
            "model",
            "processor",
            "cpu",
            Path(weights_path),
            {
                "checkpoint_path": Path(weights_path),
                "checkpoint_source": "explicit_path",
                "manifest_verified": True,
                "weight_sha256": "x",
                "weight_size_bytes": 1,
            },
        )

    import pubmedclip_pipeline.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "load_components", fake_load_components)
    monkeypatch.setattr(
        model_mod, "snapshot_download", lambda **_: (_ for _ in ()).throw(AssertionError("no hub"))
    )
    pipe = PubMedClipPipeline.from_pretrained(device="cpu", weights_dir=pinned_to_stand_in)
    assert seen["weights_path"] == pinned_to_stand_in
    assert pipe.checkpoint_source == "explicit_path" and pipe.manifest_verified is True
    (pinned_to_stand_in / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        PubMedClipPipeline.from_pretrained(device="cpu", weights_dir=pinned_to_stand_in)
    with pytest.raises(ValueError, match="not both"):
        PubMedClipPipeline.from_pretrained(
            weights_dir=pinned_to_stand_in, weights_path=pinned_to_stand_in
        )
