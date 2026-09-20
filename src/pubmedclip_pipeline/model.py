from __future__ import annotations

import hashlib
import io
import json
import os
import pickletools
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

from .config import (
    ALLOWED_CHECKPOINT_FILES,
    CKPT_ALLOWED_GLOBALS,
    DEFAULT_MODEL_KEY,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    MODEL_STATE_TENSORS,
    PICKLE_AUDIT_SHA256,
    SOURCE_FILENAME,
    SOURCE_SHA256,
    SOURCE_SIZE_BYTES,
    SOURCE_STATE_TENSORS,
    UNSAFE_WEIGHT_EXTENSIONS,
)

MANIFEST_NAME = "dimer-base-manifest.json"
#: Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): the pinned files live in a repository-
#: local snapshot directory named by the model key and described by the committed manifest; a
#: standalone notebook carries that manifest inline and stages/verifies a working-directory copy.
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / DEFAULT_MODEL_KEY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    snapshot_path: str | Path,
    *,
    require_configs: bool = False,
    return_manifest_verified: bool = False,
) -> Path | tuple[Path, bool]:
    root = Path(snapshot_path)
    if not root.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {root}")

    weight_path = root / MODEL_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(
            f"Pinned checkpoint is missing {MODEL_FILENAME}; run convert_source() on the audited "
            f"{SOURCE_FILENAME} first"
        )

    # The pinned PyTorch source is the one pickle tolerated in the directory (never loaded by
    # transformers: `use_safetensors=True`); it is digest-checked below when present.
    unsafe = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS
        and p.name != SOURCE_FILENAME
    )
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files: {unsafe}")

    manifest_path = root / MANIFEST_NAME
    manifest_verified = False

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc

        files = manifest.get("files") or []
        if not files:
            raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")

        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            target = root / rel_path
            if not target.is_file():
                if rel_path == SOURCE_FILENAME:
                    continue  # converted-only (DIMER-hosted) shape: the source is not required
                raise RuntimeError(f"Manifest file missing: {rel_path}")
            exp_bytes = entry.get("bytes")
            if exp_bytes is not None and target.stat().st_size != exp_bytes:
                raise RuntimeError(
                    f"Size mismatch for {rel_path}: {target.stat().st_size} != {exp_bytes}"
                )
            exp_sha = entry.get("sha256")
            if exp_sha is not None and _sha256(target) != exp_sha:
                raise RuntimeError(f"SHA-256 mismatch for {rel_path}")

        manifest_verified = True

    size = weight_path.stat().st_size
    if size != MODEL_SIZE_BYTES:
        raise RuntimeError(f"Unexpected {MODEL_FILENAME} size: {size}; expected {MODEL_SIZE_BYTES}")

    digest = _sha256(weight_path)
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"Unexpected {MODEL_FILENAME} SHA-256: {digest}; expected {MODEL_SHA256}"
        )

    if require_configs:
        for req in ("config.json", "preprocessor_config.json"):
            if not (root / req).is_file():
                raise RuntimeError(f"Missing required configuration file: {req}")

    if return_manifest_verified:
        return root, manifest_verified
    return root


def _read_manifest(root: Path) -> dict[str, Any]:
    """Load and identity-check ``<root>/dimer-base-manifest.json``."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing"
        )
    if not manifest.get("files"):
        raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Manifest-driven verification of a fleet snapshot directory; raise on the first mismatch.

    The identity in the manifest must be the pinned one; every manifest entry is then size- and
    SHA-256-checked by :func:`verify_checkpoint` (the existing verifier, which also asserts the
    weight file's pinned digest and byte count and refuses unsafe formats). Returns
    ``{"path": ..., **manifest}``.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    converted = (root / MODEL_FILENAME).is_file()
    if not converted:
        # Hub files only (the source pickle among them, digest-checked; it is never unpickled here).
        _check_manifest_files(root, manifest)
        return {"path": str(root), **manifest, "converted": False}
    _, manifest_verified = verify_checkpoint(
        root, require_configs=True, return_manifest_verified=True
    )
    if not manifest_verified:
        raise RuntimeError(f"manifest at {root} was not verified")  # pragma: no cover
    return {"path": str(root), **manifest, "converted": True}


def _check_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not target.is_file():
            raise RuntimeError(f"Manifest file missing: {entry['path']}")
        if target.stat().st_size != entry.get("bytes"):
            raise RuntimeError(f"Size mismatch for {entry['path']}")
        if _sha256(target) != entry.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch for {entry['path']}")


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest and
    the small files but git-ignores the weights). Returns the relative paths fetched;
    :func:`verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if (root / MODEL_FILENAME).is_file() and SOURCE_FILENAME in missing:
        missing.remove(SOURCE_FILENAME)  # converted-only shape: the source pickle is not needed
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no
    execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(
    path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS
) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and
    refuse any outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(
            f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}"
        )
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_FILENAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_SIZE_BYTES:
        raise ValueError(f"{SOURCE_FILENAME}: size {size} != pinned {SOURCE_SIZE_BYTES}")
    digest = _sha256(source)
    if digest != SOURCE_SHA256:
        raise ValueError(f"{SOURCE_FILENAME}: sha256 {digest} != pinned {SOURCE_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(
            f"{SOURCE_FILENAME}: pickle audit digest {audit['audit_sha256']} != pinned "
            f"{PICKLE_AUDIT_SHA256}"
        )
    return {"path": SOURCE_FILENAME, "bytes": size, "sha256": digest, "audit": audit}


def convert_source(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned `pytorch_model.bin` into `model.safetensors`, deterministically, after
    size,
    digest and static-audit checks: torch's weights-only unpickler, a strict load into the CLIP
    architecture built from the snapshot's `config.json` (the two non-persistent `position_ids`
    buffers dropped), and the model's own state dict saved. The pickle is unpickled exactly once,
    here."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    from safetensors.torch import save_file
    from transformers import CLIPConfig, CLIPModel

    started = time.perf_counter()
    source_path = root / SOURCE_FILENAME
    state = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{SOURCE_FILENAME} did not unpickle to a state dict of tensors")
    if len(state) != SOURCE_STATE_TENSORS:
        raise ValueError(
            f"{SOURCE_FILENAME}: state dict has {len(state)} tensors, "
            f"expected {SOURCE_STATE_TENSORS}"
        )
    buffers = sorted(k for k in state if k.endswith("position_ids"))
    config = CLIPConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = CLIPModel(config).eval()
    model.load_state_dict({k: v for k, v in state.items() if k not in buffers}, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    if len(canonical) != MODEL_STATE_TENSORS:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors; expected {MODEL_STATE_TENSORS}"
        )
    target = root / MODEL_FILENAME
    save_file(canonical, str(target), metadata={"format": "pt"})
    size = target.stat().st_size
    digest = _sha256(target)
    if size != MODEL_SIZE_BYTES or digest != MODEL_SHA256:
        target.unlink()
        raise ValueError(
            f"{MODEL_FILENAME}: converted file {size} B / {digest} != pinned "
            f"{MODEL_SIZE_BYTES} B / {MODEL_SHA256}"
        )
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "state_dict_tensors": len(state),
            "dropped_buffers": buffers,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "converted": {"path": MODEL_FILENAME, "bytes": size, "sha256": digest},
        "seconds": round(time.perf_counter() - started, 2),
    }


def resolve_weights_path(
    weights_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve weights path with precedence:

    1. Explicit argument `weights_path` -> 'explicit_path'
    2. Environment variable `PUBMEDCLIP_WEIGHTS_DIR` -> 'env_var'
    3. Source checkout convention `weights/pubmed-clip-vit-base-patch32` -> 'repo_offline'
       (only if pyproject.toml exists at repo root and weights/ contains model.safetensors)
    4. Hugging Face Hub snapshot download -> 'hf_hub'
    """
    if weights_path is not None:
        return Path(weights_path), "explicit_path"

    env_dir = os.environ.get("PUBMEDCLIP_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir), "env_var"

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        repo_weights = repo_root / "weights" / DEFAULT_MODEL_KEY
        if (repo_weights / MODEL_FILENAME).is_file():
            return repo_weights, "repo_offline"

    hub_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOWED_CHECKPOINT_FILES),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    return hub_path, "hf_hub"


_resolve_weights_path = resolve_weights_path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    return_metadata: bool = False,
) -> tuple[Any, Any, torch.device, Path] | tuple[Any, Any, torch.device, Path, dict[str, Any]]:
    """Acquire, verify, and load the one supported PubMedCLIP checkpoint."""

    candidate_path, source = resolve_weights_path(
        weights_path=weights_path,
        cache_dir=cache_dir,
    )

    verified, manifest_verified = verify_checkpoint(
        candidate_path,
        require_configs=True,
        return_manifest_verified=True,
    )
    target_device = _resolve_device(device)

    processor = AutoProcessor.from_pretrained(
        verified,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        verified,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = model.eval().to(target_device)

    weight_file = verified / MODEL_FILENAME
    metadata: dict[str, Any] = {
        "checkpoint_path": verified,
        "checkpoint_source": source,
        "manifest_verified": manifest_verified,
        "weight_sha256": _sha256(weight_file),
        "weight_size_bytes": weight_file.stat().st_size,
        "device": str(target_device),
    }

    if return_metadata:
        return model, processor, target_device, verified, metadata
    return model, processor, target_device, verified
