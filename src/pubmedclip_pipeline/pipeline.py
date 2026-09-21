# ruff: noqa: E501  -- adaptation contract written at the 110-column fleet width; this repo lints at 100
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_PROMPT_TEMPLATE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    TEXT_MAX_LENGTH,
)
from .model import convert_source, load_components, stage_missing_files, verify_snapshot

ImageInput = str | Path | bytes | Image.Image


@dataclass(frozen=True, slots=True)
class ClassificationScore:
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    index: int
    score: float


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.convert("RGB")
    path = Path(value)
    raw = str(value)
    if raw.lower().startswith(("http://", "https://")):
        raise ValueError("Remote image URLs are not accepted; provide local bytes or a local path")
    with Image.open(path) as image:
        return image.convert("RGB")


def _require_texts(values: Sequence[str], *, name: str) -> list[str]:
    result = [value.strip() for value in values]
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain at least one non-empty string")
    return result


def _clip_texts(values: Sequence[str]) -> list[str]:
    """Lowercase model-bound text before tokenization (CLIP's BPE tokenizer lowercases in its own
    cleaning step; doing it here keeps the behaviour independent of the tokenizer class resolved)."""

    return [value.lower() for value in values]


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


# --------------------------------------------------------------------------
# Adaptation contract (E2E): bounded fine-tuning of the vision tower's last blocks for a labelled
# image set, scored through the same zero-shot prompts the frozen model is scored with.
# --------------------------------------------------------------------------

PARAMETER_COUNT = 151_277_313
VISION_LAYERS = 12  # vision_config num_hidden_layers
DEFAULT_TRAINABLE_VISION_LAYERS = (
    2  # the last two vision blocks + post-layernorm + visual projection (14,570,496 params)
)
MAX_EVAL_RECORDS = 20_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.pubmed-clip-vit-base-patch32.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompts(
    classes: Sequence[str], prompt_template: str, names: Mapping[str, str] | None
) -> list[str]:
    if "{label}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal {label} placeholder")
    return [prompt_template.format(label=(names or {}).get(c, c)) for c in classes]


def _trainable_names(model: Any, trainable_vision_layers: int) -> list[str]:
    """The last `trainable_vision_layers` blocks of the vision tower, its post-layernorm and the
    visual projection. The text tower, the embeddings, the text projection and `logit_scale` stay frozen."""
    if (
        isinstance(trainable_vision_layers, bool)
        or not isinstance(trainable_vision_layers, int)
        or not 1 <= trainable_vision_layers <= VISION_LAYERS
    ):
        raise ValueError(f"trainable_vision_layers must be an int in 1..{VISION_LAYERS}")
    first = VISION_LAYERS - trainable_vision_layers
    prefixes = tuple(f"vision_model.encoder.layers.{k}." for k in range(first, VISION_LAYERS)) + (
        "vision_model.post_layernorm.",
        "visual_projection.",
    )
    return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]


def _check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Refuse an artifact whose manifest is not exactly the one this package writes: the supported format
    and version, the pinned base (id, revision, weight file, digest), exactly one file entry named
    `adapter.safetensors` that resolves inside the artifact directory, and a recorded
    `trainable_vision_layers` in range. Nothing is deserialised here. The digest check that follows detects
    corruption or drift of the weights relative to the adjacent manifest; it is not authenticity against an
    actor who can replace both files."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"artifact format_version {manifest.get('format_version')!r} is not the supported "
            f"{ARTIFACT_FORMAT_VERSION!r}"
        )
    base = manifest.get("base_model", {})
    if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
        MODEL_ID,
        MODEL_REVISION,
        MODEL_SHA256,
    ):
        raise ValueError(
            "artifact was adapted from a different base model, revision or weight file"
        )
    if base.get("weight_file", MODEL_FILENAME) != MODEL_FILENAME:
        raise ValueError("artifact was adapted from a different base weight file")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("artifact manifest must list exactly one file")
    entry = files[0]
    if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
        raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
    weights_path = (root / entry["path"]).resolve()
    if weights_path.parent != root.resolve():
        raise ValueError("artifact weight path must resolve inside the artifact directory")
    adapter = manifest.get("adapter")
    layers = adapter.get("trainable_vision_layers") if isinstance(adapter, Mapping) else None
    if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= VISION_LAYERS:
        raise ValueError(
            "artifact manifest does not record an in-range integer trainable_vision_layers"
        )
    if not isinstance(manifest.get("tensors"), list):
        raise ValueError("artifact manifest must list its tensors")
    return weights_path


class PubMedClipPipeline:
    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes
        self.adapter: dict[str, Any] | None = None
        if hasattr(self.model, "parameters"):  # injected fakes in the offline tests carry none
            for param in self.model.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> PubMedClipPipeline:
        """Load the one supported checkpoint.

        ``weights_dir`` names a fleet snapshot directory holding ``dimer-base-manifest.json``
        (normally ``weights/<DEFAULT_MODEL_KEY>/``): manifest entries that are absent are staged
        with :func:`stage_missing_files` (only when ``allow_download=True``), the directory is
        verified against the manifest and the pinned digests by :func:`verify_snapshot`, and the
        same :func:`load_components` call then loads it as an explicit path (source
        ``explicit_path``; nothing goes through ``snapshot_download``). When the serving file
        ``model.safetensors`` is absent it is produced first by :func:`convert_source` (the pinned
        ``pytorch_model.bin`` audited statically and unpickled once, weights-only); ``progress``
        receives that conversion report.
        """
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            if not (Path(weights_dir) / MODEL_FILENAME).is_file():
                report = convert_source(weights_dir)  # audit + one-time weights-only unpickle
                if progress is not None:
                    progress(report)
            verify_snapshot(weights_dir)
            weights_path = weights_dir
        model, processor, target_device, _, metadata = load_components(
            device=device,
            cache_dir=cache_dir,
            weights_path=weights_path,
            return_metadata=True,
        )
        return cls(
            model,
            processor,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
        )

    def zero_shot_classify(
        self,
        image: ImageInput,
        labels: Sequence[str],
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> list[ClassificationScore]:
        if not labels:
            raise ValueError("labels must contain at least one non-empty string")
        original_labels = list(labels)
        cleaned_labels = [label.strip() for label in original_labels]
        if any(not label for label in cleaned_labels):
            raise ValueError("labels must contain at least one non-empty string")
        if "{label}" not in prompt_template:
            raise ValueError("prompt_template must contain the literal {label} placeholder")

        prompts = [prompt_template.format(label=label) for label in cleaned_labels]
        batch = self.processor(
            text=_clip_texts(prompts),
            images=[_coerce_image(image)],
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits_per_image[0]
            scores = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        ranked = [
            ClassificationScore(label=orig_label, score=float(score))
            for orig_label, score in zip(original_labels, scores, strict=True)
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def embed_image(self, images: Sequence[ImageInput]) -> np.ndarray:
        if not images:
            raise ValueError("images must contain at least one image")
        batch = self.processor(
            images=[_coerce_image(image) for image in images],
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(**batch)
            features = F.normalize(features, p=2, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        texts_list = _require_texts(texts, name="texts")
        batch = self.processor(
            text=_clip_texts(texts_list),
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**batch)
            features = F.normalize(features, p=2, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def similarity(self, images: Sequence[ImageInput], texts: Sequence[str]) -> np.ndarray:
        image_features = self.embed_image(images)
        text_features = self.embed_text(texts)
        return image_features @ text_features.T

    def retrieve(
        self,
        query: str,
        images: Sequence[ImageInput],
        *,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not images:
            raise ValueError("images must contain at least one image")

        scores = self.similarity(images, [query])[:, 0]
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(images))]
        return [RetrievalHit(index=int(index), score=float(scores[index])) for index in order]

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        classes: Sequence[str] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        class_names_map: Mapping[str, str] | None = None,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Zero-shot classification of a validated labelled dataset: one prompt per class (the label, or its
        entry in `class_names_map`, in `prompt_template`), the scaled cosine logits as the score grid, and
        accuracy / macro F1 / per-class breakdown / text-to-image mAP from `metrics.classification_metrics`."""
        from .metrics import classification_metrics
        from .samples import class_names, validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        class_list = list(classes) if classes is not None else class_names(checked)
        unknown = sorted({r["label"] for r in checked} - set(class_list))
        if unknown:
            raise ValueError(f"records carry labels outside classes: {unknown[:5]}")
        started = time.perf_counter()
        text = self.embed_text(_prompts(class_list, prompt_template, class_names_map))
        rows = []
        for start in range(0, len(checked), batch_size):
            rows.append(self.embed_image([r["image"] for r in checked[start : start + batch_size]]))
        images = np.concatenate(rows)
        scale = float(self.model.logit_scale.detach().exp().cpu())
        scores = images @ text.T * scale  # the model's own logits_per_image (softmax-free ranking)
        gold = [class_list.index(r["label"]) for r in checked]
        metrics = classification_metrics(scores, gold, class_list)
        metrics.update(
            {
                "classes": class_list,
                "predictions": [class_list[int(i)] for i in scores.argmax(axis=1)],
                "prompt_template": prompt_template,
                "score": "logits_per_image (logit_scale × cosine; softmax over the class set gives the per-image scores)",
                "verdict": "measured"
                if len(checked) >= MIN_SCORED_RECORDS
                else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 16,
        trainable_vision_layers: int = DEFAULT_TRAINABLE_VISION_LAYERS,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        class_names_map: Mapping[str, str] | None = None,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning of the vision tower on a validated labelled dataset.

            Only the last `trainable_vision_layers` blocks of the vision tower, its post-layernorm and the
            visual projection train (2 blocks by default: 14,570,496 of 151,277,313 parameters; the text tower,
            the embeddings, the text projection and `logit_scale` stay frozen). The class prompts are embedded once
            by the frozen text tower; every batch of images is scored against them with the model's own scaled
            cosine logits and trained with the cross-entropy of the softmax over the class prompts (CLIP's
            image-to-text contrastive objective restricted to the class set). AdamW at a fixed learning rate with
            gradient clipping at 1.0, seeded shuffling, no scheduler. Epoch 0 records the frozen model's validation
            metrics; the epoch with the highest validation text-to-image mAP is kept (smoother than accuracy on a
            small validation split).
        Transactional: any failure restores the base tensors."""
        import torch
        from torch.nn.functional import cross_entropy, normalize

        from .samples import class_names, validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 64
        ):
            raise ValueError("batch_size must be an int in 1..64")
        names = _trainable_names(self.model, trainable_vision_layers)
        train_checked = validate_dataset(train)["records"]
        class_list = class_names(train_checked)
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
            if val
            else []
        )
        unknown = sorted({r["label"] for r in val_checked} - set(class_list))
        if unknown:
            raise ValueError(
                f"validation records carry labels outside the training classes: {unknown[:5]}"
            )
        model, processor = self.model, self.processor
        torch.manual_seed(seed)
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        prompts = _clip_texts(_prompts(class_list, prompt_template, class_names_map))
        with (
            torch.no_grad()
        ):  # not inference_mode: the cached prompt features feed the training graph
            tokens = processor(
                text=prompts, padding="max_length", max_length=TEXT_MAX_LENGTH, return_tensors="pt"
            )
            tokens = _move_batch(tokens, self.device)
            text_feat = normalize(model.get_text_features(**tokens), dim=-1).clone()
        scale = model.logit_scale.detach().exp()
        gold = torch.tensor(
            [class_list.index(r["label"]) for r in train_checked], device=self.device
        )

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            result = self.evaluate(
                val_checked,
                classes=class_list,
                prompt_template=prompt_template,
                class_names_map=class_names_map,
            )
            return {k: result[k] for k in ("accuracy", "macro_f1", "t2i_map", "n")}

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {
            "epoch": 0,
            "train_loss": None,
            "val": score_val(),
            "note": "frozen model",
        }
        history.append(entry)
        if progress:
            progress(entry)
        best_score = entry["val"]["t2i_map"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    index = order[start : start + batch_size]
                    pixels = processor(
                        images=[train_checked[i]["image"] for i in index], return_tensors="pt"
                    )
                    pixels = _move_batch(pixels, self.device)
                    image_feat = normalize(model.get_image_features(**pixels), dim=-1)
                    logits = image_feat @ text_feat.T * scale
                    loss = cross_entropy(logits, gold[index])
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "val": score_val(),
                }
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["t2i_map"] if entry["val"] else math.inf
                if current > best_score or not entry["val"]:
                    best_score = current
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted
                    }
                    best_epoch = epoch
        except BaseException:
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_vision_layers": trainable_vision_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "classes": class_list,
            "prompt_template": prompt_template,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation text-to-image mAP"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    def save_artifact(
        self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None
    ) -> Path:
        """Write the adapted vision-tower tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {
            k: v.detach().cpu().contiguous()
            for k, v in self.model.state_dict().items()
            if k in names
        }
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": DEFAULT_MODEL_KEY,
                "weight_file": MODEL_FILENAME,
                "weight_sha256": MODEL_SHA256,
            },
            "adapter": {
                k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")
            },
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256_file(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = _check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if (
            _sha256_file(weights_path) != entry["sha256"]
            or weights_path.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        expected = sorted(
            _trainable_names(self.model, manifest["adapter"]["trainable_vision_layers"])
        )
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith(("vision_model.", "visual_projection.")):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable vision-tower tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> PubMedClipPipeline:
        """A fresh pipeline from the pinned base with an adapter overlaid."""
        pipe = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download
        )
        pipe.load_artifact(artifact_dir)
        return pipe


def load_pipeline(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> PubMedClipPipeline:
    return PubMedClipPipeline.from_pretrained(
        device=device,
        cache_dir=cache_dir,
        weights_path=weights_path,
    )


# --------------------------------------------------------------------------
# Role stages (DIMER NOTEBOOK_SPEC 1.1 DAT24 / EVAL21)
#
# `validate_inputs` is the public validation stage: it applies exactly the checks the four core
# methods apply (image coercion through `_coerce_image`, non-empty texts through `_require_texts`,
# the label / prompt-template / top_k checks of `zero_shot_classify` and `retrieve`) and reports
# what was proven as an input manifest. `evaluation_report` is the public evaluation stage: its
# metric ids are this module's own helpers `top1_accuracy` and `recall_at_1` (the tutorial's
# sanity metrics, extracted from the notebook), and it always produces a report.
# --------------------------------------------------------------------------

#: The input contract and every named ceiling, in one readable structure.
INPUT_SCHEMA: dict[str, Any] = {
    "images": (
        "PIL.Image.Image, raw bytes, or a local path decodable by Pillow; any mode, converted to "
        "RGB; remote URLs are refused"
    ),
    "texts": "non-empty strings (labels, queries, or free text); lowercased before tokenization",
    "text_max_length": TEXT_MAX_LENGTH,
    "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    "top_k": [1, None],
    "preprocessing": (
        "the pinned processor resizes the short side to 224, centre-crops 224x224 and normalizes; text is tokenized "
        f"to max_length={TEXT_MAX_LENGTH} with padding, longer text is truncated"
    ),
}


def _check_inputs(
    images: Sequence[ImageInput] | ImageInput | None,
    texts: Sequence[str] | None,
    *,
    top_k: int | None,
    prompt_template: str,
) -> tuple[list[Image.Image], list[str] | None]:
    """The checks the core methods apply, in their order, raising exactly what they raise."""
    if images is None:
        coerced: list[Image.Image] = []
    else:
        if isinstance(images, str | Path | bytes | Image.Image):
            images = [images]
        if not images:
            raise ValueError("images must contain at least one image")
        coerced = [_coerce_image(image) for image in images]
    checked_texts: list[str] | None = None
    if texts is not None:
        if not texts:
            raise ValueError("labels must contain at least one non-empty string")
        checked_texts = _require_texts(texts, name="texts")
    if "{label}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal {label} placeholder")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    return coerced, checked_texts


def validate_inputs(
    images: Sequence[ImageInput] | ImageInput | None,
    texts: Sequence[str] | None = None,
    *,
    top_k: int | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``zero_shot_classify`` / ``embed_image`` /
    ``embed_text`` / ``retrieve`` would; a caller that wants the finding recorded catches the
    exception and stores ``str(exc)`` under ``findings``.
    """
    coerced, checked_texts = _check_inputs(
        images, texts, top_k=top_k, prompt_template=prompt_template
    )
    if names is not None and len(names) != len(coerced):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"image-{i}",
                "mode": image.mode,
                "size": list(image.size),
            }
            for i, image in enumerate(coerced)
        ],
        "texts": checked_texts,
        "top_k": top_k,
        "prompt_template": prompt_template,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def top1_accuracy(
    classifications: Sequence[Sequence[ClassificationScore]], expected_labels: Sequence[str]
) -> float:
    """Fraction of images whose top-scoring label equals the expected label (tutorial sanity)."""
    if len(classifications) != len(expected_labels):
        raise ValueError("classifications and expected_labels must have the same length")
    if not classifications:
        raise ValueError("classifications must not be empty")
    pairs = zip(classifications, expected_labels, strict=True)
    hits = sum(int(scores[0].label == expected) for scores, expected in pairs)
    return hits / len(classifications)


def recall_at_1(
    retrievals: Sequence[Sequence[RetrievalHit]], expected_indices: Sequence[int]
) -> float:
    """Fraction of queries whose top-ranked hit is the expected image index (tutorial sanity)."""
    if len(retrievals) != len(expected_indices):
        raise ValueError("retrievals and expected_indices must have the same length")
    if not retrievals:
        raise ValueError("retrievals must not be empty")
    pairs = zip(retrievals, expected_indices, strict=True)
    hits = sum(int(ranked[0].index == expected) for ranked, expected in pairs)
    return hits / len(retrievals)


def evaluation_report(
    result: dict[str, Any],
    targets: dict[str, Any] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``result`` gathers what the notebook demonstrated: ``classifications`` (one
    ``zero_shot_classify`` list per image), ``retrievals`` (one ``retrieve`` list per query), and
    optionally ``embedding_shapes`` / ``similarity_shape``. ``targets`` supplies the ground truth:
    ``labels`` (the expected label per classified image) and/or ``retrieval_indices`` (the expected
    image index per query). With targets the report carries ``top1_accuracy`` (against the
    fixed-class baseline 1/n_labels) and/or ``recall_at_1`` with the verdict ``sample-sanity``;
    without them the verdict is ``not-measurable``. Embeddings and similarity matrices are
    representations with no intrinsic metric and are always reported as not measurable.
    """
    classifications = list(result.get("classifications") or [])
    retrievals = list(result.get("retrievals") or [])
    base: dict[str, Any] = {
        "task": "zero-shot image classification, image/text embeddings, similarity and retrieval",
        "score_semantics": (
            "classification scores are a softmax over the candidate set (relative to the candidates "
            "given, not calibrated), similarity scores are cosine similarities; the decision rule "
            "used for the sanity metrics is argmax; no threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_classified_images": len(classifications),
        "n_retrieval_queries": len(retrievals),
        "embeddings": (
            "representations, not predictions: no intrinsic accuracy metric; "
            f"shapes {result.get('embedding_shapes')}"
        ),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    labels = (targets or {}).get("labels")
    indices = (targets or {}).get("retrieval_indices")
    metrics: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    estimation = "single synthetic sample set; no dispersion estimate"
    if labels is not None and classifications:
        n_candidates = len(classifications[0])
        metrics.append(
            {
                "id": "top1_accuracy",
                "value": top1_accuracy(classifications, list(labels)),
                "n_candidate_labels": n_candidates,
                "estimation": estimation,
            }
        )
        baselines.append(
            {
                "id": "fixed_class_baseline",
                "metric": "top1_accuracy",
                "value": 1.0 / n_candidates if n_candidates else None,
                "note": "always predicting one fixed candidate label",
            }
        )
    if indices is not None and retrievals:
        metrics.append(
            {
                "id": "recall_at_1",
                "value": recall_at_1(retrievals, list(indices)),
                "n_gallery_images": len(result.get("gallery_ids") or []) or None,
                "estimation": estimation,
            }
        )
    if not metrics:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no expected labels or expected retrieval indices were supplied",
            "needs": (
                "labelled images from the deployment domain (one expected label per image among "
                "the candidate labels) scored with top1_accuracy against the fixed-class baseline, "
                "and/or query-image relevance pairs scored with recall_at_1"
            ),
        }
    return {
        **base,
        "metrics": metrics,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(classifications)} labelled image(s) and {len(retrievals)} query(ies) from the "
            "tutorial sample; not a benchmark"
        ),
        "needs": (
            "a labelled evaluation set from the deployment domain, with prompt wording fixed in "
            "advance, for any generalisable accuracy or retrieval claim"
        ),
    }
