# PubMedCLIP Biomedical Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/tutorials/pubmedclip_biomedical_colab.ipynb)

DIMER-oriented inference and bounded fine-tuning wrapper for **one immutable open-weight PubMedCLIP checkpoint** — Eslami, de Melo and Meinel's CLIP ViT-B/32 fine-tuned on the ROCO radiology image–caption pairs (*Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?*, 2021), packaged as a **research and education** pipeline. It shares one code shape with the fleet's SigLIP rows (`siglip2-vision-language-pipeline`, `siglip-v1-zero-shot-pipeline`) with CLIP's softmax scoring and cross-entropy adaptation in place of SigLIP's sigmoid, and a medical-image corpus in place of the bird photographs:

- model: `flaviagiammarino/pubmed-clip-vit-base-patch32`
- pinned revision: `26c0c67f6da303ad2a38909130bd35744ea93517`
- hosted source: `pytorch_model.bin` (a torch pickle, `605,222,477` bytes, SHA-256 `4daa7650d2b47e55c37b5ca7dcabe826fd82407c26028a574bcc422f35a94aa4`) — statically audited and converted **once**, never loaded by transformers
- served weight file: `model.safetensors` (converted; `605,156,676` bytes, SHA-256 `81de21a0f1b6a3faaf1ea9e8fed4d570c671808ad72c948b4dac173eabcf676e`)
- also hosted, never fetched: `tf_model.h5` (`605,559,520` bytes, `fe5013d5…`), `flax_model.msgpack` (`605,123,003` bytes, `3b689834…`)
- upstream model license: MIT

The wrapper code in this repository is MIT licensed. The model weights retain the upstream MIT license. **Nothing here is a clinical tool**: the model is not validated for diagnosis, triage or screening, and the pipeline provides no such function.

## Status

**Candidate.** The inference contract, the adaptation contract, the pickle audit and conversion and the real pinned checkpoint have been exercised on the build workstation's CPU and, for the `E2E` standalone tutorial, the review-fix blob `066b560d` (commit `a197c8b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 15/15 code cells, 917.8 s wall, 769.9 s of it the Zenodo archive fetch; isolated uv environment Python 3.12.12, 47 locked packages, built in 53 s; torch 2.14.0+cu130, transformers 4.57.6, `cuda`; accuracy majority 0.091 / neighbour 0.511 / frozen 0.182 / adapted 0.812, text-to-image mAP 0.091 / 0.519 / 0.284 / 0.654, best epoch 4, `ap_fell` [lung-right], reload parity 8 of 8 exact; evidence in `docs/execution-evidence/2026-10-08-a197c8b/`). The previous blob `4b2fc4c4` (commit `d125896`) ran on Kaggle T4 on 2026-09-21 only after a manual restart after its in-kernel install cell (14/14 ok, 415.1 s), so it was not one-pass evidence. REL12 BYOD exercise pending before promotion. Production HTTP serving / DIMER worker packaging remains a separate serving-readiness milestone.

## Capabilities

```python
from pubmedclip_pipeline import DEFAULT_WEIGHTS_DIR, PubMedClipPipeline

# Stages the pinned source, audits and converts it once (refused unless the result matches the
# pinned digest), then strict-loads it. The Hub revision ships no model.safetensors.
pipe = PubMedClipPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR, allow_download=True)

scores = pipe.zero_shot_classify(
    "scan.png",
    ["chest X-ray", "brain MRI", "abdominal CT scan"],
)

image_embeddings = pipe.embed_image(["a.png", "b.png"])
text_embeddings = pipe.embed_text(["chest X-ray", "abdominal CT scan"])
cosine_matrix = pipe.similarity(["a.png", "b.png"], ["chest X-ray", "abdominal CT scan"])
hits = pipe.retrieve("chest X-ray", ["a.png", "b.png"], top_k=2)
```

Public inference operations:

- `zero_shot_classify()`
- `embed_image()`
- `embed_text()`
- `similarity()`
- `retrieve()`

## Adaptation contract

```python
from pubmedclip_pipeline import (
    PubMedClipPipeline, build_sample_dataset, class_names, fetch_corpus, load_byod_dataset,
    read_corpus, split_dataset, validate_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 1,100 OrganAMNIST-224 slices: 396 train / 352 validation / 176 exposed dev test / 176 fresh test
# or: splits = split_dataset(load_byod_dataset("my_images.zip"), seed=42)  # labels.csv: id, file, label
classes = class_names(splits["train"])

pipe = PubMedClipPipeline.from_pretrained(weights_dir="weights/pubmed-clip-vit-base-patch32")  # audits + converts the pickle once; CUDA when visible
prompt = "An axial abdominal CT slice showing the {label}."
frozen = pipe.evaluate(splits["test"], classes=classes, prompt_template=prompt)            # accuracy, macro_f1, t2i_map, per_class
result = pipe.adapt(splits["train"], splits["validation"], epochs=6, lr=5e-5, trainable_vision_layers=2, prompt_template=prompt)
adapted = pipe.evaluate(splits["test"], classes=classes, prompt_template=prompt)
pipe.save_artifact("outputs/adapter")                             # adapter.safetensors + manifest.json
again = PubMedClipPipeline.from_artifact("outputs/adapter", weights_dir="weights/pubmed-clip-vit-base-patch32")
```

- `validate_dataset(records)` checks `{id, image, label}` records structurally (decodable image up to `MAX_IMAGE_SIDE` = 4096 px, label of at most 64 plain characters, 8..20,000 records over 2..100 labels, unique ids) and returns a manifest with a dataset digest; `split_dataset` is a seeded, stratified, pixel-digest-deduplicated split for BYOD data; `build_sample_dataset` keeps the sample in MedMNIST's own roles; `check_split_disjoint` asserts no image is shared.
- `evaluate(records, *, classes, prompt_template, class_names_map)` embeds every prompt once and every image once, scores them with the model's own `logits_per_image` (cosine scaled by the learned `logit_scale`) and returns accuracy, macro F1, per-class recall / precision / F1 / average precision and text-to-image mAP (`metrics.classification_metrics`), plus `predictions`, `verdict` (`measured` / `measured-small-sample`) and `adapted`.
- `adapt(train, val=None, *, epochs=6, lr=5e-5, batch_size=16, trainable_vision_layers=2, seed=0, progress=None)` trains only the last `trainable_vision_layers` blocks of the vision tower, its post-layernorm and the visual projection (14,570,496 of 151,277,313 parameters by default) with the cross-entropy of the softmax over the class prompts against the frozen text tower's prompt features; AdamW, gradient clipping at 1.0, seeded shuffling, epoch 0 recorded as the frozen model, the epoch with the highest validation text-to-image mAP kept (the final epoch without validation). The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.pubmed-clip-vit-base-patch32.adapter.v1`: base id, revision and weight digest, classes, prompt template, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the vision tower and its projection, and overlays the tensors onto a freshly loaded base.
- `majority_baseline` and `colour_neighbour_baseline` (`metrics.py`) are the two non-neural references the tutorial scores beside the frozen and adapted model; on grey-scale CT the second is a 3×3 mean-intensity nearest neighbour.

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/tutorials/pubmedclip_biomedical_colab.ipynb)

`tutorials/pubmedclip_biomedical_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.5 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's seven modules, the model identity (`flaviagiammarino/pubmed-clip-vit-base-patch32` at the immutable revision `26c0c67f6da303ad2a38909130bd35744ea93517`), the 9-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). It stages and digest-verifies the snapshot, audits and converts the pickle once (the report is printed before the model loads), fetches the digest-pinned MedMNIST+ OrganAMNIST-224 archive and keeps exactly the 660 pinned CT slices in the dataset's own scan-disjoint roles, classifies the three deterministic sample shapes through the inference contract with an input manifest and a rejection probe, scores the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP on the 176 held-out slices beside the majority-floor and intensity-nearest-neighbour baselines and a second prompt set, runs the pre-registered configuration grid on 352 validation slices and a bounded fine-tuning of the vision tower's last blocks (as many as the grid selects) with validation-mAP epoch selection, scores the fresh test only under the pre-registered settings and the exposed development test per organ and the shapes, exports the adapter and reloads it with verified parity, and writes:

- `pubmedclip_biomedical_train.csv`
- `pubmedclip_biomedical_input_manifest.json`
- `pubmedclip_biomedical_evaluation_report.json`
- `pubmedclip_biomedical_shapes.json`
- `pubmedclip_biomedical_adapter/` (`adapter.safetensors`, `manifest.json`)
- `pubmedclip_biomedical_result.json`
- `provenance.json`

The default path runs on CPU and uses CUDA automatically when present (about 15 minutes of model time on the build workstation's CPU after the downloads in the 2026-10-17 CPU build record — 519 s of it the pre-registered three-configuration grid on validation and 311 s the final fine-tuning — longer on a 2-vCPU hosted runtime; a hosted T4 finishes in a few minutes). The metrics it prints are one seeded run on one 660-slice sample — evidence that the adaptation contract works, not an accuracy benchmark, not clinical evidence and not production-fitness evidence. The `main` integration workflow executes the notebook's code cells on the frozen CPU reference environment as a pre-flight; see `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate** — The 2026-10-17 review-fix revision (branch `fix/pubmedclip-tutorial-review-2026-10-17`: pre-registered grid on 352 never-used validation slices, a 176-slice fresh test scored only under the pre-registered settings, the earlier 176 test slices kept as the labelled exposed development test; see `docs/pubmedclip-preregistration.md`) is now hosted: blob `c5a0e62d` (commit `352452b`) passed on a fresh Colab Tesla T4 on 2026-10-17 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 17/17 code cells, 0 errors, 1009.0 s wall, 743.6 s of it the Zenodo archive fetch), and that run is the **pre-registered evaluation of record** (REL13; `test_look` pre-registered evaluation, `settings_differing` []): grid C2 selected on validation (mAP C0 0.772 / C1 0.678 / C2 0.888, frozen 0.296); fresh test accuracy 0.165 → 0.881, text-to-image mAP 0.248 → 0.862; frozen-embedding probe on the fresh test accuracy 0.824, mAP 0.885 (fine-tuning improves classification accuracy (fresh 0.881 vs the probe's 0.824), but a linear readout on the frozen embeddings is as good or better for retrieval on all three splits (mAP fresh 0.885 vs 0.862, exposed development test 0.851 vs 0.821, validation 0.968 vs 0.888), so the retrieval gain should not be credited to fine-tuning the vision layers); evidence in `docs/execution-evidence/2026-10-17-352452b/`. The requoted notebook (blob `15f853a5`, PR #8 head `2dfd282`) then passed on a fresh Colab Tesla T4 on 2026-10-17 (Colab account 2; Colab CLI, 17/17 code cells, 0 errors, 1091.9 s, 816.6 s of it the archive fetch). REL13: reproduction of the pre-registered evaluation, this blob: the notebook printed `test_look` `pre-registered evaluation (this runtime)` with `relation_to_record` `reproduction of the pre-registered evaluation` and no setting differing, and every grid, validation, fresh-test, exposed-test and probe figure equals the evaluation of record (evidence in `docs/execution-evidence/2026-10-17-2dfd282/`). The CPU build record is a disclosed development look. REL13: earlier-blob history follows. Earlier: the review-fix blob `066b560d` (commit `a197c8b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 15/15 code cells, 917.8 s wall, 769.9 s of it the Zenodo archive fetch; isolated uv environment Python 3.12.12, 47 locked packages, built in 53 s; torch 2.14.0+cu130, transformers 4.57.6, `cuda`; accuracy majority 0.091 / neighbour 0.511 / frozen 0.182 / adapted 0.812, text-to-image mAP 0.091 / 0.519 / 0.284 / 0.654, best epoch 4, `ap_fell` [lung-right], reload parity 8 of 8 exact; evidence in `docs/execution-evidence/2026-10-08-a197c8b/`). The previous blob `4b2fc4c4` (commit `d125896`) ran on Kaggle T4 on 2026-09-21 only after a manual restart after its in-kernel install cell (14/14 ok, 415.1 s), so it was not one-pass evidence. REL12 BYOD exercise pending before promotion; the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks are necessary but were never the evidence; the hosted run is.

The current blob `aa43cf74` (code commit `3457eba`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). On 2026-10-29 its default path (17/17 code cells, one pass, no restart, 0 errors) and its scripted DAT20 refusal (`USE_BYOD=True;BYOD_PATH='/etc/hostname'`: exactly one `Refused:` line at cell `pubmedclip_biomedical-24`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI sequential execution, not a browser `Run all`; see "Recorded executions" in `docs/release-verification.md`), and the hosted runs of earlier blobs, including `15f853a5`, are history for it (REL14).

## Score semantics

CLIP scores an image against a set of candidate prompts by the cosine of their projected embeddings scaled by the learned `logit_scale`. `zero_shot_classify()` returns a **softmax over the candidate labels**: the scores sum to one but are relative to the candidates given, depend on the prompt wording, and are not calibrated probabilities that a label is right — the model never abstains. `evaluate()` ranks and scores with the unnormalised logits, so accuracy and average precision do not depend on the candidate set's size.

The default prompt template is:

```text
A medical image of {label}.
```

The tutorial passes its own prompt for the CT slices (`An axial abdominal CT slice showing the {label}.`) and scores a second, raw-label prompt set beside it.

## Text preprocessing compatibility

CLIP's BPE tokenizer lowercases and whitespace-cleans text in its own pre-processing, and PubMedCLIP kept that tokenizer; the pipeline lowercases model-bound text itself as well, so the behaviour does not depend on which tokenizer class transformers resolves. Text is padded to CLIP's context of 77 tokens and truncated beyond it. This applies to zero-shot prompts, text embeddings, similarity, and retrieval queries. Caller-facing labels are preserved exactly as supplied.

## Embeddings and retrieval

`embed_image()` and `embed_text()` return L2-normalized 512-dimensional vectors. `similarity()` and `retrieve()` use cosine similarity through the dot product of those normalized vectors.

Retrieval is text-to-image retrieval over an in-memory image list. It returns the source index and score for each ranked hit.

## Machine-readable provenance

```python
from pubmedclip_pipeline import (
    DEFAULT_WEIGHTS_DIR, PubMedClipPipeline, build_provenance, write_provenance,
)

pipe = PubMedClipPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR, allow_download=True)
record = build_provenance(pipeline=pipe)
write_provenance("outputs/provenance.json", pipeline=pipe)
```

The record includes model ID, immutable revision, weight filename/SHA-256/size, verified checkpoint source and path, processor contract, prompt template, score/embedding semantics, Python version, platform, and runtime package versions.

## Input safety

Image inputs may be:

- a local filesystem path;
- `bytes` containing an image;
- a `PIL.Image.Image`.

Remote `http://` and `https://` image strings are rejected intentionally. The pipeline does not act as a network fetcher.

## Supply-chain controls

`load_pipeline()`:

1. resolves offline weights through `weights_path`, `PUBMEDCLIP_WEIGHTS_DIR`, dev repo `weights/pubmed-clip-vit-base-patch32`, or pinned Hugging Face revision fallback (that revision hosts only the pickle source, so the fallback is refused until `model.safetensors` exists — a fresh clone loads through `from_pretrained(weights_dir=...)` below, as the examples above do);
2. verifies snapshot files against `dimer-base-manifest.json` when present (checking hashes and sizes of configurations, tokenizer files and the pinned PyTorch source);
3. rejects every unsafe serialized format (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.h5`, `.msgpack`) except the one pinned `pytorch_model.bin`, which is tolerated only as the digest-verified conversion source and is never passed to transformers;
4. verifies the exact safetensors byte size and SHA-256 before model load;
5. loads the verified local snapshot with `trust_remote_code=False`, `use_safetensors=True`, and `local_files_only=True`.

`PubMedClipPipeline.from_pretrained(weights_dir=...)` adds the conversion step: when `model.safetensors` is absent, `convert_source()` checks the source's size and digest, lists the globals its pickle would import with `pickletools` (the fleet's four state-dict globals — `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `torch.LongStorage` — and nothing else; audit digest `5b9f0ba0…`), unpickles it exactly once through `torch.load(weights_only=True)`, loads the 400 tensors strictly into a `CLIPModel` built from the snapshot's `config.json` (the two non-persistent `position_ids` buffers dropped), and writes the 398-tensor state dict as safetensors, refusing the result if its digest is not the pinned one. The conversion is deterministic (two runs produce identical bytes).

## Weights layout

```
weights/pubmed-clip-vit-base-patch32/   pytorch_model.bin            (git-ignored, the pinned source)
                                        model.safetensors            (git-ignored, converted)
                                        config.json, preprocessor_config.json, tokenizer.json, tokenizer_config.json,
                                        special_tokens_map.json, vocab.json, merges.txt, README.md
                                        dimer-base-manifest.json
weights/organamnist/                    organamnist_224.npz + slices/   (git-ignored, the pinned archive and the 1,100 extracted slices)
```

With the converted file present the source pickle is not needed — the DIMER-hosted shape: `verify_snapshot` accepts a directory holding the digest-verified `model.safetensors` and the small files without `pytorch_model.bin`, and `stage_missing_files` does not fetch it. `docs/WEIGHTS.md` records the provenance, the audit, the conversion, the data pins and terms, and the DIMER hosting notes.

## Sample data

`fetch_corpus()` fetches `organamnist_224.npz` (1.8 GB) from the MedMNIST authors' Zenodo record at a fixed URL, verifies it by size and SHA-256, reads it with `numpy.load(allow_pickle=False)` and keeps exactly the 1,100 pinned slices (each verified against its own digest) under `weights/organamnist/slices/`; `read_corpus` decodes them into `{id, image, label}` records with their MedMNIST split, index and organ name, and `build_sample_dataset` keeps the dataset's own roles (36 / 8 / 16 per organ from the official train / val / test splits → 396 / 88 / 176; MedMNIST splits OrganAMNIST by CT scan). Nothing is vendored under `weights/`; the data are CC BY 4.0.

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions proven by the real-checkpoint `main` CI path, including the official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check .
pytest -m "not integration"
```

`tests/test_model_backed.py` (evaluate / adapt / artifact round trip / loader scope / transactional restore, and the same path on CUDA where visible; the build ran it CPU-only) runs only when the snapshot is staged and converted under `weights/pubmed-clip-vit-base-patch32/`; the archive cache `weights/organamnist/` is git-ignored and filled by `fetch_corpus()`.

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/pubmedclip_biomedical_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- any diagnostic, triage, screening or other clinical function;
- object detection;
- semantic segmentation;
- image caption generation;
- OCR;
- calibrated zero-shot probabilities;
- universal classification thresholds;
- fine-tuning of the text tower, the embeddings, the text projection or `logit_scale`, or any adaptation beyond the vision tower's last blocks and projection;
- production HTTP serving or DIMER worker packaging.

Those require separate downstream heads, models, calibration, validation, or serving work.

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
