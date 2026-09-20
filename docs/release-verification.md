# Release verification

`tutorials/pubmedclip_biomedical_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `metrics.py`, `model.py`, `provenance.py`, `samples.py`,
  `pipeline.py`, in dependency order), each equal to its source after the generator's documented rewrites (the
  `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of
  package-relative imports); the inline `MANIFEST` equal to the committed 9-entry snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `PubMedClipPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `source_overlap`, `class_names`, `write_dataset_csv`, the four dataset refusal
  probes, the ceiling print, the digest-asserted synthetic shapes, `validate_inputs` with the remote-URL refusal
  probe, `zero_shot_classify` / `retrieve` / `embed_image` with the sanity checks and the per-grid
  `evaluation_report` on the drawn shapes, `majority_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` on the
  frozen model with the CT prompt set and the raw-label prompt set and the baseline assertion, `pipe.adapt` with
  its explicit hyperparameters, `pipe.evaluate` on the validation and test splits after adaptation with the mAP
  assertion, `evaluation_report` on the shapes after adaptation, `pipe.save_artifact`,
  `PubMedClipPipeline.from_artifact` and the reload-parity assertion, `write_provenance`, and the result fields
  `weight_file` / `weight_format` / `weight_sha256` / `source_sha256` / `pickle_audit_sha256` and the `corpus` block), the seven expected `outputs/` paths, the
  learner-facing statements (MIT weights, the non-clinical statement, uncalibrated and prompt-dependent softmax scores, adaptation with
  labelled images, the CC BY 4.0 corpus, text-to-image mAP, the two non-neural baselines, no dispersion estimate, the
  leakage and prompt guidance, the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden
  patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `AutoModel` / `AutoProcessor` / `get_image_features` /
  `torch.sigmoid` / `from huggingface_hub import` / `snapshot_download` / `urllib.request` / `safetensors` imports /
  `torch.optim` / `.backward(` / `requires_grad` / `logit_scale` / `pipe.model.` / `extractall(` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_adaptation.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; no weights, injected downloader and photo fetcher —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/pubmed-clip-vit-base-patch32/` or the data cache `weights/organamnist/` (the standalone path
   writes the manifest itself, stages the missing files from the Hub, converts the pickle, and fetches the pinned
   archive from Zenodo, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 16`,
   `TRAINABLE_VISION_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the six carried module cells execute (defining `PubMedClipPipeline`, `verify_snapshot`, `stage_missing_files`,
     `audit_pickle`, `convert_source`, `validate_inputs`, `evaluation_report`, `classification_metrics`,
     `majority_baseline`, `colour_neighbour_baseline`, `fetch_archive`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `source_overlap`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `write_provenance` and the ceilings) with no import of the repository
     package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 9 manifest entries fetched from `flaviagiammarino/pubmed-clip-vit-base-patch32`
     at the immutable revision on a clean runtime (the 605 MB `pytorch_model.bin` among them), `verify_snapshot`
     returning its dict with `converted: False`, and `from_pretrained(weights_dir=WEIGHTS_DIR, progress=print)`
     printing the conversion report — the static audit (four globals, digest `5b9f0ba0…`), 400 source tensors, the two
     `position_ids` buffers dropped, `model.safetensors` 605,156,676 B / `81de21a0…` — before loading from the verified
     directory;
   - Section 4: `fetch_corpus` fetching `organamnist_224.npz` (1,803,859,544 B, SHA-256 `a19bae53…`) and
     extracting the 660 pinned slices with every digest matching; the dataset's own roles 396 / 88 / 176 (36 / 8 / 16
     per organ) with `check_split_disjoint` reporting no shared image, `source_overlap` reporting one MedMNIST split
     per role and the three dataset digests printed; `outputs/…_train.csv` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings (`TEXT_MAX_LENGTH` 77, `DEFAULT_PROMPT_TEMPLATE`, `MAX_IMAGE_SIDE` 4096,
     `MIN_RECORDS` 8, `MAX_RECORDS` 20000, `MIN_CLASSES` 2) surfaced; the three synthetic PPM images generated in
     code with SHA-256 `b38ff0c9…` / `2e3e6576…` / `f0f4c38c…` (equal to `examples/sample-data/SHA256SUMS`);
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the remote-URL probe); `zero_shot_classify` ×3, `retrieve` ×3 and `embed_image` with every sanity check
     `True` (scores in the unit interval and summing to one) and the per-grid `evaluation_report` verdict
     `sample-sanity` (the drawn shapes may rank wrongly — a radiology model owes them nothing);
   - Section 6: the majority floor (accuracy 0.091), the intensity nearest neighbour (accuracy ≈ 0.51 in the
     build record) and the frozen model's test score (accuracy ≈ 0.18, macro F1 ≈ 0.13, text-to-image
     mAP ≈ 0.28 on the CPU build record; raw-label prompts: accuracy ≈ 0.18) with the per-organ
     breakdown, and the cell's assertion that the frozen model is above the majority floor on accuracy and mAP;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 14,570,496 trainable of 151,277,313 parameters,
     and a six-epoch history with the validation mAP selecting the epoch (`best_epoch` 4 in the build
     record);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the three
     metrics, the raw-label prompts, the per-organ breakdown and `outputs/…_evaluation_report.json` written
     (the cell asserts the adapted test mAP exceeds the frozen one — 0.65 versus 0.28 in the build
     record, with accuracy 0.18 → 0.81);
   - Section 9: the three shapes re-scored by the adapted model with the `sample-sanity` report,
     `outputs/…_shapes.json` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (35 tensors, 58,286,104 bytes) and
     `PubMedClipPipeline.from_artifact` reloading it with 8/8 identical image embeddings on eight test slices (the
     cell asserts it); `outputs/provenance.json` and `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`, `source_file`,
     `source_sha256`, `pickle_audit_sha256`), the `corpus` block, the inference-contract reports, the comparison, the
     artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the data cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `pubmedclip_biomedical_colab.ipynb` (`E2E`) | — | — | — | **not yet executed** in a clean supported runtime; the first clean-room execution is queued on the workspace's Kaggle serial suite and will be recorded here |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pubmedclip_biomedical_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pubmedclip_biomedical_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0, transformers 4.57.6; snapshot staged and converted, the archive fetched from Zenodo in the run) | The notebook's default path replayed cell by cell through the package API (`fetch_corpus` → 660 slices, `build_sample_dataset(seed=42)` → 396 / 88 / 176, `validate_dataset` per split, `check_split_disjoint`, `majority_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` frozen with both prompt sets, `pipe.adapt` at the defaults, `pipe.evaluate` adapted, `save_artifact`, `from_artifact` with embedding parity): the archive already staged (extraction of the 660 slices 14.7 s), model load 2.2 s, the frozen model scored in 3.5 s, six epochs of the default recipe in 79.7 s (validation mAP 0.355 → 0.673 → 0.758 → 0.708 → 0.761 → 0.758 → 0.689, epoch 4 kept; validation accuracy 0.307 → 0.602 → 0.807 → 0.943 → 0.943 → 0.955 → 0.909), the adapter 35 tensors / 58,286,104 bytes with reload parity 0.0 (max abs difference over eight embedded test slices); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}, by_organ: {bladder: {n: 16, frozen_recall: 0.06, adapted_recall: 0.75, frozen_ap: 0.42, adapted_ap: 0.63}, femur-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.75, frozen_ap: 0.06, adapted_ap: 0.7}, femur-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.08, adapted_ap: 0.51}, heart: {n: 16, frozen_recall: 1.0, adapted_recall: 0.94, frozen_ap: 0.25, adapted_ap: 0.65}, kidney-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.62, frozen_ap: 0.08, adapted_ap: 0.62}, kidney-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.38, frozen_ap: 0.09, adapted_ap: 0.47}, liver: {n: 16, frozen_recall: 0.25, adapted_recall: 1.0, frozen_ap: 0.49, adapted_ap: 0.58}, lung-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.94, frozen_ap: 0.36, adapted_ap: 0.98}, lung-right: {n: 16, frozen_recall: 0.25, adapted_recall: 0.94, frozen_ap: 0.79, adapted_ap: 0.34}, pancreas: {n: 16, frozen_recall: 0.44, adapted_recall: 1.0, frozen_ap: 0.21, adapted_ap: 0.79}, spleen: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.29, adapted_ap: 0.92}}}; drawn shapes frozen {red_square.ppm: [[red square, 0.955], [abstract geometric shape, 0.041], [blue triangle, 0.004], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.771], [abstract geometric shape, 0.179], [red square, 0.046], [blue triangle, 0.004]], blue_triangle.ppm: [[blue triangle, 0.859], [abstract geometric shape, 0.081], [red square, 0.06], [green circle, 0.0]]} → adapted {red_square.ppm: [[red square, 0.958], [abstract geometric shape, 0.041], [blue triangle, 0.0], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.91], [abstract geometric shape, 0.087], [red square, 0.003], [blue triangle, 0.0]], blue_triangle.ppm: [[blue triangle, 0.914], [abstract geometric shape, 0.077], [red square, 0.01], [green circle, 0.0]]} | 123.3 s with the archive already on disk (the 1.8 GB download itself took 1,177 s from Zenodo at about 1.5 MB/s) s including the archive fetch | PASS — pre-flight only (no notebook, no GPU); not promotion evidence |

## Current status

**Candidate.** No clean-runtime execution of the committed `E2E` notebook blob has been recorded yet. The build
workstation's CPU pre-flight above shows the default path completing through the package API with the numbers the
notebook prose quotes; it is not REL1/REL10 supported-runtime evidence, and a GPU has not run this repository at all
(GPU execution is deferred to the hosted Kaggle / Colab run by the workspace's standing rule). The registry moves to
**Release-grade** when the committed blob runs top-to-bottom in a clean Kaggle or Colab runtime and that run is
recorded here; any later change to the carried modules or to the notebook produces a new blob and returns the
registry to **Candidate** until a clean run of that blob is recorded.

Facts a reviewer should still weigh: the frozen model is a weak zero-shot classifier on eleven abdominal organs — accuracy 0.18 against a majority floor of 0.09 and an intensity nearest neighbour of 0.51, because organs sit at characteristic positions in an axial slice and a caption model knows nothing of laterality (it answered *heart* for most slices, heart recall 1.00 on an average precision of 0.25, and never found a kidney, a femur or the spleen); the bounded adaptation moves it past both baselines (accuracy 0.18 → 0.81, macro F1 0.13 → 0.81, text-to-image mAP 0.28 → 0.65; the raw-label prompt set reaches 0.82 / 0.81 / 0.82), so the claim is that the contract works on a task the frozen model could not do, not that the adapted numbers are good; the 88-slice validation split selected epoch 4 on mAP (0.761) while validation accuracy moved between 0.91 and 0.96 across epochs 3–6, which is what "no dispersion estimate" means here; the left / right pairs stayed the weakest after adaptation (right kidney recall 0.38, right femur AP 0.51) and the heart lost the recall it only had as the default answer; and the drawn shapes ranked correctly before and after adaptation, which a radiology-caption model owed nobody. The drawn shapes re-scored after adaptation are three images
of evidence about behaviour outside the corpus, not a measurement, and nothing in the notebook is clinical evidence.
