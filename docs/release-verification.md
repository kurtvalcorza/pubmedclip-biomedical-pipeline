# Release verification

`tutorials/pubmedclip_biomedical_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.5 (REL8). This file is the durable release-gate
record for the notebook.

Current revision: the 2026-10-17 review-fix revision (branch `fix/pubmedclip-tutorial-review-2026-10-17`: pre-registered grid on 352 never-used validation slices, a 176-slice fresh test scored only under the pre-registered settings, the earlier 176 test slices kept as the labelled exposed development test; see `docs/pubmedclip-preregistration.md`) is now hosted: blob `c5a0e62d` (PR #8 head `352452b`) passed on a fresh Colab Tesla T4 on 2026-10-17 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 17/17 code cells, 0 errors, 1009.0 s wall, 743.6 s of it the Zenodo archive fetch), and that run is the **pre-registered evaluation of record** (REL13; `test_look` pre-registered evaluation, `settings_differing` []): grid C2 selected on validation (mAP C0 0.772 / C1 0.678 / C2 0.888, frozen 0.296); fresh test accuracy 0.165 → 0.881, text-to-image mAP 0.248 → 0.862; frozen-embedding probe on the fresh test accuracy 0.824, mAP 0.885; evidence in `docs/execution-evidence/2026-10-17-352452b/`. The CPU runs of 2026-10-17 in Recorded executions are disclosed development looks. REL13: the remaining figures in this paragraph describe earlier blobs. Status: **Candidate**. Previous blob: the review-fix blob `066b560d` (commit `a197c8b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 15/15 code cells, 917.8 s wall, 769.9 s of it the Zenodo archive fetch; isolated uv environment Python 3.12.12, 47 locked packages, built in 53 s; torch 2.14.0+cu130, transformers 4.57.6, `cuda`; accuracy majority 0.091 / neighbour 0.511 / frozen 0.182 / adapted 0.812, text-to-image mAP 0.091 / 0.519 / 0.284 / 0.654, best epoch 4, `ap_fell` [lung-right], reload parity 8 of 8 exact; evidence in `docs/execution-evidence/2026-10-08-a197c8b/`). The previous blob `4b2fc4c4` (commit `d125896`) ran on Kaggle T4 on 2026-09-21 only after a manual restart after its in-kernel install cell (14/14 ok, 415.1 s), so it was not one-pass evidence. Release step 7 (REL12, BYOD) has not been exercised, so the status is `Candidate` (REL14).

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.5`, a §3.3 pedagogical mode,
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
| `pubmedclip_biomedical_colab.ipynb` (`E2E`) | `a197c8b` / `066b560d` | 2026-10-08 | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-pubmedclip-a197c8b-2f7a`); kernel Python 3.13.15, isolated uv environment Python 3.12.12 (47 locked packages, 53 s), torch 2.14.0+cu130, transformers 4.57.6, `cuda` | **PASSED** — 15/15 code cells, 0 errors, no restart, 917.8 s wall (Zenodo fetch 769.9 s); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, t2i_map: {majority: 0.091, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}}; best epoch 4 (val mAP 0.355 -> 0.761); ap_fell [lung-right], recall_fell [heart]; reload parity {max_abs_difference: 0.0, identical_rows: 8, of: 8}; executed notebook sha256 `45662eec514c7172...`, evidence byte-for-byte in `docs/execution-evidence/2026-10-08-a197c8b/`; BYOD and optional journeys not exercised |
| `pubmedclip_biomedical_colab.ipynb` (`E2E`) | `d125896` / `4b2fc4c4` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pubmedclip-biomedical` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130`, `transformers 4.57.6` after, Python 3.12.13, `cuda`) | **PASSED** — 14/14 code cells ok (1 restart after install cell); 682 files, 3051 MB staged into a clean cache (the Hub snapshot, audited and converted in the notebook, plus the Zenodo OrganAMNIST archive); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}} (per-organ recall / AP in the archived result); reload parity {max_abs_difference: 0.0, identical_rows: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pubmedclip-biomedical/v1/evidence/` in the workspace |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pubmedclip_biomedical_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pubmedclip_biomedical_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-17 | `352452b` (`352452b13187cd99eaea0eea48b5d5ac8ce8f86d`, PR #8 head) / blob `c5a0e62d` (`c5a0e62d86aa073ceb9fb81c730b9c42c4ff27a7`; fetched at that commit and blob-verified before the session). Evidence under `docs/execution-evidence/2026-10-17-352452b/`: `pubmedclip_biomedical_colab_352452b_colab-cli-t4_output.ipynb` SHA-256 `4513a0f43d2706fc8e9c22f7f2006bed4a6e11c4f3ef9c01e15c39a08a5369f0`, `exec.log` SHA-256 `9134e14e055f698606cb3f22da22c28f16ca51cd5eb55d8ce5bc0ba4dcbe9e32`, `run_summary.json` SHA-256 `dcc9a22642126fa19cd921b950a16240e48b2d7057be903621cfd21e5109ce5a` (copied byte for byte) | Colab CLI 0.7.4, fresh Colab **Tesla T4** (Colab account 2; session `suite-pubmedclip-352452b-2f7a` started 2026-10-17T14:03:00Z, stopped after the run at 14:19:47Z) via the workspace `colab-cli-serial-test-suite`; isolated uv Python 3.12.12 (built in 57 s), `cuda`; not a browser Run all | Default sample path, pre-registered settings, no repository checkout; BYOD off: 1,100 slices (396 / 352 / 176 exposed development test / 176 fresh test) → frozen and baselines → frozen-embedding probe → grid C0 / C1 / C2 on validation → C2 → fresh test, exposed development test → shapes → adapter export → reload parity | 1009.0 s (743.6 s archive fetch; grid 83.3 s) | **PASSED — the pre-registered evaluation of record (REL13: hosted, this blob).** 17/17 code cells in order, 0 errors, no restart; `preregistered` True, `test_look` `pre-registered evaluation (this runtime)`, `settings_differing` []. Grid best-epoch validation mAP C0 0.7716 (epoch 2), C1 0.6782 (epoch 4), C2 0.8876 (epoch 6), frozen 0.2958 → **C2** (beat C0 by ≥ 0.02), adaptation kept. **Fresh test (176):** accuracy majority 0.091 / neighbour 0.619 / frozen 0.165 / probe 0.824 / adapted **0.881**; macro F1 0.015 / 0.61 / 0.127 / 0.819 / 0.881; mAP 0.091 / 0.538 / 0.248 / **0.885** / **0.862**. Exposed development test (176): accuracy 0.091 / 0.511 / 0.182 / 0.801 / 0.869; mAP 0.091 / 0.519 / 0.284 / 0.851 / 0.821. No confidence intervals are printed. Reload parity 8/8 exact. See the record below |
| 2026-10-17 (round 2) | working tree before the round-2 code commit (carried `metrics.py` with `fit_embedding_probe`) | Build workstation CPU, package API (`scripts`-style check, not the notebook), `CUDA_VISIBLE_DEVICES=-1` | Frozen-embedding ridge readout (amendment 1) with the **fresh test masked**: fitted on the 396 train slices, ridge chosen on the **88 retired validation slices** (the 352 new ones need the 1.8 GB archive, which Zenodo served at about 30 KB/s that day), scored on the exposed development test | 35 s | Ridge 0.01 (validation mAP by ridge 10: 0.779, 1: 0.862, 0.1: 0.919, 0.01: 0.964, 0.001: 0.953); retired-88 validation accuracy 0.943 / macro F1 0.943 / mAP 0.964; **exposed development test accuracy 0.801 / macro F1 0.802 / mAP 0.851** — against 0.858 / 0.859 / 0.817 for the C2-adapted model in the development look above: a linear readout of the frozen tower gets most of, or more than, the fine-tuning gain on the retrieval view. Informs no decision; not release evidence |
| 2026-10-17 | **uncommitted working tree** on `fix/pubmedclip-tutorial-review-2026-10-17` that was then committed as `61d464c` (the notebook it ran is byte-identical in carried code to the one generated at `61d464c`/`609ae19`, except one ruff comment line in `protocol.py` and the stamp cell) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Windows, Python 3.12, torch 2.14.0+cpu, transformers 4.57.6), `tools/run_notebook.py` with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (in-kernel, no isolated environment), snapshot and archive already on disk | Default sample path, pre-registered settings (`test_look`: pre-registered evaluation, `settings_differing` []) — a **development look** (SCORE_ONCE_RULE item 8) | 999 s (grid 519 s, final adaptation 311 s) | PASS — a development run, **not clean-runtime or release evidence**. Grid best-epoch validation mAP (352 slices): C0 0.772, C1 0.678, C2 0.892, frozen 0.296 → **C2 selected** (4 blocks, 5e-5), adaptation kept; final run best epoch 6 (validation mAP 0.892, accuracy 0.963). **Fresh test (176):** accuracy majority 0.091 / neighbour 0.619 / frozen 0.165 / adapted 0.886; macro F1 0.015 / 0.61 / 0.127 / 0.887; text-to-image mAP 0.091 / 0.538 / 0.248 / 0.865; raw-label prompts mAP 0.328 → 0.912; verdict improved; `ap_fell` [lung-right] (0.71 → 0.60), `recall_fell` []. **Exposed development test (176, reported in every run for continuity):** accuracy 0.091 / 0.511 / 0.182 / 0.858, mAP 0.091 / 0.519 / 0.284 / 0.817. Adapter 67 tensors, 114,992,888 bytes; reload parity 8 of 8, max abs difference 0.0. This fresh-test score is a disclosed development look: it changed no setting (pre-registration amendment 1 changes none), and the first hosted T4 run with the pre-registered settings is the evaluation of record; later identical hosted runs are reproductions |
| 2026-10-08 (23:30:57 UTC start) | `a197c8b` / `066b560d` (`NOTEBOOK_SOURCE.repository_revision` `9e4ca4b`, the fix commit; `9e4ca4b..a197c8b` only regenerates the notebook) | Colab CLI 0.7.4, fresh Colab Tesla T4 | Default sample path, committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; BYOD off | 917.8 s | **PASSED** — one pass, no restart, 0 errors; numbers as in the table above (the majority floor's mAP now prints the prevalence 0.091; earlier runs printed 0.119, a tie-order artefact fixed by PMC-m6) |
| 2026-09-21 | `d125896` / `4b2fc4c4` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pubmedclip-biomedical` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130`, `transformers 4.57.6` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) — the notebook's first execution anywhere and this repository's first GPU run | 415.1 s | **PASSED** — 14/14 code cells ok (1 restart after install cell); 682 files, 3051 MB staged into a clean cache (the Hub snapshot, audited and converted in the notebook, plus the Zenodo OrganAMNIST archive); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}} (per-organ recall / AP in the archived result); reload parity {max_abs_difference: 0.0, identical_rows: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pubmedclip-biomedical/v1/evidence/` in the workspace |
| 2026-09-20 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0, transformers 4.57.6; snapshot staged and converted, the archive fetched from Zenodo in the run) | The notebook's default path replayed cell by cell through the package API (`fetch_corpus` → 660 slices, `build_sample_dataset(seed=42)` → 396 / 88 / 176, `validate_dataset` per split, `check_split_disjoint`, `majority_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` frozen with both prompt sets, `pipe.adapt` at the defaults, `pipe.evaluate` adapted, `save_artifact`, `from_artifact` with embedding parity): the archive already staged (extraction of the 660 slices 14.7 s), model load 2.2 s, the frozen model scored in 3.5 s, six epochs of the default recipe in 79.7 s (validation mAP 0.355 → 0.673 → 0.758 → 0.708 → 0.761 → 0.758 → 0.689, epoch 4 kept; validation accuracy 0.307 → 0.602 → 0.807 → 0.943 → 0.943 → 0.955 → 0.909), the adapter 35 tensors / 58,286,104 bytes with reload parity 0.0 (max abs difference over eight embedded test slices); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}, by_organ: {bladder: {n: 16, frozen_recall: 0.06, adapted_recall: 0.75, frozen_ap: 0.42, adapted_ap: 0.63}, femur-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.75, frozen_ap: 0.06, adapted_ap: 0.7}, femur-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.08, adapted_ap: 0.51}, heart: {n: 16, frozen_recall: 1.0, adapted_recall: 0.94, frozen_ap: 0.25, adapted_ap: 0.65}, kidney-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.62, frozen_ap: 0.08, adapted_ap: 0.62}, kidney-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.38, frozen_ap: 0.09, adapted_ap: 0.47}, liver: {n: 16, frozen_recall: 0.25, adapted_recall: 1.0, frozen_ap: 0.49, adapted_ap: 0.58}, lung-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.94, frozen_ap: 0.36, adapted_ap: 0.98}, lung-right: {n: 16, frozen_recall: 0.25, adapted_recall: 0.94, frozen_ap: 0.79, adapted_ap: 0.34}, pancreas: {n: 16, frozen_recall: 0.44, adapted_recall: 1.0, frozen_ap: 0.21, adapted_ap: 0.79}, spleen: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.29, adapted_ap: 0.92}}}; drawn shapes frozen {red_square.ppm: [[red square, 0.955], [abstract geometric shape, 0.041], [blue triangle, 0.004], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.771], [abstract geometric shape, 0.179], [red square, 0.046], [blue triangle, 0.004]], blue_triangle.ppm: [[blue triangle, 0.859], [abstract geometric shape, 0.081], [red square, 0.06], [green circle, 0.0]]} → adapted {red_square.ppm: [[red square, 0.958], [abstract geometric shape, 0.041], [blue triangle, 0.0], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.91], [abstract geometric shape, 0.087], [red square, 0.003], [blue triangle, 0.0]], blue_triangle.ppm: [[blue triangle, 0.914], [abstract geometric shape, 0.077], [red square, 0.01], [green circle, 0.0]]} | 123.3 s with the archive already on disk (the 1.8 GB download itself took 1,177 s from Zenodo at about 1.5 MB/s) s including the archive fetch | PASS — pre-flight only (no notebook, no GPU); not promotion evidence |

### 2026-10-17 Colab CLI T4 run of `352452b` (blob `c5a0e62d`): pre-registered evaluation of record

- **Executor:** Google Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM with a Tesla T4, Colab account 2, through the
  workspace `colab-cli-serial-test-suite` (session `suite-pubmedclip-352452b-2f7a`, 2026-10-17 14:03:00–14:19:47 UTC,
  stopped after the run). Sequential CLI execution, not a browser Run all; forms and uploads were not exercised.
- **Source identity:** fetched at the full PR #8 head `352452b13187cd99eaea0eea48b5d5ac8ce8f86d`; Git blob
  `c5a0e62d86aa073ceb9fb81c730b9c42c4ff27a7` checked before the VM was allocated; the executed notebook's 45 cells have the
  committed ids, order and sources. `NOTEBOOK_SOURCE.repository_revision` `aaa64515b28d`.
- **Outcome:** 17/17 code cells in order, one pass, no restart, 0 error outputs; 1009.0 s CLI wall, 743.6 s of it the
  1,803,859,544-byte Zenodo archive fetch; isolated environment built in 57 s; checkpoint audit with 0 violations and
  conversion to `model.safetensors` (605,156,676 bytes) in 8.0 s.
- **Evaluation status (REL13):** the notebook printed `preregistered` True, `test_look` `pre-registered evaluation (this
  runtime)` and `settings_differing` []. This is the first hosted run of the pre-registered settings, so it is **the
  pre-registered evaluation of record**; a later hosted run with identical settings is a reproduction.
- **Grid (validation, 352 slices; text-to-image mAP by epoch, epoch 0 = frozen 0.296):** C0 (2 blocks, 5e-5) 0.594, 0.772,
  0.71, 0.732, 0.758, 0.675 (best 0.7716, epoch 2); C1 (2 blocks, 1e-5) 0.5, 0.602, 0.676, 0.678, 0.666, 0.672 (best
  0.6782, epoch 4); C2 (4 blocks, 5e-5) 0.705, 0.76, 0.791, 0.86, 0.858, 0.888 (best 0.8876, epoch 6). **Selected C2**
  ("C2 beat C0 by >= 0.02"), adaptation kept; the same pick as the CPU development look. Final run: 28,746,240 of
  151,277,313 parameters trainable, best epoch 6, started from the pinned base.
- **Frozen-embedding probe (supervised ridge readout of the frozen towers, amendment 1):** ridge 0.01 chosen on the 352
  validation slices (validation mAP by ridge 10: 0.7258, 1: 0.8449, 0.1: 0.9389, 0.01: 0.9679, 0.001: 0.9544).
- **Results (accuracy / macro F1 / text-to-image mAP):**

  | Split | Majority | Neighbour | Frozen (zero-shot) | Frozen-embedding probe | Adapted (C2) |
  |---|---|---|---|---|---|
  | Validation (352; chose the configuration and epoch, biased upward) | — | — | — | 0.949 / 0.948 / 0.968 | 0.96 / 0.96 / 0.888 |
  | **Fresh test (176)** | 0.091 / 0.015 / 0.091 | 0.619 / 0.61 / 0.538 | 0.165 / 0.127 / 0.248 | 0.824 / 0.819 / **0.885** | **0.881 / 0.881** / 0.862 |
  | Exposed development test (176; continuity only) | 0.091 / 0.015 / 0.091 | 0.511 / 0.502 / 0.519 | 0.182 / 0.131 / 0.284 | 0.801 / 0.802 / 0.851 | 0.869 / 0.87 / 0.821 |

  Adapted − frozen on the fresh test: accuracy +0.716, macro F1 +0.754, mAP +0.614, verdict *improved*; raw-label prompts
  mAP 0.328 → 0.91 (accuracy 0.176 → 0.864). `ap_fell` [lung-right] (AP 0.71 → 0.64, recall 0.19 → 0.94); `recall_fell` [].
  The notebook prints no confidence intervals for these figures (one seed, 176 slices from 70 shared scans).
- **Probe versus fine-tuning:** on retrieval (text-to-image mAP) the frozen-embedding probe beats the adapted model on every
  split of this run: validation 0.968 vs 0.888, fresh test 0.885 vs 0.862, exposed development test 0.851 vs 0.821. On
  accuracy the adapted model leads (fresh 0.881 vs 0.824). So fine-tuning improves classification accuracy, but a linear
  readout on the frozen embeddings is as good or better for retrieval, and the retrieval gain should not be credited to
  fine-tuning the vision layers (the CPU development look showed the same on the exposed development test).
- **Development look (disclosed, score-once rule item 8):** the CPU development look of 2026-10-17 (uncommitted working
  tree later committed as `61d464c`; second row below) scored the fresh test at accuracy 0.165 → 0.886, mAP 0.248 → 0.865,
  with the same C2 choice (validation 0.892). It is not the evaluation of record.
- **Other outputs:** synthetic shapes `sample-sanity` (top-1 1.0 before and after); adapter 67 tensors, 114,992,888 bytes;
  reload parity 8 of 8 rows, max difference 0.0, reloaded best epoch 6.
- **Notebook quotes that this run contradicts (needs requote; this record does not edit the notebook):** cells 34, 35 and 37
  C2 / selected validation mAP 0.892 (hosted 0.888); cells 40 and 44 fresh-test adapted accuracy 0.886 (0.881) and mAP 0.865
  (0.862); cell 40 raw-label adapted mAP 0.912 (0.91), lung-right AP 0.71 → 0.60 (0.71 → 0.64), exposed development test
  adapted mAP 0.817 (0.821) and the probe comparison "0.817 / 0.858 for the adapted model" (0.821 / 0.869). Equal at the
  printed precision: C0 0.772, C1 0.678, frozen 0.296, C2 − C0 0.12, fresh frozen 0.165 / 0.248, neighbour 0.619,
  majority 0.091, raw-label frozen 0.328, recall 0.19 → 0.94, exposed probe 0.851 / 0.801, neighbour 0.511. Cell 40's
  probe sentence describes the CPU check (ridge chosen on 88 retired validation slices); the hosted probe chose its
  ridge on the 352 validation slices and also reads the fresh test (0.824 / 0.885), which the notebook prose does not quote
  yet. Follow-up: requote in the generator and regenerate (changes the notebook blob; not part of this docs-only record).
  Done on 2026-10-17 in the requote commits that follow this record; the notebook now names this run in `RECORDED_EVALUATION`.
- **Evidence files** (`docs/execution-evidence/2026-10-17-352452b/`, byte-exact copies):
  - `pubmedclip_biomedical_colab_352452b_colab-cli-t4_output.ipynb` — SHA-256 `4513a0f43d2706fc8e9c22f7f2006bed4a6e11c4f3ef9c01e15c39a08a5369f0`
  - `exec.log` — SHA-256 `9134e14e055f698606cb3f22da22c28f16ca51cd5eb55d8ce5bc0ba4dcbe9e32`
  - `run_summary.json` — SHA-256 `dcc9a22642126fa19cd921b950a16240e48b2d7057be903621cfd21e5109ce5a`
- **Not exercised:** a browser Run all, release step 7 (REL12, BYOD), the next experiments and a Colab CPU runtime.

## Current status

**Candidate.** The 2026-10-17 review-fix revision (branch `fix/pubmedclip-tutorial-review-2026-10-17`: pre-registered grid on 352 never-used validation slices, a 176-slice fresh test scored only under the pre-registered settings, the earlier 176 test slices kept as the labelled exposed development test; see `docs/pubmedclip-preregistration.md`) is now hosted: blob `c5a0e62d` (PR #8 head `352452b`) passed on a fresh Colab Tesla T4 on 2026-10-17 (Colab CLI 0.7.4 sequential execution, not a browser Run all; 17/17 code cells, 0 errors, 1009.0 s wall, 743.6 s of it the Zenodo archive fetch), and that run is the **pre-registered evaluation of record** (REL13; `test_look` pre-registered evaluation, `settings_differing` []): grid C2 selected on validation (mAP C0 0.772 / C1 0.678 / C2 0.888, frozen 0.296); fresh test accuracy 0.165 → 0.881, text-to-image mAP 0.248 → 0.862; frozen-embedding probe on the fresh test accuracy 0.824, mAP 0.885; evidence in `docs/execution-evidence/2026-10-17-352452b/`. The CPU runs of 2026-10-17 in Recorded executions are disclosed development looks. REL13: the remaining figures in this paragraph describe earlier blobs. Status: **Candidate**. No current blob of this repository has release-grade evidence: the hosted runs above are for earlier blobs, and every change to the carried modules or the notebook returns the registry to Candidate until a clean run of that blob is recorded here.

History (not the current status): on 2026-09-21 the status was **Release-grade** for that blob only: The `E2E` notebook blob `4b2fc4c4` (committed at `d125896`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (14/14 ok (1 restart after install cell), 415.1 s, 682 files, 3051 MB fetched (Hub snapshot + Zenodo archive) and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The CPU pre-flight row above is what preceded it and remains history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should still weigh: the frozen model is a weak zero-shot classifier on eleven abdominal organs — accuracy 0.18 against a majority floor of 0.09 and an intensity nearest neighbour of 0.51, because organs sit at characteristic positions in an axial slice and a caption model knows nothing of laterality (it answered *heart* for most slices, heart recall 1.00 on an average precision of 0.25, and never found a kidney, a femur or the spleen); the bounded adaptation moves it past both baselines (accuracy 0.18 → 0.81, macro F1 0.13 → 0.81, text-to-image mAP 0.28 → 0.65; the raw-label prompt set reaches 0.82 / 0.81 / 0.82), so the claim is that the contract works on a task the frozen model could not do, not that the adapted numbers are good; the 88-slice validation split selected epoch 4 on mAP (0.761) while validation accuracy moved between 0.91 and 0.96 across epochs 3–6, which is what "no dispersion estimate" means here; the left / right pairs stayed the weakest after adaptation (right kidney recall 0.38, right femur AP 0.51) and the heart lost the recall it only had as the default answer; and the drawn shapes ranked correctly before and after adaptation, which a radiology-caption model owed nobody. The drawn shapes re-scored after adaptation are three images
of evidence about behaviour outside the corpus, not a measurement, and nothing in the notebook is clinical evidence.
