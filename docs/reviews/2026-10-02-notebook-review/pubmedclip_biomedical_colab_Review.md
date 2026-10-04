# PubMedCLIP Biomedical E2E Notebook — Review

**Readiness: Needs revision**  
**Review date:** 4 October 2026 (relay batch 2026-10-02)  
**Repository:** `kurtvalcorza/pubmedclip-biomedical-pipeline`  
**Notebook:** `tutorials/pubmedclip_biomedical_colab.ipynb`  
**Reviewed commit:** `626769cc12867918f27c9b53a5dd647ff0727c86` (origin/main)  
**Notebook Git blob:** `4b2fc4c4ce0ed900eae960ed3c312ae1a0e1eb56`  
**Finding prefix:** `PMC`

## Executive assessment

The notebook is a careful, standalone E2E pipeline: a digest-pinned PubMedCLIP snapshot whose pickle is audited and converted once, a digest-pinned 660-slice OrganAMNIST sample kept in the dataset's own scan-level roles, two non-neural baselines, a bounded fine-tuning of the vision tower with validation-mAP epoch selection, a held-out comparison on three metrics, per organ and on two prompt sets, and a safetensors adapter that reloads with identical embeddings. The default path has an exact-blob clean-runtime record (Kaggle T4, 2026-09-21), and the worked numbers in the prose match that record.

It is not yet ready for its declared `GUIDED` audience. The default `Run all` needed one manual restart after the in-kernel install. Re-running Section 7 for the optional experiments, or re-running from Section 4 for BYOD as the opening instructs, continues from the already-adapted model, so the "frozen" baseline and epoch 0 are no longer frozen. On realistic user data the Section 6 assertion stops the BYOD branch before fine-tuning: three of four BYOD-shaped stand-ins failed it. The guided layer is mostly absent.

The four Major findings concern different journeys. They are not four failures of the recorded default run, and this review does not establish that any recorded metric is wrong.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Profile / mode | `E2E` / `GUIDED`; declares NOTEBOOK_SPEC `2.0`. Current fleet spec is `2.2` (2026-09-26), applied here; GDL1–GDL15 and EXE1–EXE7 are `SHOULD` |
| Audience (stated) | Basic Python and PIL; knows softmax, cosine similarity, accuracy / macro F1 / average precision. No explicit intended-learner statement |
| Supported runtime (stated) | "Google Colab or Jupyter, Python 3.12"; CPU float32, CUDA used when present |
| Model | `flaviagiammarino/pubmed-clip-vit-base-patch32` @ `26c0c67f6da3…` (MIT), `pytorch_model.bin` audited and converted to a digest-pinned `model.safetensors` |
| Default data | MedMNIST+ OrganAMNIST-224 (CC BY 4.0), 660 pinned slices, 396 / 88 / 176 from the dataset's own train / val / test splits (36 / 8 / 16 per organ) |
| Adaptation | Last 2 vision blocks + post-layernorm + visual projection (14,570,496 of 151,277,313 params); AdamW 5e-5, 6 epochs, batch 16; cross-entropy over 11 organ prompts; epoch chosen on validation text-to-image mAP |
| Promised outcomes | Pinned install; snapshot staging, pickle audit and conversion; validated scan-disjoint sample; inference contract on three drawn shapes; frozen zero-shot score vs majority floor and intensity neighbour; bounded fine-tuning; held-out four-way comparison; shapes re-scored; adapter export and reload parity; BYOD through the same stages |
| Generating revision | `metadata.dimer.generated_from.revision` = `cb7f63ae69cb`. That commit is not an ancestor of main, but `git diff cb7f63ae origin/main -- src` is empty, so the carried modules are byte-identical |

### Evidence obtained

**Source inspection:** the whole notebook (31 cells) at the reviewed SHA, the six carried modules, `tools/build_notebook.py`, `tools/notebook_template.py`, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`.

**Documented execution evidence:** `docs/release-verification.md` records a clean Kaggle Tesla T4 run of **this exact blob** (`d125896` / `4b2fc4c4`, 2026-09-21): PASSED, 14/14 code cells, **1 restart after the install cell**, 415.1 s. The archived `run_summary.json` shows pass 1 raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings …, numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime…` and pass 2 succeeded. I read the archived `executed.ipynb` outputs and compared them with every number quoted in the prose. No Colab record exists.

**Direct execution (this review):** `run_probes.py` ran on a Windows workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.3). That is not the pinned torch 2.14.0, not Colab, and not the notebook itself: the probes call the repository package with the real pinned snapshot and the 660 pinned slices, on labelled stand-in subsets. It ran 20 probes in 31 s, exit 0 (`results.json`).

**Not verified:** a fresh Colab run; the upload widget itself; a full BYOD run through fine-tuning, export and reload; the default path's numbers on CPU; measured learner understanding.

## 2. Separate judgments

| Dimension | Judgment |
|---|---|
| Technical correctness | The default path is sound and recorded. The stage code is transactional and digest-checked. Defects are in the rerun state (no reset to pretrained), the BYOD-hostile assertions, and BYOD loader errors. |
| Scientific validity | The roles are the authors' scan-level splits, pixel-disjointness is asserted, and no cross-split near-duplicate was found (probe L1). Baselines and test-only reporting are correct. The majority floor's mAP is a tie-order artefact (PMC-m6). |
| Promise fulfilment | The default promises are delivered and match the record. The BYOD promise "through the same … fine-tuning … export and reload cells" fails for realistic label sets (PMC-M3). The optional experiments do not measure what they say (PMC-M2). |
| Learner experience | The prose is precise and honest about limits, and every quoted number matches the record. But it is dense reference prose, with no how-to-use, roadmap, glossary, predictions, checkpoints or troubleshooting (PMC-M4). |
| Spec conformance | Fails `MUST`: RUN10 / ENV6 (restart), DAT14 (BYOD cannot reach adaptation), DAT19 (non-actionable BYOD errors), SPL5 (no group-preserving BYOD split for patient data). `SHOULD` deviations: GDL1–GDL14, EXE2, EXE5, declared spec 2.0. |

### Promise → evidence trace (summary)

| Claim | Cell | Observable result (Kaggle record, blob 4b2fc4c4) | Status |
|---|---|---|---|
| Pinned install, no restart needed by design | 3 | Pass 1 RuntimeError, restart, pass 2 ok | Fails RUN10 (PMC-M1) |
| Snapshot staged, pickle audited and converted | 17 | 9 files fetched; audit 4 globals, `5b9f0ba0…`; `model.safetensors` 605,156,676 B `81de21a0…` | Delivered |
| 660 slices, 396/88/176, 4 refusals | 19 | As claimed; 4 probes rejected with named conditions | Delivered |
| Inference contract on drawn shapes | 21 | 5/5 sanity checks; `sample-sanity` | Delivered |
| Frozen vs baselines (0.18 / 0.51 / 0.09) | 23 | acc 0.182, neighbour 0.511, majority 0.091; mAP 0.284 | Delivered; prose matches |
| Bounded fine-tuning, epoch 4 kept | 25 | loss 1.23 → 0.074; val mAP 0.355 → 0.761 (epoch 4) | Delivered; prose matches |
| Held-out comparison; "left/right pairs stayed the weakest" | 27 | acc 0.812, mAP 0.654; lung-right AP 0.79 → 0.34 not mentioned | Partly (PMC-m5) |
| Adapter reload parity | 29 | 35 tensors, 58,286,104 B, 8/8 identical | Delivered |
| BYOD through the same stages | 19→29 | Stops at the cell-23 assert in 3/4 stand-ins (probe A1) | Fails DAT14 (PMC-M3) |
| Optional experiments compare layers / LR | 25 rerun | Continues from the adapted model (probe R1) | Misleading (PMC-M2) |

### Objective → activity trace (summary)

The opening lists 9 objectives, all phrased as things the notebook does ("install…, stage…, measure…"). Every one has executing code, but the only learner activity is running cells and reading printed dicts. No objective asks the learner to predict, explain or diagnose, and the one change-a-parameter activity (Optional experiments) gives invalid results without a restart (PMC-M2, PMC-M4).

## 3. Findings

### Major

#### PMC-M1 — Run all needs a manual restart after the in-kernel install

**Cell/section:** Section 1, cell 3 (generated by `tools/build_notebook.py`, install block around line 61).  
**Observed issue:** The cell `pip install`s torch 2.14.0, numpy 2.5.3, transformers 4.57.6 etc. into the running kernel. When a preloaded distribution changes, it raises `RuntimeError(... Restart the runtime, then rerun from the top.)`.  
**Consequence:** On a hosted image whose preloaded numpy/torch differ from the pins (Kaggle and Colab both preload them), the first Run all stops at cell 3. A learner must restart and run again, which RUN10/ENV6 forbid.  
**Evidence:** Documented execution: exact-blob Kaggle T4 `run_summary.json`, pass 1 `RuntimeError … numpy: loaded=2.0.2, installed=2.5.3`, `restarted_after_install_cell: true`. Source: probe S2 (`pip_into_kernel: true`, `uv_used: false`).  
**Recommended correction:** Replace the in-kernel install with the fleet's **uv isolated-environment pattern**. A carrier cell bootstraps uv, creates a managed-Python venv (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that env, so the kernel's preloaded NumPy/torch are never replaced. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main. Make the change in `tools/build_notebook.py`, not by hand.  
**Acceptance check:** A fresh Colab (and Kaggle) runtime completes Run all in one pass with zero restarts, and the run record says "0 restarts".  
**Spec:** RUN10, ENV6 (MUST).

#### PMC-M2 — Re-runs fine-tune the already-adapted model, so "frozen" and epoch 0 are not frozen

**Cell/section:** Opening ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Interpretation, *Optional experiments* ("set `TRAINABLE_VISION_LAYERS = 4` … raise `EPOCHS` … change `LEARNING_RATE`"); Sections 6–8 (cells 23, 25, 27). `pipe` is built once in Section 3, and `PubMedClipPipeline.adapt` (`pipeline.py` ~386–560) starts from the current weights.  
**Observed issue:** No cell restores the pretrained weights before Section 6 or 7. After the default run:
- re-running Section 6 scores the adapted model as `frozen_model_test`;
- re-running Section 7 trains the adapted model again while history epoch 0 is still labelled `'frozen model'`;
- the BYOD rerun from Section 4 makes the Section 5 shapes, the Section 6 "frozen" baseline and epoch 0 all the OrganAMNIST-adapted model. If epoch 0 is selected, the BYOD "adapter" is the organ adapter.

**Consequence:** The optional experiments compare accumulated training against a mislabelled baseline. BYOD learners see a "frozen" number that is not the pretrained model. Both are conclusions the notebook explicitly teaches learners to draw.  
**Evidence:** Direct execution (CPU stand-in: 44/22/44 pinned slices, 1 epoch; probe R1). Frozen test mAP 0.2933 → adapt → Section 6 re-run reports 0.3793, identical to the adapted score, with `adapted: True`. A second `adapt(trainable_vision_layers=4)` logs epoch 0 `note: 'frozen model'` with val mAP 0.5934, identical to the first adapt's selected state (the frozen epoch 0 was 0.4089).  
**Recommended correction:** In `tools/notebook_template.py`, rebuild the pipeline from the verified snapshot at the top of Section 6 (`pipe = PubMedClipPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)`) and again before `pipe.adapt` in Section 7, or assert `pipe.adapter is None` with a clear message. State in the opening and in *Optional experiments* exactly which cells to re-run.  
**Acceptance check:** After a default run, re-running Section 6 reproduces the first-run frozen metrics, and re-running Section 7 with any form value logs an epoch-0 validation score equal to the first run's epoch 0. A BYOD rerun reports a frozen baseline equal to a fresh pretrained model on the same split.  
**Spec:** SRC2 (hidden state dependency), GDL10, UX7.

#### PMC-M3 — The BYOD branch stops at the Section 6 assertion on realistic label sets and never reaches fine-tuning

**Cell/section:** Section 6, cell 23 (`assert frozen_test['t2i_map'] > baseline_majority['t2i_map'] and frozen_test['accuracy'] > baseline_majority['accuracy']`); Section 8, cell 27 (`assert adapted_test['t2i_map'] > frozen_test['t2i_map']`). Template lines 343 and 430.  
**Observed issue:** These sample-path sanity asserts run unconditionally, including in BYOD mode. They have no message.  
**Consequence:** BYOD is promised to flow through "baselines, fine-tuning, held-out evaluation, artifact export and reload". But BYOD is exactly the case where the frozen model may not beat the majority floor (opaque or annotator labels, left/right classes). The branch then dies with a bare `AssertionError` before adaptation, and the learner is given no explanation.  
**Evidence:** Direct execution (probe A1: fresh pretrained model, CPU; 30 pinned slices per label as BYOD-shaped records through `split_dataset(seed=42)`; prompts built as in cell 23 with `display_names = {}`). Results: `liver`/`spleen` passed; `kidney-left`/`kidney-right` (acc 0.333 vs 0.5), `class-a`/`class-b` (0.417 vs 0.5) and `L`/`R` (0.5 vs 0.5) **failed** the assertion. Only the assertion was exercised; the upload widget was not.  
**Recommended correction:** Keep the asserts for the sample path only (`if not USE_BYOD:`). In BYOD mode, print the comparison with an interpretation line ("the frozen model does not beat the majority floor on your labels; adaptation is what this notebook tests next") and continue.  
**Acceptance check:** A BYOD dataset whose frozen accuracy is at or below the majority floor reaches Sections 7–9 (adapt, evaluate, export, reload parity) without editing cells. The sample path still asserts.  
**Spec:** DAT14 (MUST), DAT13, RUN9.

#### PMC-M4 — Declared GUIDED, but the guided layer is largely missing

**Cell/section:** Opening and all sections (`tools/notebook_template.py`).  
**Observed issue:** None of the following is present: an intended-learner statement, a *How to use this notebook* section, a roadmap, an Input → Model → Output task contract, a glossary (logit_scale, mAP, macro F1, axial slice, laterality…), prediction prompts before Sections 6/8, interpretation checkpoints with sample answers, a coded Predict → Change → Run → Observe → Explain activity, Infrastructure labels on the six carried-module cells (≈ 170 kB of code, 100 kB of it in `samples.py`), troubleshooting (Zenodo download, out-of-memory, restart, BYOD upload), or a conclusion template. The objectives are "install / stage / measure" rather than observable learner actions. Probe S3: 0 of 8 markers present.  
**Consequence:** A self-paced learner new to vision-language adaptation gets accurate but dense reference prose. They have no orientation through 1,000+ lines of infrastructure, no point at which to test their understanding, and no recovery guide for the 1.8 GB download.  
**Evidence:** Source inspection; probe S3.  
**Recommended correction:** Add the GDL1–GDL14 layer in the template. Mark Section 2 and the install cell as **Infrastructure** (collapsed). Add predictions before Sections 6 and 8 (for example, "which organs will the frozen model miss, and why?") with collapsible worked answers taken from the record. Turn one optional experiment into a coded, reset-safe activity (this depends on PMC-M2). Add troubleshooting and a conclusion template.  
**Acceptance check:** Each GDL1–GDL14 item can be pointed to in the regenerated notebook, and the coded activity passes PMC-M2's acceptance check.  
**Spec:** GDL1–GDL14 (SHOULD), UX5, UX8.

### Minor

#### PMC-m1 — BYOD prompts every label as "An axial abdominal CT slice showing the …", contradicting the Prerequisites

**Cell/section:** Section 6, cell 23 (`CT_PROMPT` used unconditionally with `display_names = {}` in BYOD), cells 25 and 27; Prerequisites *Data contract* ("the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label").  
**Observed issue:** A BYOD chest X-ray or dermoscopy set is scored and fine-tuned against prompts such as "An axial abdominal CT slice showing the L." (probe A1 `prompt_example`). The documented contract says `A medical image of {label}.`  
**Consequence:** The frozen BYOD baseline is depressed by a wrong-modality prompt, and the documented contract does not match the code. This undercuts the notebook's own "the classifier is the prompt" lesson.  
**Evidence:** Source inspection; probe A1, S5.  
**Recommended correction:** Use `DEFAULT_PROMPT_TEMPLATE` (or a `PROMPT_TEMPLATE` form field) when `USE_BYOD`, and print the effective template.  
**Acceptance check:** In BYOD mode, `frozen_test['prompt_template']` equals the documented template or the form value.  
**Spec:** DAT13, DAT12.

#### PMC-m2 — BYOD: the stated minimum is refused and loader failures are not actionable

**Cell/section:** Opening and Section 4 ("at least eight images over at least two labels"); `samples.load_byod_dataset`, `split_dataset`; cell 19 upload handling.  
**Observed issue (direct execution, probes B1–B9):**
- 8 images / 2 labels pass `validate_dataset` but `split_dataset` refuses them ("split leaves 4 training records"). The smallest balanced 2-label set that splits is **12**.
- A `labels.csv` path in a subfolder (`images/i0.png`) gives a bare `KeyError: 'images/i0.png'`, because members are keyed by basename.
- Two members with the same basename in different folders collide silently: the last one is loaded (B4).
- A missing file gives a bare `KeyError`.
- A non-image gives `UnidentifiedImageError`.
- A non-zip upload is saved as `byod.zip` and gives `BadZipFile`.
- A cancelled upload gives a bare `StopIteration`.

**Consequence:** A learner following the stated minimum, or a normal nested zip, gets a refusal or a traceback that names no fix, and a basename collision trains on the wrong image without any warning.  
**Evidence:** Direct execution, synthetic PNG zips (probes B1–B9).  
**Recommended correction:** State the real minimum (or size the split so 8 works). Resolve `file` by full relative path with containment checks. Reject duplicate basenames. Wrap missing / non-image / non-zip / cancelled cases in `ValueError`s that name the file and the fix.  
**Acceptance check:** Each B-probe case either loads correctly or raises a `ValueError` naming the file and the corrective action. A zip with colliding basenames is rejected.  
**Spec:** DAT12, DAT19 (MUST).

#### PMC-m3 — BYOD requires `google.colab` although Jupyter is a stated supported runtime

**Cell/section:** Section 4, cell 19 (`from google.colab import files`).  
**Observed issue:** There is no `BYOD_ZIP_PATH` form field. On Jupyter (or Kaggle), enabling BYOD raises `ModuleNotFoundError`, although `load_byod_dataset` already accepts a path or directory.  
**Consequence:** The BYOD transfer path is unavailable on one of the two runtimes the Prerequisites name, and an executor cannot drive it.  
**Evidence:** Source inspection (probe S4).  
**Recommended correction:** Add `BYOD_PATH = ''  # @param {type:"string"}`. When it is set, read it without importing `google.colab`.  
**Acceptance check:** With `USE_BYOD = True` and `BYOD_PATH` set, cell 19 runs under plain Jupyter without importing `google.colab`.  
**Spec:** EXE2 (SHOULD), DAT16.

#### PMC-m4 — The BYOD split cannot preserve patients, although the notebook says to split by patient

**Cell/section:** Interpretation ("split by patient, scan or session when your images come from few sources"); Section 4 BYOD branch; `split_dataset`.  
**Observed issue:** The BYOD split is a seeded per-label shuffle after pixel de-duplication. `labels.csv` has no group or split column, so a learner with patient-level medical images cannot follow the advice without editing code.  
**Consequence:** Slices from one patient land in both train and test, which inflates BYOD held-out numbers. This is the specific risk the notebook warns about.  
**Evidence:** Source inspection.  
**Recommended correction:** Accept an optional `group` (or explicit `split`) column and split by group when it is present. Print which mode was used.  
**Acceptance check:** A BYOD zip with a `group` column yields splits with no group in two roles, asserted in the cell.  
**Spec:** SPL5 (MUST), SPL2.

#### PMC-m5 — Section 8 and the closing omit the one organ whose retrieval collapsed

**Cell/section:** Section 8 markdown (cell 26; template ~385–395); release-verification "facts a reviewer should weigh".  
**Observed issue:** The prose says "the left / right pairs stayed the weakest — the right kidney at 0.38, the right femur's AP at 0.51". The record shows **lung-right AP 0.79 → 0.34**: the lowest adapted AP, and the only organ whose AP fell sharply, from the frozen model's best retrieval. Probe S7 confirms Section 8 never mentions the right lung.  
**Consequence:** Learners are steered to read adaptation as uniformly beneficial on retrieval and miss the clearest regression, even though the notebook's method (read mAP first, per organ) is designed to show it.  
**Evidence:** Documented execution (archived `executed.ipynb`, cell 27 `by_organ`); source inspection.  
**Recommended correction:** Name the right-lung AP drop in Section 8 and in the closing. Better, print a computed list of organs whose AP or recall fell instead of hard-coding the narrative.  
**Acceptance check:** Section 8 output or prose identifies every organ with adapted AP < frozen AP (lung-right in the record).  
**Spec:** EVAL15, GDL8.

#### PMC-m6 — The majority floor's mAP depends on record order, not on prevalence

**Cell/section:** Section 6 (`majority_baseline` → `classification_metrics` → `average_precision`, `metrics.py` ~52–115).  
**Observed issue:** The majority baseline gives every image the same one-hot score, so every column is an all-tie. `average_precision` breaks ties with a stable argsort, so the result depends on the seeded record order. The record shows 0.119 against an 11-class prevalence of 0.091, and probe A1 shows 0.581 on balanced 2-class splits. The Section 6 assertion compares the frozen model against this value.  
**Consequence:** The "floor" on the retrieval metric is a seed artefact, and the prose calls it chance-level.  
**Evidence:** Source inspection; documented record; probe A1.  
**Recommended correction:** Score tied rankings by expected AP (equivalently, prevalence for an all-tie column), or report the majority floor's mAP as the class prevalence, with a note.  
**Acceptance check:** The majority mAP equals the mean class prevalence (0.091 on the sample) independent of `SPLIT_SEED`.  
**Spec:** EVAL3, EVAL10.

### Suggestions

- **PMC-S1** — Regenerate against NOTEBOOK_SPEC 2.2. The notebook declares 2.0 (metadata, opening, references).
- **PMC-S2** — Record a Colab run. Colab is the "supported user path" in `docs/release-verification.md`, but only a Kaggle T4 record exists.
- **PMC-S3** — Document `DIMER_NOTEBOOK_CI_PREINSTALLED` in the notebook. Cell 3 reads it, but no markdown mentions it (EXE5).
- **PMC-S4** — The recorded generating revision `cb7f63ae` is outside main's history (a pre-merge branch commit). Its `src/` is identical to main's, so regenerate to record a reachable revision.
- **PMC-S5** — Print `frozen_test['adapted']` / `adapted_test['adapted']` in Sections 6 and 8, so a stale-state rerun (PMC-M2) is visible.
- **PMC-S6** — Give an expected time for the 1.8 GB Zenodo download in the Prerequisites. The build workstation took 1,177 s at ~1.5 MB/s; Kaggle took 165.6 s.

### Checked and not raised

- **Test/train leakage:** roles come from MedMNIST's own scan-level splits, pixel-disjointness is asserted, and a 32-px cosine scan found no cross-split near-duplicate. The maximum test-to-train similarity was 0.865; the maximum test-to-test was 0.963 (probe L1, an observation, not a scan-identity check).
- **Stale worked answers:** every number quoted in Sections 6–8 and the closing (0.18 / 0.51 / 0.09; mAP 0.28 → 0.65; loss ≈1.2 → <0.1; epoch 4; 8/8 parity) matches the exact-blob Kaggle record. The only issue is the omission in PMC-m5.
- **Score semantics:** softmax-over-candidates, uncalibrated, no abstention and non-clinical use are stated correctly (UNC1–UNC4, 21.8).

## 4. Readiness

**Needs revision.** Four Major findings are open. Applicable `MUST`s fail: RUN10/ENV6 (PMC-M1), DAT14 (PMC-M3), DAT19 (PMC-m2) and SPL5 (PMC-m4). Exact-blob execution evidence exists for the default path only.

Remaining gates after fixes:
1. A one-pass fresh-runtime record (Colab preferred) of the regenerated blob with 0 restarts.
2. A rerun check per PMC-M2's acceptance check.
3. A BYOD run through reload parity with one actionable rejection (REL12).

## 5. Verified versus inferred

- **Verified (direct execution, CPU, package API, stand-ins):** rerun state carry-over (R1), the BYOD assertion failures (A1), the BYOD loader behaviours (B1–B9) and the leakage scan (L1).
- **Verified (documented execution):** the default path outcome and the restart, against the archived Kaggle record of this exact blob.
- **Inferred:** that Colab behaves as Kaggle does at the install cell; that real user BYOD sets often fall at or below the majority floor (3 of 4 stand-ins did).
- **Not verified:** the upload widget; a full BYOD run through export; learner understanding.

**Finding most likely to be wrong:** PMC-M3's severity. If most real BYOD label sets name organs or findings the model already separates (as `liver`/`spleen` did), the assertion would rarely fire, and the finding would be closer to Minor.
