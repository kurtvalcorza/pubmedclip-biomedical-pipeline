# Fleet-sweep fixes: `pubmedclip_biomedical_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `626769c`. Changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`) and, for SWP-F, in the carried `pipeline.py`; the notebook is regenerated. Status and release
labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 packages compiled from the unchanged pyproject pins for manylinux x86_64, `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. With `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (the integration job's `run_notebook.py`) the cell routes nothing and needs no IPython, so that job is unaffected. The CPU reference lock `requirements.lock.txt` is unchanged. | Section 1; generator, template, validator, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 0 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–8), What to notice + Check your reasoning after Sections 4–9 quoting the recorded Kaggle T4 run of 2026-09-21 (accuracy majority 0.091 / neighbour 0.511 / frozen 0.182 / adapted 0.812; mAP 0.119 / 0.519 / 0.284 / 0.654; raw-label mAP 0.34 → 0.815; reload parity 8 of 8), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. | opening, Sections 4–9 markdown, closing | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Fixed | Confirmed: Section 6 asserted `frozen > majority floor` (accuracy and mAP) and Section 8 asserted `adapted mAP > frozen mAP`; either aborts a BYOD run before export. Both are recorded verdicts now (`frozen_vs_majority`; `comparison['verdict']` with `improved` / `no change` / `worse`), written to the evaluation report and `result.json`. The reload-parity assert is kept. `tests/test_notebook_spec_contract.py` now looks for the verdict instead of the assert. | Sections 6, 8, 9 | `test_swp_a_*` (5 tests) |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trained the vision tower in place from whatever weights the model held, so a Section 7 re-run (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model". `pipeline.py` now applies siglip-v1's `restore_base` pattern verbatim: the pinned-base value of every tensor adapt() or load_artifact() changes is kept and restored first; a failed adapt leaves the weights as before; the result records `started_from` (printed in Section 7). | `src/pubmedclip_pipeline/pipeline.py`, Section 7 | `test_swp_f_adapt_and_load_artifact_restore_the_pinned_base_first`, `test_swp_f_restore_helpers_put_trained_tensors_back` |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (zip or folder; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, non-.zip each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes several minutes; reused afterwards). Linux x86_64 only.
- `PubMedClipPipeline.adapt()` and `load_artifact()` always start from the pinned base; new `restore_base()`; `adapt()` result has `started_from`.
- New `BYOD_PATH` field; Sections 6 and 8 print verdicts instead of raising; `result.json` carries `verdict`.
- Guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; the BYOD block and Sections 6 and 8 run with stand-ins; the restore helpers run on NumPy stand-ins (extracted from `pipeline.py`, which imports torch at module level). Plumbing evidence, not model evidence.
- CI's unit job installs `requirements.lock.txt`, which includes CPU torch from `download.pytorch.org`; that index is blocked here and torch was not installed. The torch-free part of the suite was run: `test_notebook_parity.py`, `test_notebook_spec_contract.py`, `test_weight_facts.py` 18 passed before → 18 passed after, plus 17 new `test_sweep_fixes.py` tests (35 passed). The eight torch-importing test modules (`test_adaptation.py`, `test_pipeline.py`, …) could not be collected here before or after; CI on the PR runs them.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `scripts/check_lock.py` OK; `ruff check src tests tools scripts` clean.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- CI's torch-backed unit tests on the PR (not runnable here).
- A hosted **Run all in one pass** in a fresh Colab runtime (no restart expected), then a re-run of the Section 9 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
