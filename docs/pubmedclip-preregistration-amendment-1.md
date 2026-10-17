# Amendment 1 to the PubMedCLIP pre-registration (2026-10-17, after the peer review, before any hosted run)

Committed alone, before the frozen-embedding baseline below was fitted on any data and before any hosted run.
`docs/pubmedclip-preregistration.md` is not edited; where this file differs, this file governs.

**Disclosure (SCORE_ONCE_RULE item 8).** A CPU development run on 2026-10-17 has already scored the fresh test once.
It ran on the uncommitted working tree that became commit `61d464c`; its carried code is identical to that commit's
except for one comment line in `protocol.py`. Its numbers are recorded in `docs/release-verification.md` as a
*development look*. They changed no setting, and this amendment changes no fixed setting, no grid configuration and no
selection rule. The first hosted T4 run with the pre-registered settings is the evaluation of record.

## 1. One added baseline: a supervised readout on frozen embeddings (review R71P-M4)

- **Model:** a ridge-regression readout on the frozen model's L2-normalised image embeddings. It is a one-vs-rest least
  squares fit to one-hot targets, with a bias column, fitted on the 396 training slices.
- **Ridge strength:** chosen from {10, 1, 0.1, 0.01, 0.001} by the highest validation text-to-image mAP on the 352
  validation slices; a tie goes to the stronger ridge.
- **What it is for:** it separates "using the labels at all" from "fine-tuning the vision blocks".
- **Where it is reported:**
  - on validation and on the exposed development test in every run;
  - on the fresh test only under the pre-registered settings, beside the other baselines.
- **What it does not affect:** the configuration grid, the selection rule or the keep-frozen rule. It is a secondary
  baseline, not a decision input. The fresh-test principal number remains the selected model's text-to-image mAP.

## 2. What `preregistered` requires (review R71P-M1)

In addition to the fixed settings and "the configuration in use is the grid's selection", `preregistered` now also
requires both of the following.

**The grid ran on this run's inputs.** The grid's selection carries a fingerprint of:
- the data source;
- the pin-manifest digest;
- `SPLIT_SEED`;
- the validation dataset digest;
- epochs, batch size and seed;
- the prompt template.

That fingerprint must equal the current run's.

**The model in memory is the one Section 7b trained.** `pipe.adapter` must exist and must match `adapt_result` on:
- epochs, learning rate, trainable layers and seed;
- best epoch;
- `n_train` and `n_val`.

Otherwise the run is exploratory, with the reason "model state is not the selected adapted model".

## 3. Labels (SCORE_ONCE_RULE item 8)

- **What the notebook can say.** A runtime cannot see other runtimes, so the notebook labels its scoring
  `pre-registered evaluation (this runtime)`, or `reproduction (this runtime)` when an earlier pre-registered report
  exists in the same runtime. Which run is the evaluation of record is stated only in `docs/release-verification.md`.
- **No overwriting.** A reproduction writes `*_reproduction_<n>.json` and never overwrites the first file.
- **"Reported once" is withdrawn.** The original document says the exposed development test is "reported once for
  continuity". It is in fact reported **in every run** for continuity with earlier records, and it informs no decision.

## 4. Independence (review R71P-M3)

- **The fixed settings saw the exposed test.** Epochs, batch, the prompt and the learning rate C0 and C2 share were set
  in earlier revisions with the exposed development test in view.
- **The fresh test shares those scans.** It shares the 70 official test scans with the exposed development test.
- **Consequence.** The fresh test is unseen at the slice level, but it is not an independent, patient-level
  confirmation of those earlier choices.
- **Scan versus patient.** MedMNIST documents scan-level splits; patient-level disjointness is not documented.
- **The near-duplicate guard is a coarse screen,** not a protection: its threshold is the median within-test
  nearest-neighbour distance.
