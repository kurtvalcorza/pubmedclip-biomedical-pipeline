# Pre-registration: PubMedCLIP tutorial configuration grid and fresh test (2026-10-17)

This file is committed **alone, before any grid run and before any fresh-test slice is scored**. It fixes how the
tutorial's fine-tuning configuration is chosen and how the one fresh-test evaluation is made. It answers review
finding R71-M1 (the default had been described as "the configuration that gained on the held-out split", a sentence
carried over from the SigLIP templates; whether it was ever selected on OrganAMNIST test cannot be recovered from the
record) and R71-M3 (exploratory settings scored the test split). Owner ruling R71-s1 = **A**.

## Data (all from MedMNIST+ `organamnist_224.npz`, Zenodo record 10519652, SHA-256 `a19bae53…59951e9`)

| Role | Slices | Source | Use |
|---|---|---|---|
| train | 396 (36 per organ) | official `train` split, the existing pins | fine-tuning only |
| validation | **352 (32 per organ)** | official `val` split, drawn with seed `71352` from the **6,403 val slices never used** (the 88 previously pinned val slices are excluded and retired) | configuration grid and epoch selection |
| exposed development test | 176 (16 per organ) | official `test` split, the existing pins | scored in every earlier record; reported once for continuity, labelled "exposed development test", never used for a decision |
| **fresh test** | **176 (16 per organ)** | official `test` split, drawn with seed `71176` from the 17,602 test slices never scored | scored **only** under the settings below |

- Every new slice is pinned by split, index and the SHA-256 of its raw 224 × 224 uint8 bytes in `samples.py`.
- Pixel de-duplication: a slice whose raw bytes occur more than once anywhere in the archive's train, val or test
  arrays, or equal any pinned slice, is skipped by the draw.
- Scan disjointness: MedMNIST splits OrganAMNIST by CT scan (115 / 16 / 70 scans), so validation and fresh test share
  no scan with train. The notebook asserts the role/split mapping and pixel disjointness of all five roles.
- **Shared scans (stated, not removable):** the fresh test and the exposed development test are both drawn from the
  70 official test scans; MedMNIST publishes no scan id. Before relying on an index-distance guard the archive's
  order was checked: adjacent-index slices are no more alike than random pairs (median pixel L1 74.4 adjacent vs
  73.3 random in `test`; 15,872 label runs in 17,778 slices), so the archive is **shuffled** and a ±5-index guard
  would protect nothing. It is replaced by a near-duplicate pixel guard: a candidate whose nearest exposed slice is
  closer (RMS on 56 × 56 block means) than the median nearest-neighbour distance inside the test split is skipped.
- Draw record (pins only; no slice was scored): the archive holds no duplicate-pixel slice in any split, the draw
  skipped 0 candidates, the guard threshold was 14.94 and the closest fresh slice to an exposed one is 18.64 (median 50.9).

## Fixed settings (every one is part of `preregistered`)

Data source = the pinned sample (not BYOD); the pin manifests above; `SPLIT_SEED = 42` (orders records within a role);
adaptation `seed = 0`; `EPOCHS = 6`; `BATCH_SIZE = 16`; prompt template `CT_PROMPT` =
`An axial abdominal CT slice showing the {label}.` with the organ display names; AdamW (weight decay 0.01), gradient
clipping 1.0, no scheduler; epoch selection by the highest validation text-to-image mAP (epoch 0 = frozen model);
model `flaviagiammarino/pubmed-clip-vit-base-patch32` @ `26c0c67f`.

## Grid (validation only; the fresh test is never read by it)

| Config | `TRAINABLE_VISION_LAYERS` | `LEARNING_RATE` | |
|---|---|---|---|
| **C0** | 2 | 5e-5 | current default |
| C1 | 2 | 1e-5 | smaller step |
| C2 | 4 | 5e-5 | larger trainable scope |

**Score:** best-epoch validation text-to-image mAP of each config on the 352 validation slices.

**Selection rule.**
1. C0 stays selected unless a challenger's score is **≥ C0 + 0.02** (a practical tolerance for 352 slices, not a
   significance threshold).
2. If both challengers clear the margin, the higher score wins, except that if they are within 0.02 of each other the
   one with the smaller trainable scope (fewer layers; then the lower learning rate) wins.
3. **Keep frozen unless beaten:** the selected config's score must be **≥ frozen validation mAP + 0.02**; otherwise the
   result is "adaptation did not beat frozen on validation" and only the frozen model is scored on the fresh test.

The grid is **on by default** (estimated 1–2 min on a T4; three adaptations of six epochs each). The tutorial then
adapts with the selected config.

## Fresh-test evaluation (SCORE_ONCE_RULE)

- `preregistered` is computed from **all** fixed settings above plus "the configuration in use is the grid's
  selection". Only then are the majority floor, the intensity neighbour, the frozen model and (if kept) the adapted
  model scored on the fresh test. The principal number is the selected model's fresh-test text-to-image mAP; accuracy
  and macro F1 are reported beside it.
- Any other setting (a form change, BYOD, a manual config, a smaller batch for memory, the optional experiments) is
  **exploratory**: validation and the labelled exposed development test only, written to
  `*_exploratory_<n>.json`; the pre-registered result file is never overwritten by it.
- Outputs carry `test_look` (`pre-registered evaluation` | `reproduction` | `not scored (exploratory)`) and
  `settings_differing`. The first recorded pre-registered hosted run is the result; a later run with identical
  settings is a reproduction, provided nothing was changed in response to the test number.
- Outcomes on the fresh test are recorded **verdicts**, never asserts.
