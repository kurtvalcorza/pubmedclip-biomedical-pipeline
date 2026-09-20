# Weights, conversion and data provenance

## The pinned checkpoint

| Item | Value |
|---|---|
| Hub repository | [`flaviagiammarino/pubmed-clip-vit-base-patch32`](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32) (a Hub author's transformers conversion of the ViT-B/32 PubMedCLIP checkpoint released by Eslami, de Melo and Meinel at https://github.com/sarahESL/PubMedCLIP) |
| Pinned revision | `26c0c67f6da303ad2a38909130bd35744ea93517` (repository created 2023-06-13, last modified 2023-12-28) |
| Licence | MIT (upstream code and checkpoints; the Hub card repeats it) |
| Hosted weight files | `pytorch_model.bin` 605,222,477 B, SHA-256 `4daa7650d2b47e55c37b5ca7dcabe826fd82407c26028a574bcc422f35a94aa4` · `tf_model.h5` 605,559,520 B, `fe5013d5a012ced66ce5c5536177a3467ce0319b00e15b373d0f084173e89f44` · `flax_model.msgpack` 605,123,003 B, `3b689834868221167494cb69f88b798850c37bbf9c79ddfbe4660aa302cf4c36` (all three digests read from the Git LFS pointers at the pinned revision; only the `.bin` is fetched) |
| Small files (committed) | `config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `README.md` — 9 Hub files in `dimer-base-manifest.json` with the source (608,842,164 B in total) |
| Architecture | CLIP ViT-B/32: 12-layer vision tower (width 768, 50 tokens at 224 × 224, patch 32), 12-layer text tower (width 512, context 77, 49,408-token BPE), 512-d projections, `logit_scale`; 151,277,313 parameters |

## Pickle audit and one-time conversion (asset spec §11)

The Hub revision hosts no safetensors file, and DIMER's profile upload refuses `.bin`. The pipeline therefore treats `pytorch_model.bin` as a **conversion source only**:

1. `_check_pinned_source` — byte size and SHA-256 against the pins above.
2. `audit_pickle` — the file is a torch zip archive with one pickle (`archive/data.pkl`); `pickletools.genops` lists every global it would import **without executing it**. The allow-list is the fleet's four state-dict globals — `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `torch.LongStorage` — and the sorted list's digest is pinned (`PICKLE_AUDIT_SHA256 = 5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10`). Any other global refuses the file.
3. `convert_source` — `torch.load(map_location="cpu", weights_only=True)` unpickles it exactly once (400 tensors: 398 parameters plus the two non-persistent `position_ids` buffers), loads them strictly into a `CLIPModel` built from the snapshot's `config.json` with the buffers dropped, and writes the model's own state dict as `model.safetensors` (398 tensors, contiguous, metadata `{"format": "pt"}`). The result is refused and deleted unless its size and digest are the pinned `605,156,676` B / `81de21a0f1b6a3faaf1ea9e8fed4d570c671808ad72c948b4dac173eabcf676e`.
4. `load_components` — transformers loads only `model.safetensors` (`use_safetensors=True`, `local_files_only=True`, `trust_remote_code=False`); every unsafe format in the directory other than the one pinned source is refused.

The conversion is deterministic: two conversions on the build workstation produced byte-identical files. `PubMedClipPipeline.from_pretrained(weights_dir=...)` runs steps 1–4 when the converted file is absent and reports them through `progress`; the standalone notebook prints that report before the model loads.

## DIMER hosting

Upload the converted `model.safetensors` (605 MB) with the seven committed small configuration and tokenizer files; do **not** upload `pytorch_model.bin`. With the converted file present the pipeline does not need or fetch the pickle (`verify_snapshot` tolerates the missing manifest entry, `stage_missing_files` skips it) — the DIMER-hosted shape. If the platform's own record should be the hosted `.h5` instead, its digest above identifies the revision, but that file is not what this pipeline executes.

## Tutorial data

| Item | Value |
|---|---|
| Dataset | MedMNIST+ **OrganAMNIST**, 224 × 224 (Yang et al., *MedMNIST v2*, Scientific Data 2023) — axial abdominal CT slices from the Liver Tumor Segmentation Benchmark volumes with 11 organ labels from the bounding boxes of Xu et al. (2019), abdominal Hounsfield window, grey-scale |
| Archive | `organamnist_224.npz` from Zenodo record 10519652 (MedMNIST v3.0): `1,803,859,544` B, SHA-256 `a19bae532ac0cf979f7474aba7eb923dc9bbf67c1bcba3cb941dce51a59951e9`, MD5 `50747347e05c87dd3aaf92c49f9f3170` (as published on the record and in `medmnist/info.py`) |
| Licence | CC BY 4.0 (MedMNIST; the authors list OrganAMNIST's licence as CC BY 4.0) |
| Official splits | train 34,561 / val 6,491 / test 17,778 slices; 115 / 16 / 70 CT scans — the split is by scan |
| Pinned sample | 660 slices: 60 per organ, 36 from `train`, 8 from `val`, 16 from `test`, drawn with `random.Random(20260920)` from each organ's indices within each split; each pinned by split, index and the SHA-256 of the raw 224 × 224 uint8 slice in `samples.py` |
| Reading | `numpy.load(allow_pickle=False)`, one split array at a time; the pinned slices are written to `weights/organamnist/slices/<id>.raw` (50,176 B each) and re-hashed on every use |
| Redistribution | none — the repository commits only the pins; the archive and the slices are git-ignored |

The organ keys and display names used in the prompts: `bladder` → *bladder*, `femur-left` → *left femur*, `femur-right` → *right femur*, `heart` → *heart*, `kidney-left` → *left kidney*, `kidney-right` → *right kidney*, `liver` → *liver*, `lung-left` → *left lung*, `lung-right` → *right lung*, `pancreas` → *pancreas*, `spleen` → *spleen*.
