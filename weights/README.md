# Base Model Weights Cache

This directory holds offline base-model weights, configurations, and cryptographic manifests for the PubMedCLIP biomedical pipeline.

The model is organized in its own isolated subfolder corresponding to its canonical identifier:

```
weights/
├── pubmed-clip-vit-base-patch32/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── dimer-base-manifest.json
│   ├── README.md (the upstream Hub card)
│   ├── pytorch_model.bin   (excluded from Git; the pinned Hub source — a torch pickle, audited and converted once, never loaded by transformers)
│   └── model.safetensors   (excluded from Git; produced by the conversion, or acquired from a DIMER upload)
└── organamnist/            (excluded from Git; the pinned MedMNIST+ archive and the 660 extracted slices, fetched at run time)
```

## Available Base Model Snapshots

- [**`pubmed-clip-vit-base-patch32`**](pubmed-clip-vit-base-patch32/): Dedicated snapshot for PubMedCLIP ViT-B/32 (`flaviagiammarino/pubmed-clip-vit-base-patch32`, 151,277,313 parameters; CLIP ViT-B/32 fine-tuned on ROCO by Eslami, de Melo and Meinel, 2021; MIT).
  - [**Upstream card**](pubmed-clip-vit-base-patch32/README.md): the Hub author's card (usage, training data, licence).
  - [**Manifest**](pubmed-clip-vit-base-patch32/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for the 9 Hub files, the pinned source among them.

## DIMER Architecture & Git Tracking Strategy

In the DIMER workbench ecosystem:
1. **Large binary weights:** the ~605 MB source pickle (`pytorch_model.bin`, 605,222,477 bytes) and the converted `model.safetensors` (605,156,676 bytes) are excluded from Git via `.gitignore` (`weights/**/*.bin`, `weights/**/*.safetensors`). The **converted safetensors file is the artifact to upload to DIMER** (the profile upload refuses `.bin`); with it present, `verify_snapshot` and `stage_missing_files` do not require or fetch the pickle. The Hub also hosts `tf_model.h5` (605,559,520 bytes, SHA-256 `fe5013d5…`), which DIMER's upload dialog would accept but which is not the artifact this pipeline executes; its digest is recorded in `MODEL_CARD.md` so an `.h5` upload can be matched to the revision.
2. **Configuration & tokenizers:** all accompanying configuration files (`config.json`, `preprocessor_config.json`, `special_tokens_map.json`), BPE tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`), and the cryptographic manifest are version-controlled in the repository so the pipeline and offline containers can initialize the processor and tokenizer without network dependencies.

## Management & Verification Tooling

Manage, download, and cryptographically verify model snapshots using [`scripts/fetch_weights.py`](../scripts/fetch_weights.py) (Hub files) and the package's own `convert_source` (audit + one-time conversion):

```bash
# Fetch the 9 Hub files (the pickle among them) into weights/pubmed-clip-vit-base-patch32 and verify them:
python scripts/fetch_weights.py

# Audit the pickle statically, unpickle it once (weights-only) and write the digest-pinned model.safetensors:
python -c "from pubmedclip_pipeline import convert_source; print(convert_source('weights/pubmed-clip-vit-base-patch32'))"

# Verify the existing snapshot (Hub files + the converted file when present):
python -c "from pubmedclip_pipeline import verify_snapshot; print(verify_snapshot('weights/pubmed-clip-vit-base-patch32'))"
```

`PubMedClipPipeline.from_pretrained(weights_dir=..., allow_download=True)` performs all three steps itself: it stages absent manifest entries, converts when `model.safetensors` is absent, verifies, and loads only the safetensors file.
