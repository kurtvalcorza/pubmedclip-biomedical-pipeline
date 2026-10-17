"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (six
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/pubmedclip_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` lets the pipeline pick CUDA when it is visible
(the fine-tuning stage is where that matters; CPU is the documented fallback) and prints the one-time
pickle audit and conversion report.

This template configures an E2E zero-shot-classification fine-tuning workflow: the pinned
flaviagiammarino/pubmed-clip-vit-base-patch32 snapshot is digest-verified, its PyTorch pickle audited and
converted once to safetensors and loaded, 1,100 axial CT slices of eleven organs from MedMNIST+ OrganAMNIST
are extracted from the digest-pinned archive with per-slice digests, validated and kept in the dataset's own
splits, the drawn synthetic shapes are classified through the inference contract, the frozen model's
zero-shot accuracy / macro F1 / text-to-image mAP over the exposed development-test slices is measured beside two
non-neural baselines, a bounded fine-tuning of the vision tower's last blocks runs in the kernel with the
cross-entropy over the class prompts in the configuration a pre-registered validation grid selects, the fresh test is scored
only under the pre-registered settings, the adapted model
re-scores the shapes, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pubmedclip_pipeline",
    "repo_name": "pubmedclip-biomedical-pipeline",
    "stem": "pubmedclip_biomedical",
    "notebook_name": "pubmedclip_biomedical_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.5 KND1 and DOC17-DOC19 (review R71-m3, 2026-10-17): the kind and the use notice, with the medical domain named.
    "notebook_kind": "TUTORIAL",
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. **Not a medical device; not for diagnosis, triage,\n"
        "> screening or any clinical decision**, and not validated for production or safety-critical use. Running this\n"
        "> notebook downloads third-party model weights and datasets under their own licenses (see the model card) and\n"
        "> uses compute on your own Colab, Kaggle or Jupyter runtime. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it; never upload identifiable patient data. Results here are tutorial\n"
        "> evidence, not benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum extent\n"
        "> permitted by law, the maintainers and the DIMER platform accept no liability for its use."
    ),
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "PubMedClipPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a biomedical '
            'CLIP model classifies images by comparing them with text prompts, why its zero-shot scores need baselines beside them, and what a bounded fine-tuning '
            'of its vision tower changes on a real, scan-disjoint medical-image set. No prior experience with CLIP, PubMedCLIP or medical imaging is assumed; terms '
            'are explained where they first matter and again in the **Glossary** at the end. CPU is adequate (about 15 minutes of model time on the build '
            'workstation in the CPU build record of 2026-10-17, several times longer on a 2-vCPU hosted runtime); a GPU runtime is faster. Nothing here is a clinical tool.\n\n**Input → Model → Output.**\n\n| '
            '| Zero-shot classification and retrieval | Bounded fine-tuning |\n|---|---|---|\n| Input | an image (PIL or file, sides up to 4096 px) and candidate '
            "labels filled into a prompt | labelled images: 1,100 OrganAMNIST CT slices of eleven organs (396 train / 352 validation / 176 exposed development test / 176 fresh test, each role from the dataset's own scan-level split) |\n| "
            "Model | PubMedCLIP ViT-B/32 image and text towers, loaded from audited, converted safetensors | the same model; only the vision tower's last blocks (as many as the grid selects: two = 14.6 M of 151.3 M parameters), "
            'post-layernorm and visual projection train |\n| Output | a softmax score per candidate (relative, uncalibrated, '
            'never abstains), cosine similarities, a retrieval ranking | accuracy, macro F1 and text-to-image mAP beside two non-neural baselines and the frozen '
            'model, per organ, a configuration chosen on validation by a pre-registered grid, and a safetensors adapter (58 MB with two blocks, 115 MB with four) that reloads with parity |\n\n**How to use this notebook.** Choose a runtime (CPU works; **Runtime → '
            "Change runtime type → T4 GPU** is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's "
            'own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the audited model '
            'snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are '
            'the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; '
            'after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer. Each quoted number names its run: values on the 176 '
            '*exposed development-test* slices come from the Colab CLI Tesla T4 run of notebook blob `066b560d` on 8 October 2026 (`docs/release-verification.md`); '
            'the grid, the 352-slice validation and the fresh-test values come from the **pre-registered evaluation of record**, the Colab CLI T4 run of 17 October 2026 (REL13: that run; `docs/release-verification.md` also discloses the earlier CPU development look, which changed no setting). Every adaptation starts from the pinned base, so re-running Section 7 with other settings '
            'is a fresh experiment. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** '
            '1–3 infrastructure → 4 the OrganAMNIST slices, their roles and refusals *(evaluation practice)* → 5 the inference contract on drawn shapes *(core '
            "concept: relative, uncalibrated scores)* → 6 two baselines and the frozen model's zero-shot score *(evaluation practice)* → 7 bounded fine-tuning of "
            'the vision tower, after a pre-registered configuration grid on validation *(core concept, evaluation practice)* → 8 the fresh test, scored only under the pre-registered settings *(evaluation practice)* → 9 re-scored shapes, export and fresh reload '
            '*(engineering)* → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "weights_key": "pubmed-clip-vit-base-patch32",
    "modules": ["config.py", "model.py", "metrics.py", "samples.py", "protocol.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "rewrites": [
        ["^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$", 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "PubMedClipPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, progress=print)",
    "runtime_imports": ["torch", "transformers", "numpy"],
    "title": "PubMedCLIP Biomedical Pipeline — DIMER E2E zero-shot classification fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/tutorials/pubmedclip_biomedical_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-flaviagiammarino%2Fpubmed--clip--vit--base--patch32-ffcc4d?style=flat",
            "https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-sarahESL%2FPubMedCLIP-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/sarahESL/PubMedCLIP",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2112.13906-b31b1b.svg", "https://arxiv.org/abs/2112.13906"),
    ],
    "capability": "zero-shot image classification, image/text embeddings, cosine similarity, text-to-image retrieval and bounded supervised fine-tuning of the vision tower's last blocks on a labelled medical-image dataset, using the pinned `flaviagiammarino/pubmed-clip-vit-base-patch32` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `flaviagiammarino/pubmed-clip-vit-base-patch32` snapshot, audits the 605 MB `pytorch_model.bin` statically and "
        "unpickles it exactly once through torch's weights-only loader into `model.safetensors` (the only file the model is ever "
        "loaded from), fetches the 1.8 GB MedMNIST+ `organamnist_224.npz` archive from Zenodo (verified by byte size and SHA-256, "
        "read with `numpy.load(allow_pickle=False)`), keeps exactly the 1,100 pinned slices (each verified again) in four roles drawn from the dataset's own "
        "scan-level splits (396 training, 352 validation, 176 exposed development test, 176 fresh test), classifies three drawn shapes through the inference contract with an "
        "input manifest and a rejection probe, measures the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP over "
        "the 176 exposed development-test slices beside the majority-floor and intensity-nearest-neighbour baselines, runs the pre-registered "
        "three-configuration grid (`docs/pubmedclip-preregistration.md`) on the 352 validation slices and applies its selection rule, runs a "
        "bounded fine-tuning of the vision tower's last blocks, post-layernorm and visual projection with the selected configuration, the "
        "cross-entropy over the eleven organ prompts and validation-mAP epoch selection, scores the 176-slice fresh test (only when every "
        "pre-registered setting holds) and the exposed development test per organ, re-scores the drawn shapes with the adapted model, "
        "exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify embedding parity. "
        "The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.5 §5). On CPU the model time of the whole path, grid included, was about 15 minutes on the build workstation "
        "after the downloads (CPU build record of 2026-10-17; expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically "
        "when present (the grid took 83.3 s on the 2026-10-17 T4 run of record)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply one zip (or "
        "folder) — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, through the "
        "Colab upload dialog — holding a `labels.csv` (columns `id`, `file`, `label`) beside the image files — at least twelve images over at "
        "least two labels (six per label: the split keeps one test and one validation image of each and needs eight to train), the label text being what the prompt names (filled into `BYOD_PROMPT_TEMPLATE`, by default `A medical image of {label}.`). `file` may be a path such as `images/a.png`; an optional `group` column (patient, scan or session) keeps each group in one role, and an optional `split` column (train / validation / test) sets the roles yourself. They pass through the same validation, seeded stratified "
        "split, baselines, grid, fine-tuning, evaluation, artifact export and reload-parity cells as the OrganAMNIST sample. "
        "The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path. A BYOD run is **exploratory**: it never scores the tutorial's "
        "fresh test; its numbers are your own validation and test splits, labelled as such. A refused input prints one message naming the "
        "file and the rule, with no traceback; earlier output files are kept but marked as not from this run. **Do not upload identifiable patient data to a hosted "
        "runtime**; nothing here is a clinical tool."
    ),
    "intro": (
        "`flaviagiammarino/pubmed-clip-vit-base-patch32` is PubMedCLIP [1], [6], [7]: OpenAI's CLIP ViT-B/32 [2] "
        "fine-tuned on the Radiology Objects in COntext (ROCO) image–caption pairs from open-access PubMed articles [3], released under "
        "the **MIT** licence — a ViT-B/32 image tower at 224×224 and a 12-layer text tower over CLIP's 49k-token BPE vocabulary, "
        "151,277,313 parameters. An image and a prompt are scored by the cosine of their projected embeddings scaled by the "
        "model's learned `logit_scale` and read through a **softmax over the candidate prompts**: the scores are relative to the "
        "candidates given, **not calibrated probabilities**, **prompt/label dependent**, and the model never abstains — the "
        "highest-scoring label is returned whatever the image shows. It is a research model: **not a clinical tool, not "
        "validated for diagnosis or triage**, and nothing in this notebook changes that.\n\n"
        "What this notebook adds to inference is **adaptation with labelled images**. The dataset is real, inside the model's "
        "modality and where the frozen model has room to improve: 1,100 axial abdominal CT slices of eleven organs from MedMNIST+ "
        "**OrganAMNIST** [4] at 224 px (**CC BY 4.0**; source CT volumes from LiTS [5]; 100 per organ in four roles, each from one of the dataset's own train / "
        "validation / test splits, which the authors made by CT scan), pinned by split, index and the SHA-256 of every slice and extracted from "
        "the digest-verified archive at run time. Zero-shot prompts from the organ names barely separate these slices (the "
        "Colab run of blob `066b560d` on 8 October 2026 measured accuracy 0.18 frozen on the 176 exposed development-test slices — the left / right pairs of kidney, lung and "
        "femur are indistinguishable to a caption model), so the honest question is narrow: does a bounded fine-tuning of the "
        "vision tower's last blocks on 396 slices move zero-shot accuracy and the retrieval view (**text-to-image mAP**) on a "
        "scan-disjoint test split, per organ, against two **non-neural baselines** (the **majority floor** and an **intensity "
        "nearest neighbour** — the carried `colour_neighbour_baseline`, whose three channels coincide on grey-scale CT)? Nothing "
        "here is a quality claim about your images: it is one seeded run on one sample.\n\n"
        "**Possible overlap with what the model has seen (DAT9).** Overlap between these slices and the model's training data was "
        "**not audited**. The slices are MedMNIST+ OrganAMNIST crops of the LiTS CT volumes, not ROCO figures; but ROCO is drawn "
        "from PubMed Central figures, which can include CT images from the same source collections, and CLIP's web-scale "
        "pretraining data are not enumerated, so overlap with the ROCO/PMC fine-tuning data or with CLIP pretraining cannot be "
        "ruled out. The frozen score is therefore not a guaranteed-unseen zero-shot measurement.\n\n"
        "**How the configuration is chosen.** An earlier revision said the default was the configuration that had gained on the "
        "test split; that sentence was carried over from another tutorial, and the record cannot show how the default was chosen. "
        "This revision chooses it on **validation** with a grid of three configurations fixed in advance in "
        "`docs/pubmedclip-preregistration.md`, and scores a **fresh test** of 176 never-scored slices only under those "
        "settings. The 176 test slices every earlier run scored are kept as the labelled **exposed development test**.\n\n"
        "**Snapshot note:** the pinned revision hosts the weights only as framework pickles / archives (`pytorch_model.bin`, "
        "`tf_model.h5`, `flax_model.msgpack`). Section 3 stages the 9-file manifest (the PyTorch source among them, "
        "digest-verified), and the model load audits that pickle with `pickletools` (the fleet's four state-dict globals and "
        "nothing else), unpickles it once through `torch.load(weights_only=True)`, writes `model.safetensors` (605,156,676 bytes, "
        "digest-pinned) and loads only that file, printing the audit and conversion report. The `.h5` and the `.msgpack` are "
        "never fetched."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot and read a static pickle audit before a one-time conversion; fetch a digest-pinned labelled "
        "medical-image set, validate it and keep it in its own scan-disjoint roles; classify drawn shapes through the public "
        "API and read softmax scores correctly (relative, uncalibrated, no abstention); measure the frozen model's zero-shot "
        "accuracy, macro F1 and text-to-image mAP beside two non-neural baselines and read the per-organ breakdown; run a "
        "bounded fine-tuning with the cross-entropy over class prompts, explicit hyperparameters and validation-based epoch "
        "selection; choose a configuration on validation with a pre-registered grid and a margin rule; score a fresh test only under "
        "the pre-registered settings and read the exposed development test as what it is; evaluate with two prompt sets; re-score drawings from a different image family with "
        "the adapted model; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, semantic segmentation, OCR, caption generation, calibrated probabilities or universal thresholds, "
        "any diagnostic, triage or clinical decision use, fine-tuning of the text tower, the embeddings, the text projection or "
        "`logit_scale`, training on images that are not the pinned sample or your own uploads, evaluation on MedMNIST or any "
        "benchmark proper (only one seeded 1,100-slice sample is used here), and any claim that eleven abdominal organs stand in "
        "for your classes. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the CPU build record of 2026-10-17 (build workstation, a development run of this revision, not a notebook execution) took 519 s for the three-configuration grid and 311 s for the final six-epoch fine-tuning (396 slices per epoch, scored on 352 validation slices after each epoch), so the whole default path is about 15 minutes of model time with the snapshot and slices already cached (a 2-vCPU hosted runtime will be several times slower); the 2026-10-17 Colab CLI T4 run of record took 83.3 s for the grid. Building the isolated environment (the pinned `torch==2.14.0` among its packages; reused on a re-run), the 605 MB checkpoint and the 1.8 GB MedMNIST archive are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a softmax score and a cosine similarity are; what accuracy, macro F1 and average precision measure and why none is a human judgement; why a high score is not a correct label — and, for medical images, why none of this is diagnostic evidence.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides up to `MAX_IMAGE_SIDE` (4096) px and a label of at most 64 plain characters (the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label, or its display name, filled in). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records over 2..100 labels; the sample keeps the dataset's own roles and BYOD is split stratified per label after pixel-digest de-duplication so no image lands in two splits — or by an optional `group` column (patient, scan or session) so no group lands in two splits, or by an optional `split` column — and needs at least 12 images (six per label for two labels) for the split to leave eight to train on. BYOD accepts one zip of images plus a `labels.csv` in that shape.",
        "- **Environment variables (optional; the defaults need neither):** `DIMER_ISOLATED_ENV` names the folder Section 1 builds the isolated environment in (default `dimer_isolated_env_<lock digest>` in the working directory; set it to reuse an environment elsewhere). `PUBMEDCLIP_WEIGHTS_DIR` names a folder holding an already-staged snapshot that the carried `resolve_weights_path` uses when no folder is passed; this notebook always passes the Section 3 folder, so it only matters when you call the package yourself.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right or that an image is a CT slice — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data — and never identifiable patient data — to a hosted runtime unless you are authorized to process it there. The default path uploads nothing; the sample slices are de-identified research data published under CC BY 4.0.",
        "- **External access (data):** besides the model snapshot, the default path fetches one archive, `https://zenodo.org/records/10519652/files/organamnist_224.npz` (about 1.8 GB), pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; only the 1,100 pinned slices are kept, each verified against its own digest, and every record keeps its MedMNIST split and index. The data are CC BY 4.0 (MedMNIST, Yang et al. 2023; source volumes from the Liver Tumor Segmentation Benchmark); nothing is redistributed by the repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. OrganAMNIST slices and roles\n\n"
                "`fetch_corpus` returns the 1,100 pinned slices from the per-slice cache under `weights/organamnist/slices/` or "
                "extracts them from the digest-verified `organamnist_224.npz` (fetched from Zenodo when absent; read with "
                "`numpy.load(allow_pickle=False)`, one split array at a time) — every cached slice is re-hashed and every extracted "
                "slice refused on a digest mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their "
                "MedMNIST split, index, organ name and tutorial role. `build_sample_dataset` keeps four roles, each from one official "
                "split, so no CT scan feeds training and an evaluation role: **train** 36 per organ (official train), **validation** 32 per "
                "organ (official val, drawn from slices never used before), the **exposed development test** 16 per organ (the official-test "
                "slices every earlier run of this notebook scored) and the **fresh test** 16 per organ (official-test slices never scored "
                "before) → 396 / 352 / 176 / 176; the seed only orders the records. The fresh test shares the 70 official test scans with "
                "the exposed development test (MedMNIST publishes no scan id, so that cannot be removed); the draw skipped exact pixel "
                "duplicates and applied a coarse pixel screen against the exposed slices (`docs/pubmedclip-preregistration.md`, `docs/pubmedclip-draw-log.json`). `validate_dataset` then "
                "checks every record against the contract, `check_split_disjoint` asserts no slice (by decoded-pixel digest) is "
                "shared, `source_overlap` reports which MedMNIST split feeds each role (an observation: scan identity is not "
                "published, the authors split by scan), and the training split's labels table is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 1,100 slices, the eleven organs with 36 / 32 / 16 / 16 each, four digests, the role check, and four refusal probes — a "
                "duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each "
                "rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Predict before running:** the slices come from CT scans of a limited number of patients. Why "
                "keep MedMNIST's own roles instead of shuffling the 1,100 slices into new splits?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n"
                "from dataclasses import asdict\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "# A .zip or a folder already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "# The sentence each of your labels is filled into ({{label}} marks the spot); empty = DEFAULT_PROMPT_TEMPLATE 'A medical image of {{label}}.'\n"
                "BYOD_PROMPT_TEMPLATE = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# R71P-M1: a grid selection, a Section 7b model result or a fresh-test score from an earlier data load never carries over.\n"
                "grid_selection = adapt_result = fresh_test = probe = None\n\n\n"
                "class BYODRefusal(ValueError):\n"
                "    \"\"\"A refused BYOD input: shown as this one message, without a traceback (it names the file and the rule).\"\"\"\n\n"
                "    dimer_refusal = True  # the isolated-environment router prints the message alone\n\n"
                "    def _render_traceback_(self):  # the same when the cell runs in the kernel itself\n"
                "        return [str(self)]\n\n\n"
                "def refuse_byod(reason):\n"
                "    \"\"\"DAT20: forget the previous data so no later cell reuses it, mark earlier output files as not from this run, and\n"
                "    return one actionable refusal. Earlier files are kept; the marker says they belong to an earlier run.\"\"\"\n"
                "    for stale in ('splits', 'dataset_manifests', 'train_records', 'val_records', 'test_records', 'exposed_records', 'dev_records', 'classes'):\n"
                "        globals().pop(stale, None)\n"
                "    earlier = sorted(name for name in os.listdir('outputs') if name.endswith('.json') and not name.endswith('run_invalidated.json'))\n"
                "    with open('outputs/{stem}_run_invalidated.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'reason': str(reason), 'files_from_an_earlier_run': earlier}}, handle, indent=2)\n"
                "    return BYODRefusal(f'BYOD input refused: {{reason}} Fix the upload or BYOD_PATH and run this cell again. Files already in outputs/ are from an earlier run (listed in outputs/{stem}_run_invalidated.json).')\n\n\n"
                "if os.path.exists('outputs/{stem}_run_invalidated.json'):\n"
                "    os.remove('outputs/{stem}_run_invalidated.json')\n"
                "if USE_BYOD:\n"
                "    # A refused input (missing path, wrong file, bad labels.csv, too few images) prints one message; any other error keeps its traceback.\n"
                "    try:\n"
                "        if BYOD_PATH.strip():\n"
                "            byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "            if not byod_zip.exists():\n"
                "                raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding labels.csv and the image files.')\n"
                "            file_name = byod_zip.name\n"
                "        else:\n"
                "            try:\n"
                "                from google.colab import files\n"
                "            except ImportError:\n"
                "                raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "            uploaded = files.upload() or {{}}\n"
                "            if len(uploaded) != 1:\n"
                "                raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "            file_name, payload = next(iter(uploaded.items()))\n"
                "            if not file_name.lower().endswith('.zip'):\n"
                "                raise ValueError(f'{{file_name}}: upload one .zip holding labels.csv and the image files.')\n"
                "            byod_zip = Path('work') / 'byod.zip'\n"
                "            byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "            byod_zip.write_bytes(payload)\n"
                "    except (FileNotFoundError, RuntimeError, ValueError) as exc:\n"
                "        raise refuse_byod(exc) from None\n"
                "    try:\n"
                "        records = load_byod_dataset(byod_zip)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "        for name, part in splits.items():\n"
                "            validate_dataset(part, min_records=1)\n"
                "    except ValueError as exc:\n"
                "        raise refuse_byod(f'{{file_name}}: {{exc}}.') from None\n"
                "    split_mode = byod_split_mode(records)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    display_names = {{}}\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t_fetch = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/organamnist')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    split_mode = \"the dataset's own scan-level roles\"\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    display_names = {{key: name for key, (name, _index) in ORGANS.items()}}\n"
                "    raw_rows = {{'slices': len(corpus), 'slice_bytes': sum(len(v) for v in corpus_files.values()), 'archive_bytes': CORPUS_ARCHIVE_BYTES, 'fetch_seconds': round(time.perf_counter() - t_fetch, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=1 if name != 'train' else MIN_RECORDS) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "if split_mode == 'group column':  # PMC-m4: no patient / scan / session group in two roles\n"
                "    role_groups = {{name: {{r['group'] for r in part}} for name, part in splits.items()}}\n"
                "    assert not (role_groups['train'] & role_groups['test'] or role_groups['train'] & role_groups['validation'] or role_groups['validation'] & role_groups['test']), 'a group is in two roles'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# The fresh test is `test_records`; it is scored only under the pre-registered settings (Section 8). The development set read\n"
                "# in Sections 6-8 is the labelled exposed development test on the sample, your own test split on BYOD.\n"
                "exposed_records = splits.get('exposed_dev_test', [])\n"
                "dev_records, dev_label = (exposed_records, 'exposed development test') if not USE_BYOD else (test_records, 'your own test split')\n"
                "if not USE_BYOD:  # every role from its own official split (scan-disjoint from train), and the two test sets share no slice\n"
                "    roles_ok = {{name: sorted({{r['medmnist_split'] for r in part}}) for name, part in splits.items()}}\n"
                "    assert roles_ok == {{'train': ['train'], 'validation': ['val'], 'exposed_dev_test': ['test'], 'test': ['test']}}, roles_ok\n"
                "    assert not {{r['medmnist_index'] for r in test_records}} & {{r['medmnist_index'] for r in exposed_records}}, 'a fresh-test slice is an exposed one'\n"
                "    print({{'role_check': 'train/val/test from their own official splits (MedMNIST splits by scan); fresh and exposed test share no slice, but share the 70 official test scans', 'pin_manifest_sha256': pin_manifest_digest()[:16] + '...'}})\n"
                "classes = class_names(train_records)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'split_mode': split_mode, 'splits': disjoint, 'source_overlap': source_overlap(splits), 'classes': classes}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'label': example['label'], 'display_name': display_names.get(example['label'], example['label']), 'size': list(example['image'].size), 'source': example.get('source')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'one label only': [{{**r, 'label': 'organ'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** 1,100 slices, 36 / 32 / 16 / 16 per organ, the four split digests, `role_check`, `source_overlap`, and the four refusals.\n\n<details><summary>Check your '
                "reasoning</summary>Neighbouring slices of one scan look almost identical. MedMNIST's authors split by scan, so keeping their roles means no scan feeds "
                'both training and an evaluation role; a fresh random split would put sister slices on both sides and make memorisation look like skill. `check_split_disjoint` also '
                'verifies that no decoded slice appears twice. One limit remains: the fresh test and the exposed development test both come from the 70 official test scans, '
                'so they may hold slices of the same scans; in this revision neither is used for training or selection. The fixed settings (epochs, batch, prompt, learning rate) were set in earlier revisions with the exposed development test in view; because the fresh test shares its 70 scans, it is unseen at the slice level but not an independent, patient-level confirmation of those choices. '
                'MedMNIST splits by scan; patient-level disjointness is not documented. The refusals (duplicate id, oversized image, one label, too few records) stop before the model runs and '
                'say why.</details>'
            ),
        },
        {
            "md": (
                "## 5. Classify through the inference contract\n\n"
                "The inference contract is exercised on three 32×32 synthetic shapes — a red square, a green circle and a blue "
                "triangle — rendered in code as ASCII PPM files exactly as the repository's `examples/sample-data/generate_samples.py` "
                "renders them and digest-asserted against `SHA256SUMS`: a different image family from the CT slices, images a "
                "radiology-caption model has no reason to know, and images the model will be asked to classify again after "
                "adaptation. `validate_inputs` applies exactly the checks the public operations apply and returns an input "
                "manifest; a remote URL is validated too and its rejection recorded as a finding. `zero_shot_classify` returns one "
                "softmax score per candidate label, **ordered by descending score**; `retrieve` ranks the gallery by cosine to a "
                "query. The per-grid `evaluation_report` on three drawn shapes is `sample-sanity` — plumbing evidence, not a "
                "measurement, and a wrong shape here is not a defect; whether the classifier is *right* on its own modality is what "
                "Section 6 measures on 176 slices.\n\n"
                "**Predict before running:** shown a red square and the labels *red square*, *green circle*, *blue triangle* and "
                "*abstract geometric shape*, what will the four scores add up to, and can the model answer *none of these*?"
            ),
            "code": (
                "pipe.restore_base()  # PMC-M2: the frozen shapes are scored by the pinned base, also on a BYOD re-run from Section 4\n"
                "SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n"
                "    'red_square.ppm': 'b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096',\n"
                "    'green_circle.ppm': '2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f',\n"
                "    'blue_triangle.ppm': 'f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3',\n"
                "}}\n"
                "WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n\n\n"
                "def shape_mask(shape, x, y):\n"
                "    if shape == 'square':\n"
                "        return 8 <= x < 24 and 8 <= y < 24\n"
                "    if shape == 'circle':\n"
                "        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n"
                "    if not 7 <= y < 26:\n"
                "        return False\n"
                "    half = (y - 7) // 2\n"
                "    return 16 - half <= x <= 16 + half\n\n\n"
                "def render_ppm(foreground, shape):\n"
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                "    for y in range(HEIGHT):\n"
                "        row = []\n"
                "        for x in range(WIDTH):\n"
                "            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n"
                "            row.extend(str(v) for v in pixel)\n"
                "        for start in range(0, len(row), 24):\n"
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n\n\n"
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "specs = {{'red_square.ppm': ((220, 40, 40), 'square'), 'green_circle.ppm': ((40, 170, 75), 'circle'), 'blue_triangle.ppm': ((40, 90, 220), 'triangle')}}\n"
                "shape_images = []\n"
                "for name, (foreground, shape) in specs.items():\n"
                "    path = Path('outputs/sample-data') / name\n"
                "    path.write_bytes(render_ppm(foreground, shape).encode('ascii'))\n"
                "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
                "    if digest != SAMPLE_DIGESTS[name]:\n"
                "        raise ValueError(f'Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}')\n"
                "    shape_images.append(path)\n"
                "shape_labels = ['red square', 'green circle', 'blue triangle']\n"
                "candidate_labels = [*shape_labels, 'abstract geometric shape']\n"
                "shape_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shape_images}}\n"
                "print({{'ceilings': {{'TEXT_MAX_LENGTH': TEXT_MAX_LENGTH, 'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CLASSES': MIN_CLASSES, 'image_contract': '224x224 RGB after processor resize and centre crop', 'device': str(pipe.device)}}}})\n"
                "input_manifest = validate_inputs(shape_images, candidate_labels, top_k=len(shape_images), names=[p.name for p in shape_images])\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png', candidate_labels)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'shapes': [p.name for p in shape_images], 'sha256': {{k: v[:16] + '...' for k, v in shape_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "classifications, retrievals, classification_rows = [], [], []\n"
                "t0 = time.perf_counter()\n"
                "for image_path, expected in zip(shape_images, shape_labels, strict=True):\n"
                "    scores = pipe.zero_shot_classify(image_path, candidate_labels)\n"
                "    classifications.append(scores)\n"
                "    classification_rows.append({{'image': image_path.name, 'expected_label': expected, 'predicted_label': scores[0].label, 'scores': [asdict(x) for x in scores]}})\n"
                "    print(image_path.name, '->', [(s.label, round(s.score, 4)) for s in scores])\n"
                "for query in shape_labels:\n"
                "    retrievals.append(pipe.retrieve(query, shape_images, top_k=len(shape_images)))\n"
                "shape_embeddings = pipe.embed_image(shape_images)\n"
                "checks = {{\n"
                "    'one_ranking_per_image': len(classifications) == len(shape_images),\n"
                "    'scores_in_unit_interval': all(0.0 <= s.score <= 1.0 for ranking in classifications for s in ranking),\n"
                "    'scores_sum_to_one': all(abs(sum(s.score for s in ranking) - 1.0) < 1e-4 for ranking in classifications),\n"
                "    'rankings_descending': all(ranking[i].score >= ranking[i + 1].score for ranking in classifications for i in range(len(ranking) - 1)),\n"
                "    'embeddings_unit_norm': bool(np.allclose(np.linalg.norm(shape_embeddings, axis=1), 1.0, atol=1e-4)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "result = {{'classifications': classifications, 'retrievals': retrievals, 'gallery_ids': [p.name for p in shape_images], 'embedding_shapes': {{'image': list(shape_embeddings.shape)}}}}\n"
                "targets = {{'labels': shape_labels, 'retrieval_indices': list(range(len(shape_images)))}}\n"
                "frozen_scene = evaluation_report(result, targets, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            "md": (
                '**What to notice:** each ranking in descending order, the scores summing to 1, `checks`, and the remote-URL rejection in the input manifest.\n\n<details><summary>Check '
                'your reasoning</summary>They sum to 1, and no. The softmax is taken over the candidates you give, so a score is relative to that list: add or remove a '
                'label and every score moves. There is no *none of these* — the top label is returned whatever the image shows. Whether the drawings are ranked '
                'correctly is sample-sanity only; the recorded rankings are in `docs/release-verification.md`.</details>'
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's zero-shot score on the exposed development test\n\n"
                "This section reads the **exposed development test** — the 176 test slices every earlier run of this notebook scored — "
                "so the numbers can be compared with the earlier records; it is never used for a decision, and the fresh test is not touched "
                "until Section 8. "
                "Three systems frame the adaptation, each read three ways. The **majority floor** answers every slice with "
                "the most frequent training label — on this balanced training set a tie, broken alphabetically, so it answers *bladder* (accuracy 1/11 on a balanced split, chance-level macro F1; its scores are all ties, so its mAP is exactly the class prevalence, 1/11, whatever order the slices are in). The **intensity "
                "nearest neighbour** (`colour_neighbour_baseline`) answers with the label of the training slice whose 3×3 mean-intensity "
                "grid is closest — a classifier that knows the image through nine numbers, and on axial CT a stronger one than it "
                "sounds, because organs sit at characteristic positions. The **frozen model** is scored by `pipe.evaluate`: one "
                "prompt per organ (`CT_PROMPT` with the organ's display name — *left kidney*, *right lung*, …), the model's own "
                "scaled cosine logits as the score grid, **accuracy** and **macro F1** of the top prompt, the per-organ recall, and "
                "**text-to-image mAP** — each prompt as a query ranking all 176 slices, the average precision of its own organ, "
                "averaged over the eleven — the retrieval view of the same scores and the smoother of the three on a small set. A "
                "second prompt set built from the raw MedMNIST label tokens (`kidney-left`) in the default template is scored too, to "
                "show how much the number is the prompt's. Expect the frozen model above the majority floor but **well below the "
                "intensity neighbour**: the Colab run of blob `066b560d` on 8 October 2026 measured accuracy 0.18 / macro F1 0.13 / mAP 0.28 "
                "frozen against 0.51 for the neighbour on these 176 slices. Whether the frozen model beats the majority floor is recorded as a **verdict**; "
                "on the pinned sample it is also asserted on this exposed set as a check of the snapshot and slice digests, and a run on your own labels "
                "continues to adaptation and export either way. Read the per-organ breakdown — in that run the frozen model answered *heart* for most slices (heart recall 1.00 on an average precision of 0.25), never finds a kidney, a femur or the spleen, and only the right lung ranks well (AP 0.79).\n\n"
                "*Evaluation practice.* **Predict before running:** rank the three systems — majority floor, intensity neighbour, frozen "
                "PubMedCLIP — on accuracy."
            ),
            "code": (
                "# CT_PROMPT ('An axial abdominal CT slice showing the {{label}}.') is part of the pre-registration and is carried in protocol.py.\n"
                "# PMC-m1: your own images are scored with the documented DEFAULT_PROMPT_TEMPLATE (or BYOD_PROMPT_TEMPLATE), never the CT sentence.\n"
                "PROMPT_TEMPLATE = CT_PROMPT if not USE_BYOD else (BYOD_PROMPT_TEMPLATE.strip() or DEFAULT_PROMPT_TEMPLATE)\n"
                "if '{{label}}' not in PROMPT_TEMPLATE:\n"
                "    raise ValueError(f'BYOD_PROMPT_TEMPLATE {{PROMPT_TEMPLATE!r}} must contain {{{{label}}}} where the label goes, e.g. \"A chest X-ray showing {{{{label}}}}.\"')\n"
                "# PMC-M2: the frozen model is the pinned base, whatever an earlier run of Section 7 or 9 did to `pipe`.\n"
                "restored = pipe.restore_base()\n"
                "print({{'prompt_template': PROMPT_TEMPLATE, 'model_state': 'pinned base' + (f' (restored {{len(restored)}} tensors an earlier Section 7 changed)' if restored else '')}})\n"
                "print({{'scored_here': dev_label, 'n': len(dev_records)}})\n"
                "baseline_majority = majority_baseline(train_records, dev_records, classes)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, dev_records, classes)\n"
                "METRICS = ('accuracy', 'macro_f1', 't2i_map')\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'intensity_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(dev_records, classes=classes, class_names_map=display_names, prompt_template=PROMPT_TEMPLATE)\n"
                "print({{'frozen_model_' + dev_label.replace(' ', '_'): {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'prompt_template': frozen_test['prompt_template'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'recall': round(float(v['recall']), 2), 'ap': round(float(v['ap']), 2)}} for c, v in frozen_test['per_class'].items()}}\n"
                "print({{'by_organ_frozen': frozen_fields}})\n"
                "plain_names = {{key: key for key in ORGANS}} if not USE_BYOD else {{}}\n"
                "frozen_plain = pipe.evaluate(dev_records, classes=classes, class_names_map=plain_names)\n"
                "# R71P-M4 (pre-registration amendment 1): a supervised readout on FROZEN image embeddings, fitted on train, ridge chosen on validation.\n"
                "frozen_embeddings = {{name: pipe.embed_image([r['image'] for r in part]) for name, part in (('train', train_records), ('validation', val_records), ('development', dev_records))}}\n"
                "probe = fit_embedding_probe(frozen_embeddings['train'], [r['label'] for r in train_records], frozen_embeddings['validation'], [r['label'] for r in val_records], classes)\n"
                "probe_dev = apply_embedding_probe(probe, frozen_embeddings['development'], [r['label'] for r in dev_records], classes)\n"
                "print({{'frozen_embedding_probe': {{'ridge': probe['ridge'], 'ridges_tried (validation mAP)': probe['ridges_tried'], 'validation': {{k: round(float(probe['validation'][k]), 3) for k in METRICS}}, dev_label.replace(' ', '_'): {{k: round(float(probe_dev[k]), 3) for k in METRICS}}}}}})\n"
                "print({{'frozen_model_raw_label_prompts': {{k: round(frozen_plain[k], 3) for k in METRICS}}, 'prompt_template': frozen_plain['prompt_template']}})\n"
                "# SWP-A: a quality comparison is a recorded verdict, not an assert, so a BYOD run still reaches adaptation and export.\n"
                "frozen_vs_majority = {{m: 'above' if frozen_test[m] > baseline_majority[m] else 'not above' for m in ('accuracy', 't2i_map')}}\n"
                "print({{'verdict_frozen_vs_majority_floor': frozen_vs_majority}})\n"
                "if not USE_BYOD:  # a sanity check of the snapshot and slice digests on the EXPOSED development test only; the fresh test only ever gets verdicts\n"
                "    assert frozen_test['accuracy'] > baseline_majority['accuracy'] and frozen_test['t2i_map'] > baseline_majority['t2i_map'], 'on the exposed development test the frozen model should beat the majority floor; check the snapshot and slice digests'\n"
                "elif 'not above' in frozen_vs_majority.values():\n"
                "    print('Your labels: the frozen model does not beat the majority floor here. That is a result, not an error: Section 7 tests whether adaptation changes it.')"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on accuracy, macro F1 and mAP, the frozen per-organ recall and AP, the raw-label prompt score, and the verdict line.\n\n<details><summary>Check '
                'your reasoning</summary>Neighbour first, frozen model second, majority floor last. On these 176 exposed development-test slices, in the Colab CLI T4 run of blob `066b560d` (8 October 2026), accuracy was 0.511 for the intensity neighbour, '
                '0.182 for the frozen model and 0.091 for the majority floor (macro F1 0.502 / 0.131 / 0.015; mAP 0.519 / 0.284 / 0.091 — the majority floor mAP is the class prevalence, 1/11; runs before commit `9e4ca4b`, such as the 21 September 2026 Kaggle run of blob `4b2fc4c4`, printed 0.119, an artefact of tie order). Nine mean intensities beat a '
                'caption-trained model because organs sit at fixed positions in axial CT; the frozen model mostly answers *heart*. The raw-label prompts scored a '
                "different mAP (0.34): the number is partly the prompt's.</details>"
            ),
        },
        {
            "md": (
                "## 7. A pre-registered configuration grid on validation\n\n"
                "Before the tutorial adapts the model it **chooses the configuration on validation**, by a rule written down before any "
                "run (`docs/pubmedclip-preregistration.md`). Three configurations differ in one knob each — **C0** two trainable vision "
                "blocks at learning rate 5e-5 (the earlier default), **C1** two blocks at 1e-5, **C2** four blocks at 5e-5 — and everything "
                "else is fixed: six epochs, batch 16, seed 0, the CT prompt, AdamW, clipping at 1.0. Each is scored by its best-epoch "
                "**validation** text-to-image mAP on the 352 validation slices. C0 is replaced only by a challenger that beats it by at least "
                "**0.02** (a practical tolerance, not a significance test: the 352 slices come from only 16 validation scans, so they are far from 352 independent cases); two challengers within 0.02 of each other go to the "
                "smaller trainable scope; and the selected configuration must beat the frozen model's validation mAP by 0.02, or the frozen "
                "model is kept. The grid never reads either test set; it writes `outputs/{stem}_grid.json`. On a T4 it takes about 1–2 "
                "minutes; turning `RUN_GRID` off makes the run exploratory (Section 8 then does not score the fresh test).\n\n"
                "*Evaluation practice.* **Predict before running:** which configuration will win, and by more or less than 0.02?"
            ),
            "code": (
                "RUN_GRID = True  # @param {{type:\"boolean\"}}\n"
                "GRID_EPOCHS, GRID_BATCH_SIZE, GRID_SEED = PREREGISTERED_SETTINGS['epochs'], PREREGISTERED_SETTINGS['batch_size'], PREREGISTERED_SETTINGS['adapt_seed']\n"
                "grid_results, grid_selection, grid_seconds = {{}}, None, 0.0\n"
                "if RUN_GRID:\n"
                "    t_grid = time.perf_counter()\n"
                "    for grid_name, grid_config in GRID_CONFIGS.items():\n"
                "        t0 = time.perf_counter()\n"
                "        run = pipe.adapt(train_records, val_records, epochs=GRID_EPOCHS, lr=grid_config['lr'], batch_size=GRID_BATCH_SIZE, trainable_vision_layers=grid_config['trainable_vision_layers'], prompt_template=PROMPT_TEMPLATE, class_names_map=display_names, seed=GRID_SEED)\n"
                "        by_epoch = [round(float(h['val']['t2i_map']), 3) for h in run['history']]\n"
                "        grid_results[grid_name] = {{'config': grid_config, 'best_epoch': run['best_epoch'], 'best_val_t2i_map': round(float(run['history'][run['best_epoch']]['val']['t2i_map']), 4), 'val_t2i_map_by_epoch': by_epoch, 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "        print({{grid_name: grid_results[grid_name]}})\n"
                "    frozen_val_map = float(run['history'][0]['val']['t2i_map'])  # epoch 0 = the frozen model, the same in every config\n"
                "    grid_fingerprint = run_fingerprint(data_source='pinned sample' if not USE_BYOD else 'BYOD', pin_manifest_sha256=pin_manifest_digest() if not USE_BYOD else None, split_seed=SPLIT_SEED, validation_digest=dataset_manifests['validation']['digest'], epochs=GRID_EPOCHS, batch_size=GRID_BATCH_SIZE, seed=GRID_SEED, prompt_template=PROMPT_TEMPLATE)\n"
                "    grid_selection = select_config({{n: r['best_val_t2i_map'] for n, r in grid_results.items()}}, frozen_val_map, fingerprint=grid_fingerprint)\n"
                "    pipe.restore_base()\n"
                "    grid_seconds = round(time.perf_counter() - t_grid, 1)\n"
                "    print({{'grid_selection': grid_selection, 'seconds': grid_seconds}})\n"
                "    with open('outputs/{stem}_grid.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'purpose': 'selection only (validation); no test set is read', 'preregistration': PREREGISTRATION, 'data_source': data_source, 'n_validation': len(val_records), 'fixed': {{'epochs': GRID_EPOCHS, 'batch_size': GRID_BATCH_SIZE, 'seed': GRID_SEED, 'prompt_template': PROMPT_TEMPLATE}}, 'results': grid_results, 'selection': grid_selection, 'seconds': grid_seconds}}, handle, indent=2)\n"
                "else:\n"
                "    print('RUN_GRID is off: no configuration is selected in this run, so the run is exploratory and Section 8 will not score the fresh test.')"
            ),
        },
        {
            "md": (
                "**What to notice:** each configuration's validation mAP by epoch, the three `best_val_t2i_map` values against the 0.02 margin, "
                "`frozen_val_map`, and `reason` in `grid_selection`.\n\n<details><summary>Check your reasoning</summary>In the pre-registered "
                "evaluation of record (Colab CLI T4, 2026-10-17; REL13: that run; the earlier CPU development look made the same choice with C2 at 0.892) the best-epoch "
                "validation mAP was C0 0.772, C1 0.678 and C2 0.888 against a frozen 0.296, and the rule selected "
                "**C2** (C2 beat C0 by 0.12, well over the 0.02 margin, C1 fell short of C0, and C2 beat the frozen model by far more than 0.02). A difference under 0.02 on 352 slices is treated as no difference, which is why a "
                "challenger has to clear the margin rather than merely come out ahead.</details>"
            ),
        },
        {
            "md": (
                "## 7b. Bounded fine-tuning of the vision tower's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_VISION_LAYERS` blocks of the vision tower, its post-layernorm and "
                "the visual projection — with two blocks, 14,570,496 of 151,277,313 parameters; the text tower, the "
                "embeddings, the text projection and `logit_scale` stay frozen. The eleven organ prompts are embedded once by the "
                "frozen text tower; every batch of slices is run through the vision tower, scored against the prompts with the "
                "model's own scaled cosine logits, and trained with the cross-entropy of the softmax over the eleven prompts — "
                "CLIP's image-to-text objective restricted to the class set. AdamW at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is "
                "scored on the 352 validation slices, and the epoch with the highest validation text-to-image mAP is kept. With "
                "`CONFIG = 'grid selection'` (the default) the configuration is the one Section 7 selected; any other choice, and any "
                "change to `EPOCHS` or `BATCH_SIZE`, makes the run **exploratory** — Section 8 then reads validation and the exposed "
                "development test, not the fresh test.\n\n"
                "Watch the training loss fall while the validation mAP climbs and then flattens: 396 slices are few, but the organs are "
                "visually separable and the frozen model started far from the task, so the selector's job is to stop before the last "
                "blocks memorise the training slices — when that happens within six epochs; in the T4 run of record the selected C2 kept its last "
                "epoch, so selection kept the final model. C2 is the largest configuration in the grid and the best epoch is the last of the "
                "budget: it is the best of the three pre-registered options, at the edge of both the scope and the epoch budget, not a "
                "demonstrated optimum (the optional experiments explore past that edge, on validation only). The reported validation numbers "
                "are biased upward, because the configuration and the epoch were both chosen on them. On a GPU the 7b run can differ "
                "slightly from the grid row of the same configuration (non-deterministic kernels). In the T4 run of record (2026-10-17) "
                "the selected configuration kept epoch 6 of six (validation mAP 0.296 frozen → 0.888). Every call starts from the pinned base "
                "(`started_from` in the printed result), so a re-run with other settings is a fresh experiment, not continued training, "
                "and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** will the epoch with the lowest training loss be the one kept?"
            ),
            "code": (
                "CONFIG = 'grid selection'  # @param [\"grid selection\", \"C0\", \"C1\", \"C2\", \"manual\"]\n"
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "# Used only when CONFIG is 'manual' (an exploratory run):\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "TRAINABLE_VISION_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "ADAPT_SEED = 0  # pre-registered; not a form field\n"
                "if CONFIG == 'grid selection':\n"
                "    chosen = (grid_selection or {{}}).get('config') or GRID_CONFIGS['C0']\n"
                "elif CONFIG in GRID_CONFIGS:\n"
                "    chosen = GRID_CONFIGS[CONFIG]\n"
                "else:\n"
                "    chosen = {{'trainable_vision_layers': TRAINABLE_VISION_LAYERS, 'lr': LEARNING_RATE}}\n"
                "lr_used, layers_used = chosen['lr'], chosen['trainable_vision_layers']\n"
                "print({{'config': CONFIG, 'lr': lr_used, 'trainable_vision_layers': layers_used, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'seed': ADAPT_SEED}})\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=lr_used, batch_size=BATCH_SIZE, trainable_vision_layers=layers_used, prompt_template=PROMPT_TEMPLATE, class_names_map=display_names, seed=ADAPT_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'classes': adapt_result['classes'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the training loss against `val_t2i_map` per epoch, `best_epoch`, and `selection`.\n\n<details><summary>Check '
                'your reasoning</summary>Not by design. The training loss keeps falling to the last epoch, and the selector ignores it: it keeps the epoch with the highest validation mAP. In '
                'the T4 run of record (2026-10-17) the two happened to coincide — epoch 6 of six (validation mAP 0.296 → 0.888 on 352 slices; differences of about 0.01 between epochs are noise at this size). With other settings they part (in the grid, C0 peaked at epoch 2 of six while its loss kept falling); validation, not '
                'training loss, says when to stop.</details>'
            ),
        },
        {
            "md": (
                "## 8. The fresh test, scored only under the pre-registered settings\n\n"
                "The cell first decides whether this run is **pre-registered**: every setting the pre-registration fixes (the pinned "
                "sample and its pin manifest, `SPLIT_SEED`, six epochs, batch 16, seed 0, the CT prompt, a grid run) and the "
                "configuration the grid selected. `settings_differing` lists any that differ; `test_look` is then one of "
                "`pre-registered evaluation (this runtime)`, `reproduction (this runtime)` (an earlier report in this runtime already holds a "
                "pre-registered scoring; it is written to `*_reproduction_<n>.json`, never over the first) or `not scored (exploratory)`. A "
                "runtime cannot see other runtimes: the evaluation of record is the first hosted pre-registered run, named in "
                "`docs/release-verification.md` and in `RECORDED_EVALUATION`, and a run of yours with identical settings is a reproduction of it (the report's `relation_to_record` says so). The check also "
                "requires that the grid ran on this run's data and settings and that the model in memory is the one Section 7b trained "
                "(re-running Section 5, 6 or 7 after 7b puts the pinned base back, and the run becomes exploratory).\n\n"
                "- **Pre-registered:** the majority floor, the intensity neighbour, the frozen model and — if the grid kept adaptation — "
                "the adapted model are scored on the **176-slice fresh test**, not used for training, configuration or epoch selection in "
                "this revision (a CPU development look scored it once on 2026-10-17, before the T4 run of record; see `docs/release-verification.md`). The **frozen-embedding "
                "readout** (a ridge readout on the frozen model's image embeddings, fitted on train, its ridge chosen on validation; "
                "pre-registration amendment 1) is scored beside them: the gap between it and the adapted model is what fine-tuning the vision "
                "blocks adds over simply using the labels. The fixed settings (epochs, batch, prompt, learning rate) were set in earlier revisions with the exposed development test in view; because the fresh test shares its 70 scans, it is unseen at the slice level but not an independent, patient-level confirmation of those choices. Its slices come from the official test scans, which no training or validation "
                "slice shares; they do share those 70 scans with the exposed development test. The report is "
                "`outputs/{stem}_evaluation_report.json`.\n"
                "- **Exploratory** (any form change, BYOD, `RUN_GRID` off, a smaller batch for memory, the optional experiments): the "
                "fresh test is **not** scored. The cell reads validation and the development set (the labelled exposed development test, or "
                "your own test split on BYOD) and writes `outputs/{stem}_evaluation_report_exploratory_<n>.json`, never the "
                "pre-registered file.\n\n"
                "Either way the adapted model is also read on the exposed development test, in every run, for continuity with the earlier "
                "records (it informs no decision). Read **text-to-image mAP** first (the metric the configuration and the epoch were selected on), then "
                "accuracy and macro F1, then the per-organ rows: `ap_fell` and `recall_fell` name the organs whose AP or recall went down, "
                "so read those rather than trusting a mean. Outcomes are recorded **verdicts** (`improved`, `no change`, `worse`), never "
                "asserts, so a run that does not improve still exports and reloads. One hundred and seventy-six slices from one seeded "
                "run give **no dispersion estimate**; a gain on eleven abdominal organs says nothing about your classes — or about any "
                "clinical use — until you measure them.\n\n"
                "*Evaluation practice.* **Predict before running:** will the adapted model pass the intensity neighbour on the fresh "
                "test? Will its gain carry over to the raw-label prompts it was never tuned with?"
            ),
            "code": (
                "import shutil\n\n"
                "fresh_test = None\n"
                "run_settings = {{'data_source': 'pinned sample' if not USE_BYOD else 'BYOD', 'pin_manifest_sha256': pin_manifest_digest() if not USE_BYOD else None, 'split_seed': SPLIT_SEED, 'adapt_seed': (adapt_result or {{}}).get('seed'), 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'prompt_template': PROMPT_TEMPLATE, 'grid_run': grid_selection is not None, 'trainable_vision_layers': layers_used, 'lr': lr_used}}\n"
                "run_fingerprint_now = run_fingerprint(data_source=run_settings['data_source'], pin_manifest_sha256=run_settings['pin_manifest_sha256'], split_seed=SPLIT_SEED, validation_digest=dataset_manifests['validation']['digest'], epochs=PREREGISTERED_SETTINGS['epochs'], batch_size=PREREGISTERED_SETTINGS['batch_size'], seed=PREREGISTERED_SETTINGS['adapt_seed'], prompt_template=PROMPT_TEMPLATE)\n"
                "# R71P-M1: the fresh test needs the model Section 7b trained, not a base restored by a later re-run of Section 5, 6 or 7.\n"
                "model_state = adapter_mismatch(pipe.adapter, adapt_result)\n"
                "differing = settings_differing(run_settings, grid_selection, run_fingerprint_now, model_state)\n"
                "preregistered = not differing  # SCORE_ONCE_RULE: computed from ALL pre-registered settings, not one field\n"
                "report_path = Path('outputs/{stem}_evaluation_report.json')\n"
                "prior_report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None\n"
                "test_look = look_at_test(differing, prior_report)\n"
                "adaptation_kept = bool(grid_selection['adaptation_kept']) if grid_selection else True\n"
                "print({{'preregistered': preregistered, 'test_look': test_look, 'settings_differing': differing, 'adaptation_kept': adaptation_kept}})\n"
                "if not preregistered:\n"
                "    print('Exploratory run: the fresh test is NOT scored. Compare on validation; the development view below is the ' + dev_label + '.')\n\n\n"
                "def fields(result):\n"
                "    return {{c: {{'n': v['n'], 'recall': round(float(v['recall']), 2), 'ap': round(float(v['ap']), 2)}} for c, v in result['per_class'].items()}}\n\n\n"
                "def compare(majority, neighbour, frozen, adapted, frozen_raw, adapted_raw, readout=None):\n"
                "    systems = {{'majority': majority, 'neighbour': neighbour, **({{'frozen_embedding_probe': readout}} if readout else {{}}), 'frozen': frozen, **({{'adapted': adapted}} if adapted else {{}})}}\n"
                "    table = {{metric: {{name: round(float(r[metric]), 3) for name, r in systems.items()}} for metric in METRICS}}\n"
                "    table['raw_label_prompts_note'] = 'secondary analysis, not pre-registered'\n"
                "    table['raw_label_prompts'] = {{metric: {{'frozen': round(float(frozen_raw[metric]), 3), **({{'adapted': round(float(adapted_raw[metric]), 3)}} if adapted_raw else {{}})}} for metric in METRICS}}\n"
                "    before = fields(frozen)\n"
                "    if adapted:\n"
                "        after = fields(adapted)\n"
                "        delta = float(adapted['t2i_map']) - float(frozen['t2i_map'])\n"
                "        table['delta_vs_frozen'] = {{metric: round(float(adapted[metric]) - float(frozen[metric]), 3) for metric in METRICS}}\n"
                "        table['verdict'] = {{'adapted_vs_frozen_t2i_map': 'improved' if delta > 0 else ('no change' if delta == 0 else 'worse')}}\n"
                "        table['ap_fell'] = sorted(c for c in classes if after[c]['ap'] < before[c]['ap'])\n"
                "        table['recall_fell'] = sorted(c for c in classes if after[c]['recall'] < before[c]['recall'])\n"
                "        table['by_organ'] = {{c: {{'n': before[c]['n'], 'frozen_recall': before[c]['recall'], 'adapted_recall': after[c]['recall'], 'frozen_ap': before[c]['ap'], 'adapted_ap': after[c]['ap']}} for c in classes}}\n"
                "    else:\n"
                "        table['verdict'] = {{'adapted_vs_frozen_t2i_map': 'adaptation not kept (did not beat frozen on validation by the margin)'}}\n"
                "        table['by_organ'] = before\n"
                "    table['verdict']['frozen_vs_majority_floor'] = {{m: 'above' if frozen[m] > majority[m] else 'not above' for m in ('accuracy', 't2i_map')}}\n"
                "    return table\n\n\n"
                "adapted_val = pipe.evaluate(val_records, classes=classes, class_names_map=display_names, prompt_template=PROMPT_TEMPLATE)\n"
                "adapted_dev = pipe.evaluate(dev_records, classes=classes, class_names_map=display_names, prompt_template=PROMPT_TEMPLATE)\n"
                "adapted_dev_plain = pipe.evaluate(dev_records, classes=classes, class_names_map=plain_names)\n"
                "development_view = {{'label': dev_label, 'n': len(dev_records), 'note': 'reported in every run for continuity with earlier records; informs no decision' if not USE_BYOD else 'your own test split (exploratory)', **compare(baseline_majority, baseline_neighbour, frozen_test, adapted_dev, frozen_plain, adapted_dev_plain, probe_dev)}}\n"
                "print({{'validation': {{'adapted': {{k: round(float(adapted_val[k]), 3) for k in METRICS}}, 'frozen_embedding_probe': {{k: round(float(probe['validation'][k]), 3) for k in METRICS}}}}, 'n': adapted_val['n'], 'note': 'the configuration and the epoch were chosen on these slices, so these numbers are biased upward'}})\n"
                "print({{'development_view': {{k: development_view[k] for k in ('label', 'n', 'note', *METRICS, 'verdict')}}}})\n"
                "fresh_test = None\n"
                "if preregistered:\n"
                "    # THE fresh test: scored only here, only under the pre-registered settings. The adapted state is parked in work/ so\n"
                "    # the frozen model can be scored on the same slices, then put back.\n"
                "    parked = Path('work/{stem}_adapter_parked')\n"
                "    shutil.rmtree(parked, ignore_errors=True)\n"
                "    adapted_fresh = pipe.evaluate(test_records, classes=classes, class_names_map=display_names, prompt_template=PROMPT_TEMPLATE) if adaptation_kept else None\n"
                "    adapted_fresh_plain = pipe.evaluate(test_records, classes=classes, class_names_map=plain_names) if adaptation_kept else None\n"
                "    pipe.save_artifact(parked)\n"
                "    pipe.restore_base()\n"
                "    frozen_fresh = pipe.evaluate(test_records, classes=classes, class_names_map=display_names, prompt_template=PROMPT_TEMPLATE)\n"
                "    frozen_fresh_plain = pipe.evaluate(test_records, classes=classes, class_names_map=plain_names)\n"
                "    probe_fresh = apply_embedding_probe(probe, pipe.embed_image([r['image'] for r in test_records]), [r['label'] for r in test_records], classes)\n"
                "    pipe.load_artifact(parked)\n"
                "    fresh_test = {{'n': len(test_records), **compare(majority_baseline(train_records, test_records, classes), colour_neighbour_baseline(train_records, test_records, classes), frozen_fresh, adapted_fresh, frozen_fresh_plain, adapted_fresh_plain, probe_fresh)}}\n"
                "    print({{'fresh_test': {{k: fresh_test[k] for k in ('n', *METRICS, 'verdict')}}}})\n"
                "comparison = fresh_test if fresh_test is not None else development_view\n"
                "for key in ('raw_label_prompts', 'delta_vs_frozen', 'ap_fell', 'recall_fell', 'by_organ'):\n"
                "    if key in comparison:\n"
                "        print({{key: comparison[key]}})\n"
                "# The pre-registered evaluation of record (docs/release-verification.md); an identical pre-registered run is a reproduction of it.\n"
                "RECORDED_EVALUATION = 'Colab CLI T4 run of 2026-10-17 (commit 352452b, notebook blob c5a0e62d; docs/release-verification.md)'\n"
                "relation_to_record = ('reproduction of the pre-registered evaluation' if RECORDED_EVALUATION else 'the pre-registered evaluation') if preregistered else 'not scored (exploratory)'\n"
                "evaluation_report_payload = {{\n"
                "    'test_look': test_look,\n"
                "    'evaluation_of_record': RECORDED_EVALUATION,\n"
                "    'relation_to_record': relation_to_record,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': differing,\n"
                "    'preregistration': PREREGISTRATION,\n"
                "    'frozen_embedding_probe': {{'ridge': probe['ridge'], 'ridges_tried': probe['ridges_tried'], 'validation': {{k: probe['validation'][k] for k in METRICS}}}},\n"
                "    'run_settings': run_settings,\n"
                "    'grid_selection': grid_selection,\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'prompt_template': PROMPT_TEMPLATE,\n"
                "    'fresh_test': fresh_test,\n"
                "    'development_view': development_view,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "run_suffix = output_suffix(report_path, test_look)  # '' / _reproduction_<n> / _exploratory_<n>, once for every file this run writes\n\n\n"
                "def suffixed(path):\n"
                "    path = Path(path)\n"
                "    return path.with_name(path.stem + run_suffix + path.suffix)\n\n\n"
                "written = suffixed(report_path)  # never overwrites the first pre-registered report (this runtime)\n"
                "with open(written, 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': str(written), 'test_look': test_look, 'relation_to_record': relation_to_record}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `preregistered`, `test_look` and `settings_differing` first; then the four systems on the three metrics, `delta_vs_frozen`, the raw-label prompt rows, `ap_fell` / `recall_fell`, and the `verdict`.\n\n<details><summary>Check '
                'your reasoning</summary>The evaluation of record is the first hosted pre-registered run, the Colab CLI T4 run of 2026-10-17 (REL13: that run; `docs/release-verification.md`, which also discloses a CPU **development look** of the same day, uncommitted working tree later committed as `61d464c`, that read accuracy 0.886 and mAP 0.865 and changed no setting). It read '
                'accuracy 0.165 frozen → 0.881 adapted (neighbour 0.619, majority 0.091), text-to-image mAP 0.248 → 0.862, '
                'verdict *improved*; on the raw-label prompts (a secondary analysis, not pre-registered) mAP 0.328 → 0.910, so the gain transfers to a prompt set the model was never tuned with. `ap_fell` was [lung-right]: its AP fell 0.71 → 0.64 while its recall rose 0.19 → 0.94 — found as the top answer more often, ranked lower by its own prompt. '
                'On the exposed development test the same model read mAP 0.821 (the 8 October 2026 Colab run of the earlier default read 0.654 there). The intensity neighbour, '
                'a deterministic baseline, read accuracy 0.619 on the fresh slices and 0.511 on the exposed ones — two 176-slice draws from the same 70 scans, '
                'so a swing of about 0.1 is sampling noise, a free lesson in why one seeded run has no dispersion estimate. '
                'The frozen-embedding readout is the sharper comparison, and it changes the reading. On retrieval mAP it is as good as or better than the fine-tuned model on all three splits: fresh test 0.885 against 0.862, exposed development test 0.851 against 0.821, validation 0.968 against 0.888. On accuracy fine-tuning leads: fresh test 0.881 against 0.824, exposed development test 0.869 against 0.801. So fine-tuning the vision blocks improves classification accuracy, but a linear readout on the frozen embeddings is as good or better for retrieval, and the retrieval gain over the frozen zero-shot prompts should not be credited to fine-tuning the vision layers. '
                'A higher mean can hide a regression in one organ, one seeded run on 176 slices has no dispersion estimate, and none of this is clinical '
                'evidence.</details>'
            ),
        },
        {
            "md": (
                "## 9. Re-score the drawn shapes, export the adapter and reload it\n\n"
                "The three shapes from Section 5 are classified again by the adapted model — drawings, a different image family "
                "from the CT slices it was tuned on, so this is a small look at what the adaptation did *outside* its corpus "
                "(earlier runs' rankings are in `docs/release-verification.md`; a changed ranking here is a finding to record, not a "
                "failure) — and reported with the per-grid `evaluation_report` (`sample-sanity`). Both score sets are written "
                "as JSON.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the vision tower's trained blocks, post-layernorm and "
                "visual projection, about 58 MB with two blocks and 115 MB with four — as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digest of the base `model.safetensors`, the classes and prompt "
                "template, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `PubMedClipPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest "
                "and its exact tensor set **before** deserialising, refuses any tensor outside the vision tower and its projection, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical image embeddings on eight validation slices (VER4)."
            ),
            "code": (
                "adapted_classifications = [pipe.zero_shot_classify(image_path, candidate_labels) for image_path in shape_images]\n"
                "adapted_retrievals = [pipe.retrieve(query, shape_images, top_k=len(shape_images)) for query in shape_labels]\n"
                "adapted_scene = evaluation_report({{'classifications': adapted_classifications, 'retrievals': adapted_retrievals, 'gallery_ids': [p.name for p in shape_images]}}, targets, sample_kind='synthetic')\n"
                "for image_path, before, after in zip(shape_images, classifications, adapted_classifications, strict=True):\n"
                "    print({{'image': image_path.name, 'frozen': [(s.label, round(s.score, 3)) for s in before[:2]], 'adapted': [(s.label, round(s.score, 3)) for s in after[:2]]}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open(suffixed('outputs/{stem}_shapes.json'), 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'frozen': classification_rows, 'adapted': [{{'image': p.name, 'scores': [asdict(x) for x in ranking]}} for p, ranking in zip(shape_images, adapted_classifications, strict=True)]}}, handle, indent=2)\n\n"
                "artifact_dir = suffixed('outputs/{stem}_adapter')\n"
                "artifact_role = 'the adapted model' if adaptation_kept else 'exported for inspection only: the grid did not keep adaptation, so the pre-registered result is the frozen model'\n"
                "print({{'artifact_role': artifact_role}})\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PubMedClipPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.embed_image([r['image'] for r in val_records[:8]])\n"
                "after = reloaded.embed_image([r['image'] for r in val_records[:8]])\n"
                "parity = {{'max_abs_difference': float(np.abs(before - after).max()), 'identical_rows': int((np.abs(before - after).max(axis=1) < 1e-5).sum()), 'of': int(before.shape[0])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rows'] == parity['of']\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'test_look': test_look,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': differing,\n"
                "    'grid_selection': grid_selection,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256, 'source_file': SOURCE_FILENAME, 'source_sha256': SOURCE_SHA256, 'pickle_audit_sha256': PICKLE_AUDIT_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'url': CORPUS_URL, 'archive_bytes': CORPUS_ARCHIVE_BYTES, 'archive_sha256': CORPUS_ARCHIVE_SHA256, 'pinned_slices': len(tutorial_records()), 'pin_manifest_sha256': pin_manifest_digest(), 'organs': {{k: list(v) for k, v in ORGANS.items()}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'labels': shape_labels, 'candidate_labels': candidate_labels}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'verdict': comparison['verdict'],\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'role': artifact_role}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "result_path = suffixed('outputs/{stem}_result.json')\n"
                "with open(result_path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the frozen and adapted top-2 labels for each drawing, the artifact's size and tensor count, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>A changed ranking on the drawings is a finding, not a failure: tuning the vision tower on CT slices can move how it sees '
                'anything else, and three images are evidence, not a measurement. In the recorded run the adapter reloaded into a fresh pipeline with identical '
                'embeddings on all eight slices (`max_abs_difference` 0.0, 8 of 8 rows).</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is a partial zero-shot classifier on eleven abdominal organs whose names it was not trained to "
        "classify — above the majority floor, but well below the intensity neighbour — and a bounded fine-tuning of the vision "
        "tower's last blocks and projection, in the configuration a pre-registered grid chose on validation, moves both scores "
        "clearly: on the fresh test, text-to-image mAP 0.248 → 0.862 and accuracy 0.165 → "
        "0.881 in the pre-registered evaluation of record (Colab CLI T4, 2026-10-17; REL13: that run), with an adapter (115 MB with the four selected blocks) that reloads to "
        "identical embeddings. That is the claim: the adaptation contract works end to end on a real labelled medical-image set, "
        "and the numbers it produces are read on three metrics, per organ, on two prompt sets, against two non-neural baselines "
        "and the frozen model rather than in isolation [1], [2].\n\n"
        "The fresh test is 176 slices from the dataset's own test scans, scored only under the pre-registered settings; the "
        "validation split that chose the configuration and the epoch is 352 slices; the exposed development test (the 176 slices "
        "every earlier run scored) is reported in every run for continuity and informs no decision. The fresh and exposed test sets "
        "share the 70 official test scans, which MedMNIST does not identify, so they are not independent of each other — only of "
        "training and validation (scan-disjoint; patient-level disjointness is not documented). The fixed settings (epochs, batch, prompt, learning rate) were set in earlier revisions with the exposed development test in view; because the fresh test shares its 70 scans, it is unseen at the slice level but not an independent, patient-level confirmation of those choices. The frozen-embedding "
        "readout is the reference that splits the gain into \"a supervised readout of the frozen tower\" and \"fine-tuning the "
        "vision blocks\", and here it splits it unevenly: on the fresh test the readout reaches mAP 0.885 and accuracy 0.824, so fine-tuning "
        "improves classification accuracy (0.881) but a linear readout on the frozen embeddings is as good or better for retrieval "
        "(on all three splits); the retrieval gain should not be credited to fine-tuning the vision layers. On the data: LiTS is a cohort of liver-tumour patients from several centres, not a general population; "
        "OrganAMNIST labels come from bounding boxes; and by radiological convention the patient's left appears on the image's right, "
        "which is why left / right are the hard classes (the adaptation applies no flip augmentation, which is correct here). The metrics are three reference-based scores (own numpy implementations; none a human judgement), accuracy "
        "moves in steps of one slice, and there is no dispersion estimate. Overlap between these slices and the ROCO/PMC "
        "fine-tuning data or CLIP's pretraining data was not audited and cannot be ruled out (DAT9). So a gain here says the "
        "contract works, not that the adapted model is better on your images, that its scores are calibrated, or that a softmax "
        "score is a probability of being right — it still returns a best label for every image, and it can be wrong confidently. "
        "Fine-tuning on a narrow set can also erode the model elsewhere; the drawn shapes re-scored in Section 9 are three images of "
        "evidence about that, not a measurement. **None of this is clinical evidence:** OrganAMNIST slices are 224-pixel crops with "
        "organ labels derived from bounding boxes [4], the model is a research artefact trained on figure captions [1], and no "
        "reading of these numbers supports a diagnostic, triage or screening use; this notebook is not a medical device.\n\n"
        "Three things to carry to real data. **Baselines first:** the majority floor, the intensity neighbour and the frozen "
        "model's score on *your* labels are the numbers to read before any adapted one, per class and on the retrieval view. "
        "**Leakage and selection:** keep every image in one split (the contract de-duplicates by decoded pixels), split by patient, "
        "scan or session when your images come from few sources, and choose settings on validation — a test split that has guided a "
        "choice is a development set from then on. **Prompts:** the classifier is the prompt as much as the tower; a second prompt "
        "set is scored here so the difference is visible, and a label an annotator wrote is not a prompt the model understands — the fixed CT prompt itself says *abdominal* CT for the femur, the heart and the lungs, structures that are not abdominal.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, audit and convert its pickle once, fetch and digest-verify a real "
        "labelled medical-image set, validate the demonstrated dataset contract without leakage, execute the inference contract, "
        "a pre-registered selection on validation and a bounded fine-tuning, evaluate against two trivial baselines and the frozen "
        "model on a scan-disjoint fresh test, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, zero-shot accuracy on any other population, scanner or window "
        "setting, calibration, or clinical or production fitness.\n\n"
        "**Scope of a re-run.** Sections 5, 6, 7 and 7b each start from the pinned base (`restore_base`), so re-running any of them never continues from an earlier adaptation. "
        "The fresh test is scored only under the pre-registered settings (see `docs/release-verification.md` for the evaluation of record and its reproductions); this runtime labels a later run with "
        "identical settings a *reproduction (this runtime)*, and any changed setting makes the run *exploratory*: it compares on validation and writes a separate "
        "`*_exploratory_<n>.json`. For an experiment, change a Section 7b form value and run Sections 7b, 8 and 9 in order; for your own data, set `USE_BYOD` and run from Section 4 to the end.\n\n"
        "**Activity — Predict → Change → Run → Observe → Explain.** *Predict:* if a configuration the grid did **not** select trains instead, will the **validation** mAP fall, and how will the adapter's size change? "
        "*Change:* in Section 7b set `CONFIG` to one of C0, C1, C2 other than `grid_selection['selected']` (printed in Section 7) — choosing the selected one would repeat the pre-registered run. *Run:* Sections 7b, 8 and 9. *Observe:* epoch 0 still reads the frozen validation mAP (0.296 in the CPU build record — the proof that the run started from the pinned base), "
        "`best_epoch`, the validation `t2i_map`, `test_look` (*not scored (exploratory)*) and the development-view rows, and the adapter's bytes and tensor count in Section 9. "
        "*Explain:* compare the validation number with that configuration's row in Section 7's grid — the same configuration, the same seed — and say why only a difference of at least 0.02 on 352 slices was allowed to change the default.\n\n"
        "**Optional experiments (exploratory; they never score the fresh test):** set `CONFIG = 'manual'` with `TRAINABLE_VISION_LAYERS = 6` and compare the "
        "artifact size and the validation mAP; raise `EPOCHS` and watch the validation mAP pick the epoch while the training loss keeps falling; set "
        "`LEARNING_RATE` to another value under `manual` and compare its validation curve with the grid's C0 and C1 rows; or bring your own images through BYOD and "
        "read the two baselines before the adapted number. Each writes its own `*_exploratory_<n>.json`.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        "names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 4 is slow or stops on the archive's size or "
        'SHA-256** — the 1.8 GB Zenodo download was cut short or altered. Run Section 4 again; slices already extracted and verified are reused from `weights/organamnist/slices/`.\n- '
        '**Section 3 reports a pickle-audit refusal** — the audit allows the four state-dict globals only and names any other; the downloaded '
        '`pytorch_model.bin` is not the pinned one. Delete it and run Section 3 again.\n- **Section 7 is very slow** — you are on a 2-vCPU CPU runtime; switch '
        'to a T4 GPU. Lowering `EPOCHS` (or `BATCH_SIZE`, for memory) gives a quick look, but the run becomes exploratory: the fresh test is not scored and the numbers will differ from the recorded run. Turning `RUN_GRID` off saves the three grid adaptations, with the same consequence. The grid always uses the pre-registered batch of 16; if it runs out of memory, use a T4 (or turn `RUN_GRID` off and accept an exploratory run).\n- **BYOD: "BYOD zip must contain labels.csv" or a '
        'label/size refusal** — the message names the rule: a `labels.csv` with columns `id`, `file`, `label`, at least twelve images over at least two labels (six per label), '
        'sides up to 4096 px.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the '
        'upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip (or folder) in the runtime and set `BYOD_PATH` to its path.\n- **BYOD: '
        '"Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "BYOD input refused: …"** — a refused input prints this one message (no traceback) naming the file and the rule; '
        'files already in `outputs/` are from an earlier run and are listed in `outputs/{stem}_run_invalidated.json`. Fix the input and run Section 4 again.\n'
        '- **Environment variables** — `DIMER_ISOLATED_ENV` moves the isolated environment Section 1 builds (default: a folder named after the lock digest in the '
        'working directory); `PUBMEDCLIP_WEIGHTS_DIR` points the package at an already-staged snapshot when no folder is passed. Neither is needed for Run all.\n\n## Glossary\n\n- **CLIP / PubMedCLIP:** a '
        'pair of encoders, one for images and one for text, trained so matching image–caption pairs have similar embeddings; PubMedCLIP is CLIP fine-tuned on '
        'radiology figures and their captions.\n- **Zero-shot classification:** naming an image by comparing it with a text prompt for each candidate label, '
        'with no training on those labels.\n- **Prompt template:** the sentence a label is filled into (`An axial abdominal CT slice showing the {{label}}.`); '
        'the classifier is the prompt as much as the model.\n- **Softmax score:** the scores over the candidates, summing to 1; relative to the candidate list, '
        'not calibrated probabilities, and the model never abstains.\n- **Cosine similarity / embedding:** the angle between two vectors the encoders produce; '
        'the basis of every score here.\n- **Accuracy, macro F1:** the share of slices whose top prompt is right; the F1 averaged over the eleven organs so each '
        'counts equally.\n- **Text-to-image mAP:** each organ prompt ranks all test slices; the average precision of its own organ, averaged over organs — the '
        'retrieval view.\n- **Pre-registration / exploratory / fresh test / exposed development test:** settings written down before any run '
        '(`docs/pubmedclip-preregistration.md`); a run with any other setting, which compares on validation only; 176 test slices scored only under the '
        'pre-registered settings; the 176 test slices earlier runs scored, kept for continuity and never used for a decision.\n- **Majority floor / intensity neighbour:** two non-neural baselines — always answer the most frequent label; answer with the label of '
        'the training slice whose 3 × 3 mean-intensity grid is closest.\n- **Scan-disjoint split:** no CT scan contributes slices to two roles, so neighbouring '
        'slices cannot leak between training and test.\n- **Frozen / adapted / pinned base:** the packaged model; the model after Section 7; the verified '
        'packaged weights every adaptation starts from (`restore_base`).\n- **Pickle audit / safetensors:** the upstream checkpoint is a pickle that could run '
        'code when loaded; it is statically checked and converted once to safetensors, a format that stores only tensors.\n- **Isolated environment:** the '
        'separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete these in your own '
        "words; the quoted values, and the run each comes from, are in the **Check your reasoning** answers above.\n\n- The grid selected ___ on validation (C0 ___, C1 ___, C2 ___).\n"
        "- On the 176 fresh-test slices the frozen model scored accuracy ___ against the intensity neighbour's ___ and the majority floor's ___ (`test_look`: ___).\n"
        "- Fine-tuning moved fresh-test text-to-image mAP from ___ to ___ (verdict: "
        '___); the organs that stayed hardest were ___.\n- The number I would not trust on its own is ___, because ___.\n- Before using this on my own images I '
        'would split by ___, compare against ___, and remember that none of it is clinical evidence.\n\n'
        "## References\n\n"
        "[1] S. Eslami, C. Meinel, and G. de Melo, \u201cPubMedCLIP: How much does CLIP benefit visual question answering in the medical domain?,\u201d in *Findings of the Association for Computational Linguistics: EACL 2023*, 2023. Preprint: \u201cDoes CLIP benefit visual question answering in the medical domain as much as it does in the general domain?,\u201d arXiv:2112.13906, 2021. [Online]. Available: https://arxiv.org/abs/2112.13906\n\n"
        "[2] A. Radford *et al.*, \u201cLearning transferable visual models from natural language supervision,\u201d in *Proc. 38th Int. Conf. Machine Learning (ICML)*, 2021. [Online]. Available: https://arxiv.org/abs/2103.00020\n\n"
        "[3] O. Pelka, S. Koitka, J. R\u00fcckert, F. Nensa, and C. M. Friedrich, \u201cRadiology Objects in COntext (ROCO): A multimodal image dataset,\u201d in *Intravascular Imaging and Computer Assisted Stenting and Large-Scale Annotation of Biomedical Data and Expert Label Synthesis (LABELS 2018)*, LNCS 11043, Springer, 2018, pp. 180\u2013189.\n\n"
        "[4] J. Yang *et al.*, \u201cMedMNIST v2 \u2014 A large-scale lightweight benchmark for 2D and 3D biomedical image classification,\u201d *Scientific Data*, vol. 10, art. 41, 2023, doi: 10.1038/s41597-022-01721-8. Data: https://zenodo.org/records/10519652\n\n"
        "[5] P. Bilic *et al.*, \u201cThe Liver Tumor Segmentation Benchmark (LiTS),\u201d *Medical Image Analysis*, vol. 84, art. 102680, 2023, doi: 10.1016/j.media.2022.102680.\n\n"
        "[6] F. Giammarino, \u201cpubmed-clip-vit-base-patch32,\u201d Hugging Face model card, revision `{MODEL_REVISION}`. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        "[7] S. Eslami, \u201cPubMedCLIP,\u201d GitHub repository. [Online]. Available: https://github.com/sarahESL/PubMedCLIP\n\n"
        "[8] K. Valcorza, \u201cpubmedclip-biomedical-pipeline: README, model card and sample dataset card,\u201d GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline (model card: https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/MODEL_CARD.md)\n\n"
        "[9] K. Valcorza, \u201cPre-registration: PubMedCLIP tutorial configuration grid and fresh test,\u201d `docs/pubmedclip-preregistration.md`, 2026-10-17.\n\n"
        "[10] DIMER, \u201cDIMER Notebook Specification 2.5\u201d and \u201cModel Card Specification 1.1,\u201d `integrations/dimer/fleet-specs/`, ml-worker repository, revision `bfe4150`."
    ),
}
