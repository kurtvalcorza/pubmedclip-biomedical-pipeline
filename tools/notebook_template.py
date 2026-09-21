"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

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
converted once to safetensors and loaded, 660 axial CT slices of eleven organs from MedMNIST+ OrganAMNIST
are extracted from the digest-pinned archive with per-slice digests, validated and kept in the dataset's own
splits, the drawn synthetic shapes are classified through the inference contract, the frozen model's
zero-shot accuracy / macro F1 / text-to-image mAP over the held-out slices is measured beside two
non-neural baselines, a bounded fine-tuning of the vision tower's last blocks runs in the kernel with the
cross-entropy over the class prompts, the held-out split is scored again per organ, the adapted model
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
    "pipeline_class": "PubMedClipPipeline",
    "weights_key": "pubmed-clip-vit-base-patch32",
    "modules": ["config.py", "model.py", "metrics.py", "samples.py", "pipeline.py", "provenance.py"],
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
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `flaviagiammarino/pubmed-clip-vit-base-patch32` snapshot, audits the 605 MB `pytorch_model.bin` statically and "
        "unpickles it exactly once through torch's weights-only loader into `model.safetensors` (the only file the model is ever "
        "loaded from), fetches the 1.8 GB MedMNIST+ `organamnist_224.npz` archive from Zenodo (verified by byte size and SHA-256, "
        "read with `numpy.load(allow_pickle=False)`), keeps exactly the 660 pinned slices (each verified again) in the dataset's own "
        "396 / 88 / 176 training, validation and test roles, classifies three drawn shapes through the inference contract with an "
        "input manifest and a rejection probe, measures the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP over "
        "the 176 test slices beside the majority-floor and intensity-nearest-neighbour baselines, runs a bounded fine-tuning of the "
        "vision tower's last two blocks, post-layernorm and visual projection with the cross-entropy over the eleven organ prompts and "
        "validation-mAP epoch selection, scores the held-out slices again per organ, re-scores the drawn shapes with the adapted model, "
        "exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify embedding parity. "
        "The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). On CPU the model time of the whole path was about a minute and a half on the build workstation "
        "after the downloads (expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically when present and "
        "finishes in a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `labels.csv` (columns `id`, `file`, `label`) beside the image files — at least eight images over at "
        "least two labels, the label text being what the prompt names. They pass through the same validation, seeded stratified "
        "split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the OrganAMNIST sample. "
        "The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path. **Do not upload identifiable patient data to a hosted "
        "runtime**; nothing here is a clinical tool."
    ),
    "intro": (
        "`flaviagiammarino/pubmed-clip-vit-base-patch32` is PubMedCLIP (Eslami, de Melo and Meinel, 2021): OpenAI's CLIP ViT-B/32 "
        "fine-tuned on the Radiology Objects in COntext (ROCO) image–caption pairs from open-access PubMed articles, released under "
        "the **MIT** licence — a ViT-B/32 image tower at 224×224 and a 12-layer text tower over CLIP's 49k-token BPE vocabulary, "
        "151,277,313 parameters. An image and a prompt are scored by the cosine of their projected embeddings scaled by the "
        "model's learned `logit_scale` and read through a **softmax over the candidate prompts**: the scores are relative to the "
        "candidates given, **not calibrated probabilities**, **prompt/label dependent**, and the model never abstains — the "
        "highest-scoring label is returned whatever the image shows. It is a research model: **not a clinical tool, not "
        "validated for diagnosis or triage**, and nothing in this notebook changes that.\n\n"
        "What this notebook adds to inference is **adaptation with labelled images**. The dataset is real, inside the model's "
        "modality and where the frozen model has room to improve: 660 axial abdominal CT slices of eleven organs from MedMNIST+ "
        "**OrganAMNIST** at 224 px (**CC BY 4.0**; Yang et al., 2023; 60 per organ, kept in the dataset's own train / validation / "
        "test roles, which the authors split by CT scan), pinned by split, index and the SHA-256 of every slice and extracted from "
        "the digest-verified archive at run time. Zero-shot prompts from the organ names barely separate these slices (the "
        "build record measured accuracy 0.18 frozen on the 176 test slices — the left / right pairs of kidney, lung and "
        "femur are indistinguishable to a caption model), so the honest question is narrow: does a bounded fine-tuning of the "
        "vision tower's last blocks on 396 slices move zero-shot accuracy and the retrieval view (**text-to-image mAP**) on a "
        "scan-disjoint test split, per organ, against two **non-neural baselines** (the **majority floor** and an **intensity "
        "nearest neighbour** — the carried `colour_neighbour_baseline`, whose three channels coincide on grey-scale CT)? Nothing "
        "here is a quality claim about your images: it is one seeded run on one sample.\n\n"
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
        "selection; evaluate on the held-out split with two prompt sets; re-score drawings from a different image family with "
        "the adapted model; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, semantic segmentation, OCR, caption generation, calibrated probabilities or universal thresholds, "
        "any diagnostic, triage or clinical decision use, fine-tuning of the text tower, the embeddings, the text projection or "
        "`logit_scale`, training on images that are not the pinned sample or your own uploads, evaluation on MedMNIST or any "
        "benchmark proper (only one seeded 660-slice sample is scored here), and any claim that eleven abdominal organs stand in "
        "for your classes. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the build record measured about 3.5 s to embed and score the 176 test slices and 80 s for the six epochs of fine-tuning (396 slices per epoch through the full vision tower, the last two blocks and the projection training), including the per-epoch validation scoring, so the whole default path is about a minute and a half of model time on the build workstation's CPU with the snapshot and slices already cached (a 2-vCPU hosted runtime will be several times slower); a hosted T4 finishes it in a few minutes. The pinned `torch==2.14.0` install, the 605 MB checkpoint and the 1.8 GB MedMNIST archive are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a softmax score and a cosine similarity are; what accuracy, macro F1 and average precision measure and why none is a human judgement; why a high score is not a correct label — and, for medical images, why none of this is diagnostic evidence.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides up to `MAX_IMAGE_SIDE` (4096) px and a label of at most 64 plain characters (the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label, or its display name, filled in). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records over 2..100 labels; the sample keeps the dataset's own roles and BYOD is split stratified per label after pixel-digest de-duplication so no image lands in two splits. BYOD accepts one zip of images plus a `labels.csv` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right or that an image is a CT slice — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data — and never identifiable patient data — to a hosted runtime unless you are authorized to process it there. The default path uploads nothing; the sample slices are de-identified research data published under CC BY 4.0.",
        "- **External access (data):** besides the model snapshot, the default path fetches one archive, `https://zenodo.org/records/10519652/files/organamnist_224.npz` (about 1.8 GB), pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; only the 660 pinned slices are kept, each verified against its own digest, and every record keeps its MedMNIST split and index. The data are CC BY 4.0 (MedMNIST, Yang et al. 2023; source volumes from the Liver Tumor Segmentation Benchmark); nothing is redistributed by the repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. OrganAMNIST slices and roles\n\n"
                "`fetch_corpus` returns the 660 pinned slices from the per-slice cache under `weights/organamnist/slices/` or "
                "extracts them from the digest-verified `organamnist_224.npz` (fetched from Zenodo when absent; read with "
                "`numpy.load(allow_pickle=False)`, one split array at a time) — every cached slice is re-hashed and every extracted "
                "slice refused on a digest mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their "
                "MedMNIST split, index and organ name. `build_sample_dataset` keeps the dataset's own roles (36 / 8 / 16 per organ from "
                "the official train / val / test splits → 396 / 88 / 176; the seed only orders the records). `validate_dataset` then "
                "checks every record against the contract, `check_split_disjoint` asserts no slice (by decoded-pixel digest) is "
                "shared, `source_overlap` reports which MedMNIST split feeds each role (an observation: scan identity is not "
                "published, the authors split by scan), and the training split's labels table is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 660 slices, the eleven organs with 36 / 8 / 16 each, three digests, and four refusal probes — a "
                "duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each "
                "rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n"
                "from dataclasses import asdict\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    display_names = {{}}\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t_fetch = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/organamnist')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    display_names = {{key: name for key, (name, _index) in ORGANS.items()}}\n"
                "    raw_rows = {{'slices': len(corpus), 'slice_bytes': sum(len(v) for v in corpus_files.values()), 'archive_bytes': CORPUS_ARCHIVE_BYTES, 'fetch_seconds': round(time.perf_counter() - t_fetch, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "classes = class_names(train_records)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'source_overlap': source_overlap(splits), 'classes': classes}})\n"
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
                "Section 6 measures on 176 slices."
            ),
            "code": (
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
                "## 6. Baselines and the frozen model's zero-shot score on the test slices\n\n"
                "Three systems frame the adaptation, each read three ways. The **majority floor** answers every slice with "
                "the most frequent training label (accuracy 1/11 on a balanced split, chance-level macro F1). The **intensity "
                "nearest neighbour** (`colour_neighbour_baseline`) answers with the label of the training slice whose 3×3 mean-intensity "
                "grid is closest — a classifier that knows the image through nine numbers, and on axial CT a stronger one than it "
                "sounds, because organs sit at characteristic positions. The **frozen model** is scored by `pipe.evaluate`: one "
                "prompt per organ (`CT_PROMPT` with the organ's display name — *left kidney*, *right lung*, …), the model's own "
                "scaled cosine logits as the score grid, **accuracy** and **macro F1** of the top prompt, the per-organ recall, and "
                "**text-to-image mAP** — each prompt as a query ranking all 176 slices, the average precision of its own organ, "
                "averaged over the eleven — the retrieval view of the same scores and the smoother of the three on a small set. A "
                "second prompt set built from the raw MedMNIST label tokens (`kidney-left`) in the default template is scored too, to "
                "show how much the number is the prompt's. Expect the frozen model above the majority floor but **well below the "
                "intensity neighbour**: the build record measured accuracy 0.18 / macro F1 0.13 / mAP 0.28 "
                "frozen against 0.51 for the neighbour, and read the per-organ breakdown — the frozen model answers *heart* for most slices (heart recall 1.00 on an average precision of 0.25), never finds a kidney, a femur or the spleen, and only the right lung ranks well (AP 0.79)."
            ),
            "code": (
                "CT_PROMPT = 'An axial abdominal CT slice showing the {{label}}.'\n"
                "baseline_majority = majority_baseline(train_records, test_records, classes)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records, classes)\n"
                "METRICS = ('accuracy', 'macro_f1', 't2i_map')\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'intensity_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names, prompt_template=CT_PROMPT)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'prompt_template': frozen_test['prompt_template'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in frozen_test['per_class'].items()}}\n"
                "print({{'by_organ_frozen': frozen_fields}})\n"
                "plain_names = {{key: key for key in ORGANS}} if not USE_BYOD else {{}}\n"
                "frozen_plain = pipe.evaluate(test_records, classes=classes, class_names_map=plain_names)\n"
                "print({{'frozen_model_test_raw_label_prompts': {{k: round(frozen_plain[k], 3) for k in METRICS}}, 'prompt_template': frozen_plain['prompt_template']}})\n"
                "assert frozen_test['t2i_map'] > baseline_majority['t2i_map'] and frozen_test['accuracy'] > baseline_majority['accuracy']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the vision tower's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_VISION_LAYERS` blocks of the vision tower, its post-layernorm and "
                "the visual projection — two blocks by default, 14,570,496 of 151,277,313 parameters; the text tower, the "
                "embeddings, the text projection and `logit_scale` stay frozen. The eleven organ prompts are embedded once by the "
                "frozen text tower; every batch of slices is run through the vision tower, scored against the prompts with the "
                "model's own scaled cosine logits, and trained with the cross-entropy of the softmax over the eleven prompts — "
                "CLIP's image-to-text objective restricted to the class set. AdamW at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is "
                "scored on the 88 validation slices, and the epoch with the highest validation text-to-image mAP is kept — "
                "accuracy on 88 slices moves in steps of about 1 %, which is why the retrieval view selects and the 176-slice test "
                "split is what the numbers are read from.\n\n"
                "Watch the training loss fall from about 1.2 to below 0.1 within six epochs while the validation mAP "
                "climbs: 396 slices are few, but the organs are visually separable and the frozen model started far from the "
                "task, so the selector's job is to stop before the last blocks memorise the training slices. The build record kept "
                "epoch 4 of six (validation mAP 0.36 frozen → 0.76); the default is the "
                "configuration that gained on the held-out split."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_VISION_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_vision_layers=TRAINABLE_VISION_LAYERS, prompt_template=CT_PROMPT, class_names_map=display_names, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'classes': adapt_result['classes'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test slices were never used for training or epoch selection, no slice appears in two roles, and the authors "
                "split OrganAMNIST by CT scan, so no scan feeds two roles either. The adapted model is scored exactly as the frozen "
                "model was in Section 6 — the same eleven prompts — the four systems are put side by side on the three metrics, the "
                "per-organ breakdown is repeated, and the raw-label prompt set is scored again (the vision tower was adapted, not the "
                "prompts, so a gain that carries to a prompt set it never saw is the more general one). Read it in this order: "
                "**text-to-image mAP** first (the metric the epoch was selected on — the build record measured 0.28 → "
                "0.65), then accuracy and macro F1 (0.18 → 0.81 and 0.13 → 0.81), "
                "then the per-organ recall, where every organ but the heart gained recall (liver and pancreas to 1.00, both lungs to 0.94, the spleen 0.00 → 0.81) while the left / right pairs stayed the weakest — the right kidney at 0.38, the right femur's AP at 0.51 — and the heart lost the recall it had only as the default answer. The cell asserts the adapted mAP is above the frozen one. "
                "One hundred and seventy-six slices from one seeded run give **no dispersion estimate**; the deltas are sample-sanity "
                "evidence that the adaptation contract works, not a benchmark, and a gain on eleven abdominal organs says nothing "
                "about your classes — or about any clinical use — until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names, prompt_template=CT_PROMPT)\n"
                "adapted_val = pipe.evaluate(val_records, classes=classes, class_names_map=display_names, prompt_template=CT_PROMPT)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in adapted_test['per_class'].items()}}\n"
                "adapted_plain = pipe.evaluate(test_records, classes=classes, class_names_map=plain_names)\n"
                "comparison = {{metric: {{'majority': round(baseline_majority[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['raw_label_prompts'] = {{metric: {{'frozen': round(frozen_plain[metric], 3), 'adapted': round(adapted_plain[metric], 3)}} for metric in METRICS}}\n"
                "comparison['by_organ'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_recall': frozen_fields[c]['recall'], 'adapted_recall': adapted_fields[c]['recall'], 'frozen_ap': frozen_fields[c]['ap'], 'adapted_ap': adapted_fields[c]['ap']}} for c in classes}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'prompt_template': CT_PROMPT,\n"
                "    'baselines': {{'majority': baseline_majority, 'intensity_neighbour': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_test_raw_label_prompts': frozen_plain,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'test_metrics_raw_label_prompts': adapted_plain,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['t2i_map'] > frozen_test['t2i_map']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-score the drawn shapes, export the adapter and reload it\n\n"
                "The three shapes from Section 5 are classified again by the adapted model — drawings, a different image family "
                "from the CT slices it was tuned on, so this is a small look at what the adaptation did *outside* its corpus "
                "(the build record's rankings are in `docs/release-verification.md`; a changed ranking here is a finding to record, not a "
                "failure) — and reported with the per-grid `evaluation_report` (`sample-sanity`). Both score sets are written "
                "as JSON.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the vision tower's last two blocks, post-layernorm and "
                "visual projection, about 58 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digest of the base `model.safetensors`, the classes and prompt "
                "template, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `PubMedClipPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest "
                "and its exact tensor set **before** deserialising, refuses any tensor outside the vision tower and its projection, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical image embeddings on eight test slices (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_classifications = [pipe.zero_shot_classify(image_path, candidate_labels) for image_path in shape_images]\n"
                "adapted_retrievals = [pipe.retrieve(query, shape_images, top_k=len(shape_images)) for query in shape_labels]\n"
                "adapted_scene = evaluation_report({{'classifications': adapted_classifications, 'retrievals': adapted_retrievals, 'gallery_ids': [p.name for p in shape_images]}}, targets, sample_kind='synthetic')\n"
                "for image_path, before, after in zip(shape_images, classifications, adapted_classifications, strict=True):\n"
                "    print({{'image': image_path.name, 'frozen': [(s.label, round(s.score, 3)) for s in before[:2]], 'adapted': [(s.label, round(s.score, 3)) for s in after[:2]]}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_shapes.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'frozen': classification_rows, 'adapted': [{{'image': p.name, 'scores': [asdict(x) for x in ranking]}} for p, ranking in zip(shape_images, adapted_classifications, strict=True)]}}, handle, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PubMedClipPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.embed_image([r['image'] for r in test_records[:8]])\n"
                "after = reloaded.embed_image([r['image'] for r in test_records[:8]])\n"
                "parity = {{'max_abs_difference': float(np.abs(before - after).max()), 'identical_rows': int((np.abs(before - after).max(axis=1) < 1e-5).sum()), 'of': int(before.shape[0])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rows'] == parity['of']\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256, 'source_file': SOURCE_FILENAME, 'source_sha256': SOURCE_SHA256, 'pickle_audit_sha256': PICKLE_AUDIT_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'url': CORPUS_URL, 'archive_bytes': CORPUS_ARCHIVE_BYTES, 'archive_sha256': CORPUS_ARCHIVE_SHA256, 'pinned_slices': len(SAMPLE_RECORDS), 'organs': {{k: list(v) for k, v in ORGANS.items()}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'labels': shape_labels, 'candidate_labels': candidate_labels}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is a partial zero-shot classifier on eleven abdominal organs it was never told about — above the "
        "majority floor, but well below the intensity neighbour — and a bounded fine-tuning of the vision tower's last two blocks "
        "and projection on 396 slices moves the retrieval view clearly (text-to-image mAP 0.28 → 0.65 in the "
        "build record) and the top-1 accuracy by 111 slices of 176 (0.18 → 0.81), with a "
        "58 MB adapter that reloads to identical embeddings. That is the claim: the adaptation contract works end to end on "
        "a real labelled medical-image set, and the numbers it produces are read on three metrics, per organ, on two prompt sets, "
        "against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 176 slices from the dataset's own test scans, the validation split that picks the epoch is 88, the "
        "metrics are three reference-based scores (own numpy implementations; none a human judgement), accuracy moves in steps of "
        "one slice, and the build record's own epoch history shows the estimate's fragility: validation accuracy moved between "
        "0.91 and 0.96 from one epoch to the next. So a gain here says the contract works, not that the adapted "
        "model is better on your images, that its scores are calibrated, or that a softmax score is a probability of being right — "
        "it still returns a best label for every image, and it can be wrong confidently. Fine-tuning on a narrow set can also erode "
        "the model elsewhere; the drawn shapes re-scored in Section 9 are three images of evidence about that, not a measurement. "
        "**None of this is clinical evidence:** OrganAMNIST slices are 224-pixel crops with organ labels derived from bounding "
        "boxes, the model is a research artefact trained on figure captions, and no reading of these numbers supports a "
        "diagnostic, triage or screening use.\n\n"
        "Three things to carry to real data. **Baselines first:** the majority floor, the intensity neighbour and the frozen "
        "model's score on *your* labels are the numbers to read before any adapted one, per class and on the retrieval view. "
        "**Leakage:** keep every image in one split (the contract de-duplicates by decoded pixels) and split by patient, scan or "
        "session when your images come from few sources — the sample keeps MedMNIST's own scan-level split for exactly that "
        "reason. **Prompts:** the classifier is the prompt as much as the tower; a second prompt set is scored here so the "
        "difference is visible, and a label an annotator wrote is not a prompt the model understands.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, audit and convert its pickle once, fetch and digest-verify a real "
        "labelled medical-image set, validate the demonstrated dataset contract without leakage, execute the inference contract "
        "and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on a scan-disjoint split, and emit "
        "the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, zero-shot accuracy on any other population, scanner or window setting, calibration, or clinical or "
        "production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_VISION_LAYERS = 4` and compare the "
        "artifact size and the held-out mAP; raise `EPOCHS` and watch the validation mAP pick the epoch while the training "
        "loss keeps falling; change `LEARNING_RATE` to `1e-5` and read a smaller, steadier gain; or bring your own images "
        "through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/MODEL_CARD.md\n"
        "- Sample dataset card (synthetic shapes): https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code and checkpoints: https://github.com/sarahESL/PubMedCLIP\n"
        "- Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain? (Eslami, de Melo and Meinel, 2021): https://arxiv.org/abs/2112.13906\n"
        "- Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021): https://arxiv.org/abs/2103.00020\n"
        "- MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification (Yang et al., Scientific Data 2023): https://doi.org/10.1038/s41597-022-01721-8 — data: https://zenodo.org/records/10519652\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
