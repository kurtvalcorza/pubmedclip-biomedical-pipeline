---
license: MIT
model_card_spec: "1.1"
pipeline_tag: zero-shot-image-classification
task: "Others - Biomedical Zero-Shot Image Classification"
tags:
  - vision-language
  - zero-shot-image-classification
  - biomedical
  - radiology
  - embeddings
  - fine-tuning
base_model: flaviagiammarino/pubmed-clip-vit-base-patch32
date_published: "2023-06-13"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/flaviagiammarino/pubmed-clip-vit-base-patch32); the PubMedCLIP paper and original checkpoints date from 2021-12 (arXiv:2112.13906)"
---

# PubMedCLIP ViT-B/32 — Biomedical Vision-Language Encoder (Zero-Shot Classification, Embeddings, Retrieval & Bounded Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-flaviagiammarino%2Fpubmed--clip--vit--base--patch32-ffcc4d?style=flat)](https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-sarahESL%2FPubMedCLIP-181717?style=flat&logo=github&logoColor=white)](https://github.com/sarahESL/PubMedCLIP)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2112.13906-b31b1b.svg)](https://arxiv.org/abs/2112.13906)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only — not a medical device, not a clinical tool.** Model weights are redistributed unmodified in content (converted once from the hosted pickle to safetensors) under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, diagnostic, triage or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, audit and convert its pickle once, fetch and validate a digest-pinned labelled medical-image set, measure the frozen model against two non-neural baselines, run a bounded fine-tuning, evaluate on a scan-disjoint split, and export and reload the adapter:

- **E2E Fine-tuning Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/tutorials/pubmedclip_biomedical_colab.ipynb) [`pubmedclip_biomedical_colab.ipynb`](https://github.com/kurtvalcorza/pubmedclip-biomedical-pipeline/blob/main/tutorials/pubmedclip_biomedical_colab.ipynb) \
  *Zero-shot classification, embeddings, similarity and retrieval with the pinned `flaviagiammarino/pubmed-clip-vit-base-patch32` weights, then bounded supervised fine-tuning of the vision tower's last blocks on 660 MedMNIST+ OrganAMNIST CT slices of eleven abdominal organs: the frozen model's accuracy, macro F1 and text-to-image mAP beside the majority-floor and intensity-nearest-neighbour baselines, the cross-entropy over the organ prompts with validation-mAP epoch selection, held-out evaluation per organ on two prompt sets, the drawn shapes re-scored, and a safetensors adapter that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. Its clean-runtime execution record and the promotion requirements are in [release verification](docs/release-verification.md).

---

#### Description

PubMedCLIP ViT-B/32 is an open-weight biomedical vision-language dual encoder: OpenAI's CLIP ViT-B/32 fine-tuned by Eslami, de Melo and Meinel (2021) on the Radiology Objects in COntext (ROCO) image–caption pairs from open-access PubMed articles, released under the MIT licence and hosted at `flaviagiammarino/pubmed-clip-vit-base-patch32` (pinned revision `26c0c67f6da303ad2a38909130bd35744ea93517`). This repository packages it as a verified DIMER inference pipeline — the fleet's SigLIP code shape with CLIP's softmax scoring — for research and education. Built with separate vision and text Transformer towers (151,277,313 parameters; a 12-layer ViT-B/32 image tower at 224 × 224 and a 12-layer text tower over CLIP's 49,408-token BPE vocabulary), it maps images and text into a shared 512-dimensional space. Zero-shot classification is a softmax over the scaled cosine between an image and prompt-expanded label strings; retrieval and similarity are cosine distances between L2-normalized embeddings. The Hub revision hosts the weights only as framework pickles (`pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack`); this repository contributes a hardened supply chain — the PyTorch source digest-pinned and **statically audited** (`pickletools`; the fleet's four state-dict globals and nothing else), **unpickled exactly once** through `torch.load(weights_only=True)` and converted deterministically to `model.safetensors` (605,156,676 bytes, SHA-256 `81de21a0…`), the only file the model is ever loaded from; local-only snapshot execution (`trust_remote_code=False`), text lowercasing, strict rejection of remote HTTP(S) image fetches, and full provenance tracking — and a bounded adaptation contract: `evaluate` scores a labelled image set with the model's own `logits_per_image` (accuracy, macro F1, per-class breakdown, text-to-image mAP), `adapt` fine-tunes only the last blocks of the vision tower, its post-layernorm and the visual projection (14,570,496 of 151,277,313 parameters by default) with the cross-entropy over the class prompts against the frozen text tower's prompt features, and `save_artifact` / `from_artifact` export the trained tensors as a digest-manifested safetensors adapter that reloads onto a freshly verified base. The tutorial demonstrates the contract on 660 axial abdominal CT slices of eleven organs (MedMNIST+ OrganAMNIST, CC BY 4.0), where the frozen model's zero-shot prompts barely separate the organs (accuracy 0.18 against an intensity-nearest-neighbour baseline of 0.51) and the bounded adaptation carries it past both baselines (0.81).

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline. **No clinical use is intended.**

###### Primary Intended Uses

The primary intended uses of this pipeline comprise seven technical vision-language capabilities, all for research, teaching and evaluation:
1. Zero-shot image classification (`PubMedClipPipeline.zero_shot_classify`): Ranking arbitrary candidate textual labels against an input image by a softmax over the scaled cosine scores, without task-specific training.
2. Image feature extraction (`PubMedClipPipeline.embed_image`): Generating dense, unit-norm 512-dimensional visual vectors for indexing, clustering, and vector search over medical figures and images.
3. Text feature extraction (`PubMedClipPipeline.embed_text`): Generating dense, unit-norm 512-dimensional vectors from natural-language descriptions up to 77 tokens.
4. Multimodal cosine similarity (`PubMedClipPipeline.similarity`): Computing pairwise similarity matrices across image and text batches.
5. In-memory semantic image retrieval (`PubMedClipPipeline.retrieve`): Finding top-k matching images from a local candidate corpus for a given text query.
6. Labelled-set evaluation (`PubMedClipPipeline.evaluate`): Scoring `{id, image, label}` records with one prompt per class and reporting accuracy, macro F1, per-class recall / precision / F1 / average precision and text-to-image mAP beside the predictions.
7. Bounded supervised fine-tuning (`PubMedClipPipeline.adapt`, `save_artifact`, `from_artifact`): Adapting the vision tower's last blocks and projection to a small labelled image set with validation-based epoch selection, exporting the adapter, and reloading it with verified parity.
Target application domains include modality and anatomy tagging of medical figures for research catalogues, teaching material about vision-language models in the medical domain, dataset curation and semantic retrieval of open-access biomedical images within the DIMER platform, and reproducible experiments on domain adaptation — none of them clinical decision-making.

###### Primary Intended Users

Primary intended users are machine learning researchers, medical-imaging informatics researchers, data scientists, and educators building or studying multimodal indexing, image classification, or retrieval over biomedical images. Users are expected to understand dual-encoder vision-language principles — that CLIP's per-image scores are a softmax over the candidates supplied rather than calibrated probabilities that a label is right, that the prompt is part of the classifier, and that the model never abstains — and, for the adaptation contract, why a fine-tuning gain on one sample is evidence that the contract works rather than a benchmark, why a split must be image-disjoint and patient- or scan-disjoint, and why the two non-neural baselines are read before the adapted number. Users must also be familiar with vector-space geometry (cosine similarity over L2-normalized embeddings), and must have the domain and regulatory competence to know that nothing here is validated for patient care.

###### Out-of-scope use cases

1. **Capability boundaries:** This model is an encoder-only vision-language dual tower. It cannot generate reports or captions, explain images, read text in figures, or output bounding boxes or masks. It must not be marketed or deployed as a detector, a segmenter, a report generator or a visual assistant.
2. **Input boundaries:** Accepts local image paths, raw image bytes, and PIL Image instances. Resolution is fixed to 224 × 224 pixels through the upstream processor (short-side resize and centre crop); high-resolution radiographs, whole-slide images or images with drastic aspect ratios lose detail. Texts exceeding 77 tokens are truncated. Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Adaptation boundaries:** `adapt` trains the vision tower's last `trainable_vision_layers` blocks, post-layernorm and visual projection only; the text tower, the embeddings, the text projection and `logit_scale` stay frozen, so a label the prompt does not describe cannot be learned through the text side, and fine-tuning on a narrow set can erode the model outside that set (the tutorial re-scores three drawn shapes as a small look at this, not a measurement). Datasets are validated structurally, never semantically: a mislabelled set is fine-tuned on without complaint. Mixed prompt sets, calibration and per-class thresholds are not provided.
4. **Decision boundaries:** Any diagnostic, triage, screening, prognostic or treatment-related use, autonomous or human-supervised, is out of scope; so is any unreviewed deployment in legal, forensic or punitive workflows. The pipeline is a research artefact.

---

#### Factors

This section describes factors influencing model representation and behavior, including demographic categories, capturing instruments, and operational runtime environments.

###### Groups

PubMedCLIP was fine-tuned on ROCO, roughly 80,000 radiology figures with their captions from open-access PubMed Central articles, on top of CLIP's web-scale pretraining. Neither corpus was demographically balanced or audited for parity: the published cases, the anatomy shown, the diseases named in captions and the imaging equipment reflect what open-access authors chose to publish. Representation quality and zero-shot label alignment can therefore differ systematically across age, sex, body habitus, ancestry, disease prevalence and region, and the captions carry the vocabulary and case mix of the medical literature rather than of any patient population. The pipeline introduces no demographic filters; anyone applying the model to images of people bears the responsibility for independent fairness and false-positive-disparity audits on their target data, and no such audit exists for the tutorial sample (MedMNIST publishes no demographics for OrganAMNIST).

###### Instrumentation

ROCO figures come from journal illustrations — radiographs, CT, MRI, ultrasound, fluoroscopy, PET and other modalities — as authors reproduced them: cropped, annotated, window-levelled, compressed and often at low resolution. The tutorial sample (OrganAMNIST) is 224-pixel axial CT crops rendered with an abdominal Hounsfield window from the LiTS volumes. Key instrumentation factors are scanner and protocol, slice thickness, window and level settings, reconstruction kernel, compression, and — in the model's training figures — the arrows, labels and colour overlays authors add. Because the processor resizes and centre-crops to 224 × 224, fine structures and image borders can be lost; grey-scale inputs are replicated to three channels. The pipeline validates image decoding and enforces RGB, but cannot detect scanner miscalibration, wrong window settings or embedded annotations.

###### Environment

1. **Operating environment:** Designed to run on Python 3.12 with `torch==2.14.0` and `transformers==4.57.6` (the pinned reference environment). Supported hardware includes x86_64 CPUs and NVIDIA GPUs supporting CUDA 12.x. A single inference instance requires approximately 0.6 GB of memory for model weights and minimal RAM for batch activations; the one-time conversion needs the 605 MB source and the state dict in memory. Float32 precision is default and fully qualified on CPU; half-precision formats require compatible GPU accelerators.
2. **Data environment:** Assumes medical figures or images resembling what appears in the biomedical literature — radiographs, CT and MRI slices, ultrasound frames, illustrations. Behaviour degrades on natural photographs, on modalities absent from ROCO, on raw DICOM intensities without windowing, and on images whose relevant finding occupies a few pixels of a 224-pixel crop.

---

#### Metrics

This section details performance metrics, decision thresholds, and uncertainty management applied across pipeline operations.

###### Performance Measures

The pipeline reports softmax classification scores (`scores = softmax(logits_per_image)`) bounded in `[0.0, 1.0]` and summing to one over the candidate set for zero-shot classification, and dot-product cosine similarity bounded in `[-1.0, 1.0]` for embedding retrieval. `evaluate` scores a labelled set with the unnormalised `logits_per_image`: accuracy and macro F1 of the top prompt, per-class recall / precision / F1 / average precision, and text-to-image mAP (each class prompt as a query ranking every image); `majority_baseline` and `colour_neighbour_baseline` (a 3 × 3 mean-intensity nearest neighbour on grey-scale CT) are scored on the same images. Upstream, PubMedCLIP was evaluated as a pre-trained encoder inside medical VQA systems (accuracy on VQA-RAD and SLAKE), not as a zero-shot classifier; no upstream zero-shot number exists for the tutorial task. Softmax scores are ranking scores over the supplied candidates; the public `evaluation_report` helper packages the tutorial's per-grid `top1_accuracy` (against a fixed-class baseline) and `recall_at_1` on the drawn shapes as `sample-sanity` evidence, never as a measurement.

###### Decision thresholds

The pipeline deliberately applies **no default binary classification threshold** and enforces **no argmax decision rule**. Because CLIP logits depend on prompt phrasing, candidate label cardinality and image domain, a universal threshold (such as 0.5 on a softmax score) is mathematically ungrounded and operationally misleading — a softmax score rises when a weak candidate is added and falls when a strong one is. The API emits raw scores and ranks labels comparatively. Downstream operators own threshold selection, which must be calibrated empirically against task-specific validation sets by balancing the asymmetric operational costs of false positives against false negatives; for anything touching patient care, that is a regulated validation, not a notebook cell.

###### Approaches to uncertainty and variability

Inference across all pipeline methods is strictly deterministic on CPU: no random sampling, temperature perturbation, or stochastic dropout is active during inference (`model.eval()`). Variations across runs can arise solely from floating-point kernel differences across distinct hardware architectures or non-deterministic GPU BLAS routines. Adaptation is seeded (`seed=0`: shuffling order) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one 660-slice sample in the dataset's own roles, with an 88-slice validation split that moves accuracy in steps of about 1 % — the build record's epoch history moved validation accuracy between 0.91 and 0.96 from one epoch to the next, which is the size of the uncertainty a reader should attach to any single number here. Output softmax scores are ranking scores and do not represent calibrated posterior probabilities or confidence intervals. Deployments requiring rigorous uncertainty quantification must employ post-hoc calibration techniques, such as Platt scaling, isotonic regression, or conformal prediction frameworks evaluated on domain-specific holdout splits.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The model weights were produced by fine-tuning OpenAI's CLIP (web-scale image–text pairs, not publicly enumerated) on ROCO, a corpus of radiology figures and captions extracted from open-access PubMed Central articles; the figures are de-identified journal illustrations, but neither corpus provides an instance-level manifest, so exposure to identifiable or sensitive imagery cannot be ruled out. This repository distributes only open-source Python code, tests, configuration manifests and the checkpoint's small configuration and tokenizer files; no weight blob and no image is distributed through git. The tutorial's adaptation corpus is 660 axial abdominal CT slices from MedMNIST+ OrganAMNIST at 224 px (bladder, left and right femur, heart, left and right kidney, liver, left and right lung, pancreas, spleen; 60 per organ in the dataset's own train / validation / test roles), published by Yang et al. under CC BY 4.0 from the de-identified Liver Tumor Segmentation Benchmark volumes and fetched from the authors' Zenodo record at run time, verified by archive and per-slice digests; nothing is redistributed, and every record keeps its MedMNIST split and index. Operators supplying inference images and text prompts must verify that their input data complies with data-protection and health-information law (e.g., GDPR, HIPAA) and does not contain identifiable patient information they are not authorized to process.

###### Human Life

PubMedCLIP is a research encoder and is **not** certified, tested, or approved for any medical or life-critical application. It must never be used to diagnose, screen, triage, monitor or treat anyone, to prioritise care, to read images in a clinical workflow, or as an input to any device or software that does so; the tutorial's organ labels are a machine-learning demonstration, not a radiology reading. Any secondary medical use would demand regulatory validation, clinical evaluation, and continuous human oversight that this repository neither provides nor claims.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to immutable commit `26c0c67f6da303ad2a38909130bd35744ea93517`; the PyTorch source verified by exact byte size (`605,222,477`) and SHA-256 (`4daa7650…`) and the converted `model.safetensors` by exact byte size (`605,156,676`) and SHA-256 (`81de21a0…`) prior to instantiation.
2. **Pickle audit and one-time conversion:** The source pickle's globals are listed statically with `pickletools` (no execution) and must equal the fleet's four state-dict globals (audit digest `5b9f0ba0…`); the file is unpickled exactly once through `torch.load(weights_only=True)`, loaded strictly into a `CLIPModel` built from the snapshot's `config.json`, and written as safetensors; transformers never sees the pickle (`use_safetensors=True`), and every other unsafe serialized format in the snapshot directory is refused.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes, or PIL images. The public `validate_inputs` helper applies exactly these input checks and returns an input manifest of the schema, ceilings, per-image observations and verdict before the model runs.
4. **Preprocessing fidelity:** Lowercases model-bound text before tokenization (CLIP's tokenizer does so itself; the shim removes the dependence on the tokenizer class), pads to CLIP's 77-token context, and preserves caller label casing in returned outputs.
5. **Deterministic normalization:** Enforces explicit L2 normalization on image and text feature embeddings before cosine scoring.
6. **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, trains only the named vision-tower tensors with every other parameter's `requires_grad` false, restores the frozen weights on any exception, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digest, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the vision tower and its projection, and overlays onto a freshly loaded base.
7. **Scope statements everywhere the model is reachable:** the notebook, the README and this card state that the model is not a clinical tool and that no output is diagnostic evidence.

###### Risks and harms

Key identified risks include:
1. **Clinical misuse and automation bias:** A user may read a high softmax score as a radiological finding; the scores are relative rankings of prompts by a caption-trained encoder, and acting on them for care is the primary harm this card warns against.
2. **Domain and publication bias:** ROCO figures are the cases authors chose to publish — unusual, annotated, cropped — so the model's notion of "normal" and of any modality is skewed toward the literature, not toward routine imaging; laterality (left / right) is essentially invisible to a caption model, which the tutorial's kidney, lung and femur pairs make explicit.
3. **Adversarial susceptibility:** Like all dual-encoder CLIP-style models, PubMedCLIP is susceptible to typographic attacks (text or annotations on an image overriding its content) and subtle adversarial perturbations.
4. **Search and retrieval bias:** Semantic search over large uncurated image collections can surface misleading associations for sensitive queries (disease names, body parts) and can re-identify cases when combined with other data.
5. **Adaptation risks:** fine-tuning on a small labelled set learns that set's labelling, including its errors, its scanner and its window settings; a gain measured on the sample's own test scans can overstate transfer to other scanners or protocols; and a narrow adaptation can erode zero-shot behaviour on classes and image families it never saw.

###### Use cases

The following use cases are strictly prohibited by policy and developer intent:
1. Any diagnostic, screening, triage, prognostic or treatment-related use, whether autonomous or as decision support, and any integration into a medical device or clinical information system.
2. Re-identification of patients, or profiling of individuals or groups from medical images.
3. Automated demographic profiling or discriminatory filtering in insurance, employment, benefits access or any other consequential decision.
4. Deceptive systems, such as fabricated medical imagery, misleading medical claims or manipulated retrieval results presented as evidence.
5. Any application that violates the upstream MIT license terms, the CC BY 4.0 terms of the tutorial data, or applicable health-data and privacy regulations.

---

## Technical Specifications and Architecture

### Architecture Overview

PubMedCLIP keeps CLIP's two parallel Transformer backbones:
- **Vision Tower:** Vision Transformer (ViT-B/32) with patch size 32 × 32 pixels and input resolution 224 × 224 pixels (50 tokens), 12 layers, width 768, post-layernorm and a linear visual projection to 512 dimensions.
- **Text Tower:** 12-layer Transformer text encoder, width 512, over CLIP's 49,408-token BPE vocabulary, context 77 tokens, with a linear text projection to 512 dimensions.
- **Embedding Projection:** Both modalities project into a shared 512-dimensional latent space.
- **Loss Formulation:** Contrastive InfoNCE over image–text pairs — the softmax over the batch of the scaled cosine similarities, symmetrised over images and texts:
  $$\mathcal{L} = \tfrac{1}{2}\left(\mathrm{CE}_{\text{image}\to\text{text}} + \mathrm{CE}_{\text{text}\to\text{image}}\right)\quad\text{with logits } s\, (v_i \cdot t_j)$$
  where $s = \exp(\texttt{logit\_scale})$. The pipeline's adaptation uses the image-to-text term over the class prompts only.

### Checkpoint Invariants and Loading Controls

The snapshot loader (`pubmedclip_pipeline.model.load_components`) enforces strict supply-chain controls:
1. Pinned Hugging Face repository: `flaviagiammarino/pubmed-clip-vit-base-patch32`
2. Pinned commit revision: `26c0c67f6da303ad2a38909130bd35744ea93517`
3. Hosted source: `pytorch_model.bin` — `605,222,477` bytes, SHA-256 `4daa7650d2b47e55c37b5ca7dcabe826fd82407c26028a574bcc422f35a94aa4`; a torch zip archive whose single pickle imports exactly `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `torch.LongStorage` (audit digest `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10`); 400 tensors, two of them the non-persistent `position_ids` buffers. Also hosted and never fetched: `tf_model.h5` (`605,559,520` bytes, SHA-256 `fe5013d5a012ced66ce5c5536177a3467ce0319b00e15b373d0f084173e89f44`) and `flax_model.msgpack` (`605,123,003` bytes, SHA-256 `3b689834868221167494cb69f88b798850c37bbf9c79ddfbe4660aa302cf4c36`) — the digests are recorded so a DIMER upload of the `.h5` can be matched to this revision, but the executed artifact is the converted file below.
4. Served weight file: `model.safetensors` — converted deterministically by `convert_source` (398 tensors); expected byte size `605,156,676`; expected SHA-256 `81de21a0f1b6a3faaf1ea9e8fed4d570c671808ad72c948b4dac173eabcf676e`
5. Upstream parameters: `151,277,313` F32 parameters (vision 87,849,216 + projection; text 63,428,096 + projection; `logit_scale`)
6. Execution policy: `trust_remote_code=False`, `use_safetensors=True`, `local_files_only=True`; the source pickle is tolerated in the snapshot directory only as the digest-verified conversion input and is never passed to transformers; with the converted file present it is not required (the DIMER-hosted shape)
7. Adapter artifact format: `org.valcorza.pubmed-clip-vit-base-patch32.adapter.v1` — `adapter.safetensors` (the trained tensors only; 35 tensors for the default two blocks, post-layernorm and visual projection, 14,570,496 parameters, about 58 MB) plus `manifest.json` naming the base id, revision and `model.safetensors` digest, the classes and prompt template, the tensor names, the file size and SHA-256, the training configuration and the epoch history
8. Tutorial corpus: 660 MedMNIST+ OrganAMNIST-224 slices (CC BY 4.0; eleven organs, 60 each, in the dataset's own roles 36 / 8 / 16 per organ → 396 / 88 / 176), pinned in `pubmedclip_pipeline/samples.py` by split, index and slice SHA-256 and extracted from `organamnist_224.npz` (`1,803,859,544` bytes, SHA-256 `a19bae53…`, MD5 `50747347e05c87dd3aaf92c49f9f3170` as published) fetched from Zenodo record 10519652
9. Tutorial prompts: `An axial abdominal CT slice showing the {label}.` with the organ display names (*left kidney*, *right lung*, …); the second prompt set is the raw MedMNIST label token in the default template `A medical image of {label}.`
10. Build record (CPU, 2026-09-20): the default tutorial path run through the package API on the build workstation's CPU (`torch 2.14.0`, `transformers 4.57.6`, Python 3.12, `CUDA_VISIBLE_DEVICES=-1`, snapshot converted and archive pre-staged) — the archive already staged (extraction of the 660 slices 14.7 s), model load 2.2 s, the frozen model scored in 3.5 s, six epochs of the default recipe in 79.7 s (validation mAP 0.355 → 0.673 → 0.758 → 0.708 → 0.761 → 0.758 → 0.689, epoch 4 kept; validation accuracy 0.307 → 0.602 → 0.807 → 0.943 → 0.943 → 0.955 → 0.909), the adapter 35 tensors / 58,286,104 bytes with reload parity 0.0 (max abs difference over eight embedded test slices); comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}, by_organ: {bladder: {n: 16, frozen_recall: 0.06, adapted_recall: 0.75, frozen_ap: 0.42, adapted_ap: 0.63}, femur-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.75, frozen_ap: 0.06, adapted_ap: 0.7}, femur-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.08, adapted_ap: 0.51}, heart: {n: 16, frozen_recall: 1.0, adapted_recall: 0.94, frozen_ap: 0.25, adapted_ap: 0.65}, kidney-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.62, frozen_ap: 0.08, adapted_ap: 0.62}, kidney-right: {n: 16, frozen_recall: 0.0, adapted_recall: 0.38, frozen_ap: 0.09, adapted_ap: 0.47}, liver: {n: 16, frozen_recall: 0.25, adapted_recall: 1.0, frozen_ap: 0.49, adapted_ap: 0.58}, lung-left: {n: 16, frozen_recall: 0.0, adapted_recall: 0.94, frozen_ap: 0.36, adapted_ap: 0.98}, lung-right: {n: 16, frozen_recall: 0.25, adapted_recall: 0.94, frozen_ap: 0.79, adapted_ap: 0.34}, pancreas: {n: 16, frozen_recall: 0.44, adapted_recall: 1.0, frozen_ap: 0.21, adapted_ap: 0.79}, spleen: {n: 16, frozen_recall: 0.0, adapted_recall: 0.81, frozen_ap: 0.29, adapted_ap: 0.92}}}; drawn shapes frozen {red_square.ppm: [[red square, 0.955], [abstract geometric shape, 0.041], [blue triangle, 0.004], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.771], [abstract geometric shape, 0.179], [red square, 0.046], [blue triangle, 0.004]], blue_triangle.ppm: [[blue triangle, 0.859], [abstract geometric shape, 0.081], [red square, 0.06], [green circle, 0.0]]} → adapted {red_square.ppm: [[red square, 0.958], [abstract geometric shape, 0.041], [blue triangle, 0.0], [green circle, 0.0]], green_circle.ppm: [[green circle, 0.91], [abstract geometric shape, 0.087], [red square, 0.003], [blue triangle, 0.0]], blue_triangle.ppm: [[blue triangle, 0.914], [abstract geometric shape, 0.077], [red square, 0.01], [green circle, 0.0]]}. Recorded in `docs/release-verification.md` as a pre-flight. Executed 2026-09-21: the **committed notebook blob** (`d125896` / `4b2fc4c4`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-pubmedclip-biomedical` v1, `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 14/14 ok (1 restart after install cell), 415.1 s, 682 files, 3051 MB fetched (Hub snapshot + Zenodo archive) and digest-verified inside the notebook; comparison {accuracy: {majority: 0.091, neighbour: 0.511, frozen: 0.182, adapted: 0.812}, macro_f1: {majority: 0.015, neighbour: 0.502, frozen: 0.131, adapted: 0.81}, t2i_map: {majority: 0.119, neighbour: 0.519, frozen: 0.284, adapted: 0.654}, delta_vs_frozen: {accuracy: 0.631, macro_f1: 0.679, t2i_map: 0.37}, raw_label_prompts: {accuracy: {frozen: 0.182, adapted: 0.818}, macro_f1: {frozen: 0.096, adapted: 0.811}, t2i_map: {frozen: 0.34, adapted: 0.815}}} (per-organ recall / AP in the archived result); reload parity {max_abs_difference: 0.0, identical_rows: 8, of: 8}. Not executed: any corpus other than the one 660-slice OrganAMNIST sample, repeated seeds (no dispersion), BYOD, and the adapted model on any images but that test split.

### Public Inference API

```python
from pubmedclip_pipeline import load_pipeline

pipe = load_pipeline(device="cpu")

# 1. Zero-shot classification (returns a softmax over the candidate labels)
scores = pipe.zero_shot_classify(
    image="figure.png",
    labels=["chest X-ray", "brain MRI", "abdominal CT scan"],
    prompt_template="A medical image of {label}."
)

# 2. Dense feature embeddings (L2-normalized)
img_emb = pipe.embed_image(["figure1.png", "figure2.png"])
txt_emb = pipe.embed_text(["chest X-ray", "abdominal CT scan"])

# 3. Cross-modal cosine similarity
sim_matrix = pipe.similarity(["figure1.png"], ["chest X-ray", "abdominal CT scan"])

# 4. In-memory semantic image retrieval
hits = pipe.retrieve(
    query="abdominal CT scan",
    images=["figure1.png", "figure2.png", "figure3.png"],
    top_k=2
)
```

### Public Adaptation API

```python
from pubmedclip_pipeline import (
    PubMedClipPipeline, build_sample_dataset, class_names, fetch_corpus, read_corpus,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)  # 396 / 88 / 176 slices, the dataset's own roles
classes = class_names(splits["train"])
pipe = PubMedClipPipeline.from_pretrained(weights_dir="weights/pubmed-clip-vit-base-patch32")  # audits + converts once
prompt = "An axial abdominal CT slice showing the {label}."

frozen = pipe.evaluate(splits["test"], classes=classes, prompt_template=prompt)      # accuracy, macro_f1, t2i_map, per_class, predictions
result = pipe.adapt(splits["train"], splits["validation"], epochs=6, lr=5e-5,
                    batch_size=16, trainable_vision_layers=2, prompt_template=prompt)
adapted = pipe.evaluate(splits["test"], classes=classes, prompt_template=prompt)
pipe.save_artifact("outputs/adapter")                       # adapter.safetensors + manifest.json
again = PubMedClipPipeline.from_artifact("outputs/adapter", weights_dir="weights/pubmed-clip-vit-base-patch32")
```

Dataset contract (`samples.py`): records `{id, image, label}` (`id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; a PIL image or a decodable file with sides up to `MAX_IMAGE_SIDE = 4096`; a label of at most 64 plain characters); `validate_dataset(records, *, min_records=8, max_records=20000)` (2..100 labels); `split_dataset(records, *, val_fraction=0.15, test_fraction=0.2, seed=0)` (stratified, pixel-digest de-duplicated; for BYOD data); `check_split_disjoint(splits)`; `source_overlap(splits)`; `load_byod_dataset(path)` (directory or zip with `labels.csv`: `id`, `file`, `label`); `write_dataset_csv(records, path)`; `fetch_archive(cache_dir=None)`, `fetch_corpus(cache_dir=None)`, `read_corpus(files)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`. Metrics (`metrics.py`): `classification_metrics(scores, gold, classes)`, `average_precision`, `majority_baseline(train, records, classes)`, `colour_signature(image)`, `colour_neighbour_baseline(train, records, classes)`. Conversion (`model.py`): `audit_pickle(path)`, `convert_source(weights_dir)`.

### Upstream References and Citations

- **PubMedCLIP Paper:** Eslami, de Melo and Meinel, *"Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?"*, arXiv:2112.13906 (2021); code and original checkpoints at https://github.com/sarahESL/PubMedCLIP (MIT).
- **CLIP:** Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, ICML 2021, arXiv:2103.00020.
- **ROCO:** Pelka et al., *"Radiology Objects in COntext (ROCO): A Multimodal Image Dataset"*, MICCAI LABELS 2018.
- **Hosted checkpoint:** `flaviagiammarino/pubmed-clip-vit-base-patch32` (Hugging Face Hub; the ViT-B/32 variant converted to the transformers format by the Hub author).
- **Tutorial corpus:** Yang et al., *"MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification"*, Scientific Data 10, 41 (2023), https://doi.org/10.1038/s41597-022-01721-8 — MedMNIST+ data at https://zenodo.org/records/10519652 (CC BY 4.0); OrganAMNIST source volumes: Bilic et al., *The Liver Tumor Segmentation Benchmark (LiTS)*, with organ labels from Xu et al. (2019).
- **Fleet siblings sharing the code shape:** `siglip2-vision-language-pipeline`, `siglip-v1-zero-shot-pipeline`.
