from __future__ import annotations

MODEL_ID = "flaviagiammarino/pubmed-clip-vit-base-patch32"
MODEL_REVISION = "26c0c67f6da303ad2a38909130bd35744ea93517"
MODEL_LICENSE = "MIT"

# The Hub revision hosts the PubMedCLIP ViT-B/32 weights only as framework pickles / archives:
# `pytorch_model.bin` (a torch zip archive whose pickle names the fleet's four state-dict globals),
# `tf_model.h5` and `flax_model.msgpack`. The pipeline stages the PyTorch source, audits its pickle
# statically, unpickles it once through torch's weights-only loader and writes the safetensors file
# below, which is the only file the model is ever loaded from (asset spec §11; the `.h5` and the
# `.msgpack` are never fetched — their digests are recorded here for the card).
SOURCE_FILENAME = "pytorch_model.bin"
SOURCE_SHA256 = "4daa7650d2b47e55c37b5ca7dcabe826fd82407c26028a574bcc422f35a94aa4"
SOURCE_SIZE_BYTES = 605_222_477
PICKLE_AUDIT_SHA256 = "5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10"
CKPT_ALLOWED_GLOBALS = frozenset(
    {
        "collections.OrderedDict",
        "torch._utils._rebuild_tensor_v2",
        "torch.FloatStorage",
        "torch.LongStorage",
    }
)
SOURCE_STATE_TENSORS = 400  # 398 parameters + the two non-persistent position_ids buffers
TF_H5_SHA256 = "fe5013d5a012ced66ce5c5536177a3467ce0319b00e15b373d0f084173e89f44"
TF_H5_SIZE_BYTES = 605_559_520
FLAX_MSGPACK_SHA256 = "3b689834868221167494cb69f88b798850c37bbf9c79ddfbe4660aa302cf4c36"
FLAX_MSGPACK_SIZE_BYTES = 605_123_003

# The converted serving file (deterministic; `model.convert_source`).
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "81de21a0f1b6a3faaf1ea9e8fed4d570c671808ad72c948b4dac173eabcf676e"
MODEL_SIZE_BYTES = 605_156_676
MODEL_STATE_TENSORS = 398

DEFAULT_MODEL_KEY = "pubmed-clip-vit-base-patch32"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)

ALLOWED_CHECKPOINT_FILES = (
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    SOURCE_FILENAME,
)

DEFAULT_PROMPT_TEMPLATE = "A medical image of {label}."
TEXT_MAX_LENGTH = 77
