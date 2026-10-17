"""Labelled-image dataset contract for adapting the zero-shot classifier: the pinned OrganAMNIST sample,
validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and inside the checkpoint's domain (radiology) while outside anything its
captions describe precisely: 660 axial abdominal CT slices from MedMNIST+ **OrganAMNIST** at 224 × 224 (Yang et
al., 2023; source volumes from the Liver Tumor Segmentation Benchmark, organ labels from the bounding boxes of
Xu et al.; CC BY 4.0), 60 per organ over the 11 organ classes, drawn with a fixed seed from the dataset's own
train / validation / test splits (36 / 8 / 16 per organ) — MedMNIST splits by CT scan, so no scan contributes
slices to two roles — and pinned here by split, index and the SHA-256 of the raw 224 × 224 uint8 slice. The
archive `organamnist_224.npz` (1.8 GB) is fetched from the authors' Zenodo record at run time, verified by byte
size and SHA-256, and read with `numpy.load(allow_pickle=False)`; only the pinned slices are kept and each is
verified again. The repository redistributes none of the slices.

A record is ``{id, image, label}``: a PIL image (or a path to one) and the organ key of its gold label;
`ORGANS` maps the key to the display name the prompts are built from and to the MedMNIST class index.
"""
# ruff: noqa: E501  -- fleet dataset module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .config import MODEL_ID

MAX_IMAGE_SIDE = 4096  # pixels; larger images are rejected before any decode-to-tensor work

CORPUS_NAME = "MedMNIST+ OrganAMNIST, 224 px (axial abdominal CT slices, 11 organs)"
CORPUS_RELEASE = "MedMNIST v3.0, Zenodo record 10519652, organamnist_224.npz (2024-01-18)"
CORPUS_URL = "https://zenodo.org/records/10519652/files/organamnist_224.npz?download=1"
CORPUS_LICENSE = "CC BY 4.0 (MedMNIST; Yang et al., Scientific Data 2023) — source CT volumes: LiTS"
CORPUS_ARCHIVE_NAME = "organamnist_224.npz"
CORPUS_ARCHIVE_BYTES = 1_803_859_544
CORPUS_ARCHIVE_SHA256 = "a19bae532ac0cf979f7474aba7eb923dc9bbf67c1bcba3cb941dce51a59951e9"
CORPUS_ARCHIVE_MD5 = "50747347e05c87dd3aaf92c49f9f3170"  # as published on the Zenodo record and in medmnist/info.py
SLICE_SIDE = 224
DEFAULT_CACHE_DIR = Path("weights") / "organamnist"
# organ key -> (display name used in the prompts, MedMNIST class index)
ORGANS: dict[str, tuple[str, int]] = {
    "bladder": ("bladder", 0),
    "femur-left": ("left femur", 1),
    "femur-right": ("right femur", 2),
    "heart": ("heart", 3),
    "kidney-left": ("left kidney", 4),
    "kidney-right": ("right kidney", 5),
    "liver": ("liver", 6),
    "lung-left": ("left lung", 7),
    "lung-right": ("right lung", 8),
    "pancreas": ("pancreas", 9),
    "spleen": ("spleen", 10),
}
SPLIT_ROLE = {"train": "train", "val": "validation", "test": "test"}
# (id, organ key, MedMNIST split, index in that split's image array, sha256 of the raw 224 x 224 uint8 slice)
SAMPLE_RECORDS: tuple[tuple[str, str, str, int, str], ...] = (
    ("bladder-train-00", "bladder", "train", 1033, "2286ac7a2846a6be0302d7a85d6510ae7cd0d9f9be2856ca820495bf35a0ffdb"),
    ("bladder-train-01", "bladder", "train", 1506, "9deeec4377cef8abfc358705a6276e0f2f931743c8f7030f0c6b100d9e6081c4"),
    ("bladder-train-02", "bladder", "train", 3265, "ca320ac84b917342ccd6233a6c741324b15ce46c84eea7de8d5cc1110f53c9d0"),
    ("bladder-train-03", "bladder", "train", 3942, "1f220cf3fdf80a6e0189278969ff2c33bd3f587794f4ad9a1521de052ee5386e"),
    ("bladder-train-04", "bladder", "train", 4703, "0a524f9efd56e31483274dceda89473444db644e0c51ece9799e2f50dc0e5f9e"),
    ("bladder-train-05", "bladder", "train", 5618, "3bf5b0f8fe2bfbb3793458a2c66286576aba07e8b5700030361ad7e38af0593e"),
    ("bladder-train-06", "bladder", "train", 7839, "c59f9262a614a5f5a7b41742c55327847ecb45e4d6fd2d48b0a94822cc536f64"),
    ("bladder-train-07", "bladder", "train", 10040, "eb0d73d9ac5b75aad377cbf1ee211add4544e255b6d3be1c9ad345932479f426"),
    ("bladder-train-08", "bladder", "train", 11430, "891c8b6e17d24209a0216d0374e7d5d89e5c15b47a350c9d07f91a67f83d565b"),
    ("bladder-train-09", "bladder", "train", 11600, "311ef2828fc7aa9ba73decefe3b1cc9b72680049e20386205d6af973c0604321"),
    ("bladder-train-10", "bladder", "train", 11745, "4cad80ac6103d69e2e079d23de1f0041e87f9888d8be1962232b524d13258e32"),
    ("bladder-train-11", "bladder", "train", 12287, "8a8832b26b4dfc1f86942bb7a43c4c00f4d4e65ebd5d40281a306a68d0711158"),
    ("bladder-train-12", "bladder", "train", 12955, "8647fd206027e57a692a97a5c1f0138de070dc33623bd8fd60a82ffb76024f8c"),
    ("bladder-train-13", "bladder", "train", 13955, "1b19185405120c4a172268e0ea8e2f8e876efb219e6949780c0df50bd3171a19"),
    ("bladder-train-14", "bladder", "train", 15462, "8c14d41681557e01321df2b6e40698b366c489efe968d9de4c464762651d30d5"),
    ("bladder-train-15", "bladder", "train", 15668, "dfa65f8fa8b1c9676762c1a2ba944600261c933ada3c51361af00c887b9b172e"),
    ("bladder-train-16", "bladder", "train", 16169, "f0fbec94b7c36bae1624c151f6198049551fa2bdf259780426a3297bc9fd7d83"),
    ("bladder-train-17", "bladder", "train", 17364, "a32a41e753a93daf863198eb3c9649924010067815ddbe7676f5c074ee76c242"),
    ("bladder-train-18", "bladder", "train", 18694, "ca70d889d425e3d7aa25301e7d8f7ec8aac9382a4c3596035f13e66957d6f457"),
    ("bladder-train-19", "bladder", "train", 19436, "3fc78be4d1c78c6d02fa1955d1fe7c53d8081af9175026697ae2b83267018c21"),
    ("bladder-train-20", "bladder", "train", 19771, "038bb496dc5efbbd28786d6de2984083688954b11d68b388cac9c866e9861fbf"),
    ("bladder-train-21", "bladder", "train", 22346, "a0130be7d8f122f627da2c4e7839786a5cae36f0ab4dde3b3741362e379c2531"),
    ("bladder-train-22", "bladder", "train", 22489, "9a4e2587a6cb8abb7defe35f18a778ba029ce7b38ee984904931ac2c8e6b0f95"),
    ("bladder-train-23", "bladder", "train", 23728, "c44d331d04ae51bb3a13b95bfb0d7589ea935c2a6153a859ca3051deaa7b9c1e"),
    ("bladder-train-24", "bladder", "train", 23922, "59e3f25c8c747abca06568a3054997e1fc266e5952b0469a762d2ad2c14cb70b"),
    ("bladder-train-25", "bladder", "train", 26113, "fe0d935a55415868f24502f16e43dec8185206c7c50741f86cbb27132bf48424"),
    ("bladder-train-26", "bladder", "train", 26689, "7e71f2a24aed6b26828b899705c54d6d44c82f57900a85e861c8aa1483fef427"),
    ("bladder-train-27", "bladder", "train", 26942, "64022ba5cf74f72facd6e90bee5b2d2a36c26d22f7fae440c227612df97b26b4"),
    ("bladder-train-28", "bladder", "train", 27124, "7a3c543b92eb2c1ab2efc09fdd2cede42887d53399706648846e692144ee9fef"),
    ("bladder-train-29", "bladder", "train", 27822, "cb37623029ed705ce4a68107adf9255fbb3d02b5976a7b4a94360eebab4f2b23"),
    ("bladder-train-30", "bladder", "train", 28466, "81d824c9cb253f4dc51b1527f8e9f47ff63dfd60cda2f9cab34b80b97082dfce"),
    ("bladder-train-31", "bladder", "train", 30553, "b1e487b23d5e174743595cf95c4a944492fd78f2318bd80df49d16ca2c7e7e10"),
    ("bladder-train-32", "bladder", "train", 31445, "f71d4426e460be32c253f7e73c4ed4c65bc90c1b7c916a8c3d53d99478eb337a"),
    ("bladder-train-33", "bladder", "train", 32110, "a15f9d2e1b4ec9671e5f62ce4ebc34d20381bf805874960ce016fa6eec9e80bf"),
    ("bladder-train-34", "bladder", "train", 33094, "1d5ebf9860cad253ad9d5ff1844280727519747aeaa19cb127936047e91b7bcb"),
    ("bladder-train-35", "bladder", "train", 34084, "0b55250f24b1a6719f1e125930715dc6ad61606700d0ac71ce1e907ccbd76e60"),
    ("femur-left-train-00", "femur-left", "train", 1104, "14015c3a6b0c36f1f40a930c647462200ea0b92c1ccd504d2e0c642c68f89133"),
    ("femur-left-train-01", "femur-left", "train", 2245, "6af20c51a3667d673ec9c5217af3444be51e56cf3d5eb1a4af136551d47dae7f"),
    ("femur-left-train-02", "femur-left", "train", 3329, "2c1cbe502a33cacc12cdb264117c37050368438d9b0d1760e33bbff40c958fd4"),
    ("femur-left-train-03", "femur-left", "train", 3615, "2152afd48fe62d68106fdb994aef3f4e2de7087893f3218580d926f2bfe95205"),
    ("femur-left-train-04", "femur-left", "train", 7148, "5f959ccfafbc80c64b3fed12024816a421361be6e9e39a6febe63b9cfbfc218e"),
    ("femur-left-train-05", "femur-left", "train", 7631, "79c7883384fe47a7a642ce506618d653256f70c32f7afb331b0e64e696efc8f4"),
    ("femur-left-train-06", "femur-left", "train", 9037, "9f7d78dd289c49e6c11671d9361f9e690d5ecd6cea89b7bbd0de46b892dd96fc"),
    ("femur-left-train-07", "femur-left", "train", 9297, "ddbf1f2ebc40f14b81d7f37fc1fdcb36d3ec39f6f205db34cd22805780bf341d"),
    ("femur-left-train-08", "femur-left", "train", 11047, "5ba8918ca0463afe11fa83f86e777a0eaf333aefbe7ce41e2435e434734b892d"),
    ("femur-left-train-09", "femur-left", "train", 12840, "5ea1e7572555003bdac0f9888c2248b61543dfe3ca81a8293b7085761e6ba292"),
    ("femur-left-train-10", "femur-left", "train", 13085, "28bfa0925e719e08f2367b82b48a6cbd2deeb10fca48126aad190cbe414b7e98"),
    ("femur-left-train-11", "femur-left", "train", 13649, "ac730ccc4865ae2842cf050ea2da33b170c57c4704fc8254bd6e31221f56cbf5"),
    ("femur-left-train-12", "femur-left", "train", 13971, "6a8052a8d889582bec0102d6b09d6b5783a8073adf4dba2580a5dca9a0cf29b1"),
    ("femur-left-train-13", "femur-left", "train", 15846, "e7b4728b6f127743a985eba55464db1ec2c9715803cb47ca07342a5e8b5fed18"),
    ("femur-left-train-14", "femur-left", "train", 19165, "54c861f3b457b66b1fd640f14c3103dd64052e8b7768a6842cd10589422e439b"),
    ("femur-left-train-15", "femur-left", "train", 19270, "d48ba3492a6a77c750daf4c5f7e282462bbe54fd03aab29abdc52638d4d71dd3"),
    ("femur-left-train-16", "femur-left", "train", 19533, "a548fdd9a5b51f17b71e7d34ad62397e31cd2049dfd1826fe94ca2c42eedda87"),
    ("femur-left-train-17", "femur-left", "train", 19798, "978d24c3da67499e0782f7625ec34da68655c1e6b446111c43a287ddf5a30d74"),
    ("femur-left-train-18", "femur-left", "train", 21824, "412ccfd71ac3eb0b64df9b104bd4989ba5435221e17ca26a4d9ad7dfea5b332a"),
    ("femur-left-train-19", "femur-left", "train", 22417, "96fa4b19ed5182956b3ba9d9ca074cda5fd256f7dc9ee9cfbd24b86f4d9a53fb"),
    ("femur-left-train-20", "femur-left", "train", 22814, "bf1110c638609870c51aa6cec824240f6a574614823509b07ddc861b0805f25d"),
    ("femur-left-train-21", "femur-left", "train", 23245, "e5ea2364927e665793ba4ca23e69f89333322d7ed642b5f8686c4aba6870c269"),
    ("femur-left-train-22", "femur-left", "train", 23546, "a784dac8e240f929787b1049d33887e09fa486297bd60746b74382993c4c931c"),
    ("femur-left-train-23", "femur-left", "train", 24425, "a0801e6b67c83706e3093ee773dc1f36304af656c60890dab8f1dc82840b7101"),
    ("femur-left-train-24", "femur-left", "train", 24670, "9d5ea30a552dd41ccc5dc3b0a6f0f3f3620185f11cbe123944d55a16969ee6c2"),
    ("femur-left-train-25", "femur-left", "train", 25233, "e102586b69e95920e05f79a104a2810f884678ad4f9beeb9b70577f601e85f9d"),
    ("femur-left-train-26", "femur-left", "train", 25444, "e6d2291e386c4b2c3a71eb0cdf9990d6b29c7a017f9107bd0fb5895ac9eb9073"),
    ("femur-left-train-27", "femur-left", "train", 25485, "c514482216e3d818e73886f85be36c0c31dc8a3b5370aa9eb41a0a2b66cb21be"),
    ("femur-left-train-28", "femur-left", "train", 26655, "7ec9de47f3eb75639a0f83684a7a93dab031767787d4e60c6e612216dc95f7ae"),
    ("femur-left-train-29", "femur-left", "train", 27061, "95f521f3dd7819842cd4392e3a3f2381a86f6e70133b8e02b32a77826e9c1f86"),
    ("femur-left-train-30", "femur-left", "train", 27222, "e74d6c4e2d85bb7631d83f38eb74edd6def72612db850589cf9710d669930245"),
    ("femur-left-train-31", "femur-left", "train", 28276, "e09c04887f49b8ff3211df19a12d97901656ec75c64d01f04b9cb4958823ab80"),
    ("femur-left-train-32", "femur-left", "train", 30021, "9ebe3e77aada99d8a74e1d8621c27cd24f136bc4cea63515d40e9d0eed32d5ba"),
    ("femur-left-train-33", "femur-left", "train", 30687, "b79f54f5ddcfa5eea0049631a28f6fddae4d9f213847b5d5bead4e1f2a53f903"),
    ("femur-left-train-34", "femur-left", "train", 32055, "99504301ca4190187c6a4bc0983ee241187b877c1d4abc87adb322adbe757796"),
    ("femur-left-train-35", "femur-left", "train", 33964, "ee88f8eb84b6e05379bac8bd45d2dfa0c129a5e668066920b9c5875a2bdb9466"),
    ("femur-right-train-00", "femur-right", "train", 181, "8e8e7edebc86eefb396a116de7de6c18fc8a6a91d4e2816f19f79c9dfcb6ea16"),
    ("femur-right-train-01", "femur-right", "train", 561, "cb7587dc446f8450af00ba4c63877b18f131c9fb51e4469578ebb66eb024a82e"),
    ("femur-right-train-02", "femur-right", "train", 1942, "d18e9a06faafaa013a1a8c97109e921af1fd0dd51e5bf77dfd2d14433c5b3ac0"),
    ("femur-right-train-03", "femur-right", "train", 1969, "c9093ae4c8d8ee95a1d895640ab75028ae55007ed0d027397f6684d3cce88114"),
    ("femur-right-train-04", "femur-right", "train", 2696, "88b527cc6a16acb265b876e077228b5b505eecffccd0b233901e68e0cb96fc34"),
    ("femur-right-train-05", "femur-right", "train", 4150, "0a1609f7995bd6b66cdfd4a7595b6501b3f987bf8c40a450f5565621e5e8c6a6"),
    ("femur-right-train-06", "femur-right", "train", 5080, "710f11d64ba48aa32f7cd23d2f74c16672c5d7fe1e05530d5416e35e139beaca"),
    ("femur-right-train-07", "femur-right", "train", 8227, "39151ce4d16a96dc1ae8d6125f564e80d3d705968e53a27fa5aef34641a7611a"),
    ("femur-right-train-08", "femur-right", "train", 9747, "58f4be24daf70cd8d020b9b73f4b01ebe4c9620efa181725529e6d5c1c9d8d69"),
    ("femur-right-train-09", "femur-right", "train", 10464, "d6d59cbc2b002ede824ca9cd528c0236567e4ec69a25cf7bc18aafc639692914"),
    ("femur-right-train-10", "femur-right", "train", 10584, "68585fef6e50f93906fe9b01e9250a9a8bbfc422d68a8b7347c764e5d60a3cd2"),
    ("femur-right-train-11", "femur-right", "train", 12436, "2aac3148cbe264c494f22c3ab9b6a091b6d5d9e6f1f8c88a4dcd0cfe57a256c2"),
    ("femur-right-train-12", "femur-right", "train", 12837, "922c2a4dc7687af453559febdb01103f60219e6290e86554ceda2f28c551801a"),
    ("femur-right-train-13", "femur-right", "train", 13230, "8267a24d6e6aacc9b4eaf1ed1f4d374fe82778f665be3ff759a4342da9903f17"),
    ("femur-right-train-14", "femur-right", "train", 13251, "9d0cb85bd0a5b5347aa5edb493003a6d5158e9f234a4b1c9ca0fa3bdc1df790e"),
    ("femur-right-train-15", "femur-right", "train", 14499, "6dda2d7b34b277ce967c3da2d2c168b8cea334fe32c88e2291ca680db13a7edc"),
    ("femur-right-train-16", "femur-right", "train", 14766, "e4d23f12dc654f7b039d203900ab2a6fad46b04125de77ba97792f5e6c462346"),
    ("femur-right-train-17", "femur-right", "train", 15234, "13c2eaf124e712c7467234b459c2de7683230421a184abb28046d1a941795af5"),
    ("femur-right-train-18", "femur-right", "train", 16454, "6ee4a5bf74be37f3780d1bd9a0b5bcdcdf572a6e8d67a3a025d10491812b436d"),
    ("femur-right-train-19", "femur-right", "train", 17443, "04c9c45e97cacec4b13c250a594c0157bc24e83418f06379e5770e1c0cc8c392"),
    ("femur-right-train-20", "femur-right", "train", 19388, "39ee45becf7dcd1a2a2f3940cbd1c7986213ab974ff4b878b24f77992ee52583"),
    ("femur-right-train-21", "femur-right", "train", 19589, "324f589b01ce412915d9144063ceab91aaf645fbe0d9e4887fa8139b25c72f84"),
    ("femur-right-train-22", "femur-right", "train", 21988, "07780cda3d3c2e570d059ae9dc17f7b6e8f466fa9ba49d8f1c35e062871ee2ee"),
    ("femur-right-train-23", "femur-right", "train", 22788, "49b3111715dbdecbbd0e84b2f3f5a5f25fd8cea7f23d15f8edd40ec5a4e082dd"),
    ("femur-right-train-24", "femur-right", "train", 23396, "d2b133659e89f09043f69b5074e8b84892d2a82d4216f8b7670f317fa9b720a2"),
    ("femur-right-train-25", "femur-right", "train", 24441, "e3b0ab288b6fa43521921d42ec2ca751eb23f52af95c0f09447c488763beda5a"),
    ("femur-right-train-26", "femur-right", "train", 25260, "cb36e059a2f2b2e030e21749dfa1cbcdd1ea5577a535700e30e13ca118171c27"),
    ("femur-right-train-27", "femur-right", "train", 27497, "f269e2b59b15efff296c81ec02d76e7f7172338a57ed414b35ded1d7778a3916"),
    ("femur-right-train-28", "femur-right", "train", 28702, "423ef8f754cdc748a5caa6fa183b27e3c3aba050aed67c07c45845e1f681fa55"),
    ("femur-right-train-29", "femur-right", "train", 28893, "5ce378734d3ec8d011421a992b6c8063983e78bbf87aba1820da7a84e3696eb7"),
    ("femur-right-train-30", "femur-right", "train", 29831, "7e06af6249730fda1e81746925acb636878dff81fff6d1d331d298e31a7c0ea2"),
    ("femur-right-train-31", "femur-right", "train", 29980, "aa418eed4b1eb20527de6ceef1aa35bfcccbb44dc0d32950f7b9b7e2666a67e5"),
    ("femur-right-train-32", "femur-right", "train", 31047, "da4ad4ddcef471d3ca0cbef8ea4bd7c5fd592cb731a0f580ae747ea95f1b66f4"),
    ("femur-right-train-33", "femur-right", "train", 32737, "4ee04be5a93f2be4a01517cf7aec182c55c6e80c3b1031135b85e384b6dda468"),
    ("femur-right-train-34", "femur-right", "train", 33182, "f424da3ab611866b84c406466ea86700a0bdc1df604ecfface7a68ffa39f8013"),
    ("femur-right-train-35", "femur-right", "train", 33206, "d8c8b8e82901693b473959a472fea99ddcd9a14b48fcda3a297e96f8f28bad95"),
    ("heart-train-00", "heart", "train", 79, "cd49933f38119e5740d05cd64a1f42cd98a2644b53dd75a3893bcd16419bc305"),
    ("heart-train-01", "heart", "train", 2480, "3a58a59b2651efadab092c3c5c156693fc61a311cd7fad59e0fa8f209f12986b"),
    ("heart-train-02", "heart", "train", 2995, "30ffa7bab11455e8f362aad008489a620d5c8760640fb82aa352301159ded9c6"),
    ("heart-train-03", "heart", "train", 4056, "657224999ffbec0cd9b049251e1455b391bd62ddad3795b9f22235b6f12f2ed5"),
    ("heart-train-04", "heart", "train", 5304, "7e4a71c7ef808c3636deb29eb031e4fc04e34e86881b697af0a05283a4fbf728"),
    ("heart-train-05", "heart", "train", 6069, "55a39d1f5798672bbe2369565353aa6fa65704de5071cdbba43107f04d5d4fd4"),
    ("heart-train-06", "heart", "train", 6190, "76fa8fd7fef756943bd4f41cfdf2c5cfd65d3bbb2cf3c187223c3b1059e1ebd8"),
    ("heart-train-07", "heart", "train", 6193, "36792e9638e6d957c3971cd6ced60eb205656e673d969ce126f4207fc6ce6359"),
    ("heart-train-08", "heart", "train", 6891, "370ec05af5be720c46ed71be871bc7be6094458175dfb7094ae4cf8d41de5da9"),
    ("heart-train-09", "heart", "train", 8804, "59bc3fcbe4b32f4be2acc730762b041eb0a3b0c871d4358162c37212618f0092"),
    ("heart-train-10", "heart", "train", 9573, "d7a0387e7613ff50a40a5b325aaceecdd722454b9411f1a9e1ba91816803e8e2"),
    ("heart-train-11", "heart", "train", 9735, "79fe49a0295878b8efcf49b168db1ddb8364d3bf6e66f868f6a9bd6c7edb6675"),
    ("heart-train-12", "heart", "train", 10121, "99824ba2b33a91bd7991b7ff878a4d6589b2bc7db5b52e6fde1d3e8f03d1cbf7"),
    ("heart-train-13", "heart", "train", 11560, "2db712964789184242a3bde0e160e7b0f7b1f1430ab040a57428dd587b0d3c00"),
    ("heart-train-14", "heart", "train", 11643, "5d1468b16f649c095aeb85ab538028fddbdc7041f49e1f039cc846eab39eb950"),
    ("heart-train-15", "heart", "train", 12228, "54721fd110166659bd7cf86779288dc290939556b5830aebdab4e00f88e24cf9"),
    ("heart-train-16", "heart", "train", 14365, "38dae71cba0bcd9893b66782191a664b451c267b9034d4809a1abdf422b1c0c6"),
    ("heart-train-17", "heart", "train", 17079, "aa2221b4d8676a8ac1129198f72ac552c35d9f9ea57914ac537166231aa2d329"),
    ("heart-train-18", "heart", "train", 17194, "e6dd4bc84a8837fb7049b148186275bd7066eedecdcb0488cad75b42207983ff"),
    ("heart-train-19", "heart", "train", 17274, "dfea6edc3bfac50896abec26d109e215719a49f172b86faeb9ca1dc6f5658fd5"),
    ("heart-train-20", "heart", "train", 17867, "9e7d7509c3f1afa963f08fffc59d1cf8236af30edf01859efda9ff6981d056c7"),
    ("heart-train-21", "heart", "train", 18716, "4a0417afa634bf7ce6c21cb3e751b694b2306c10c530ef00378f8f18e7953b61"),
    ("heart-train-22", "heart", "train", 19421, "ada7a419f43f8fafe8e31cd4be8745bdd75e98ecb66b7994a6957437b7621caa"),
    ("heart-train-23", "heart", "train", 19665, "e92eabed429869f6e9bb4ef5bc82226fe34ad024ed0a157e3b7f0557fe1266bb"),
    ("heart-train-24", "heart", "train", 22007, "0bdebdd70ad1da63b6916c446734a7127908b0aca0b671a86d0764569d4d845e"),
    ("heart-train-25", "heart", "train", 24752, "5d0efd02ecb2882cb69c7c8f0c828647b9190fbc8f2747746310627683d284a9"),
    ("heart-train-26", "heart", "train", 26198, "677c4fcb1492603ddbf8316a38e6c93b5be2fec45516390123f76ea4a8d7deed"),
    ("heart-train-27", "heart", "train", 26401, "b02356910abe87bf3133fd953ef25d95b1c19df45dcfce63884555dfd95f5e6d"),
    ("heart-train-28", "heart", "train", 26996, "ec83dd14efa1c82171f140170fa7e2d3324752baa7e4c65a68b0de4b7eff3249"),
    ("heart-train-29", "heart", "train", 27386, "ad74b93dfadd728b72f1a56bcffc2bfa255de6eddcbe09f1eef3bd44d3f6fb6d"),
    ("heart-train-30", "heart", "train", 28200, "078fbcd5d54b8da441891fd81caf5e6354a3c4f52525fc70b52e6deedb3cd6cc"),
    ("heart-train-31", "heart", "train", 30242, "5a020a3bbf7dde0b1faa55ac30d9196b922d7b3b8d42b3ede37bed9081306506"),
    ("heart-train-32", "heart", "train", 30794, "a55e3dfb6329eee86486aa3402011740ab9976936678d55acf13992315a9e17e"),
    ("heart-train-33", "heart", "train", 31390, "1ae981cadb7fbfeaaaae65de500a8b956d2df454cc48eec150bafcfd07e353bd"),
    ("heart-train-34", "heart", "train", 34166, "7b69cc30b40dbe8cd454262d4dd47cce378a15eb634615cc465bdc8959d041e5"),
    ("heart-train-35", "heart", "train", 34170, "4c024c3217699f5660fc7141910e44d1865533c2384ea907969b1a5b593ffb50"),
    ("kidney-left-train-00", "kidney-left", "train", 1725, "f67b3a3567d2c5a54c1a96a1a0ce626a9a9b4bcb2a00edd7a7e99c0c73775b33"),
    ("kidney-left-train-01", "kidney-left", "train", 2617, "4bec8b918eb60c5673d88b7c3634289096f939dc052d726ee02ab5df608b2394"),
    ("kidney-left-train-02", "kidney-left", "train", 3304, "61781fbaf449deb32958dc25aa048895369a20694a64c76f1ba2f6635f9c4eb5"),
    ("kidney-left-train-03", "kidney-left", "train", 4524, "97ce2408cfd6bb97282a4148aa6b11150ddbf78d2c86e763edd419ea905310f6"),
    ("kidney-left-train-04", "kidney-left", "train", 5554, "cc7191531f1e5a83d5176b08eaf6df886f26cbb3d0722c9437b256d71c45062e"),
    ("kidney-left-train-05", "kidney-left", "train", 5699, "84bd60bc85d2d6c7d84f70720758543c3488eee9e88d49124a376afc9a64898d"),
    ("kidney-left-train-06", "kidney-left", "train", 5876, "e35d48e76d0591a98672a49dc42cf6f8b8f0e4c190d6405e966affe2105b4421"),
    ("kidney-left-train-07", "kidney-left", "train", 6468, "57033e3ec4d51c5630520a6892136b1d927c12ebe140012c410b03df463217db"),
    ("kidney-left-train-08", "kidney-left", "train", 8118, "406abcc06c6b722a0c1b19a843c7bea018fa360dc513117f0671e8155a61154b"),
    ("kidney-left-train-09", "kidney-left", "train", 9490, "69cc95eb8787b29c1502c43cd79f8e9a4d8783bbe35de6e3a1e851bc817884a1"),
    ("kidney-left-train-10", "kidney-left", "train", 9844, "012d2df273b989f223868d1e517e9c0a9db3c0a816d1aef54745f883cef6fe8f"),
    ("kidney-left-train-11", "kidney-left", "train", 10897, "f01f20a45acb3f9ecd3133aa78cadc2215132a12ce8cc51e35000f49299da552"),
    ("kidney-left-train-12", "kidney-left", "train", 11797, "b8aa09013e82245f37317b02d2afade684977b8b28eecedf4e9dbfffcb6111c5"),
    ("kidney-left-train-13", "kidney-left", "train", 12357, "eba70c62b97c3b8e94d716cca4802bda0e5e398f1edcb62a54ccdd8d483da13d"),
    ("kidney-left-train-14", "kidney-left", "train", 12789, "70314a54c4bad75038fa5bf2f4ff87e37ff1cf81f733870784f1fd6b7bbd188d"),
    ("kidney-left-train-15", "kidney-left", "train", 13205, "3bea246485786c1b58bc198a8ef33bbb9ad09a70c6fd29d39945108cfbe27142"),
    ("kidney-left-train-16", "kidney-left", "train", 15679, "8707530209cc3974d9c0ca63591ea7f65a8dc4a0c4f4e1c521d8c1967c1a6928"),
    ("kidney-left-train-17", "kidney-left", "train", 16025, "5af78275676c5cf1ecf29c533e7957cd9990bc5a6e67a1edec0846208aabeec5"),
    ("kidney-left-train-18", "kidney-left", "train", 16760, "0aa7d2d0e4247a4d67f05af44ec1443a1984d59ee5b07be669922b6f4ab72af9"),
    ("kidney-left-train-19", "kidney-left", "train", 17946, "48768831185402817a2b312e7d178932500a6f4222a58deb03e39222194ef467"),
    ("kidney-left-train-20", "kidney-left", "train", 18443, "4c95b3d57b96e0d86605b0bf582bdb44376cad53f4eceadc6429be080323bcc3"),
    ("kidney-left-train-21", "kidney-left", "train", 19156, "dff3fb7a7ee14cdd854ecfb25e1156465935cd38db5aa0a66fa95d2da4e17844"),
    ("kidney-left-train-22", "kidney-left", "train", 22161, "782e634a2355e78cf495e0123b7fef809dd4763535c932d7a1440113a1aad725"),
    ("kidney-left-train-23", "kidney-left", "train", 24250, "bab2920c58adbae608cfddfbd22e8de247d46cbca8b3e5f5fadcc61388350cbd"),
    ("kidney-left-train-24", "kidney-left", "train", 24529, "c62decb3b787f30b4e535aaf9eb8ea6a0de77a4cff81c0ceb136f9f995cedbc9"),
    ("kidney-left-train-25", "kidney-left", "train", 24572, "405f94dd9692169d05672c18a40b18b3853c1426df715d06ed4a03c16d4a136b"),
    ("kidney-left-train-26", "kidney-left", "train", 25065, "f0d8444c3aa25d1f2dfa7a027dd90fdfae10fbda0fe2331e5c3e944b3916f13d"),
    ("kidney-left-train-27", "kidney-left", "train", 26602, "7efb86fa601b0f2a3f3bb5f65bdc1088771bd13af41ee120d7b643c586993c8f"),
    ("kidney-left-train-28", "kidney-left", "train", 26691, "b1a5e17ae27639068b6e71a624676b645ac47062376bced1bb07b998886557a9"),
    ("kidney-left-train-29", "kidney-left", "train", 28053, "215add963fb2517a3d4cbc09aca0b3b9f3605924ef355f29d5abd96d18528b1d"),
    ("kidney-left-train-30", "kidney-left", "train", 28100, "f5262e9bfea3f901e2da2faef753143b71ca72492a6802f28e9804ab89465aae"),
    ("kidney-left-train-31", "kidney-left", "train", 29485, "e19fc1a4c0b4f866fffa5dd334bd8841ebee134e183b1f93bc88935f41bbf75a"),
    ("kidney-left-train-32", "kidney-left", "train", 30546, "3a88a8536f76abd2ee62f5c19665cc599b846c9535e9e2843063fc9a7d5dfc11"),
    ("kidney-left-train-33", "kidney-left", "train", 30861, "69f4ddddcdb421bacc05cc3e4064f7829ecafea6d19d9a8a9625644e8abd6cb1"),
    ("kidney-left-train-34", "kidney-left", "train", 31208, "7dd09ede61d0b47a05fcb4610cc480edadbed7f95410e17eeb3ea9776bfd9eb3"),
    ("kidney-left-train-35", "kidney-left", "train", 32484, "23192e87d43db4d0e7a766f6e41ecae394aec15cc4042cbf43a5297fae763f01"),
    ("kidney-right-train-00", "kidney-right", "train", 107, "5b3dc24e40dd3b1785951787990724efb597365a7f6bf8e763fedd1876c2a325"),
    ("kidney-right-train-01", "kidney-right", "train", 1086, "a7eff4174b574450a4dbd8e683bf3538736e960203c88dd598b67a686f22f830"),
    ("kidney-right-train-02", "kidney-right", "train", 2827, "4213b189f40daf3e4e8ade84ead2df4893bc9995e5bcabf86e143a26668bf8ca"),
    ("kidney-right-train-03", "kidney-right", "train", 3028, "c7741bfe46b9acc3d052513d31b0c8e3a70e4cffa4ba2a79aa6b9dbaf5b1ccbe"),
    ("kidney-right-train-04", "kidney-right", "train", 4162, "3f00d2151bb2316b9c93a1b8220b939101b7f988c102577ad3b2a5fa75cfe5c1"),
    ("kidney-right-train-05", "kidney-right", "train", 4537, "0cb79d7b37aa4064a85f116018bdcd27a044d426e9009b77fc21b192359e5cee"),
    ("kidney-right-train-06", "kidney-right", "train", 8257, "8e39622d8a53cff8c29ff07fa989cb0b7687558bc6d91441564ca2067184942c"),
    ("kidney-right-train-07", "kidney-right", "train", 9466, "daf5a8764758e9efe57993ca77854c5c7eec76f58047bf4c3a34390e8b7422aa"),
    ("kidney-right-train-08", "kidney-right", "train", 9643, "7cb7d1e857e92049827c80c73a68550a591de3a371fe6af53dff7f296192cdd6"),
    ("kidney-right-train-09", "kidney-right", "train", 10112, "eb644c665aca8a662c98cd7533f862e278ebe7287f3ed4fbad1e2315774c5e2a"),
    ("kidney-right-train-10", "kidney-right", "train", 10180, "39aa63a381acc5e8a4d5e671835b04e30041a959f30d7286f72822362355838c"),
    ("kidney-right-train-11", "kidney-right", "train", 10956, "969b17796f7d0830c00cbee828678e87422c874024c4184050ad443c4dc9763d"),
    ("kidney-right-train-12", "kidney-right", "train", 13524, "f0ae1e2ae4d029c3c245f71000168636d4801a19f863b5790c39d9d578c85d52"),
    ("kidney-right-train-13", "kidney-right", "train", 14215, "085e1c17f74eaa265d545730ba1bdb9b16fa47d3007d1fd50d14747655e8496a"),
    ("kidney-right-train-14", "kidney-right", "train", 15864, "baff32c50f4e7576587212f8eabc81cdad313c61095063642df2a18737ee3b90"),
    ("kidney-right-train-15", "kidney-right", "train", 16136, "aa232522c40c8895565776b83b6e4655cb20df7be974546d00a78a1f4f36ee6c"),
    ("kidney-right-train-16", "kidney-right", "train", 17404, "76f1cc743ec33a678a584c9d2a81f35477b79972d15bf77cb7931dc98e6774df"),
    ("kidney-right-train-17", "kidney-right", "train", 17497, "41852094ebcf4aa1598aad02bde7a2206f5e8cec9db2e6b8dad7de9e3a458f08"),
    ("kidney-right-train-18", "kidney-right", "train", 18170, "d6adc973dde2f26a05bffd68a04192334fe1836c656160dd3a049e7890b0b74f"),
    ("kidney-right-train-19", "kidney-right", "train", 18328, "063feb2c689e1f431cc0a5f12f91e08dc4213ccd71797cc3b05a1cc2e0dfc291"),
    ("kidney-right-train-20", "kidney-right", "train", 18436, "a8703c03df72b70eac647cbb306e620720330e6f8a1e275aff30e696608e2cf2"),
    ("kidney-right-train-21", "kidney-right", "train", 18739, "68e16d2edb97e37d3e8dcc62b58bcc6560ac2bfebebdff86582a2d38217eae48"),
    ("kidney-right-train-22", "kidney-right", "train", 20219, "02149b4302862bf0fad70fc16f942cd08590ba77a5293c8d0188740ff2b747ce"),
    ("kidney-right-train-23", "kidney-right", "train", 22728, "3b080f972eed51490f43078a01c03a5bbd1a76ea4b907832ed865c5d9ecc7fb2"),
    ("kidney-right-train-24", "kidney-right", "train", 23508, "bb56ac896346839ee54148634cb35253f2c1e3e721782f5bc7d2b343065f5fdc"),
    ("kidney-right-train-25", "kidney-right", "train", 24589, "408c4366f2b42deb4594026b2c4ff19a5869de9a944311ff0216943467f515ef"),
    ("kidney-right-train-26", "kidney-right", "train", 26245, "144630dd19db8f904cad0883ea88a04f2d591e38d224964b344dfb86e1f74cbe"),
    ("kidney-right-train-27", "kidney-right", "train", 26407, "59cc0d4c210be6bdd733b8aebe28c18a82193577230cc246701ce820da235bfc"),
    ("kidney-right-train-28", "kidney-right", "train", 27703, "28841a6cb2fd8be6fb4e22da4be63546c33deb4e25a76e7a13e261f7016ac78b"),
    ("kidney-right-train-29", "kidney-right", "train", 28998, "9b8ad1a81249c9e72039b3e465c0200d1a4dc246b0f0439e65abd2cde046df02"),
    ("kidney-right-train-30", "kidney-right", "train", 30875, "ac1b590f56ae5ca14ceaa5fd82c5579259a8f92d5bb8bf781be8e1837f68d51b"),
    ("kidney-right-train-31", "kidney-right", "train", 31501, "3d0cefb6bcce57e131a9d508e74c12e31d3d3372feb56b8dbda4eeff1493e56c"),
    ("kidney-right-train-32", "kidney-right", "train", 32806, "9f783b0b5a8c27d2ba04f32fc076b9fa61acbd6dd71239f8d592b9e8677d7914"),
    ("kidney-right-train-33", "kidney-right", "train", 33625, "0e317cd117540bfba322be57a3005e1fb238d1c8df97a3382298240ad6ff2fff"),
    ("kidney-right-train-34", "kidney-right", "train", 33862, "a9cf2c410fec210fe50a3040dcc516c66eb7ff21527468b72537884427e24f1b"),
    ("kidney-right-train-35", "kidney-right", "train", 34495, "7aa608f2833a60bdbba2488f252becb9efc417d77c5f393701678887018c62ce"),
    ("liver-train-00", "liver", "train", 717, "5ea15b7c3f100d0ae397123f8c4f8ecdfad3cd34be79d2565557cee6417b5db5"),
    ("liver-train-01", "liver", "train", 2163, "aff6934a61732dc4677db1fb0f13b3a4fc79eaef232d653912f56a26ea508e7a"),
    ("liver-train-02", "liver", "train", 3294, "f8fd90853d68940661b299eb3ef1bf100bc8ade39f8d295593b4a46305b96153"),
    ("liver-train-03", "liver", "train", 8511, "affa356cf2023b3b2b21b444bbd32f5389f1c5ef19d9210b4b58b559db522f2a"),
    ("liver-train-04", "liver", "train", 8525, "e50e523debde8f65cc6948f3bd75acbdb0776c3e66f5527dfc367e464ea27258"),
    ("liver-train-05", "liver", "train", 9228, "504c2aab4477335884ac8447047a28ea7e65f47dd176e6371316045c87389363"),
    ("liver-train-06", "liver", "train", 12185, "cc51a09124b97bde34b4ae821471a8fd8430400280f3464d49185217d5f56f64"),
    ("liver-train-07", "liver", "train", 12678, "ef2c39da77b8fe7bba297f0f498417324993d982aeba36fb4f145eb3c9177416"),
    ("liver-train-08", "liver", "train", 12874, "6d64878d549b3e527d1ae05be858820dd9588de2fd1ba9a6e9d6f1851ddbd5f9"),
    ("liver-train-09", "liver", "train", 15556, "df28ecae24f7e5509e3dd5dac83c5c15546056257b0ee7ec4aa32fc454d8385c"),
    ("liver-train-10", "liver", "train", 16659, "63239f6d4ecfafc9416693b83ee17c4ca0de891262b8c3c99c09253dbbc75570"),
    ("liver-train-11", "liver", "train", 16881, "495853ebec2d159cccf40ab80a93faeefece831fef3091654e4d53ac3eb4e6df"),
    ("liver-train-12", "liver", "train", 18145, "9990b4b882f3a50833014d0d586a6c683a031d3cb614cd2192c44b18c403f558"),
    ("liver-train-13", "liver", "train", 19238, "ff97976952004451cc3255f5d0dd41f0d5dbf7e4f5a0070cc49fd3f34f284c8c"),
    ("liver-train-14", "liver", "train", 19267, "6cccc4550441a8bf52d3ca218a125c4375d653c3b8baba338ef2b741d75cbae1"),
    ("liver-train-15", "liver", "train", 20126, "8c95fe1a35df56898250aa4045344aa4bb84d5234eeb349dadd9703d5fb0ced5"),
    ("liver-train-16", "liver", "train", 20783, "cad805d8cd51dba2dbe0863c4756c85bdd52fefd12ac290c7451833635ca13d4"),
    ("liver-train-17", "liver", "train", 22390, "474ca2114633d2865994674f09bf47788fd7ae42c29909e0bfc48473c5381315"),
    ("liver-train-18", "liver", "train", 22546, "9e204edc1b4b9fa4eb9234b90fad7109642c090340abc14febc3e39a55299ac6"),
    ("liver-train-19", "liver", "train", 22763, "5cb906fb482bdc3dfec1d302585dc9f16129ae11d20111f77dc7ad912a699cee"),
    ("liver-train-20", "liver", "train", 23117, "4a9b31625814a16210438300345270a4a244521e46cd13ecd320d9df59329622"),
    ("liver-train-21", "liver", "train", 23205, "1554b495427e6c86631892f31bee3d1463447c09e6b0dc91a1426fcd18187c4b"),
    ("liver-train-22", "liver", "train", 24358, "c0aa39973be566d76fb2620ac529c5f53032761573d2a3e5a4eaf965f3c3e4f6"),
    ("liver-train-23", "liver", "train", 25445, "47032364045712e1f7788db9ab13732bc03895ad487915b9fe85b6b105b75354"),
    ("liver-train-24", "liver", "train", 25802, "1c46f2b439b498fc39cb815853a7b8dc8aad39e0187f3f4bc640085612da0d6e"),
    ("liver-train-25", "liver", "train", 26070, "8648f795018e341c981aef4c9b4350fba6952e4e41e4830dd81c70a9f0103d59"),
    ("liver-train-26", "liver", "train", 26452, "59a671fe91eabb40b0d827b411e55b6536a47f8132ad2320b86f5371d934635a"),
    ("liver-train-27", "liver", "train", 27796, "b7ea68f64e10f44eae83624fb30a17a9c6a74b4d3b346d0615586e0dd2c7b38c"),
    ("liver-train-28", "liver", "train", 28037, "bde1f2b37bbb8a8244c87349cb6a58efedd9c0a31fb442d81b97680b600bd08e"),
    ("liver-train-29", "liver", "train", 29086, "e9c1f30498e4018f2a1c0db29d1c6c0b6eef010af01d5b5bbbab447650559a75"),
    ("liver-train-30", "liver", "train", 29793, "26906a47454f59df66851b9c7237101f2a19d1b17750d2c71b09c522fc19f3bf"),
    ("liver-train-31", "liver", "train", 29815, "4d127f5438f52b287d6e083492d91745a38ed0831c247c47dcd0dd07979f405d"),
    ("liver-train-32", "liver", "train", 30025, "be976fe2b324ad9e57395b19235695ac1bb579e4fc6ac4058de0c39def6c868b"),
    ("liver-train-33", "liver", "train", 31507, "ac9174129c613b9f182510ac4bbe07d47bbceaf242f12240ea408f7e604449a6"),
    ("liver-train-34", "liver", "train", 31654, "8d3cbefef56c1842b0051641cfa45c739677d1ef4c3e3564c7ddc88dcc1291cb"),
    ("liver-train-35", "liver", "train", 32466, "4986c178ea13eeb477790f927318e109eac31339005c4310f34651a6aa06478b"),
    ("lung-left-train-00", "lung-left", "train", 504, "a113a28206d351ca613513f1beaa5cb07e2ca65f6ce2d74530f890dc3d66d8f4"),
    ("lung-left-train-01", "lung-left", "train", 669, "46c894d580238875bb266a2df4dd1b7876a9019f5e6a9b6d75011345f13c1738"),
    ("lung-left-train-02", "lung-left", "train", 3883, "d2de6763705fed18bcc70773d378dbec19bddba9fb3596fb645fe905de5f73ce"),
    ("lung-left-train-03", "lung-left", "train", 7362, "f8db6ef1586fa7a31dbad24e5891473131dee7b187d2bc181a5628ac58b09c83"),
    ("lung-left-train-04", "lung-left", "train", 7989, "d2efcfa561395af7c0014d9342886b2d95dc451cb66c8aa901c10d4530a2ba5f"),
    ("lung-left-train-05", "lung-left", "train", 9757, "b7405a16449115e1aed6ff665e38498baf2c64faf0ef5688990a719d4444a2ba"),
    ("lung-left-train-06", "lung-left", "train", 10794, "54c276b0c66d217c8ede6b5c616aba5a8f216631dbfc1d7286b518e1ff123989"),
    ("lung-left-train-07", "lung-left", "train", 11353, "f1b806e0c6c0ec3f63dc0b20637936ba5fb432018a013bb095bbf9c79e63a6be"),
    ("lung-left-train-08", "lung-left", "train", 11599, "18d1eba76c6502ec7ad96baa5bf2841a3e86a7680695cedf55a64f3656d78a3e"),
    ("lung-left-train-09", "lung-left", "train", 12274, "293452ef673c87768523e00df6a4ff047f4f039180c28744e179ab857797fa7c"),
    ("lung-left-train-10", "lung-left", "train", 13105, "7d5deb993ba1136ec053448c8905cd6957355bf4ee2efddc861af3117bb4e833"),
    ("lung-left-train-11", "lung-left", "train", 13187, "8a6e19e518d04cbb0e525f8a5cbbda0fa93a668f212584b1bcb4a4afce2a981b"),
    ("lung-left-train-12", "lung-left", "train", 13374, "b5b385132c26778458fe9b48dc571fa7b534bcd06c8102e4b18c12b9fd50498b"),
    ("lung-left-train-13", "lung-left", "train", 14962, "03a3d100862f7820a2cc323296589de68b769bfb532346e6d44d3186bc11ca0c"),
    ("lung-left-train-14", "lung-left", "train", 15181, "ebbfe977a029fff4748293ae90f03797fa4beafcbb7f61be145613390e0fee6b"),
    ("lung-left-train-15", "lung-left", "train", 16081, "71e241d89b080f71a1815ef80fb00e1056a5aab19dd4cb9853e1f623c5587287"),
    ("lung-left-train-16", "lung-left", "train", 17008, "eb42d863788a4012720bde8ae88378a3ba71ad3849081a2f756e0cc03a282bbf"),
    ("lung-left-train-17", "lung-left", "train", 17335, "a5a97c7d6f0cff12852631dbb2348c9ad2a76c9a5c3c302bf2d66e8fe12e6a72"),
    ("lung-left-train-18", "lung-left", "train", 17552, "4af0fb542834c8178e80e6e0b29866b6ce737a5bab89cfd150e9045510d2f34c"),
    ("lung-left-train-19", "lung-left", "train", 17679, "93df27c15a6d1f2bb2395e4925dde6903b784244e35d3aad831e8f818e83fa6c"),
    ("lung-left-train-20", "lung-left", "train", 19430, "a8efa33c8aca76ccd4b66bc78ddcf4a7cb46900974824cfa0f844f02420231cd"),
    ("lung-left-train-21", "lung-left", "train", 20891, "4f171e954c719faeaf470207e911b4441926ab15958edeb8fc62aeb2655a4c7f"),
    ("lung-left-train-22", "lung-left", "train", 21733, "cd50eb582acfc1deb84d9df165b48d6b82cf11a33044e948cd57846f24b188c5"),
    ("lung-left-train-23", "lung-left", "train", 23087, "c08622823f85867b83b298f64fc6ae5c92b9e51a79194f7d76ebc6b433b93227"),
    ("lung-left-train-24", "lung-left", "train", 23132, "6031d45aa41bcbfae42eb934e9d4cf9ce43712af578a9543b3e3d6d4b44b2fbf"),
    ("lung-left-train-25", "lung-left", "train", 25835, "7ea660b082f6a5d64e8eeedaa3e85b7647f0823d94c1219ede37433650ebf7d5"),
    ("lung-left-train-26", "lung-left", "train", 28001, "647a75e616c04939679677088fdab4d139659c02381f11ade23872bdebd79ed0"),
    ("lung-left-train-27", "lung-left", "train", 28417, "7db120bb8bfc50efadc5bfd5c2c224cd1a141192f849132f829525ae4367f540"),
    ("lung-left-train-28", "lung-left", "train", 30256, "370322ea25fe396067498d6652afcace6fbd90e244987c744defeb4118048226"),
    ("lung-left-train-29", "lung-left", "train", 30453, "d4fcf412ee246a62779d6a68ef85c8afbbc2e4cecdb38f97243e64fccd2a10d8"),
    ("lung-left-train-30", "lung-left", "train", 30634, "9c00bafa3314a3723a1a4047181b0a59e3107d321c6b4206f74b59fc81786117"),
    ("lung-left-train-31", "lung-left", "train", 30909, "583d78830fd89d9d5b30ef1dbf12ccea83587b9316e42c085ae5464a8b1dbc1f"),
    ("lung-left-train-32", "lung-left", "train", 31582, "c2020c3e9049f4664ba03bcf599ad54dc0719151b5cf0e73a0ff3b467b787c06"),
    ("lung-left-train-33", "lung-left", "train", 32378, "6060413083bd66773aad15ac17a7bca3d21d9c37d2adfd8f3dec50abd9164106"),
    ("lung-left-train-34", "lung-left", "train", 32919, "8aba68927d48f4bd26e33210c2962ae26ea9e33f016c1fa15ecd7c7388dcf852"),
    ("lung-left-train-35", "lung-left", "train", 33658, "34d08178af4628d78eac208e4909ff33327fddf5dd970b34d508a8b9cdfaff92"),
    ("lung-right-train-00", "lung-right", "train", 205, "cad61ea2d491a6ac177cd4ee67a0f650753497625d66c7c5e8f6dc1031c4a387"),
    ("lung-right-train-01", "lung-right", "train", 577, "9faee439ebf076e5b7f05d02dfd0b5190780893003d114e707cecdd3f4ceacdb"),
    ("lung-right-train-02", "lung-right", "train", 4810, "f534cdb0819afd9ece730a9d973417aa8ae4c35916f7c2d914732ba29f02c25a"),
    ("lung-right-train-03", "lung-right", "train", 6967, "fbba7a4cb9e17fee08097d9c17afdcd1aa4b6c3fa417a98ed244449cfe8e20d1"),
    ("lung-right-train-04", "lung-right", "train", 8137, "ff9e9675a5460c15e2819904a604d63ba013b18ab4ec0bb7c969bfe719513bf9"),
    ("lung-right-train-05", "lung-right", "train", 8777, "db4d9904a9a9d03c3b7d065ecf272e933cc6e39411a674b99ae42b98f135dd49"),
    ("lung-right-train-06", "lung-right", "train", 9668, "224c7eff3e21104ec1e9f32e43652b2922e85bbac32f7635b88c775734ebe1b2"),
    ("lung-right-train-07", "lung-right", "train", 10381, "e69ff9cd6271705b5fd4e4378c40c964b6d7c27e316fcaa6401ecdfa7e81e3a5"),
    ("lung-right-train-08", "lung-right", "train", 11030, "6e5010c5060801216f2ce41efd1b12385d1ad1b49155a34cb1c281a48b832a68"),
    ("lung-right-train-09", "lung-right", "train", 12505, "e5823f0f6386c5abfccde981ae5e3629499079c0a5afa23fe5d3e850c607da56"),
    ("lung-right-train-10", "lung-right", "train", 12931, "8da5af919dfc187ff1cc96f73ed68bfa6189b9b155d520e8b4b8811c429dc556"),
    ("lung-right-train-11", "lung-right", "train", 13299, "f464476814b7747f9ac3b34cf62201392e3228d72466f6108e1bb79c3118e9d7"),
    ("lung-right-train-12", "lung-right", "train", 14659, "5eca16cd3c16a68239ec1d5a8a7d06acc4a82253e42751f2951e14eb55830b1f"),
    ("lung-right-train-13", "lung-right", "train", 15660, "366f70d8167e550267960771676d184bb12fbab3ae05374078a4d5295481dc07"),
    ("lung-right-train-14", "lung-right", "train", 16204, "299da23c84ff71db294cccf94e0be9d6bf68bb6983b04660c9f178acaad9511f"),
    ("lung-right-train-15", "lung-right", "train", 16236, "4b56ed7f86ac8f41060646099c418f5470c8fd0e8608386d1c3d424ae9f06314"),
    ("lung-right-train-16", "lung-right", "train", 16277, "9715864f05f5f2b1ec3ccacddb12eb03b750b6d21a97768e049c7d3262c40c18"),
    ("lung-right-train-17", "lung-right", "train", 17054, "b95df9093f422c7f1b04fc2b207fe19f84406fcba3d013d393318f3327f542a2"),
    ("lung-right-train-18", "lung-right", "train", 19352, "e26c423995dc9d8e92d0f70faebbcb791093f6525d22a226c842e74da34fe625"),
    ("lung-right-train-19", "lung-right", "train", 20527, "baf3b9a3669194ce9d21f831f2ffda2c0d28d7d3c70c7db175c642e61c86f2f1"),
    ("lung-right-train-20", "lung-right", "train", 21027, "c55e8e38e153a8507601b8939cfb99210ee4de91baa42cb572f1c2c90c01503d"),
    ("lung-right-train-21", "lung-right", "train", 22877, "3083264c144662f222c6a81a5445d80492bf9734d5f990f5c8ad5cc57fbfbede"),
    ("lung-right-train-22", "lung-right", "train", 23385, "bcd45de97776374fbc8fcda3e4f1532596080173636a1ce68a6b27a960b6eb40"),
    ("lung-right-train-23", "lung-right", "train", 24229, "20100823a9cfb7c519465f6fd2c5701c3f99c2f66250d523dd42e0d043fd0ff7"),
    ("lung-right-train-24", "lung-right", "train", 25255, "4cc4438d3d33fe100d6d02bd2d7b688249c964d6363258f8a2be5043ee2310ee"),
    ("lung-right-train-25", "lung-right", "train", 25460, "cc72fd341512032d3206a285630831695193f65955ce3edcced4bfdde501604b"),
    ("lung-right-train-26", "lung-right", "train", 27809, "8d2726247051210058c346179ebf95b738a94b999f15baf3eccadb6028beb949"),
    ("lung-right-train-27", "lung-right", "train", 28306, "9981d900280002b727358afa1da65a58372b966dfa65c7bd508944c361da3ddf"),
    ("lung-right-train-28", "lung-right", "train", 29291, "48c37604acca57c7f4f138b4640a29ac81b481aab51b0547f843fcc8af193da0"),
    ("lung-right-train-29", "lung-right", "train", 30349, "ba1c4486765a6a42ef1eae1478e58e3a30ea0e2cd130ddcce6bd9dc8faf63351"),
    ("lung-right-train-30", "lung-right", "train", 30856, "bbb164911cef5b198c64dbff98bbec4295af7e919c4f4241fb34f8dbbfd971bd"),
    ("lung-right-train-31", "lung-right", "train", 31465, "ed554005fd329f72d7b1db805e65352980a2e9e66f49a563df922cfbac2bf8bf"),
    ("lung-right-train-32", "lung-right", "train", 32490, "169814da70705f5d1d99ca41030cd67ba4efb367b349688e2106fd9e475f9209"),
    ("lung-right-train-33", "lung-right", "train", 33277, "761c6d7d1cb56df9c9ecd1671c97e77384ab1426a6334c8f300db2b5dbbdba2f"),
    ("lung-right-train-34", "lung-right", "train", 33541, "5b16ec953a20feeed9d17fd1c23c35f77d05bfdd5824e24a349c88ed8efc0d20"),
    ("lung-right-train-35", "lung-right", "train", 34459, "f832b08c551752867310ff329e9c03022cd2ccdcba191df69fab4a7bfd58dbdf"),
    ("pancreas-train-00", "pancreas", "train", 925, "819107d33a9a501939b01fb57bdcd81b13411c448514ebcf65250a3be8c59838"),
    ("pancreas-train-01", "pancreas", "train", 1655, "e454d703db96c44ed5c3ccbfca80f0010124183c0cb1c47caf728c3aeb013973"),
    ("pancreas-train-02", "pancreas", "train", 2976, "b2e73025eaafdc42a1fbeebb5a4a4b70eee1ec952043cd27ba98784519fcc497"),
    ("pancreas-train-03", "pancreas", "train", 3366, "5470c63284225196fcb5bfe60b07c01c51be3b7d2f6a363c7066308ce9f53370"),
    ("pancreas-train-04", "pancreas", "train", 4261, "4976947b25f170263045978be4b2bbabdc487aeb4ca75b96522b2d6808ea95ab"),
    ("pancreas-train-05", "pancreas", "train", 4943, "176c55f2f5546195963d7a8ad6ba8b6910b70c561705c9660f9909d80987b5c4"),
    ("pancreas-train-06", "pancreas", "train", 6504, "ad8513b13c5baca58df4864ad45c936d572d0c8e921331ef86e4695e1bc56478"),
    ("pancreas-train-07", "pancreas", "train", 10031, "3e1d8b68c7e945aaf731ce2597d53b4fb621dc4f012fb77b2be23408fa71f47a"),
    ("pancreas-train-08", "pancreas", "train", 10101, "a4751c1a2994eaa260a6146b7532aefd37b81f8149158fa64851137622cf17dd"),
    ("pancreas-train-09", "pancreas", "train", 11264, "9a4508cf429d2614c1b06f540becea5807f0526ded5990f055358cafe37a1710"),
    ("pancreas-train-10", "pancreas", "train", 11539, "8a4bb7c5c16deb0a9edd77ad5c5a5b5676afe8bd008b420687afaf36890367fc"),
    ("pancreas-train-11", "pancreas", "train", 12320, "c48635398f13cb653805f39633154296783f6143204b5eedceaa37cfe61a9666"),
    ("pancreas-train-12", "pancreas", "train", 12384, "effe84f7e42baecb49727b7f733f51c93ab3a5b8706295c349f01eb491d2edb6"),
    ("pancreas-train-13", "pancreas", "train", 13888, "6d9e3f579f58f3ef7804f98fbd911022de4483acaef0308858af983988472c04"),
    ("pancreas-train-14", "pancreas", "train", 14300, "a5a60c532d26c2649f6d7786885bbcdb29b422d40ab4e6c106fe114a3de2ef88"),
    ("pancreas-train-15", "pancreas", "train", 14429, "2eb24d4f5d4e700784a4b081bfb90f595174376b140820b82ff6c46ee31a00fd"),
    ("pancreas-train-16", "pancreas", "train", 14972, "5d2c90832fd0c988df60705886bad5ad8b8ba74fe0c5c301b86be8cd27ffd1b8"),
    ("pancreas-train-17", "pancreas", "train", 16699, "1253c2729e03719eabb15c3f47a32f89a6ab481c73af11528ccc384982926b25"),
    ("pancreas-train-18", "pancreas", "train", 17106, "f2ffde15d5931e22414481ae6d8b41ab0db84ffcec55737255f363070db0f525"),
    ("pancreas-train-19", "pancreas", "train", 17735, "897318408e7f9ff0c19e95b91f4806b15f2d44239ab95f8c33e9e09be726ddff"),
    ("pancreas-train-20", "pancreas", "train", 19529, "80dd247853e10de14245a26394e84d04f5a880df8465aa3e1e3932d757e7bbc0"),
    ("pancreas-train-21", "pancreas", "train", 19583, "cac052ffaa0dcd2a1bdea5b0b00003cf3d4b70efcbbbe7df158c0096d0ee35d1"),
    ("pancreas-train-22", "pancreas", "train", 20526, "42c1bb829c704392facc280fdd49edc47a8acf8b991fce5718d32d4bc79212f8"),
    ("pancreas-train-23", "pancreas", "train", 21395, "2002e39fdfe34993fb2a678fb1c77a0f511f946d8f7c01196c8f9d1bc79ccce8"),
    ("pancreas-train-24", "pancreas", "train", 24566, "e5fd41cd94071b187640af3499910ab245a030d943387d5110171ef4ba25169b"),
    ("pancreas-train-25", "pancreas", "train", 25837, "7cf33f1a827ce4c306755b3ef76348a7167e353adcb5f7b0da903b79fef7e18f"),
    ("pancreas-train-26", "pancreas", "train", 26653, "967df370f8da5ba82f72903e145b85053fd82e57d03d50662943a2175904422f"),
    ("pancreas-train-27", "pancreas", "train", 26866, "7a7bea2e7177f76cf21ad7fa51cbb58135943acd1897b6609b7eee15989634a3"),
    ("pancreas-train-28", "pancreas", "train", 27168, "15df950bd67753e34da85ccdda5a847d8bdb60085a630b66511de4d6809e1e8f"),
    ("pancreas-train-29", "pancreas", "train", 28295, "fe4faee6c7dcc41d75b35750822037b84df5d796c54f788af5f0994a10919b5a"),
    ("pancreas-train-30", "pancreas", "train", 28861, "1f0d1fff91a94d1b33a296c404501bc10c4ab745537f1e5217696d0215d76174"),
    ("pancreas-train-31", "pancreas", "train", 29270, "c2651137bc42be3be83391b277fa4c1743475adb05a4ab5b9d0a48bb1d12f9ac"),
    ("pancreas-train-32", "pancreas", "train", 30204, "0246a8f50091d70581219b4799ec36a01843d1b0ef02dc84a3bb04dbcdad8b49"),
    ("pancreas-train-33", "pancreas", "train", 31905, "96003cfcfe593c61531ccea6c2d72c710a9061a46d8f22b0085f0640c289727b"),
    ("pancreas-train-34", "pancreas", "train", 32269, "e1ee1a8347529588b2551643e9032ad58b6505c0c477ba0d7cf1a4129311c1c4"),
    ("pancreas-train-35", "pancreas", "train", 34310, "505ecde1983c0feb9b46c3690f0ae91971991b8385f813b3f380b40f7ff66957"),
    ("spleen-train-00", "spleen", "train", 100, "1ef0ae2c65d5c121156ef93d9e697bf742a41e34c0f819255720de35b9fc0ab0"),
    ("spleen-train-01", "spleen", "train", 1372, "392d94654d1c3e8058af258b82c628f17cdff019c5ad4433711e5e3ad8f1df6e"),
    ("spleen-train-02", "spleen", "train", 1839, "c8291f9d75f0fa5b517206c90858b0dec66c171cfad40f6adf1a78509066452d"),
    ("spleen-train-03", "spleen", "train", 2959, "957f39f7b41734e87b037779a6e5576edcf87482223ae77ca7af03039662662f"),
    ("spleen-train-04", "spleen", "train", 3160, "54178e85c302459a31d2681163368d02f0c217739da6149557e9d461863d96e8"),
    ("spleen-train-05", "spleen", "train", 3418, "37f32c8291d08eeb2d86d94c1272461745e35e8d4fe5fb59f5d4ae163c656f30"),
    ("spleen-train-06", "spleen", "train", 5656, "3d75e1ffd6c7a039c2166b429fbe960c57440f33ab4f1c4e103d6de842812351"),
    ("spleen-train-07", "spleen", "train", 6683, "d75e02c8b76a1b381cc243dc6ded30b67b1162e2693fee58091d64a364136ca6"),
    ("spleen-train-08", "spleen", "train", 7390, "429557f66dc9fdd06623c14445a8141a7f864a081487350cf933dd3efef3bbeb"),
    ("spleen-train-09", "spleen", "train", 9045, "b7f8d9a85603e813af3abeb3aa617d5eec3aa590b4350c9be1ef5be45dc46312"),
    ("spleen-train-10", "spleen", "train", 9541, "bb7f5b1984c66a6276ed9a14f4f0ecb7dadb2b864416337042f03edad73727bf"),
    ("spleen-train-11", "spleen", "train", 11522, "25458bc7669d0bdc33290360ea3f0617201253a0836232a7df859140c9d7d35c"),
    ("spleen-train-12", "spleen", "train", 11572, "a1615564fc3f5840dc22e11a3780301746daae45d39955a472087a7b12e1dfcd"),
    ("spleen-train-13", "spleen", "train", 12795, "8f7bcff49b200f1ac284c05fd619e407df710c87fe32d86abd28ad953eb72faf"),
    ("spleen-train-14", "spleen", "train", 13180, "a3c16dfb9c1b43247023badf5be58179c53616f3aa9fb0e1254e77320a33113b"),
    ("spleen-train-15", "spleen", "train", 13418, "6c65c8b1a5fd4b969e31ec22de839724d9865b179df06a6c32552dde25536522"),
    ("spleen-train-16", "spleen", "train", 16646, "c4fbb2d7d4c474013e0f30df3ac4dd3881c0b598670fcd74dc9bad1a2f99e668"),
    ("spleen-train-17", "spleen", "train", 17578, "4f2640a7c2bb53971d3b4d58f5b3c4a1d48ebfccf0b72ce2c5051665d679b834"),
    ("spleen-train-18", "spleen", "train", 18139, "dfaaed010ccd1bcf041c27548a6d00516a575b0522dd4a07e2fae9dffd1ab81e"),
    ("spleen-train-19", "spleen", "train", 19339, "2aae05f4d1ae3c03f5587377d4e3828a46f06ea3a8df20b85333e0bcf4940d25"),
    ("spleen-train-20", "spleen", "train", 20519, "d69ffd5279f2cfb2c9606b2fc94406258bc752f9bfdc54f3b1b78ada97a67070"),
    ("spleen-train-21", "spleen", "train", 20662, "8b57a12de5226ce402c29e0121d65f5d3aad76c410d2105ef49adbb2c43aa1d9"),
    ("spleen-train-22", "spleen", "train", 22674, "cf9d3ce5633185b224ff1617a5ee650a24ff09edb50ef2b8a97cf176bdcf104f"),
    ("spleen-train-23", "spleen", "train", 24616, "5b2d51a08ca23c75857865473d9aa371a8fa81554f993493b999ed2a3696840b"),
    ("spleen-train-24", "spleen", "train", 24866, "08da5f5aa3a26549e7d48c4da3fe0e6beb4c0922e8e4a111863ee3c6d7915222"),
    ("spleen-train-25", "spleen", "train", 25299, "859c65621333a8914be66791fe80eb3fb78a59e4a816e730585cde53dc028222"),
    ("spleen-train-26", "spleen", "train", 25839, "fa5ce913c4cc9cf85a2eec56231971120f533d9e9c088c42771f75d3ba748ac2"),
    ("spleen-train-27", "spleen", "train", 26316, "0a315efe97583a7b7b197abc9555ca8247f1f8344a043208e9be0dc40bbe1748"),
    ("spleen-train-28", "spleen", "train", 29115, "914ae19ebc865260d541a082d962bbe762fe57f9469869e047fb70670065a3de"),
    ("spleen-train-29", "spleen", "train", 30031, "6022e31f51d22779742a5a8f39cc2055faa9e7ed46cc96019efadadf9609adf7"),
    ("spleen-train-30", "spleen", "train", 30732, "e75d5ed6f8be1a45c861413450f731972ba17bee9f033ef5bcdc43284c4a5232"),
    ("spleen-train-31", "spleen", "train", 32817, "15d888101791581468c8b3579a8fb501180daa4f92ed426c0fe218abacecd1c4"),
    ("spleen-train-32", "spleen", "train", 32875, "dc7e606efab848e88c4d2d8f098d970f8927223d02b2a09756370373bf7d8582"),
    ("spleen-train-33", "spleen", "train", 33008, "f751a01b1a525c1c95c599ea22c2755e75abdffe99f38fe87a8d26785a5fdf75"),
    ("spleen-train-34", "spleen", "train", 33239, "41f71c9f4fe9fe8bc57ae67e662b6fe2f51982f8f97b3ccda69131ff72c10a07"),
    ("spleen-train-35", "spleen", "train", 33795, "93dade1a022560474a2125e95446dd429cd7f99c07c8db3a78d38821c933db68"),
    ("bladder-val-00", "bladder", "val", 272, "be7cbc3364d24f770a8e03827697de892d7c5bacf063cedd7fceb459c83fb606"),
    ("bladder-val-01", "bladder", "val", 409, "e8d62dd4ab85deb9b3cb19c29f4b9c8c55149bc03949d613e3eee583889c291e"),
    ("bladder-val-02", "bladder", "val", 631, "a151749806e136fa698fb339dc237ef9c4c40500ef89f18149ccf187193b2137"),
    ("bladder-val-03", "bladder", "val", 2199, "59e3bea10288a11e60751aa9a75a340f376dcb22f8c77de3a39172a1bbc852a6"),
    ("bladder-val-04", "bladder", "val", 2768, "a22970620d90c7f55d6f22672a2e6f1b0e0b30da2cca9d61576f11cab73b98e3"),
    ("bladder-val-05", "bladder", "val", 3459, "916cb18ed09931657b36b8b27a1250ef193f075785972cc3259b9772f9b660c4"),
    ("bladder-val-06", "bladder", "val", 5289, "f838a015b7f840a9cc59b9cef083a9b21241582ba344fffaaf70331e1eec2ab0"),
    ("bladder-val-07", "bladder", "val", 5659, "ee64d228b32b43d5563df751dc2629063294f2d00565f79a615f96ab1ff0f287"),
    ("femur-left-val-00", "femur-left", "val", 177, "49e750c8182eecab708a40e0153f91eb11d26c416bfe4d50ee87b6fdfbdee324"),
    ("femur-left-val-01", "femur-left", "val", 795, "d394c9a633f0dccb2b00381d13024a46e0c389bcb4c855e7668165a66912f2e6"),
    ("femur-left-val-02", "femur-left", "val", 1543, "93357bc49ecc33197ff0bfb3e32b51ddc7cbcb7dc49428aec93082f4abcb7a86"),
    ("femur-left-val-03", "femur-left", "val", 3015, "df3106b681107b1c01271537646a7a7394ad1d6348d7490585c5ea12b35c6407"),
    ("femur-left-val-04", "femur-left", "val", 4223, "17f872f9396a7a669be87ccbe787fcb8f9fd7f22afd07e08ff8e707a406083aa"),
    ("femur-left-val-05", "femur-left", "val", 4723, "9a5f0e314e0cc44fbf61c94815da1239415d4c5fd349a8119036d8c9924dbda6"),
    ("femur-left-val-06", "femur-left", "val", 5118, "96cac67a2fd68259c317304664be94fd0ed2e7c8a4138f9a58238144e68b05ae"),
    ("femur-left-val-07", "femur-left", "val", 5776, "9ceb1a2fa6bb1b628931e465da36ad59631f668b4a915ea0d0ba7d578b27068e"),
    ("femur-right-val-00", "femur-right", "val", 1339, "d389e21a91ebd51d07d6166edcbd00d0d4d67464ca9e339f4b1b2b4f5844c38c"),
    ("femur-right-val-01", "femur-right", "val", 1430, "636bd175a87c6d988b1404c0f24cc12a737e7ca608a5d49956e1ee02ec9a3a36"),
    ("femur-right-val-02", "femur-right", "val", 2284, "8c3360c6c332591cd2fa5d31e6eac33c7a156bd97ede94d82d54ed4fa40b5fa6"),
    ("femur-right-val-03", "femur-right", "val", 2326, "0d37fa4e618444794f967a9d6d62e3947545a08bd8787ea78776e32a81168cca"),
    ("femur-right-val-04", "femur-right", "val", 2350, "71b071bd81d98c4e6480665e5276134d3df37f7efaae960b201413dcab7b6772"),
    ("femur-right-val-05", "femur-right", "val", 2835, "8e3eae34a8f080a7cfdc099e960c1ff48d55e5b08da65cd473fce70ba24e4936"),
    ("femur-right-val-06", "femur-right", "val", 3146, "7bd1db0415d354099f98833fce81a97f0c7433424040bc3378ca38705fbbac3e"),
    ("femur-right-val-07", "femur-right", "val", 4121, "d50e389e1d0c194de3085f38159c637cfd521b6abb246e00b52d506deadd5a8e"),
    ("heart-val-00", "heart", "val", 298, "d37f7ad4da126937e7edde417f374891edc9c95d92620a2aecfb784328cc5b1d"),
    ("heart-val-01", "heart", "val", 397, "8b7ee27dc772ed65e9714c7635e0fe77a10d267b583716681097aa7e0de281f6"),
    ("heart-val-02", "heart", "val", 1292, "1664f4047e1fd0330a9743a8fcb6cca5533ba336383758ac7be7a1556b7afefc"),
    ("heart-val-03", "heart", "val", 2722, "639fb21b776195de29fbf86f33c3009cbf3cdfd84fc53279c4db2b1b6d3b9b95"),
    ("heart-val-04", "heart", "val", 3751, "0064e40d3fdc604f8b0ae32109b816474d12f14dd814df15d683f7871785da69"),
    ("heart-val-05", "heart", "val", 5180, "5553a09c12de9136c2db6f32898aab3c15ba02dd9497aa4a7ad73117bacc29fa"),
    ("heart-val-06", "heart", "val", 6025, "2ae75aed51652c797a9e8ac4f60dd34cf56b207a9d9834ff468a9bddb099a752"),
    ("heart-val-07", "heart", "val", 6066, "4b24787f07c26ca9f1c3820c7dd8ffbf3515ae8aa770c4425595ff59573c024d"),
    ("kidney-left-val-00", "kidney-left", "val", 1769, "dabfe9c8c7b8dc338499e56c342bfe0ae09f4e4d2f70f08d95ee22cbaa67e833"),
    ("kidney-left-val-01", "kidney-left", "val", 1989, "632d3fb3a1ddee7252ded42126073f59cfdca54123a436806832434f2c254fa3"),
    ("kidney-left-val-02", "kidney-left", "val", 2376, "41eb9168dfa9df078835c44451cfdee6f1e60c72c5c52237715918eed31e7b28"),
    ("kidney-left-val-03", "kidney-left", "val", 3592, "e364f4e7fc8298828323a54f4bd476308c9c0741841338d4c4e8443ed6e788a9"),
    ("kidney-left-val-04", "kidney-left", "val", 3700, "80838e0527c92334cd501893305d6f8a397c11bb9c58fc8bbb5f4ae6bb81c5c8"),
    ("kidney-left-val-05", "kidney-left", "val", 4276, "5da2621e92e5279bc3b64fbe3413636e7e0f6b3b641b5e3f27687401bb0bbfc3"),
    ("kidney-left-val-06", "kidney-left", "val", 5348, "f66c54bb34c855d460e01f74c9b2de6d9d02bcbfc363affeefd05b614eb17cce"),
    ("kidney-left-val-07", "kidney-left", "val", 5397, "a39828e3cad890eab7932f7153a901e4eece486a532c300b7fe012059f4e5aba"),
    ("kidney-right-val-00", "kidney-right", "val", 842, "698400ab1d21e08534a16149e09d0c7b29d5a0f4580b317c34b929d64c202a8d"),
    ("kidney-right-val-01", "kidney-right", "val", 2057, "1a95d735a771da38b63e5cc42cb3ae08a33bce318531a160c570f4a9bc897739"),
    ("kidney-right-val-02", "kidney-right", "val", 2400, "da5d9546a2d55d4278076d89009cf2ddce9e837d4ffc9ecc588c72a0fd17d4a5"),
    ("kidney-right-val-03", "kidney-right", "val", 4387, "4c4b1a464197126ea1aff609c12bd0262855f08e544e51050e18bbc014cafcbd"),
    ("kidney-right-val-04", "kidney-right", "val", 4618, "9e1f1d84f0c1de2b0a60d53cdd6ce9136455cafcca41982a34e1e20c43451dd5"),
    ("kidney-right-val-05", "kidney-right", "val", 5669, "aaee44b2f759114481f15e887e08d105d589e86b111abc0f71087d383df03929"),
    ("kidney-right-val-06", "kidney-right", "val", 6043, "e1ee636c24d332c35eec0871a9c72fd6cb747d3c075c3aa7fba30ea75a88056f"),
    ("kidney-right-val-07", "kidney-right", "val", 6397, "6414a20056a7261cc9bd94e80ebb7dcc49dfaf69d0447aed171759c3aa4bb6e6"),
    ("liver-val-00", "liver", "val", 99, "95ab8d6798d0d028b936679740995de3b96b33e1402ff80dbc392b9f8f6b17d3"),
    ("liver-val-01", "liver", "val", 1865, "a10533451a2e808be8afd692b17a984c492c265b46e3d1a27411eba8025727cc"),
    ("liver-val-02", "liver", "val", 1963, "2ef790251a6668c70efd54a3419c62394dbde6a357dd678aa3600c23af0d670a"),
    ("liver-val-03", "liver", "val", 2479, "530090070ed8252003a2ad2c7f6d53616721c8c376874a7c728ee437a6c42d7b"),
    ("liver-val-04", "liver", "val", 2488, "89f2bf279951af2fda7f2840f08c0686d46dbb0c94514b7d107d966e5c56f424"),
    ("liver-val-05", "liver", "val", 3885, "07f88e95bfc7f07331b50df3d036b8c43113fca5be5336d082ad2390756066ed"),
    ("liver-val-06", "liver", "val", 5617, "5e1637a2be0c4033304920d1e362f101effcb4834781a024a92f41b0b3f208cf"),
    ("liver-val-07", "liver", "val", 6466, "2c8c8edae417cf1645b1a19b53abfa7d523032f47b552e8b10fa3447fecb9e4d"),
    ("lung-left-val-00", "lung-left", "val", 543, "b818bee6b4dd71ac9622ca1487fed59641994180e36cb93c15cb57be69dca2df"),
    ("lung-left-val-01", "lung-left", "val", 1157, "9b24722d767a1929f25770d9c29a5c724dc98f013b04a7b0fc7ab27c81ccd0a0"),
    ("lung-left-val-02", "lung-left", "val", 1197, "21b1c6413aa885cb944c0206516e40e8e2eec7ca9389dec91ce5b8ec7d7d7dfa"),
    ("lung-left-val-03", "lung-left", "val", 1582, "86dc58b590f5aee5c238998576400487d5a0bec33f28e7f22645e417619d3ed6"),
    ("lung-left-val-04", "lung-left", "val", 1817, "e4bbfe223c09a7eb677bf9f411627796ee65f065485dcaca18c54477c513a0c6"),
    ("lung-left-val-05", "lung-left", "val", 4157, "d6a2405e8a4ab50860f8f59b5e77877c32ec202c016459157fac4f714d6bc57e"),
    ("lung-left-val-06", "lung-left", "val", 4944, "662c0ad1e6333530ee1714a5487bc0ea8908eb787a0c7ccd3619052486ffb05d"),
    ("lung-left-val-07", "lung-left", "val", 6108, "c90e1255fce219b1987f4fe78454aab61d8bea62169837db2bcaeb9f91d023de"),
    ("lung-right-val-00", "lung-right", "val", 610, "68ee5b224b039cc6d8b8ba846176bb294217b3347452c657b0d1137a51ecf03c"),
    ("lung-right-val-01", "lung-right", "val", 1358, "5d28620f4c5527ffb2dd00bae9d4929a7c4240e605bb7f92b67b4fb2adec8085"),
    ("lung-right-val-02", "lung-right", "val", 1606, "75346efd0c6672e6fff5a7cba82a88c00cbbb89160b02413a53355cb0d329927"),
    ("lung-right-val-03", "lung-right", "val", 2161, "68bf82fad98619fbb3178e339b943790d338936285ca00f463f2164f56ba194b"),
    ("lung-right-val-04", "lung-right", "val", 2454, "f7f01d82038ec89503d58477a75188ba0004994a13f708d8f61110ecf759b766"),
    ("lung-right-val-05", "lung-right", "val", 4296, "433bc93ee8720390dbeaaf9a6a078572ecbfc01b9421584a108b4c4c61af7821"),
    ("lung-right-val-06", "lung-right", "val", 5023, "e519625f91cfd0cb14a8d8bade7b1d506d209e71f6842707fbe91beb7634558c"),
    ("lung-right-val-07", "lung-right", "val", 6206, "cf25929f3d0bc1a7cbd52c7772ebb71c3fd483de0e4735ce073006eda912ff5d"),
    ("pancreas-val-00", "pancreas", "val", 495, "ee69a45b803872a44fc3955bba760b988106c1a38a2561667c908bf6395de31c"),
    ("pancreas-val-01", "pancreas", "val", 1443, "987beb60ac1a68aefbca245c24d7079fe030619ff9fe5d7ce6b470d45bc2f2df"),
    ("pancreas-val-02", "pancreas", "val", 2171, "e9005e70db81f9ee70a394508e0780d503a1621957b80be67447a21b33952502"),
    ("pancreas-val-03", "pancreas", "val", 2600, "7903e19ae8f9c0d331d724cb7cb45e33e4bd3c46d55cb39c56afce55cbbaf651"),
    ("pancreas-val-04", "pancreas", "val", 3196, "054e59de8868838c41bbc4adc85f463ccdc4b09589a2ff7858966d32062c46b9"),
    ("pancreas-val-05", "pancreas", "val", 3726, "c7a8d23b5746b2e07258fd81367bc938c8ce7b3ce5aa930b2cc8177c0d200232"),
    ("pancreas-val-06", "pancreas", "val", 3850, "5afb20d31deb1bfee56d7694c2fb87b30b32ca810b0191b904bf7c4860953bcd"),
    ("pancreas-val-07", "pancreas", "val", 5059, "73ce5387806387334b0de46ed5c824c273bb150c0905c1ea337a537338f0a59e"),
    ("spleen-val-00", "spleen", "val", 333, "bd28bdc21c690c68f47c289a9a74846b3faf27cf79c857ec3b793bc7876f13d6"),
    ("spleen-val-01", "spleen", "val", 723, "ed31a209b4eecfed1ecbdffd9271588c40a38f0a483f1d21fc56b7a4e3254989"),
    ("spleen-val-02", "spleen", "val", 782, "428b3fbe5681270523f9d44d80c2e9465c63ac43b235bb1db2ec040140290865"),
    ("spleen-val-03", "spleen", "val", 1278, "2515eacd17b7785ed9bc6f1eb3cfe15813ba89ecd368d5f2de675d58248e68e9"),
    ("spleen-val-04", "spleen", "val", 1770, "5654b0a2059ad237dd134a5aade104e18ebf1cc4166d1cf74a41f2346d505202"),
    ("spleen-val-05", "spleen", "val", 2781, "5f2fffc764ebfeb956a4e1c366348dbc31a0133e478c7dcc3722a7871268d557"),
    ("spleen-val-06", "spleen", "val", 4162, "4bc286ce3521717b895f4ed04d06af4d264f9d1f1f76ee941227d10214c45501"),
    ("spleen-val-07", "spleen", "val", 4799, "f418c25907b6273f20d4ca950fc04e51dacac3d163392016ff6723962ed47a3a"),
    ("bladder-test-00", "bladder", "test", 2017, "5ba1b06555f4534c68aff7048481c8d7d63984d31d8a907b1ac2b76e1dc31723"),
    ("bladder-test-01", "bladder", "test", 3354, "6780fda73d610e22738f6cfd5f3017fdc81ef9dcb9e81a89025ce4448d9b9160"),
    ("bladder-test-02", "bladder", "test", 3859, "5a421a82157235c8d2f4e6c7521e65e3f7b1c53d7ef96f6a1b96fcc7ebf62848"),
    ("bladder-test-03", "bladder", "test", 4378, "7d7a7b254b79f811f0c2e461219c95ae1011dcaf592d1aae5bc3dd3e87ec2c56"),
    ("bladder-test-04", "bladder", "test", 4884, "2142873b9cfa29978e4335618b30332a0bb989d82e8d7f508790d216ecbebdcd"),
    ("bladder-test-05", "bladder", "test", 5116, "c468a5d832f48bf5500fdf03edd5143d8e7b45ebce1dd17828dc00bec0ba6dd1"),
    ("bladder-test-06", "bladder", "test", 5307, "8c95a15ab2e378f2c020d5dbc7c4ecf535156123006b152e76f09a6acb8b163c"),
    ("bladder-test-07", "bladder", "test", 6835, "b4aa991076053b3cc374c4612be74737c673732bf372838a1bd7bc87f75d29d8"),
    ("bladder-test-08", "bladder", "test", 7051, "34916226096a1fa1a4bd9253b238ca313b22c1b7e58c54ae7f52766914d8b463"),
    ("bladder-test-09", "bladder", "test", 7652, "a30830b3dde9ba7d5fb59f0c8bdc34127613f7e1947b6e63c67d3ad95a8997d6"),
    ("bladder-test-10", "bladder", "test", 8303, "ac50477b238c7a80419228c02ca8e6e836f9edbb3295c9bad7f04f0d6a061e01"),
    ("bladder-test-11", "bladder", "test", 8869, "8a108ce0608988d6ae878b60550ff834a65615691402a244d71f62ecd9ca7691"),
    ("bladder-test-12", "bladder", "test", 10893, "9ccf8e54422524ed67743261b3c850d2ad4427975ab88f77646bcc7367dd10dd"),
    ("bladder-test-13", "bladder", "test", 11634, "88f193886b1e3e71ac2c37630609a9af3f9655c7c3ec1212ee513748fff130af"),
    ("bladder-test-14", "bladder", "test", 13671, "18dede073695fed22c8fbd1226eda2349731a1d2bb7acfa61277544bf00addce"),
    ("bladder-test-15", "bladder", "test", 15586, "d5a9d743830b156af77733d720ef3f9cd06fa2b19088118e79183ec42d89fe29"),
    ("femur-left-test-00", "femur-left", "test", 225, "f18dca9b433add028a6ec013e1e973345a1d7f9b5cf6b69ef3dd3521b63e64d0"),
    ("femur-left-test-01", "femur-left", "test", 334, "f2af5242205ce29e736d68956e74cfc727c16a014728b78c1825da3f8801bded"),
    ("femur-left-test-02", "femur-left", "test", 958, "931dce7f811bc51e32349ce9536257aaa6d523c03964269b800c06dfe29e2879"),
    ("femur-left-test-03", "femur-left", "test", 1058, "29e92778f1b34afabb83a49bc26dc4b00da6ed5a34c4b6f4a88320adac0da06f"),
    ("femur-left-test-04", "femur-left", "test", 1088, "1441c536387101dff9ac547c531303e9b75d0f6898dbfdc2de292a619f23fe5a"),
    ("femur-left-test-05", "femur-left", "test", 6261, "74ac5c144010cc2382112a1cfab8fb9fa1a13e8a05cdff2c368318b3c48a01fc"),
    ("femur-left-test-06", "femur-left", "test", 6317, "a5306ae61e280653db3db3fae1dfc9bbb81015f9a74ac761c9243f45152a03d2"),
    ("femur-left-test-07", "femur-left", "test", 6587, "be90c442cbd9860fd1b2ad52da2740a87668dfaa70b030adde692043fdb2fea4"),
    ("femur-left-test-08", "femur-left", "test", 8174, "815c5fa13ee7a409342ec509c5b1cefbcf0193165734b21313151f9f92923724"),
    ("femur-left-test-09", "femur-left", "test", 9230, "e023c4db98ad17f48005294572e3a3169d294589da0343d91e88684c0b599b27"),
    ("femur-left-test-10", "femur-left", "test", 9283, "a079542b7767f953942230f294502419bfbdfe7fd510f8b5b5222c369202d973"),
    ("femur-left-test-11", "femur-left", "test", 10410, "7e82253d7421e5461f21d9fd2e029a37450e7a34748bf9d418eabf2053da5378"),
    ("femur-left-test-12", "femur-left", "test", 12439, "8444231810afb352bd82e29c2f5f46ec67901913b21c99765ff9c3e6fd300a60"),
    ("femur-left-test-13", "femur-left", "test", 13462, "80c5f3fbc9cc7ea1ce16169946c2d52ecfe8baccfe3c42420bc7a448ff5697a4"),
    ("femur-left-test-14", "femur-left", "test", 14271, "b049799288ead381782e8e36e8e2e4623533d6a21ec05a2fcdb20272839aead4"),
    ("femur-left-test-15", "femur-left", "test", 15269, "a549d7c7cfac61c34871da4c132b3331691a16d5916e3726be35b469fcd16471"),
    ("femur-right-test-00", "femur-right", "test", 1834, "8f8432d384bb1434336063cf46ecc953e66351f39e8f5c8a34c5da1aace01896"),
    ("femur-right-test-01", "femur-right", "test", 2310, "32924d648271b40f29f85a60a61b86f1592c3bda40fe128223289c5f4cafc94a"),
    ("femur-right-test-02", "femur-right", "test", 2422, "fd751d13b6ff6f0ae26ad3d339fc59a4df47a44eed71b4da3df10b9131a6c860"),
    ("femur-right-test-03", "femur-right", "test", 3791, "df863d0eb593dd49828db35fb632115e6729941f0fe0a56b7190b2af2295170c"),
    ("femur-right-test-04", "femur-right", "test", 4539, "fbb0d149ed5465e2b74685fb745fc844d7e7ed8d76ec12b6e16bf3417c7f6dad"),
    ("femur-right-test-05", "femur-right", "test", 4596, "5f8e799a758723f9f53e3cb21db7df275ecce6c293a8a06afca0e7d099561517"),
    ("femur-right-test-06", "femur-right", "test", 6015, "616dc8979d0c8156c474acc8ba011fb912b89a033cc8ddc168c9d9646e13216b"),
    ("femur-right-test-07", "femur-right", "test", 8092, "953e1af24f265e751937c33ae4ab5adafbc8f7cb48e9fdd7b4638ca6a4b20667"),
    ("femur-right-test-08", "femur-right", "test", 8691, "8a91ef0a0434a6c46b71f38ee0666d7bf7b947e0fa6cbb2f7e19240dcd5a9017"),
    ("femur-right-test-09", "femur-right", "test", 9643, "70b0e726063fe65b3429ff5d58b3f70b873e53294caa5bf9e29ac031f0537f73"),
    ("femur-right-test-10", "femur-right", "test", 10166, "1329f1527de51002bc96635da8ac480b05ca1b7c1295fcdab5fcee32eacedfb8"),
    ("femur-right-test-11", "femur-right", "test", 11275, "6ae99ed931cef7824148a8eda4a123f33dca0c70c3084de44a7272dccf340ef0"),
    ("femur-right-test-12", "femur-right", "test", 14450, "d6f3581d1f2bd7e05d7bd7ddbefeb3c815b8d0a90ce0f001d7e47bc1e384529b"),
    ("femur-right-test-13", "femur-right", "test", 16700, "ce5c321510519baf8a3e9b28ac592a26f58be1378abf6dccb7a83b1321b5f36b"),
    ("femur-right-test-14", "femur-right", "test", 16994, "980dedc7b1b3b800249eaae6488cc989dc5b2a950847d05ab29af1580f4511b4"),
    ("femur-right-test-15", "femur-right", "test", 17170, "95bf9c3d85efda3675afc41a0c882e2e07ad11ebaac868389eccaee894ad87b0"),
    ("heart-test-00", "heart", "test", 81, "fbc91747ab60dbf2ecee9d61f03ed2196509c18f2c3764e031d34cbec312a4e0"),
    ("heart-test-01", "heart", "test", 1013, "80d92d5b4418d1030d8b396118c47bc6cfefa2dd66e2ab1c5a3cb99b0d251fbd"),
    ("heart-test-02", "heart", "test", 1425, "8e69b83552e015553d12da1f462fad5b75e1a7511fd8fc645fde0a8d1632b348"),
    ("heart-test-03", "heart", "test", 2281, "5667d13d4c742857daac1557e2c98da74adc7c211f403eea981d48f46a7316e0"),
    ("heart-test-04", "heart", "test", 2689, "ce44e1210c83f732626037e6e94c3330c84aa7a85710611776cba38288edf7cf"),
    ("heart-test-05", "heart", "test", 3580, "53642d9c9c6205cb0867dbbd722bcd8e35c65e883873ff2065a499b076cdeda4"),
    ("heart-test-06", "heart", "test", 5070, "d19fd450b0b5b6ceb487cc22cf44dc6d00842806608f31a998dc7a173d95299e"),
    ("heart-test-07", "heart", "test", 6334, "e89794d75e56c808088598f31f5c24a1da4f85142d48ab089931809dd785db49"),
    ("heart-test-08", "heart", "test", 6973, "c9096e811b456d99b2fb4122e5b2f4e12dbd826ccdf5be40b0e238efa06b5ae3"),
    ("heart-test-09", "heart", "test", 12210, "3f815489ba60b748d9367099b6c2d2b7c4d0516ec41dea231f6129e21686f9c6"),
    ("heart-test-10", "heart", "test", 13037, "80c8cf3e77536044854aff0f5dc2dda9d5bd84c945525d1bf08010b63f500365"),
    ("heart-test-11", "heart", "test", 13366, "33c33e05fa86df2b2d1284090ecba9bfe64c70d3b933dbe52ab6a69c472ff6f4"),
    ("heart-test-12", "heart", "test", 15124, "68eea769bafb9adaed8640d4f3b423a31c3a28426b0a4d74a1c4f9492ddfdb85"),
    ("heart-test-13", "heart", "test", 16071, "48092b8fa77c254f3d8b9d08b56ccc39ca1bb4e6239631aa839f901ca2b4ca74"),
    ("heart-test-14", "heart", "test", 16427, "9cfa7ff56a7b9e90100daf332a3580eb31ed503ce5e1374824b4e6106619b900"),
    ("heart-test-15", "heart", "test", 16976, "448c93e0569eef158bfb42378d576d1ce3cb9de81f5fa7d3dc58119b3e54588f"),
    ("kidney-left-test-00", "kidney-left", "test", 1370, "8f3a19e8f621050357bcc73558e38125a3fd1452354965042a9e9d2afee91c24"),
    ("kidney-left-test-01", "kidney-left", "test", 1946, "517f159383549f26b1128a299f3d7fd8bf676527a049b4e771bb297e3686c9e2"),
    ("kidney-left-test-02", "kidney-left", "test", 2533, "cd79629e4eb71b2e78e7071b5433ebdc7237b5e193dd9ab48eea8f5bfff92421"),
    ("kidney-left-test-03", "kidney-left", "test", 2809, "e98df43517660c6eb189c018ec372a617b2756daaa715e3f3ec296cb1aca332c"),
    ("kidney-left-test-04", "kidney-left", "test", 2812, "fd2cc831bde6ec7e815bd6dbfff04ced414d556e8acc3248d28aaedf74579324"),
    ("kidney-left-test-05", "kidney-left", "test", 3584, "d91370952002b27dd35c83d5a1d947f1c8026a2694b99b3772750f965f7b0d38"),
    ("kidney-left-test-06", "kidney-left", "test", 4202, "e206997dada8ecd75c16a5559a56b477515fdf077f01e3da58c22d2284ae62cb"),
    ("kidney-left-test-07", "kidney-left", "test", 7419, "f7eada82b06af5a577d03377568f8dabcb8035e6420f20cd3462bdffe9e9e235"),
    ("kidney-left-test-08", "kidney-left", "test", 8278, "310f871c70592dedae393fd9c98728006fd2c4510d7d25df6622cb19c1b7f64e"),
    ("kidney-left-test-09", "kidney-left", "test", 8628, "cf37c9b9bce734fe5237004e42e679378d87bce0bdf3e09635c29bed979ed0d4"),
    ("kidney-left-test-10", "kidney-left", "test", 10520, "9a72161496f9d12bd80d3d55904eacf42676c8adc2939979a81435db6d56ce5a"),
    ("kidney-left-test-11", "kidney-left", "test", 10533, "0a9a320cbecfedd518efda621400d2b029624b42ec4b1e817e057662f6e753e9"),
    ("kidney-left-test-12", "kidney-left", "test", 10865, "ddf9c062a872beea7be721ffb2886fbf387c81fdc76075cb60b9dadd865b9ff6"),
    ("kidney-left-test-13", "kidney-left", "test", 11174, "c319dbd048964bec57cf8445b8788a3e13fc27f7302ce8e2f7797017d32d2179"),
    ("kidney-left-test-14", "kidney-left", "test", 11335, "789ff8e895022ae2c6db5351ef2096ebf97108dff0c06b4de8d3fd206e98f4a3"),
    ("kidney-left-test-15", "kidney-left", "test", 16005, "f5849e084e53c7f919003879bdff64e6aa9fe856d21f2015bdf0356bb76613ff"),
    ("kidney-right-test-00", "kidney-right", "test", 29, "4a0f967b0f450c94e154316cdf69ca2ecd32057f321493ee12c395774ba956a2"),
    ("kidney-right-test-01", "kidney-right", "test", 1334, "3f84f0df8eeebdefa29736b8267e7a27b1def696d1cede660ac020dfeb6f2d28"),
    ("kidney-right-test-02", "kidney-right", "test", 2856, "e34759466c2e722e8fca121fb989b1bdfe9128ca4352d99094f4ab41d966d1ce"),
    ("kidney-right-test-03", "kidney-right", "test", 3476, "b5388bd2862b19327f8920da1f286994f6f62047b4e594294357db713f21a2a4"),
    ("kidney-right-test-04", "kidney-right", "test", 5027, "2dca0cd8f439a45aa1757648d7eac0e6d9fe620cb14b6e3115cdcd17489149c1"),
    ("kidney-right-test-05", "kidney-right", "test", 5158, "4787157ea276f34cb88def1fe2a3ad7cd45a20860245247da962eb619b122fec"),
    ("kidney-right-test-06", "kidney-right", "test", 5201, "999a76945aaccdc56f65c30689a2061eb9e71fae06d033cd945deb54f31f1968"),
    ("kidney-right-test-07", "kidney-right", "test", 6264, "661823ca9f5f9fac3c9da819e5834ff208456518f154bd9846b1aa5b23cde3cc"),
    ("kidney-right-test-08", "kidney-right", "test", 8567, "5372f1dd60d81ae4fd3ae93405fe02e4e4d9f345f8bd1aae1996a27d4a6c36fa"),
    ("kidney-right-test-09", "kidney-right", "test", 10504, "45699316d445d4063b8648b3ab449c74a61520cb13cd6890d4f48ddc48deec9e"),
    ("kidney-right-test-10", "kidney-right", "test", 11920, "9057d990003bbe3325b742dc01063b901539d48b9b8304f432ea648093aafe25"),
    ("kidney-right-test-11", "kidney-right", "test", 14510, "7f7ad7e974ad1860fce1ade4a85e21d689e361b82b7c48caabb1d72dfa2b01b7"),
    ("kidney-right-test-12", "kidney-right", "test", 14668, "aabbbbf11ef88a32a9ac2762f29880c50522677e08c33b0a3ba437c983923f2b"),
    ("kidney-right-test-13", "kidney-right", "test", 15236, "673762368781d4b2dc0dda4b2a41613531299603b172b7b5d1968e6b544a5bbd"),
    ("kidney-right-test-14", "kidney-right", "test", 16260, "0d4a6d5ab4e8e17e5fd7ac0955326c2dcd02b7f1b8dea06c89d3ecd035efec5c"),
    ("kidney-right-test-15", "kidney-right", "test", 17373, "5e5fc7004cd3cae719fef923d7f6e9e4b83e8231abaeacda656e8f6ff04b2381"),
    ("liver-test-00", "liver", "test", 495, "0aac96f5c98a487580497746eb8275c2e90abbc8802af3b1ea138474a777e135"),
    ("liver-test-01", "liver", "test", 597, "0897b5b32ef6bf95e408b1c5905c995159f9d109563046c8daa507fed2a0d0eb"),
    ("liver-test-02", "liver", "test", 770, "ffb5e3e6c66bd06d455208123866cfa7e1db4e23c615fe5b08550741ccc499a3"),
    ("liver-test-03", "liver", "test", 1174, "d555765dcd9f87c018f0049c87b319e4a5f9c88b74afcf0bd1b664f72bfc1934"),
    ("liver-test-04", "liver", "test", 3830, "021d6e5e4894b7c09fe43ddb9905c52599e26626aa2fb550df9f22e0bfd35250"),
    ("liver-test-05", "liver", "test", 4003, "020b33f6e6792f61d60672ece090d908d525222615a6bd86f43d28f60a8fb7ad"),
    ("liver-test-06", "liver", "test", 4086, "20f2a316896c597eb82b8f091756305ae70daafb64781b57e91c5cb373674ed1"),
    ("liver-test-07", "liver", "test", 5770, "15bd23d23d2fd15a71d8d902eb7dfe8c9cc0f8a556ba4962593a17641e5b8899"),
    ("liver-test-08", "liver", "test", 8263, "edcb2aeba3cbc69f04050e5755e3198fa21ddf392d6106fdbfc0acd40298ab1c"),
    ("liver-test-09", "liver", "test", 10290, "2f4c7dfd587ff3b5d4c2e22ac67220757e2c7a01a48baa77b5889604355f4609"),
    ("liver-test-10", "liver", "test", 11692, "f018091f3d017514f7f68db1aaeeccc46bc2d5f0834944229101e25b946fd3ff"),
    ("liver-test-11", "liver", "test", 13230, "ca57ac75282776acecc0bf5b1c39aa9a4da18b18fe81df8743a0099d41035183"),
    ("liver-test-12", "liver", "test", 13309, "aa818688ef70d9f9ecc9c4ad7177d0f7856f54d73d5881ac754f5f63c547dd73"),
    ("liver-test-13", "liver", "test", 15817, "7507fa5049e57333839a2f35f957abba5eba031fd16b7a64c7d812a589f81acb"),
    ("liver-test-14", "liver", "test", 16022, "dbbaae659518ee0f4ef3bb076fe43a2625e531bd8b371851f3772a12f98d6456"),
    ("liver-test-15", "liver", "test", 16411, "4cbecdfe638207b58d4d6f3d9d9a21c23e5293b4974c3f8ad466acb8f78f8901"),
    ("lung-left-test-00", "lung-left", "test", 1463, "3ed26e81326a0c4c76898842504cc37516cde0d6dd5825223c6fd9860ad3a957"),
    ("lung-left-test-01", "lung-left", "test", 2598, "141af754ac6968a0a0aa0d9c45d14fa396dded386f82bb22a1b916ecb64a0692"),
    ("lung-left-test-02", "lung-left", "test", 4553, "04968c3d67a0fbd9763204b0f3bf11a7c6be0bd41527057ba8b431ad64f4d644"),
    ("lung-left-test-03", "lung-left", "test", 7092, "80e6029b4c1298dc2728674365c0aae1ad33e9a1a8b78f74eb477ea869657f93"),
    ("lung-left-test-04", "lung-left", "test", 7427, "b1121d9c9322aa972f600629cf7095eb1501ac677221da3e856be64bf79d011b"),
    ("lung-left-test-05", "lung-left", "test", 7625, "1bc434c366ddf76c82e4a917fd38f442ba633f9fc432f9125fd0189ef2305a8f"),
    ("lung-left-test-06", "lung-left", "test", 7879, "b9d6511a8891d3c8588bfedd644896492bbbc24f67e62371d1e906af71f6571f"),
    ("lung-left-test-07", "lung-left", "test", 7899, "a96ddc055e67bcf16f9765c75b03fb21c0eee16a94a7ca0b846fc244264cef0b"),
    ("lung-left-test-08", "lung-left", "test", 9144, "9ae2c1d81ccd8392962d5e10934786d57159f798ed7e0523644eabe32d410a65"),
    ("lung-left-test-09", "lung-left", "test", 11222, "de6da67cdeb45452dbd6fa47078d52a33ad4059649cdb914985f23f1ba959af9"),
    ("lung-left-test-10", "lung-left", "test", 13228, "54fd8bf19ed95d3fdc61a3cdc9f813f42850ac17654de9a4d79999f10f51c1e5"),
    ("lung-left-test-11", "lung-left", "test", 13718, "11c126b6792c2dc5882dc798cfa0c14c509b4da846544cab5770e2acdcf69593"),
    ("lung-left-test-12", "lung-left", "test", 14051, "3b4977a16201ab6a553e1377f0469117543cff73d241c2ee6fae70789a011fd9"),
    ("lung-left-test-13", "lung-left", "test", 14139, "65509282fda376aeb7a72656bf5fbd07afb2af017d32607d81960f7eb2a9d499"),
    ("lung-left-test-14", "lung-left", "test", 14273, "885512c96cf47c4959d686b7dd2693ce543bf66ff8ab7c5e09fc818603c6cd81"),
    ("lung-left-test-15", "lung-left", "test", 15162, "6f71fa0edb0b3d59337e97a13cafc5a41dcea70dfc2e251992b6f0b99cd2bdc3"),
    ("lung-right-test-00", "lung-right", "test", 1288, "a89865523f52ed268e94d4b03eb847ef84e5aa3d04e052b0a0327821e8d43273"),
    ("lung-right-test-01", "lung-right", "test", 1791, "179d552082c1e4aa879cb8911132d6e3d90fd9b502907f3758f2ad494e486fb4"),
    ("lung-right-test-02", "lung-right", "test", 2071, "95a4a61cb2e2c3ae694ee80f8a03d36ddd3d01af8ba623b06cad7ab4d1b9b393"),
    ("lung-right-test-03", "lung-right", "test", 2935, "52d3aae46c1dfd3d1f813cf519b8f6143cf78d839b58dc094539c3fcdc638925"),
    ("lung-right-test-04", "lung-right", "test", 4018, "44e8883aa648d668a1f2428618f6a5ac5c224d69d8905b66b81ed51a016a558c"),
    ("lung-right-test-05", "lung-right", "test", 4607, "786454078d0cdac23732fb781127e35dad67d6ba4746acefb4cfa2c22d6d593d"),
    ("lung-right-test-06", "lung-right", "test", 5106, "46725f79a24a052178613c2beaf33b250def8aaf6e1d2703e8a113cede07734f"),
    ("lung-right-test-07", "lung-right", "test", 5179, "60de9c9aa1b1114aeb434a5c2f5040d0cda73c0efede6dc696bf6d8235deebfd"),
    ("lung-right-test-08", "lung-right", "test", 8473, "8cb0bb109dac15d135ee0f1bd2fddb5be1f43fe985cfeb50f7c09d01b607f24b"),
    ("lung-right-test-09", "lung-right", "test", 9478, "ba75e8c2c341cbc7eabf3bea07d943cef93387b6aeed4455bd16fcfb2672cc0e"),
    ("lung-right-test-10", "lung-right", "test", 10099, "39aa8707a5f81b7587289551f773c748c380d1894f9c8c88d3cd2755305866a6"),
    ("lung-right-test-11", "lung-right", "test", 10918, "a6134802327a439c45acf5710235da15bc6579a918de528e84f151daa906f5e4"),
    ("lung-right-test-12", "lung-right", "test", 11565, "6e3b66dc74cd8d6dfa6fae13dd1ae3d8b8983c4e0f04118a413b85cc18b5946a"),
    ("lung-right-test-13", "lung-right", "test", 11909, "e1eb819f69a7cca33242fc976b3168e51a88e739c159bfb800544c683922d621"),
    ("lung-right-test-14", "lung-right", "test", 14686, "0abc579e1d87b584e55fa4cc6d508f4dfe8426b5c3402b5c7f93b1bed132fac2"),
    ("lung-right-test-15", "lung-right", "test", 16377, "39cdec8cd1e0f990ed7e3cbbcd9bde66fcdb4d8d57d33ac3546cbfa5018f6f55"),
    ("pancreas-test-00", "pancreas", "test", 1535, "9f0df8401e45c3d96a31ac54c7fafeb0b4bb1eb2933cfa8108e44e03fe07afdd"),
    ("pancreas-test-01", "pancreas", "test", 4055, "f9dff3d146432205023cc13d87dc09221733abb87a582630ef055e56a8bf039f"),
    ("pancreas-test-02", "pancreas", "test", 5497, "c92dc99c676948336120ee2a675bcf200d732a294a383d14e566deac2335b69b"),
    ("pancreas-test-03", "pancreas", "test", 5691, "c2035866b476dbac8f4dde0018d7019cf7281c2948a7a12fc0ae95eea361c502"),
    ("pancreas-test-04", "pancreas", "test", 5766, "ed04190bc876e7007e68213b3c4dc94a5d3b8ebbb65d49cc13d70d125a112141"),
    ("pancreas-test-05", "pancreas", "test", 5821, "1784132134d5d069a416c72880a05123facf941b465607648b32895ceabb689a"),
    ("pancreas-test-06", "pancreas", "test", 6088, "dd3212172b5c57ba7357920c9e0ac085365f27a7cd4498b1e452fadab510115b"),
    ("pancreas-test-07", "pancreas", "test", 12002, "13fd899098fe5ef968f67fe73b89e088c01334ee3826970cb7d3baeced8c2962"),
    ("pancreas-test-08", "pancreas", "test", 12304, "ce75f761e08a6a067e45578234c1db81a9c1b8a54c43cb407da629d39067c5b7"),
    ("pancreas-test-09", "pancreas", "test", 12481, "0bf23e3d4b80eca43752c3230be1cd4705eaa23fab02190dd6c0e49c1019574a"),
    ("pancreas-test-10", "pancreas", "test", 13619, "93f5563f21d401dfe6354cd287814982cec9c66cd79402075f29ce3f0502c57b"),
    ("pancreas-test-11", "pancreas", "test", 13857, "ecece3f428cae94babb31d05b70479f9c5ac9d9bfff8469b9d3c4b4c86f2d520"),
    ("pancreas-test-12", "pancreas", "test", 15169, "67ce0662e195b2d29cf17234609080ad19e76bcabae06ca615a70ee621901e0a"),
    ("pancreas-test-13", "pancreas", "test", 15282, "3082e6e3d7567ba350c42d261630bbbd0f14575ff0662fb368208ef893443faf"),
    ("pancreas-test-14", "pancreas", "test", 17038, "5eaa3deece4e5cb9d891aabb2709fbd13f9208bc00118445c9d3e4bf9c8427cf"),
    ("pancreas-test-15", "pancreas", "test", 17659, "321b72bd61e492971c53bd674fdc5385633c4a9d8bf29465407a631182b4f9f4"),
    ("spleen-test-00", "spleen", "test", 360, "e043b1dadcd5ba6f742fe2bf6e5c95c647f766e8c3c02cc133e5514f8813bfad"),
    ("spleen-test-01", "spleen", "test", 683, "4c8e1f968b9e3e7ce999ba0271ab77cf0fa2492734a954e7b47dfe89ff4e486d"),
    ("spleen-test-02", "spleen", "test", 3153, "965a168d4f6e88f7c948e20f66b7249b690cc154018b9d211ee4845eac62d4f1"),
    ("spleen-test-03", "spleen", "test", 3395, "8ac698d2e0799b5eb2aab6c7174e41d3623ba8e0f48ce16084abd1c926a41281"),
    ("spleen-test-04", "spleen", "test", 4038, "874f04936a5bfa63650a81beac8b92bc5e31e95102e070f7e44dd3bdf3cde27f"),
    ("spleen-test-05", "spleen", "test", 6159, "0f63909f7e0a53dd075a1b64a76874e57ff8ad29ce76e38409d3b21f9d9e8a0c"),
    ("spleen-test-06", "spleen", "test", 7293, "4d05ac979c12cc98016cad0fa1c0a6d0c7bb974e323fa8632b4c8aa0230721a2"),
    ("spleen-test-07", "spleen", "test", 7462, "441558429fe80397a415cc99394b70342d16dda305de364d44a8cc1d49e06aee"),
    ("spleen-test-08", "spleen", "test", 7870, "f92c9189f784f73b41938d1a89ad2dc8a95801c0a5e63e383f936c87e5dbbac3"),
    ("spleen-test-09", "spleen", "test", 8836, "2c764390333101026e8bf4e9a118315c39c0f3d56e26549c06f6f75dac8a63ce"),
    ("spleen-test-10", "spleen", "test", 9019, "b452f7a476dbf84676dbb506b9e31f380bb888ddaf6326cfd7ce3f4f113a51ea"),
    ("spleen-test-11", "spleen", "test", 10912, "e00a738d8aba446866efd3c1605eaec647541a5c52453bef5f3f6482981af7c5"),
    ("spleen-test-12", "spleen", "test", 12329, "7538e153e0684bea709b93518eeec60fd6f9c8e78af83d87b846e40dc6b701ba"),
    ("spleen-test-13", "spleen", "test", 12854, "2270044ed6f7403b65d910de95ae9c1ac70dd8d8bffa01d4d80936eeb777be95"),
    ("spleen-test-14", "spleen", "test", 15501, "d86c35868fec57b63fd4d4175dfa30dac020163e0de550e0a2e036efed8da136"),
    ("spleen-test-15", "spleen", "test", 16909, "06593001a6e1009c6902998db304065443f6f72ed53e17731b023c6eaf0b17c2"),
)
# Validation for the pre-registered grid and epoch selection (docs/pubmedclip-preregistration.md; owner ruling R71-s1 = A): 32 per organ
# drawn with VALIDATION_DRAW_SEED from the 6,403 official val slices never used before (the 88 earlier val pins are retired).
VALIDATION_RECORDS: tuple[tuple[str, str, str, int, str], ...] = (
    ("bladder-gridval-00", "bladder", "val", 220, "af513605198dcb2e633002f3b2b7710bfd9f36c02314bf4a4d03577e11098eb0"),
    ("bladder-gridval-01", "bladder", "val", 280, "66b2123704c6f61630e30d33bf9096ee87ac465cdb1a3fe9176ecfe23983f927"),
    ("bladder-gridval-02", "bladder", "val", 315, "7315287d480b31221da0eb5d26cef517745b31a9bd8b617b63bf657bea67d5da"),
    ("bladder-gridval-03", "bladder", "val", 361, "594f7c2d089e5be5aae1bb7cfa94f33907f0fdbb9916b0cd84947475b6c44f55"),
    ("bladder-gridval-04", "bladder", "val", 794, "a916eddca5c73234c12be4768194146f92cd4675102dd68180d28e536c047c1f"),
    ("bladder-gridval-05", "bladder", "val", 928, "1c9c43da406f31de783e5b2a58f8b9ac48564a9a6ae81dfc20840b5106fb98fa"),
    ("bladder-gridval-06", "bladder", "val", 1007, "9b6f1b504dc078e4c94a6ad49c43d00027e6fe32dee8ecb29d6ba6505a510f2d"),
    ("bladder-gridval-07", "bladder", "val", 1117, "d1f741703a021e231a0fae5369d1e4781698b08b1608b8bf5eb6d6bc8e1b977f"),
    ("bladder-gridval-08", "bladder", "val", 1131, "03a52f2b0ad176c2e0263b67d0158cdba327869bdd917ea8ff64d9e43d60bd27"),
    ("bladder-gridval-09", "bladder", "val", 1453, "485997fe0e32c418c6b7ca5adfa9133ce20e8fbd12bbd97696d6b35b3e4966a9"),
    ("bladder-gridval-10", "bladder", "val", 1571, "0474223eb5f7fb83162f285566d97de8c24f97cb39dbe95bcec621a8152c472b"),
    ("bladder-gridval-11", "bladder", "val", 1572, "9731a32ac36e50dc53d136c1fb633bff543cdd03b71e8fb899071a6a37190ab1"),
    ("bladder-gridval-12", "bladder", "val", 1833, "4cafaf4ff9c5308763941f53aae1e5a44ec8c09a34f4256bb2bb1be1af036bb3"),
    ("bladder-gridval-13", "bladder", "val", 2021, "beaa0b4354baa8fa8966f1e08b0bbb898d24d2cc3a8783deadf709c25bca5851"),
    ("bladder-gridval-14", "bladder", "val", 2212, "d01f3a8ebe446f31fb77d7d092dd1d762cf2e84bbff9a6b6f31fde9941d43ac5"),
    ("bladder-gridval-15", "bladder", "val", 2349, "a095ca16400acc432cfb697c13647436fffde61a776cb5b58994d5012a75c8e0"),
    ("bladder-gridval-16", "bladder", "val", 2670, "8a8e87b5ff91516c5a7669cb9278cfbc44503c6c8cfcc65765f59303d705052d"),
    ("bladder-gridval-17", "bladder", "val", 3102, "b6b526dbbf32cea41fcc08442d6f22c29dda70feb15baa21ea75f42b2a86d9be"),
    ("bladder-gridval-18", "bladder", "val", 3156, "cbca79828740139f07fc3615cee098905ef5c68c3de686aee1c53c0f88210c7d"),
    ("bladder-gridval-19", "bladder", "val", 3427, "5a3af4a2b55a3fd056b1a80a9db097ff3ad9348cb80b2d140a7f9cae9c42c14a"),
    ("bladder-gridval-20", "bladder", "val", 3799, "b88ecd28137cfde8f9506e4db72cff6c704a3d42d4cd72661a0415ce73e99f2f"),
    ("bladder-gridval-21", "bladder", "val", 3822, "333e0c1dae38a628249a305fc01e6cb5d7da951370cbd053ef2de7023dfac29f"),
    ("bladder-gridval-22", "bladder", "val", 3932, "7761ff01d20c4d53767d28d9420aac23f0b2ee22ee52c07d9d4bf1167bfa9664"),
    ("bladder-gridval-23", "bladder", "val", 4370, "8b619534b55611bc7e9dbb133a253a054deb5aeb2f4927d8cbd1e38c5dffb9e2"),
    ("bladder-gridval-24", "bladder", "val", 4434, "82a6908a1ee01ac26f47fc6b61156b4567be9184c0ff85f6540e0e7685fa41ec"),
    ("bladder-gridval-25", "bladder", "val", 4655, "fe536149de07eca62e3b886ced3871370ac147f25b114bd5d957c3d8b1728938"),
    ("bladder-gridval-26", "bladder", "val", 4692, "ef14894ca43f1b80eefa040530bdcb3bcdbca64b3761572699d23d002e08674a"),
    ("bladder-gridval-27", "bladder", "val", 4708, "854cbc908d9597446ae7f4eef3521770ea5fe734fd73606d887dae21082fe56f"),
    ("bladder-gridval-28", "bladder", "val", 5120, "6e7b8850af911e2a2d75b0b724bcc282d98c25ae8ff7076ae36f6076ba5fe3de"),
    ("bladder-gridval-29", "bladder", "val", 5325, "3e2aa25eaf235b363dd3f69f6aee9b8a5a49d7acc956cda6f2ec7615da21db5a"),
    ("bladder-gridval-30", "bladder", "val", 5896, "a57c379ba52c5d99391b5b42ebf2ddc84d1b47f1f8338c5579a98bdb1c888644"),
    ("bladder-gridval-31", "bladder", "val", 6359, "4d7f6ad290270e9eec258ff5e10414656b72cb6767f0628548a442b9ac56697d"),
    ("femur-left-gridval-00", "femur-left", "val", 26, "5a5b13e5956161214822f2de33c4a03a54246b90863081b411c2802641b485aa"),
    ("femur-left-gridval-01", "femur-left", "val", 36, "89aefabfe62d4e987ccf9accdc4c4bef5877d5b71c28477c8d6d27c69388963e"),
    ("femur-left-gridval-02", "femur-left", "val", 780, "ac79eb044530017d21a40090f87f0632c81b3d35a86e218ee9a4eaf19d5a7f61"),
    ("femur-left-gridval-03", "femur-left", "val", 1555, "a16c853be4d67f6df15e50b94fdaddfa3eeb32a1da5993bb33726f2cdcba439b"),
    ("femur-left-gridval-04", "femur-left", "val", 1706, "e250ecff6afe3ad2793bd3d15c83d1025b1462544b122594596766794d02ecc3"),
    ("femur-left-gridval-05", "femur-left", "val", 2143, "35cacc4669de5256b2fe2e44bddd93698860b7f354f6c6a19dea72ae342a0f54"),
    ("femur-left-gridval-06", "femur-left", "val", 2379, "293fef8152a80719967b7049db624d253ca4aeb1a33e0251547205e14cd9962d"),
    ("femur-left-gridval-07", "femur-left", "val", 2441, "64d6da93050a7e22108f1ec1d4cd10ba5a0889a6454ad39c8c13db30a1f47556"),
    ("femur-left-gridval-08", "femur-left", "val", 2550, "c07c557704671818c030da07e1e8e91b82df79b13868b92158722f6ed104c011"),
    ("femur-left-gridval-09", "femur-left", "val", 2564, "449d64329154ca96a6500f5bd638ab93ea79c62296e371956e0e69c840796637"),
    ("femur-left-gridval-10", "femur-left", "val", 2733, "144fc8c444afa76d8fb643d60584202cfc7828b3030b7e46b6d95fbf3651dfa1"),
    ("femur-left-gridval-11", "femur-left", "val", 2740, "a0e1c22cb1efaed5d4859e65acd70129b09934c16eb4b29a9e65fcf4c6d9786d"),
    ("femur-left-gridval-12", "femur-left", "val", 2943, "22addb3b6f789cd1db93f6c5df5dd85cdb178fe9a4bb8a47b6b79af48abf4a74"),
    ("femur-left-gridval-13", "femur-left", "val", 2988, "806f14941dbd9e5458e0aabfcb97a129db967ccb0264c263e49e082443a997ad"),
    ("femur-left-gridval-14", "femur-left", "val", 3107, "9a9548774ace48dc6344162d6af29e246134d5c0f79608debf42a01116ef75a6"),
    ("femur-left-gridval-15", "femur-left", "val", 3203, "1bf02573a5ca0d410568f49faef5af31a3dfcaa7612b5fa16d3f8562cd787d98"),
    ("femur-left-gridval-16", "femur-left", "val", 3682, "88cee29478af1cc149a62d654f402b45c6ede83aca1759c6c331f6241ff9ea8f"),
    ("femur-left-gridval-17", "femur-left", "val", 3877, "d8fef1c18976e85a8c23cc829fe01025054029821f86c2a96b98aa7c39ddc130"),
    ("femur-left-gridval-18", "femur-left", "val", 3924, "797d0ffb2073c17a58cd2def651ae0c662615ccb5a91ce2ec27d81a63f69fbda"),
    ("femur-left-gridval-19", "femur-left", "val", 3928, "e635b708b979cf1b9bab56ee74d5732ed677de8c361108d096e920c025720dfa"),
    ("femur-left-gridval-20", "femur-left", "val", 4207, "57f3150a05b53b54df72006bf028d9a08d960157d6b08c5ea3fdb39618707859"),
    ("femur-left-gridval-21", "femur-left", "val", 4282, "3767dedda862ac569b5625aad729c5eca92462bbd15e4421adb238f015304852"),
    ("femur-left-gridval-22", "femur-left", "val", 4324, "3571aa94834964d9ecce1fd3270b0cf748f1c7da1ca9dc28bccf66d8a1997fcc"),
    ("femur-left-gridval-23", "femur-left", "val", 5131, "f4282d505903c990ae3d7822c4faca09f323ffc2e10cf4f6402d2455016baf25"),
    ("femur-left-gridval-24", "femur-left", "val", 5146, "61be23134c8ac36a84932ad39e93bdc7f7808efd086d238476272b1d39106dfd"),
    ("femur-left-gridval-25", "femur-left", "val", 5206, "82189a7be55aab02eae24b1943025e8f69013d3d04d24e12baadb9b17f383898"),
    ("femur-left-gridval-26", "femur-left", "val", 5281, "6d1123555c076b647763a1a5ad96ebe48009fd61da331f871bc4e32fd800af1c"),
    ("femur-left-gridval-27", "femur-left", "val", 5704, "339a37eec803717b82c89dc495caab1c647ac75ecdc0574fe53fa6124f0764c8"),
    ("femur-left-gridval-28", "femur-left", "val", 5991, "7b0548278c26aa11a2977c806eced48fb7f8ac8de28dd4afffa5fb876e2fb8c7"),
    ("femur-left-gridval-29", "femur-left", "val", 6125, "57384c8d1aba8090b2fb090fd3c00de29dfa09933e45635f599e7b173165d89d"),
    ("femur-left-gridval-30", "femur-left", "val", 6329, "7dad724e844529181f77a165e4e1aeb3549ee72304525311607a6d9c9ea8fea7"),
    ("femur-left-gridval-31", "femur-left", "val", 6472, "1c59ff2c04bd5a348a0eeddb5b414353b44948b64fe81d0c39b44c1bf0513791"),
    ("femur-right-gridval-00", "femur-right", "val", 112, "af6e88bb39885d1d698f78723512119f6ca19d95d05446d0c8d04c9507464145"),
    ("femur-right-gridval-01", "femur-right", "val", 115, "fb3a0d7f71faa576c6a75714090756469eb0d68036c77d95ae83723436627af0"),
    ("femur-right-gridval-02", "femur-right", "val", 185, "5bbc443217c34dfdb7bc5ae75d4a6789042dc1af32e6827ab3d9612f06701cfa"),
    ("femur-right-gridval-03", "femur-right", "val", 550, "476b313a67599726434e27aea12bc04c7c9a44a3dddc16b9711080e77ba2b5c4"),
    ("femur-right-gridval-04", "femur-right", "val", 654, "6e6bb55d7c3a8bd4b63e58376dc404fa0a4ad5e7dbcef61db8d82a5a03b5d831"),
    ("femur-right-gridval-05", "femur-right", "val", 867, "cbfbbe860454a303b2915196410e706ac9ad89b4d6bd81ee553eaf0d406ea720"),
    ("femur-right-gridval-06", "femur-right", "val", 985, "c39942208ef7098b4c0cd6e3c837bd22df030bfa844c3d392ecf9c62106655cc"),
    ("femur-right-gridval-07", "femur-right", "val", 1665, "87661471c8ff1e49e9e0f58d61efcbb3e96cc7a695708d82e1fa0b80b0bb1340"),
    ("femur-right-gridval-08", "femur-right", "val", 1748, "962aa2fef8cfac72790a2e0ef5175734a384543bc9c66ad692d335abea80141f"),
    ("femur-right-gridval-09", "femur-right", "val", 1990, "1bb533296602f5f053603fbac9083a15de0c6a2e86cd9f292861c55feb53ba58"),
    ("femur-right-gridval-10", "femur-right", "val", 2098, "3b69f80d6002b9811e0abca9ecc2052d074ad4434fafa26e9bfea518038758b5"),
    ("femur-right-gridval-11", "femur-right", "val", 2475, "1a560c50394baa6a43a932519f5a72d894993f943498f835db7b803bdad1f2bf"),
    ("femur-right-gridval-12", "femur-right", "val", 2583, "c045e2b694063d39712fe91d199d77889fce8175ef61c10824c58af9f0df8fab"),
    ("femur-right-gridval-13", "femur-right", "val", 2662, "7fa07e41248a451c2746003db48bf83cbe5328d413d0a1dfb431030167a61f8e"),
    ("femur-right-gridval-14", "femur-right", "val", 2687, "0ba119d3c797612954a4432f572950f7b85a8191fe1393c3b6a477d7ad969bb6"),
    ("femur-right-gridval-15", "femur-right", "val", 3057, "59c367520925cd465365f86a8b4b687bcb586849a83698cd8e5901a55a06fbf4"),
    ("femur-right-gridval-16", "femur-right", "val", 3101, "1bdaca2c4d05de7ed874ed0e578151f886a05f6c3063b3063b9e181af872e225"),
    ("femur-right-gridval-17", "femur-right", "val", 3312, "be40a92ee1beb92984aac013d9c3a06d737052b80f928d56cc4bde9cb0cb18e3"),
    ("femur-right-gridval-18", "femur-right", "val", 3599, "cf291b5ac77cdca287c725c31f81610d93e67722d2cd3c704d82c7a92d4e352d"),
    ("femur-right-gridval-19", "femur-right", "val", 3719, "1a1016e7d54a890ce2ea0a16ecd8dbe08071c2d8f1e252bf528dc1a328bc66b3"),
    ("femur-right-gridval-20", "femur-right", "val", 4973, "5123884855469dece5103dd92573e61872b8957fb64b57545b9fed0bc409ac3e"),
    ("femur-right-gridval-21", "femur-right", "val", 5435, "b3c1d1ca8649905f0242767f53223a4c2fd11126beee24ef175541f3f967655d"),
    ("femur-right-gridval-22", "femur-right", "val", 5442, "f6a30b6e604f57daf65bb1ddbeade9d0a884a9c3ebf8778a40139c92b36e6473"),
    ("femur-right-gridval-23", "femur-right", "val", 5645, "2ebc6a51a3b197af594c1b304a4b12af7075b02a7c00421a30ae853cbffd44d4"),
    ("femur-right-gridval-24", "femur-right", "val", 5946, "53cc65f953cb593538ddb1b63660dd1edbf0cbca516f408af030d6fe1215b1c9"),
    ("femur-right-gridval-25", "femur-right", "val", 5962, "bf246c29d04e0088bc008d6341bee389b99a759637a0c24ce7d78ec4c571d584"),
    ("femur-right-gridval-26", "femur-right", "val", 6078, "d1d140492ad252d5229876f34e4342e214dc3c1f012461b9c1eba08b0131cd1b"),
    ("femur-right-gridval-27", "femur-right", "val", 6176, "04fd475ede843fb380924474803392339f5a716aed254d86cdc7bfe66883704c"),
    ("femur-right-gridval-28", "femur-right", "val", 6321, "6fe7c4542894d82a2962da3e4f4eda88ad64957dc47da73b0b2ba8a2d0ffcde5"),
    ("femur-right-gridval-29", "femur-right", "val", 6436, "66d038e4d8c57da35f0c4ba8e742ca57fdfa7660084b1366af68dc9c1384aa33"),
    ("femur-right-gridval-30", "femur-right", "val", 6467, "a60bb13cccc10e72f713d4a78bd8f202551440b5dd87c5823c64a4bd6bb07580"),
    ("femur-right-gridval-31", "femur-right", "val", 6490, "4b1ee2527e2d37b17b8c65cd87d7cf76a882af78ab99ba304505ac3c34867b1b"),
    ("heart-gridval-00", "heart", "val", 171, "6687ab08b2b9689eb4a443e130d7bb96b30dc1fa33f3f080f28cc1bd57309dd6"),
    ("heart-gridval-01", "heart", "val", 815, "e8602abd3180643cc5dad47aebc3f29336b40fde026f00f1dbbcc368456a5aca"),
    ("heart-gridval-02", "heart", "val", 1118, "7144af1db088461721df30c302d4e8c579572f3fa8f3daed0c343f26e8f35d5e"),
    ("heart-gridval-03", "heart", "val", 1890, "50e316a3a8e23d4156d3c251fcbf17a2b67f42532f636b5a95b1357b90437fdf"),
    ("heart-gridval-04", "heart", "val", 1941, "bb6d12b126017f1d71dc327e3cb98103f1b425ad31462d7d3aada0a32670d393"),
    ("heart-gridval-05", "heart", "val", 2096, "c59faa4c36bc45fd0207c8f3ee94f2b699fc02c40d0ead919f8f96789c0370d1"),
    ("heart-gridval-06", "heart", "val", 2425, "a52b6f72f313868613a9bf860f5c8bf7bcd80e743fbfcd88c991009ee5da6eb7"),
    ("heart-gridval-07", "heart", "val", 2726, "b2c38e5a00768c3cad7254c48ca043cb50c267d49d7f4dae9b5d26192f08c745"),
    ("heart-gridval-08", "heart", "val", 2868, "fe0be48cf693cbf59f7456330a89827666158e33237412eb5c8764224e9f53a5"),
    ("heart-gridval-09", "heart", "val", 3078, "a5b5007e193a5d188837bafc72b4225ad1f7a897c4fb01cebd3bb5e2aed454af"),
    ("heart-gridval-10", "heart", "val", 3142, "e3b9834223529d1b01ca078b17a41baf6564705a35d03af1ca2433bd385c020e"),
    ("heart-gridval-11", "heart", "val", 3419, "ac1b67835c9bfdfa18dd515aa8aba94e2fb617c0a7b58f23bb160d74ecc8605e"),
    ("heart-gridval-12", "heart", "val", 3563, "a084f514ba635b9fd8cdf0596b26faf708d18ac7c21a62cdcabf5b1bf670c27c"),
    ("heart-gridval-13", "heart", "val", 3654, "d7289d0329f1a84e357e55d2f697bffdb3061c11b69de286e1c6cdd95a13ecc8"),
    ("heart-gridval-14", "heart", "val", 3664, "5ffca6884c6a26467b953f540ef9088a2c6f69a39e39bfb3351819bc76f4ddef"),
    ("heart-gridval-15", "heart", "val", 3848, "ce8401522aeed9b237d93dd7000ff27a40328170a2155b3e92dc848cc1f0cb20"),
    ("heart-gridval-16", "heart", "val", 4025, "18f732b95fd8d383e1392bf540a7d40365146d459355e0b24e413ef078170340"),
    ("heart-gridval-17", "heart", "val", 4189, "5e474285d84a1f4bb1f73f93e2b8239eb45c63002d039884db5383875f4f8e12"),
    ("heart-gridval-18", "heart", "val", 4243, "c668b5f5016754644c43f1a99eb9cd93bbc3dd959daa8ce4b5c49d364112591b"),
    ("heart-gridval-19", "heart", "val", 4326, "0062d54d2aaf31b8c4b945986ec1d9c886213572ca22d7b8d1a3c035b9700820"),
    ("heart-gridval-20", "heart", "val", 4476, "aa22e9bcf3eb11295c9f1aa437eff01ac50fb51d52493c64999caa75354e83a7"),
    ("heart-gridval-21", "heart", "val", 4523, "e021c6b500a87e98524874aef5e07344d98a34534367d30047235d7d6801923b"),
    ("heart-gridval-22", "heart", "val", 4853, "8f90c5d2f432e94b7e2fb8cb99ca0b159d4b2bd3fca539148c13491f07fcfe40"),
    ("heart-gridval-23", "heart", "val", 4867, "655eff55b99650aea932f4c097fa2bd8aff4a7df17ccf1314e86760f18f00d99"),
    ("heart-gridval-24", "heart", "val", 5163, "e2c68140002ac9333b684bee09ea5ea418c33c6bf486bd8625cb562d06b99ed5"),
    ("heart-gridval-25", "heart", "val", 5413, "5febfd889ea3d7e19d53de2682044bbcf061a0169017509de0c7d580746bba4e"),
    ("heart-gridval-26", "heart", "val", 5423, "3508f1d649dc03fc06f772ed9f90807308f02e59abfefc1e0d53072607c087b5"),
    ("heart-gridval-27", "heart", "val", 5562, "483bd53c222e40472b00b2b12fad2e533b095f7d5fc34a92caf2894466607704"),
    ("heart-gridval-28", "heart", "val", 5581, "e7e68c40b5ce4256fca1ebfc8deead9b7059772358e8681984bec23bf53f1ed0"),
    ("heart-gridval-29", "heart", "val", 5594, "6a0348df8d32fdb8f3289f857ba475cfeedea103036fc70585bc9d97a27829a4"),
    ("heart-gridval-30", "heart", "val", 5615, "2373befaa5beba8f1a74bb6c94171f7ceb3cb63fab7eaa921d4e0a6247be7a37"),
    ("heart-gridval-31", "heart", "val", 6129, "95159d6d50bf43b7142a21021c45b37ac1f7a6812b4db969ee2548dd0d164bea"),
    ("kidney-left-gridval-00", "kidney-left", "val", 175, "b3dff921304ec4034f485b740c6ddbf0c3ea191f60f45ce255f4deab6522ba64"),
    ("kidney-left-gridval-01", "kidney-left", "val", 428, "ed437fe200922feaa11d27a59ca84168aeda3d460290218066a728ffeb6e30b1"),
    ("kidney-left-gridval-02", "kidney-left", "val", 789, "691695a0b99b0d48d6c8828cb6ca72e8cc8707ab19cc7a4bcacb1d4cad5762b2"),
    ("kidney-left-gridval-03", "kidney-left", "val", 1144, "28ec042751cacd657626066d961bee6bd9f0e8d02689570ae926a4b35ab538b2"),
    ("kidney-left-gridval-04", "kidney-left", "val", 1382, "255882bb4f3185ee018b0969e4fa5a5a4903bf5e8da5218a4c43b23ac5546327"),
    ("kidney-left-gridval-05", "kidney-left", "val", 1406, "a1014c8fbf5710e4eaf7912347ecde7edd20d2a414e81086c062f3b22f2386ae"),
    ("kidney-left-gridval-06", "kidney-left", "val", 1584, "4fccdd909ce5511663efd22b89a76a6746a145d03f8b9a9ca02a625c9c91c1d6"),
    ("kidney-left-gridval-07", "kidney-left", "val", 1700, "38189cf2159a5b4519eb90b43af5b356e78a9558c275ac3e40c0d96e32926759"),
    ("kidney-left-gridval-08", "kidney-left", "val", 1742, "ef8836061bea5547309211cbc6896e1d355c539b64ee5cbb3584adccf572ac68"),
    ("kidney-left-gridval-09", "kidney-left", "val", 2086, "309041fd6d0219f4301f87a5825c177571dc5f86c85e1a04479566e0b0500913"),
    ("kidney-left-gridval-10", "kidney-left", "val", 2091, "61261da961da0046dd0939d75cc8bdeedc863fcb0576a453278e04098feb7b9f"),
    ("kidney-left-gridval-11", "kidney-left", "val", 2264, "e2759deff456dc9b1ffb0bb3fac39e9b6e552eb27918ac0e0481c443cd0e802a"),
    ("kidney-left-gridval-12", "kidney-left", "val", 2347, "c2a94d4f771992d25e55dde259276c9e657aed7e4eefa60de4c8baefed781ce2"),
    ("kidney-left-gridval-13", "kidney-left", "val", 2640, "9833d63baa18959edde81e85333451d0a99f6e9463ab01cba422b89f1a24ab5a"),
    ("kidney-left-gridval-14", "kidney-left", "val", 2806, "6c1815ccd2d34b8a6a4153c85dbd1611443fb268b9465c303e62a99482ec3e23"),
    ("kidney-left-gridval-15", "kidney-left", "val", 2840, "bc4c6da4ad3703889b6f7a52e43b61aa8d8c8da76313d8f3739ac8a30068f47d"),
    ("kidney-left-gridval-16", "kidney-left", "val", 3037, "5578384536b79937971345de720234fd86f17c7464d24c79f7b4014b8afb3477"),
    ("kidney-left-gridval-17", "kidney-left", "val", 3058, "4a3aa81fb720eceee43ab65be30ba6ba945d45dc24cae4b45d7e781994d12916"),
    ("kidney-left-gridval-18", "kidney-left", "val", 3110, "01f450d98543813e053b0b84b7387d16263b0d9461fe6a4314da48d794af6869"),
    ("kidney-left-gridval-19", "kidney-left", "val", 3223, "ce2f50ea6e879687d3cbf28d22b436fbeca45c5042551db19e8a4108f404b249"),
    ("kidney-left-gridval-20", "kidney-left", "val", 3783, "cd34c52f9ca25e5f9435919a88fa87f5cb19cb431a103ccc9c887c71c57917e0"),
    ("kidney-left-gridval-21", "kidney-left", "val", 3934, "916fd7cae63e2bc0a74853e239f1483ac4a36b5402af4ad4b6dd11318272c404"),
    ("kidney-left-gridval-22", "kidney-left", "val", 3936, "9199ab7406a9efb01c59449ef3180cf57a46d5b433acdaf5c9e0bc7a40cc9d4c"),
    ("kidney-left-gridval-23", "kidney-left", "val", 4756, "d63bb4b9fcb80bfaba2d1145f918aa866b68619278668ba8e75189f3454aa806"),
    ("kidney-left-gridval-24", "kidney-left", "val", 4907, "2e4fb05865d28a79aba726f3046e57dd3736294d2c47c9def31ae01721ea2192"),
    ("kidney-left-gridval-25", "kidney-left", "val", 4971, "9036c25a6ae1298a97d9bd6ca03d49661121a5a4b81643e8b20278c9f62b2265"),
    ("kidney-left-gridval-26", "kidney-left", "val", 5123, "cee45b015ad86c19ea9f61d7104ece551efec25cada5ff80359087374233bbf5"),
    ("kidney-left-gridval-27", "kidney-left", "val", 5201, "92594edce10d392d097b1c5026f265132da76c6335d2ee7b800305a3cccda49f"),
    ("kidney-left-gridval-28", "kidney-left", "val", 5209, "5678b3331c2efbbe750395fe95539d7d91917a9f9f8c6347c86034252edcc7b8"),
    ("kidney-left-gridval-29", "kidney-left", "val", 5587, "ce033d7c9e5ba17fc07c3d73b0d31e6d87c4f95b998a6c006c04cbd98dec8e51"),
    ("kidney-left-gridval-30", "kidney-left", "val", 6048, "3c480827d2d592d579a5fb15aed3f4787ca728557189b890ed51c6ca7f93f405"),
    ("kidney-left-gridval-31", "kidney-left", "val", 6400, "ffc8d2eeff1ab6388e9abe2e1db8d80f0bfad8562169ffdb4ae718dff30570a6"),
    ("kidney-right-gridval-00", "kidney-right", "val", 721, "e9d1f2063a29bf296bd6aac06899eac666ff823d5ca883e07cb1bd0523502a08"),
    ("kidney-right-gridval-01", "kidney-right", "val", 1098, "4f022fe8469e7f69b5418e991ac0535b52c48840b036ab538afb4a1e80172715"),
    ("kidney-right-gridval-02", "kidney-right", "val", 1248, "919de1a254ff3ceef22464e57ea4769331beb7f72c8d043b58a8600b08d1ee23"),
    ("kidney-right-gridval-03", "kidney-right", "val", 1560, "f88665bb3ac8cb5e93125e4b8f9ffb9476f087348f9a2483f4ac7c896e38900a"),
    ("kidney-right-gridval-04", "kidney-right", "val", 1607, "c04e9b23e62660efaea26f91028e9b45943316fe909e761423b2778e87ee2da7"),
    ("kidney-right-gridval-05", "kidney-right", "val", 1629, "95fa4056394134933bb7fa3ce5cf01eb64967f90cba146bec244a785c67e4e76"),
    ("kidney-right-gridval-06", "kidney-right", "val", 1752, "48394616c60b5884b13adde636587ef37bea2f385b3bd7a70d9c573a5bebd375"),
    ("kidney-right-gridval-07", "kidney-right", "val", 1868, "ee86d7883081817ea1397573ff67dc3156252d1b67d35610916f2bef46834e94"),
    ("kidney-right-gridval-08", "kidney-right", "val", 2590, "5e7865fcbb40ccd2c70b38fd2d6b93ca669045dea7daf8a9a655656552fc64cb"),
    ("kidney-right-gridval-09", "kidney-right", "val", 2644, "2e98e25fef79a8bb34bc213f41116d98d8de54426feee8f4ddabe65aef21fe84"),
    ("kidney-right-gridval-10", "kidney-right", "val", 2744, "87f314840faecaebdbe94f124678aa17933fb34e625ecfc2217d368e4f529e9c"),
    ("kidney-right-gridval-11", "kidney-right", "val", 2912, "29389f153ad820f879ba46456e1d976ec5bafc8cffe9f35d9200e10676b5b8d1"),
    ("kidney-right-gridval-12", "kidney-right", "val", 3041, "dc1820558e9fcffd2c7c9430caf0cc47ae889c3177d6a6728ecefac59ba1f5f8"),
    ("kidney-right-gridval-13", "kidney-right", "val", 3184, "1f10a33be0ce1621a29ec1efcb95f265facaab538b8277ce0239847dcf5bd25f"),
    ("kidney-right-gridval-14", "kidney-right", "val", 3473, "32cade4a92b0ff4006c30f9c5c98bad6531876989627fc136d5e779de2ee1051"),
    ("kidney-right-gridval-15", "kidney-right", "val", 3503, "e38fb65d6c202bd1b38a494862e5aa53fa9f8b86530c5ff0fead503eb0091c37"),
    ("kidney-right-gridval-16", "kidney-right", "val", 3670, "a1960592108824db95ddf9dd90cc9cbc07eba958f565764142ae7f81ab95399b"),
    ("kidney-right-gridval-17", "kidney-right", "val", 3817, "cb708199ab871bdb4d66116eff74d4d400d219978f2d8f4ce025d9650dfd7d70"),
    ("kidney-right-gridval-18", "kidney-right", "val", 3876, "dace5558a02c981c0095295dd5969025faa648b5de9b52a99fd280cce7da889b"),
    ("kidney-right-gridval-19", "kidney-right", "val", 4277, "9bb1d9b1fa0d95489907d3a489b6e20ef831c86d2e68749cb5196166b1ae8607"),
    ("kidney-right-gridval-20", "kidney-right", "val", 4289, "524df37ba3514d9ad19bf92146ca53223b5ae1c13eb9d52b1c490b8bcc3f68f2"),
    ("kidney-right-gridval-21", "kidney-right", "val", 4405, "eb5aa701d95da9a91acd1e8f55a96604aabfe3e41994a651985bf3fe2f22ada2"),
    ("kidney-right-gridval-22", "kidney-right", "val", 4575, "e53ec62747a4e15714f8e739d16c92c4b8ac99e42367e03fbb365069071ee2b9"),
    ("kidney-right-gridval-23", "kidney-right", "val", 4657, "289e5f8c9f4b7851569815b59d8a4f0f2de712cb66a10eeade702cb1a3486736"),
    ("kidney-right-gridval-24", "kidney-right", "val", 4715, "80400eef680e7d7fe44197528b083cfcb2637e5de0ffbfa4674448e69b7ba85d"),
    ("kidney-right-gridval-25", "kidney-right", "val", 5199, "df463f0a75b4906c39bf4087a29a63b462cba860b24508926515b18734dd3582"),
    ("kidney-right-gridval-26", "kidney-right", "val", 5255, "8bfe2d8cacc6a32ae4d84063315bf7cd0c40841e17e5295742b2f4d989185457"),
    ("kidney-right-gridval-27", "kidney-right", "val", 5338, "d6560db1a7d37f176568beb2024d83d4ac36acfd89a6efceed1813529aef5741"),
    ("kidney-right-gridval-28", "kidney-right", "val", 5799, "484d116dff3ffebb37e88069471d050d49ca79321c995c662941b61576206537"),
    ("kidney-right-gridval-29", "kidney-right", "val", 5960, "53953ef5407e156978a17549bc19f89fa6fbfd34285dad38ea9097b44f78287c"),
    ("kidney-right-gridval-30", "kidney-right", "val", 6096, "5571d376d6bd7e56d27ec1f67f82e409165804ae4f2a95d16e24143bf1366f76"),
    ("kidney-right-gridval-31", "kidney-right", "val", 6200, "90f2a71f32b89e697be59d1893caeff9c7e2710f33158bc270ac2468ef9b5987"),
    ("liver-gridval-00", "liver", "val", 537, "5180c5405859220f63a7ab25c00d12cf90670c7c58eeaf16a5be986e0f8644e7"),
    ("liver-gridval-01", "liver", "val", 1125, "d56a98281eda921a422846471b3d3fbb95c2a66273b425d23cd0a1de88352699"),
    ("liver-gridval-02", "liver", "val", 1444, "61cb6cc635d7e562ffe55e654f87f0f2a18f97f1ee753d0c8d4e758f2fd1db1e"),
    ("liver-gridval-03", "liver", "val", 1640, "3b49af053af994627072edc56ddc006438c8f8bec8c2ace7cac18ae4697db0ae"),
    ("liver-gridval-04", "liver", "val", 1738, "a6a40f26b14e1af935ae4ee6a91e8bc701f1934ce4b7801bb5ef41f5c767ffae"),
    ("liver-gridval-05", "liver", "val", 1808, "43cc48478581f6ff21195b229298440e787dbdf2bb2e602520fd316bf6130cec"),
    ("liver-gridval-06", "liver", "val", 1811, "3d31a1f78f38315fd8a1f59c7889d552a2407f53949bb535bdce5b7307d676ff"),
    ("liver-gridval-07", "liver", "val", 1899, "37a770354f180fe88616f361999b99c2f7657c399f4d3f05e6f38448b83c218c"),
    ("liver-gridval-08", "liver", "val", 2382, "8140556e713f6297f4ce22205883a560be61bbf94e375d886c44f21574f8c6fc"),
    ("liver-gridval-09", "liver", "val", 2788, "86aa9ab8eb184379e78534f30ba504aa2c43914455c580bb4495122efc44b501"),
    ("liver-gridval-10", "liver", "val", 2819, "ae1749c8733feadb2920585160e29e4f5e35b18d3e7fa01c7055551b18d2e38f"),
    ("liver-gridval-11", "liver", "val", 3000, "14acccf6092ed3211bb25d9b8a1e433f99cb39caadd854ad611136d9490be646"),
    ("liver-gridval-12", "liver", "val", 3332, "e14ba8d46545bccee5e5718052d4fb96a4c816fa28ad1e7251071dac8e0ed1fe"),
    ("liver-gridval-13", "liver", "val", 3423, "2783dd7de856e4069d1a727ee57ac531a8ad712593da2b4eb82ddb8d59ab43bd"),
    ("liver-gridval-14", "liver", "val", 3481, "a3eea7be3bbd5cd3c9c98290c855cbb65828f30842b759aff50ba1ad9f94960f"),
    ("liver-gridval-15", "liver", "val", 3578, "2b449c64ca6fff30a32b1a70282242766634f9757746001877637afab23b7fb8"),
    ("liver-gridval-16", "liver", "val", 3789, "6fe2b7cdf56ddb3cb928bc6579f499276310b23ad321cd172b270ff0782cd07f"),
    ("liver-gridval-17", "liver", "val", 3874, "9c5e7e526f184fedf7dd8f622f5ad4eaac34689f3eb3879176d8466186e54a11"),
    ("liver-gridval-18", "liver", "val", 4178, "f4742278844e0742f1b30751854272ef96d7d9b8ed8539d459caec8e122f9845"),
    ("liver-gridval-19", "liver", "val", 4445, "bd5cf1cb1f2fa130e657ff4d1fc3b356fc6720a8a229480107157840800a5f8e"),
    ("liver-gridval-20", "liver", "val", 4704, "81ae8b39b728513f20b4896e134746cc2f273068f3bccde3f1606cf1fbc19804"),
    ("liver-gridval-21", "liver", "val", 4846, "5be81ba285c9129a02ac96b7eac3a6b6c908036d0b00101e4cf5ff23db7e4340"),
    ("liver-gridval-22", "liver", "val", 5047, "43e903afec3dec7b30117fdd3377a63fc4e13e28125d034c576dc79d152f6e56"),
    ("liver-gridval-23", "liver", "val", 5649, "24bf42b7f11da7cea3f4d7f16f3ec92b0d2144f3d88efbadf121247cf72ac3d0"),
    ("liver-gridval-24", "liver", "val", 5794, "195424ac4fb7a342e5dc08f2a2301290553ff913ec84ed73c22d07831dd862f8"),
    ("liver-gridval-25", "liver", "val", 6015, "7600bafc7507ac215c99f8b3c1228ad8d9fbfaa5020492e1564858cc1e0dfb4b"),
    ("liver-gridval-26", "liver", "val", 6273, "ace90236c23e0a6d9aaaa6c569d2d9f8014b603db7002238c20b10ab987681a9"),
    ("liver-gridval-27", "liver", "val", 6274, "51db01f3b7db39650f00b95848687a44b24502d2e0bc56d296d49015effd4722"),
    ("liver-gridval-28", "liver", "val", 6424, "205a649b8cde093f53d0c42b684c5d13ac5ab7d5cb026162a6b9870aac07481e"),
    ("liver-gridval-29", "liver", "val", 6429, "6ba01d68fb685afb075561691a689915b97af899a9b9d03c5ed2e0336bfb3b9d"),
    ("liver-gridval-30", "liver", "val", 6463, "478c0733b679de09e6cb24d12bcf5686e827386cf1949bdffcf5e6d7469eb416"),
    ("liver-gridval-31", "liver", "val", 6481, "af9135f8f226900e693d64a4ad3dfe7a430bc781f38ce59b7d50e06c4227fa3a"),
    ("lung-left-gridval-00", "lung-left", "val", 16, "d00159d0c4221f2c1819b723372e8b60b296a4413baf7ed8d322d9d11ad958ac"),
    ("lung-left-gridval-01", "lung-left", "val", 21, "235284dfb54ac86cfed0223527d778106a9f0b6cfd883764f508dea449ebf232"),
    ("lung-left-gridval-02", "lung-left", "val", 59, "4fee8c696842b91c8ec63034a5caaf9639c7dde29fc573bc3eab00ee9714ca9f"),
    ("lung-left-gridval-03", "lung-left", "val", 588, "9a69e48258363c508778c7a8273a196a91d80c0c92d3e17ac211d9d942623370"),
    ("lung-left-gridval-04", "lung-left", "val", 875, "e4e62bc4ae71274487570d48af904f06c003e6b2e8614198ae9f1d67c6ad2d23"),
    ("lung-left-gridval-05", "lung-left", "val", 891, "eb1ccd9589b3267f6b0c35d80d1591c43068ed613c494194111f76ed071eb276"),
    ("lung-left-gridval-06", "lung-left", "val", 998, "64d3c2c47c742e75f2413ee086d607d2e83e75ae7d2df898d60daa986ac091ed"),
    ("lung-left-gridval-07", "lung-left", "val", 1082, "e238af85799f6997bec04aa2601a71b81c1280ca5a36544cb8ce730f92f34baa"),
    ("lung-left-gridval-08", "lung-left", "val", 1484, "28572ff67b007b63e944e00a53e9b01df842a6146d47073876e06d7d5c077107"),
    ("lung-left-gridval-09", "lung-left", "val", 2304, "2e04a58ddf2bddac26cb9d74045d9aaf3b4d7678b5b5b3f9e047e94daeffff4c"),
    ("lung-left-gridval-10", "lung-left", "val", 2321, "a8acfed26747c699bbcf0488451fa4c148436c1e93b62bd27d8005cb8316a249"),
    ("lung-left-gridval-11", "lung-left", "val", 2497, "462344e8962cfacf1aa654ac24240a317f136dc5f72acaceb868d156dd932486"),
    ("lung-left-gridval-12", "lung-left", "val", 2789, "822759b38e7bc0ff60f6956dcb07464f69769b6f17a0400ea3e4c3e19e40863e"),
    ("lung-left-gridval-13", "lung-left", "val", 2898, "7295af653420ac820b5952286b753d85a54ad9d924ff846e46ed770b6beaf2bc"),
    ("lung-left-gridval-14", "lung-left", "val", 3355, "0fc930f463b585945fe67c3efe8c655fb2f1176984a12ea5fcdcd3462fc42de5"),
    ("lung-left-gridval-15", "lung-left", "val", 3679, "77b96b9518d31630dfb2365154cf79395f00e6cfce700a05c3662ff5642c9076"),
    ("lung-left-gridval-16", "lung-left", "val", 3740, "7cc4b6e9ac5a56eb6fca2892c247b9c12c4dfb41e9ff21c3f07ca4b85e15f2cc"),
    ("lung-left-gridval-17", "lung-left", "val", 4228, "6cb5a0015e0403112979f175b836f76dc6012be41c54534ab7dfa8d2d2de99d4"),
    ("lung-left-gridval-18", "lung-left", "val", 4262, "011bcc3f7e75e5d70d20aed00db9ff62ce15acc89b87015104cc14d109d85a0b"),
    ("lung-left-gridval-19", "lung-left", "val", 4268, "3096523270c5c24181ca6b72bfdecaabe9d5167975c532643d24886b2c0b3a64"),
    ("lung-left-gridval-20", "lung-left", "val", 4292, "93cb06007dbd8a4759abb1d5dde9b2cbd7fbade686e4e6210c29882e608a3500"),
    ("lung-left-gridval-21", "lung-left", "val", 4435, "ea387fce1ef6f589a1a44d5f066d03d3f7e6482b5508dfa5b636564efb61576c"),
    ("lung-left-gridval-22", "lung-left", "val", 4709, "69ea2f769499ed31bb55dbe2f668410a2a925c496a8913c453a5440d341fbb49"),
    ("lung-left-gridval-23", "lung-left", "val", 4755, "e9026e1523e83d86d7d316fc13f38f843343ea0146e471f21e2aa4656d7293fa"),
    ("lung-left-gridval-24", "lung-left", "val", 4771, "64daca01afe4ef235abe9c14dac76320ec25baf89a17fde721618db7849fc7ee"),
    ("lung-left-gridval-25", "lung-left", "val", 5069, "06ab8bf3a46b77476886df005faf962c750429e3ec18b3c558b52dd251eaf02c"),
    ("lung-left-gridval-26", "lung-left", "val", 5449, "3c42f0ee1a4670c44aa574483bcf42e4022d8d5b39f6effbefa91bc4352b3bb1"),
    ("lung-left-gridval-27", "lung-left", "val", 5461, "931882b398b81c168f094e447e7ffdaeec06a0208fbee8d6288ddef7e1065651"),
    ("lung-left-gridval-28", "lung-left", "val", 5572, "9932f7d88f58cae083d89b5ad8e972281ce6f9fb5e936cfdd13fa3eeae38f32a"),
    ("lung-left-gridval-29", "lung-left", "val", 5690, "897ca1efb49a44467ff1f794ab192f11511bad25b417049a13e9a296d243a00c"),
    ("lung-left-gridval-30", "lung-left", "val", 5757, "f8d5a5870570cf9a277e1f63e9b828db70c8d89ad6163163fd679252f8326b60"),
    ("lung-left-gridval-31", "lung-left", "val", 5952, "f53204054d9685ca78cafc14416e7077d9b830adf340f1429f3bdf1d41e500a0"),
    ("lung-right-gridval-00", "lung-right", "val", 235, "675aee484140468e6514832d4e4bb6a19cfc051330adb179903d4438653007eb"),
    ("lung-right-gridval-01", "lung-right", "val", 392, "fa09234f296056ad8395d152b0a29956625de20e663f78c165997354f5c81add"),
    ("lung-right-gridval-02", "lung-right", "val", 554, "83ca8e67fabf43b246721be14cfe2d01bed1af3c5c464891fa73959d8d4a6786"),
    ("lung-right-gridval-03", "lung-right", "val", 624, "5c0fea389a682ea5d3ebc98093d86bd77eccbd3d94e1005890f34302bd7d7996"),
    ("lung-right-gridval-04", "lung-right", "val", 1075, "7040ab14d8f26f588fcb4ebda9a0cf23ed31abbe3df11733aaa3fa06056aabe5"),
    ("lung-right-gridval-05", "lung-right", "val", 1115, "762134d46a6626f892fcf9d686941bed1c5b760916a4fdc8c6fd1534f30e56a6"),
    ("lung-right-gridval-06", "lung-right", "val", 1384, "05ca718106f7f15147b783d094d8892fb8644b2e9433e009922879dcbdc8325f"),
    ("lung-right-gridval-07", "lung-right", "val", 2014, "834a22ea35ffca7cfa1ad778867ec98fdba2dfa89cf13186a93f64ff5438eebd"),
    ("lung-right-gridval-08", "lung-right", "val", 2145, "76e72353c60854e0753b2896ac7465a3f59f02d95c1d24606fce6e11c95ed37b"),
    ("lung-right-gridval-09", "lung-right", "val", 2299, "9ef230027b83912a702660ca8da387bf105639f308f9f139401ea60adc87d5a9"),
    ("lung-right-gridval-10", "lung-right", "val", 2302, "5292c1fb485823202934cdce6dc42903acef8e024b35560abb246a987f2a4957"),
    ("lung-right-gridval-11", "lung-right", "val", 2374, "46954d0be3c19cf5b383d8f7577541b546d5219f02e5b7c8efd32494219256c9"),
    ("lung-right-gridval-12", "lung-right", "val", 2427, "2e8befbf77fc6d2de5e297104808cfaf9f264baa16421d8e85e9dbfed6fb092c"),
    ("lung-right-gridval-13", "lung-right", "val", 2743, "801fe19b0b25270363df8fa071ed739912264a58fa58ea2a101b90d628bef2e5"),
    ("lung-right-gridval-14", "lung-right", "val", 3192, "6e806e717ee68640f755ea65932a6b0cdc2b46c0d5b99c65810188a39de187b1"),
    ("lung-right-gridval-15", "lung-right", "val", 3236, "e5af0d024fda111e7539d169301ed5074e49d3df664fbfc13ddd239e1b84697c"),
    ("lung-right-gridval-16", "lung-right", "val", 3341, "fb45d04466740500f52bf73da1fe9e6340618a3ba2e8001de7f2e16365eb1d9a"),
    ("lung-right-gridval-17", "lung-right", "val", 3429, "349428eef9b3bda1357c13fc0529b45ce4218a3a9b6972046d8ea47a8f82cc45"),
    ("lung-right-gridval-18", "lung-right", "val", 3442, "03d4d27f83d57386ca7bf2ecd1423e485d0a52966b2622ca2ab60836bb98a400"),
    ("lung-right-gridval-19", "lung-right", "val", 3595, "7374743631fca49ce852d5f21cbb8211cb862d9e5efb161e6425c36ab14ed669"),
    ("lung-right-gridval-20", "lung-right", "val", 3620, "13de4083a76899ed617c4f0e3f34039b7020dfeebe65be4120aba7e9502e69ff"),
    ("lung-right-gridval-21", "lung-right", "val", 3678, "902ac35817cfd6f8ce71c18bee83264fb06aed737dc92474166d5930491895a7"),
    ("lung-right-gridval-22", "lung-right", "val", 3698, "f28b2189974494c7a8406de99c838cfc011f337c27d39f79e87f04c83aac8d38"),
    ("lung-right-gridval-23", "lung-right", "val", 3927, "53735c9db1c57a49d021d2ff364babbdd9c9bffe76250ec3a34fd1831cda1bc2"),
    ("lung-right-gridval-24", "lung-right", "val", 4354, "22c85d08210da57f645c347fc12d39e0614516696fb2eac54a320dbd63c514c9"),
    ("lung-right-gridval-25", "lung-right", "val", 5262, "cd001a586eef9f037ff29dbb470933ea8eba966ec8276316dbf35b2ed6e0153c"),
    ("lung-right-gridval-26", "lung-right", "val", 5277, "95a3cc97474214b574174256c0f40a3551fd6e9de916cd79cf30b83840cf5d2b"),
    ("lung-right-gridval-27", "lung-right", "val", 5367, "9fcfff785d9fad51117700ff5b29c30382a2f79df9ae8dba6be1d9faeefe12a5"),
    ("lung-right-gridval-28", "lung-right", "val", 5480, "110c1efcd9cf6457cf99b9c25cd76d1e070d6f6ed1632cef6ef36dcb4dcbb200"),
    ("lung-right-gridval-29", "lung-right", "val", 6112, "89a13c00a722548b37b1b3781b6cd6f9dc8e5a6dab50757c5a5dd5f692e60c64"),
    ("lung-right-gridval-30", "lung-right", "val", 6245, "3a79cecc3f3f624a1dddd6533c3af0f4d08d15f2296797436eae6a8922635102"),
    ("lung-right-gridval-31", "lung-right", "val", 6459, "a13b727db96f04eb64082afc5fae9b4a87c5d8412bb91ec28e70c900f2f0b6bd"),
    ("pancreas-gridval-00", "pancreas", "val", 22, "445b36e8170b056999fbfdf9a4d5974749f38e2b7066d0d2ff7e0abadfd4d479"),
    ("pancreas-gridval-01", "pancreas", "val", 23, "d5cae3fa8f93542ddc5b4ce12c01eb2937d0d2a03a1cbbbed91f06187b2ed223"),
    ("pancreas-gridval-02", "pancreas", "val", 64, "67f41b2ce71525355bb826a4f4bc89ef374c4b12e4297e7bf75e87ddc66d8cba"),
    ("pancreas-gridval-03", "pancreas", "val", 130, "7693035b0ee82e030d45fc74a193d851c47c65aa0e6e81e59723312d663ac843"),
    ("pancreas-gridval-04", "pancreas", "val", 226, "7d1824cdf43cdf9daa39923da773b58aa249953093e3e8bb2c68f3150866d138"),
    ("pancreas-gridval-05", "pancreas", "val", 294, "901c8e558f98c7bdd25b685067a14e6730639f15199657ee63b2e772d169ba5f"),
    ("pancreas-gridval-06", "pancreas", "val", 844, "e6d786debbb75721ea9a4af9aeeb1faac10ef1940b414746267ac451cef59c41"),
    ("pancreas-gridval-07", "pancreas", "val", 972, "8b0ae829379e70841861271450ef9f01918c0d5f8d688ab796cd5edb2f194360"),
    ("pancreas-gridval-08", "pancreas", "val", 975, "18fb119c11b928d4f096790c3d588b6f490401f610a563e77c67e3d809d71a4d"),
    ("pancreas-gridval-09", "pancreas", "val", 1303, "766ebb224d0896c93b1f18e8999e60ba3a5dea4521a74550264d5c46a8f46d44"),
    ("pancreas-gridval-10", "pancreas", "val", 1553, "53201754e7b8124c11d17624e2c44eb9ed36cf352894728744f050a1e0faf659"),
    ("pancreas-gridval-11", "pancreas", "val", 1644, "abd29f0fe463efd3d99985c654ddb89563fde3c0543661694f24867595e3fa5e"),
    ("pancreas-gridval-12", "pancreas", "val", 1691, "01bb5b6a694160827ef5268681fec34f13927951212c5025ae05e4cc3906c792"),
    ("pancreas-gridval-13", "pancreas", "val", 1755, "3612155e5200c1d71aa020f0d953b5614b3f629bb35f7afbb20825d4ea39a61a"),
    ("pancreas-gridval-14", "pancreas", "val", 1923, "b4f461e72cca76487c571b906cb77c44a9e9964f3ed4cf85aac343e9fbf6bce9"),
    ("pancreas-gridval-15", "pancreas", "val", 1987, "2ef67b109b501f64fb176f4352017a04137cb7ed0d2250c25429c20b24822772"),
    ("pancreas-gridval-16", "pancreas", "val", 2004, "b0c3d79935532b40ca01410a5784a58852147bf33f462356f715481329bd8e00"),
    ("pancreas-gridval-17", "pancreas", "val", 2045, "86d7a55abcbd238939749c640f9b74fd490a57cc6db26abd79375473cad7c76e"),
    ("pancreas-gridval-18", "pancreas", "val", 2180, "daf15f11e479a0a8277ea80fe7a9be88566fae904d107504ad2d40e39e1868b4"),
    ("pancreas-gridval-19", "pancreas", "val", 2184, "29d215f93089f9da5b1ed6a69a57ac77c62c700bbb1c399442fbd6abb495a14b"),
    ("pancreas-gridval-20", "pancreas", "val", 2778, "8751476103d17f95ea6f6fc73ccca3c64303e449ad7fb94065dc384a2e082aa4"),
    ("pancreas-gridval-21", "pancreas", "val", 3014, "3c322bb49a22400b7eea4d66d67f08d88d19f47dbf5bb0b4ad5f41efb5795076"),
    ("pancreas-gridval-22", "pancreas", "val", 3226, "3dae6408c8147c47b7b2bc1a4a8b14f7c0f54fd551338153d98a39c45a6c2ecf"),
    ("pancreas-gridval-23", "pancreas", "val", 3245, "d711845d1863796e930a959e9a902b6f873658268a68b76990bcbbf06e342f64"),
    ("pancreas-gridval-24", "pancreas", "val", 3424, "64c8199b26c29902beb712193af7cc02ae9d522f8edc17faa09cba5477ed8dcc"),
    ("pancreas-gridval-25", "pancreas", "val", 3660, "eb0d8247d09788267a25b07ae7552f63e70e371c395700024a4092c720bacf27"),
    ("pancreas-gridval-26", "pancreas", "val", 3771, "949d1ef8426941b11d08fd2a924531efced06b26114d452df160f35947a11f78"),
    ("pancreas-gridval-27", "pancreas", "val", 4501, "0c8ce1f9d196c626cae1b36c57c96cf2cf09f3604fdc02d261feecc58d7c3f90"),
    ("pancreas-gridval-28", "pancreas", "val", 5472, "f7aecdbdd97920d835058feaf2ad086ce79a02e7966e3ded3512cff922e0e017"),
    ("pancreas-gridval-29", "pancreas", "val", 5979, "c8d15bee582ea480393b011daff049cb5ed3ac0650dfcf39aaa372c9a479248c"),
    ("pancreas-gridval-30", "pancreas", "val", 6038, "185f101697ac01c25fd53cf75c85b9d5bd1484ee59fcff1ba18d7c1379b70559"),
    ("pancreas-gridval-31", "pancreas", "val", 6218, "ddcecb2a1b5879c15aa0d29421a5eb703aff1d4f8b0ba525c6795d3e69e54ea0"),
    ("spleen-gridval-00", "spleen", "val", 317, "086528b9c0c0a9dcd78a795b9a9cf207103975d5727a5041d040543943fb5b10"),
    ("spleen-gridval-01", "spleen", "val", 352, "ef3ca6d0aa642d1aeb0efef8abf3780a6ce1b403832c82957b7f5ff4086a4c75"),
    ("spleen-gridval-02", "spleen", "val", 1139, "161d2980d2c6ed93de0cb24d7b5ef1574d4260949a314e7b11355f178a3b2ed2"),
    ("spleen-gridval-03", "spleen", "val", 1632, "e3cbbe1905e787451cf03e0d3fb8d31da02a453ae11b936ccc7a8d68d3e413ca"),
    ("spleen-gridval-04", "spleen", "val", 1686, "a85ad55e8482b19337b31ac7d91ea1377a8f958f77fa7970651bd8a8ba73dbae"),
    ("spleen-gridval-05", "spleen", "val", 1851, "b66352017cce7e611bcedf1283947f58bcbc689e35a86b2074e12def6f6cd168"),
    ("spleen-gridval-06", "spleen", "val", 2102, "61ebdaf0cbc44316615648431fa4c9099fc823b1f356f97da939dfb9a25f4e9b"),
    ("spleen-gridval-07", "spleen", "val", 2391, "1836acd2fec3413140693e03ecb15b917293eb25aaeee813dfd94966494b9091"),
    ("spleen-gridval-08", "spleen", "val", 2445, "640291a5411d80cb310c9e5fcab234099e4f4cde86f40011e7a67f40b65514fa"),
    ("spleen-gridval-09", "spleen", "val", 2494, "92df0e5d07dc49cd74977adc2c3923dc5bff298fefab681954fc676a31c0cba2"),
    ("spleen-gridval-10", "spleen", "val", 2607, "5010cc116c40e8d9d16a01be069d2af17314e98328e79f30532d837ce7d064fe"),
    ("spleen-gridval-11", "spleen", "val", 2882, "c0b8808da584c6fa5be4e289e25691e28d2514ece5c073df99a876b3dc5687f6"),
    ("spleen-gridval-12", "spleen", "val", 3307, "d064b6c713b867c16bc8a72f38ea69d3529ea0284d5a3c8239042180948410ef"),
    ("spleen-gridval-13", "spleen", "val", 3455, "3e32de526d3b36b82866838fe84bdf3ec66987f16faeeb1fcf0328e896ccf022"),
    ("spleen-gridval-14", "spleen", "val", 3819, "f89b48e0d7830a1f4fe0b09a4744b7f6a1b3f6e143608bae0ea6a26d92744ef3"),
    ("spleen-gridval-15", "spleen", "val", 4216, "887ddacbd1b6fa986d037b4948b9789a4ad4ab4be87894a351bb5bada7055b2e"),
    ("spleen-gridval-16", "spleen", "val", 4461, "6eecf30893dd44ec6e29532a9d81c7792269de9ffaab040105be28d3f9f21b3a"),
    ("spleen-gridval-17", "spleen", "val", 4565, "d615bf5493d90a2afc784f3274a2610374d5c9775a2caf6608eede4f65dccdd0"),
    ("spleen-gridval-18", "spleen", "val", 4686, "af94493b60b75aa8110d0a57a7f0f5eb7c6e122410c798ac0c98adf9c94d2efe"),
    ("spleen-gridval-19", "spleen", "val", 4784, "26e0c852c17f0db4e5e705e71dbcc263e7ac29f247bcadba22b1bc60f6d4fb8d"),
    ("spleen-gridval-20", "spleen", "val", 4818, "6732665823aab596f3b2aaa8fe7023cc2ab549c304d037cfe51ce1d95737aadc"),
    ("spleen-gridval-21", "spleen", "val", 5020, "084e6526f8093cdd3dd38660a35f73f28fe8af895e7881313d21a57b8c1f552b"),
    ("spleen-gridval-22", "spleen", "val", 5026, "523cd7d6260da43906672ded3c8cd21e822d0644bf8840e012945f0830985bb0"),
    ("spleen-gridval-23", "spleen", "val", 5237, "0e2ac717fba85a8d95974f43fa7424232afd2b8018860490936292519e3d7ecb"),
    ("spleen-gridval-24", "spleen", "val", 5257, "e259eb4fc5ed300bbf43d5d167b85bd225e4edfbfadfc02a8511783e4c547a5a"),
    ("spleen-gridval-25", "spleen", "val", 5294, "c85167a4140acdb8d8e8157975595b789341328981a6c09cec878d5a426c91f4"),
    ("spleen-gridval-26", "spleen", "val", 5452, "2a5ec041349a78f5eea983099e6d1c9765746988537e7b470c6e01699eb0ea1e"),
    ("spleen-gridval-27", "spleen", "val", 5839, "264a62a9b49997f3903ac5e12209afa478a39c7f6483e1dfbc8ac0b615194e7c"),
    ("spleen-gridval-28", "spleen", "val", 5856, "031285ee91ef3016909c519ee5dc476f36a46677e9471f223c09db49710e9472"),
    ("spleen-gridval-29", "spleen", "val", 5921, "9b5f862ed0db670ae93bedb2ca788fb48796833c8b5eddcbf0f2c6144a862c01"),
    ("spleen-gridval-30", "spleen", "val", 6269, "8cc2e874eb81eb88382d3b12a1250731f3d6aa8939a5f7b1e9ea6278100c1ae6"),
    ("spleen-gridval-31", "spleen", "val", 6401, "f8de01d4c4ef69c0322bc5aa0d94d529e96935f371d46f6f03c1a9d6f955f2df"),
)
# The FRESH TEST: 16 per organ drawn with FRESH_TEST_DRAW_SEED from the 17,602 official test slices never scored, skipping
# duplicate-pixel slices and near-duplicates of the exposed 176 (the archive is shuffled, so index distance means nothing).
# Scored only under the pre-registered settings. It shares the 70 official test scans with the exposed development test.
FRESH_TEST_RECORDS: tuple[tuple[str, str, str, int, str], ...] = (
    ("bladder-fresh-00", "bladder", "test", 1085, "33c614c442b57f8a8f7d055c876cbf7dcf5f5d8f3ee14dc613e62e2d6aa3c280"),
    ("bladder-fresh-01", "bladder", "test", 5246, "65181488bd3c70a7ff315a84bfa7d70bb5dba145db75da4848170b92ef7289a9"),
    ("bladder-fresh-02", "bladder", "test", 7077, "d08c7306d7e54e511ba6f63e82df02ab405ddecb9650d8733ba24e78de9f801b"),
    ("bladder-fresh-03", "bladder", "test", 7081, "fe87b72c5cf361b96c70d3eaf24e1f3f026d5f806457088a4d49833c1f342b21"),
    ("bladder-fresh-04", "bladder", "test", 7739, "8bac5a3030c5b2b4f56194dc560b97811408bce9b7aa210306bc06e8d70e8d8e"),
    ("bladder-fresh-05", "bladder", "test", 8020, "1ab43da1e41b646d494f091936e95237d4072d0c0fd6f454cd87b10ff239489e"),
    ("bladder-fresh-06", "bladder", "test", 8458, "253cf9dbd67431f9015cb7e1aa0d263224a963e7920d4dcf40104221b0429ad8"),
    ("bladder-fresh-07", "bladder", "test", 9610, "992035261459a436b4095676f5a496df2b0b20f14f3ceaada8b25fc74f14a563"),
    ("bladder-fresh-08", "bladder", "test", 10172, "58d15ac0f8aa4136199e01482457eebd4915fb1d44d2384698327f0e8bf5e6f0"),
    ("bladder-fresh-09", "bladder", "test", 10559, "7cdb522221b199c671f71868873f4f78d08b76f8516a4def8e742c2fb7f95cac"),
    ("bladder-fresh-10", "bladder", "test", 10685, "adf8d1f5b62d18f5bbb8e6624e93b42dd84b38be830ed41b080c2437abfa3a2e"),
    ("bladder-fresh-11", "bladder", "test", 13244, "e893a7b75d5d38ad4eaebe0dfaab2fc29cf2da0a87bbb21857b49dc4b85c268d"),
    ("bladder-fresh-12", "bladder", "test", 15032, "d099c3aa2d92ec82b7e2e8b4e45106ba39444e9f1e22bf405fbd89ecdfcdcc66"),
    ("bladder-fresh-13", "bladder", "test", 15450, "d10f30d3224030da64931e095ec485107b95163011f37d36722205cd639c0123"),
    ("bladder-fresh-14", "bladder", "test", 15763, "dee31bb52481c6735a03328c121e04000c48e7f7fe58e9b6b82ceec9836e9a47"),
    ("bladder-fresh-15", "bladder", "test", 17037, "85586c65e99a7a38470cb66f8f5d3962330c16a33254ddbf952d157cbd0310ff"),
    ("femur-left-fresh-00", "femur-left", "test", 3639, "8cae42ae30926ae933a3434d91c6dedfb4df2722036be114143012ba08a85a95"),
    ("femur-left-fresh-01", "femur-left", "test", 3726, "0fed03e0e3a1869a0c0d6420fc26f6280a6f9e536f3e4fa8e38347e68c774815"),
    ("femur-left-fresh-02", "femur-left", "test", 4658, "3c1a524c6c54f387e861e381b1f2584fdfc2f7bfdb83c08490d52d346d5edf16"),
    ("femur-left-fresh-03", "femur-left", "test", 6110, "d22f3a9716939d2d34796985a0b2bbaa393d93307ac5b96e6bda423266af3a7f"),
    ("femur-left-fresh-04", "femur-left", "test", 6430, "f5252d37c3a386bfc9008ecf444c3b8eba49859e7cea05ad14d18d1eff89c8cc"),
    ("femur-left-fresh-05", "femur-left", "test", 6943, "59eea795cbc8045d3b0fc3347c0dc2b2060ec8f7de5af00cf7f6e5a9dcbd9252"),
    ("femur-left-fresh-06", "femur-left", "test", 7467, "bb32e5f40c8bb9f25478affee881043bf33fe2b9fdd49d5009c9ba22b4553fb7"),
    ("femur-left-fresh-07", "femur-left", "test", 9869, "9c9c519732a45e38d6910f9fb008ad1dbe7052f87f1a9c8c97d37bbd6be3b4c2"),
    ("femur-left-fresh-08", "femur-left", "test", 10007, "1dcd51d03f00adf2da993e6fb3913373ca70e65a30fc95899cc8f17c785e27cd"),
    ("femur-left-fresh-09", "femur-left", "test", 10162, "858c2c8ab5a9772a899eec2cfed245f75aaa5e171e01354b942002a0a9b18175"),
    ("femur-left-fresh-10", "femur-left", "test", 12837, "4839d250151bcf040575c43d93d3783cab0c247b1196bccad95ed5e0df237824"),
    ("femur-left-fresh-11", "femur-left", "test", 13039, "365bf1f8c314930bb8341236de17cc4a8eacb0711c0ceb5c298a84ae2a0bfc61"),
    ("femur-left-fresh-12", "femur-left", "test", 13283, "6fe09cdd70ac61c12f48801899da58c41f947a978712cb598a1e73fcdaa40c33"),
    ("femur-left-fresh-13", "femur-left", "test", 14060, "c4ee8ae968d74782061819eaef9a496103ce2905a8bac91d962854a1c0cd3bf2"),
    ("femur-left-fresh-14", "femur-left", "test", 15363, "9918c1100bd89abe25f35862961c93804efb1a55868fe63cfc6cfbada4e5d556"),
    ("femur-left-fresh-15", "femur-left", "test", 15984, "e9f3ed6fd647b2bcedbf5db7e22919d10d1fc6c337e7ba097134fe26ef861efc"),
    ("femur-right-fresh-00", "femur-right", "test", 230, "2acc44ccbc6b74e68f34fa339f1b4cdfe60c9e7a883023ef19222d4c71329d00"),
    ("femur-right-fresh-01", "femur-right", "test", 1312, "d56e89c741280b331e6086fa2c6b6074268016000072f45daed4595dad38a623"),
    ("femur-right-fresh-02", "femur-right", "test", 3240, "d933a64d0cf47ce2f7d113566f19db85f7facb8575954914a5564911c39db35b"),
    ("femur-right-fresh-03", "femur-right", "test", 3809, "a4d3e842d8ce162a1ccb141a4c69ba2ee4ab9f111b938ff9c9e60cb6e631467b"),
    ("femur-right-fresh-04", "femur-right", "test", 4653, "4f9dd01fbb7644f5bde50ab1dd651aac37bc651e5c798dc7ff95f7faa13aaa86"),
    ("femur-right-fresh-05", "femur-right", "test", 4814, "f8e2402132ee528edd915a96f57b34a0a7fafe055286ef7cbca5271cbb5d07ca"),
    ("femur-right-fresh-06", "femur-right", "test", 5329, "ade16d314151de355a53cac4e7dba648d1e4a63e9907070352593dcd30dc7995"),
    ("femur-right-fresh-07", "femur-right", "test", 5892, "1980531e689e21468518db2fdbe104202b62acd38df7f7a3159672467d49ef00"),
    ("femur-right-fresh-08", "femur-right", "test", 8528, "5aa8ada13a166bc82efe334c4190b286e22c6adc3c8916dc57309e63893a0f12"),
    ("femur-right-fresh-09", "femur-right", "test", 9942, "04e2a4911e2123f1988f5f18818af508d6acd09efbfe4f2833afcbdb9d2c8a18"),
    ("femur-right-fresh-10", "femur-right", "test", 10319, "756068a8263f0f150a28f8d20af525a4b1924e40de359904b97946fe997b9e0d"),
    ("femur-right-fresh-11", "femur-right", "test", 12270, "40ed1f495148418fe9a54f762f0ee7d0658ff3e099baf89ba5d49f3d64ccaadc"),
    ("femur-right-fresh-12", "femur-right", "test", 13200, "8545aad72f30abf055f0c305cb978cfd88308ad16a30b89bc1654f3cb16239a7"),
    ("femur-right-fresh-13", "femur-right", "test", 14192, "243f50eb49de62d917e6fd48229b11950362d639e89cfe012887c35c07b48629"),
    ("femur-right-fresh-14", "femur-right", "test", 14585, "d460bc2a119aa6e69073248059cd5d3cab1248709183f616a264218b973a9e88"),
    ("femur-right-fresh-15", "femur-right", "test", 15641, "aff82d2b2bbf317e4fc45768fd3fc3ca1dec2284b33179e2a9b31d2cc563a65b"),
    ("heart-fresh-00", "heart", "test", 533, "ebf30d96c1a1eae2353752b06ac5e9f26cfde018f9c55968612ab1404b88223e"),
    ("heart-fresh-01", "heart", "test", 878, "d3cdd954f2ecbd461adb515d1164382498c5ea4dafa8abe9e8566cd28654a1c2"),
    ("heart-fresh-02", "heart", "test", 2067, "81df80def470e33cf0d825ce4d221359ac9f22e9261d9f7882996b1e755fdf44"),
    ("heart-fresh-03", "heart", "test", 4555, "010a65dd1eeb593800ac359247e20ad4850b6146755affd603adcc6f51dfec1d"),
    ("heart-fresh-04", "heart", "test", 6024, "5ed7b0b70af3150bbd70510efcdd74c3a8ec492df90ef6c7d9b0edbe8b5fa08f"),
    ("heart-fresh-05", "heart", "test", 6734, "dd542c7f8bad5ae6f5127797c1765299c63d70a91b98d53c6ff96d7a298a48ba"),
    ("heart-fresh-06", "heart", "test", 7382, "76d5086ef272e249fe7588fced56ca332e21064e05921b16cfa00fbee2b97c79"),
    ("heart-fresh-07", "heart", "test", 9483, "0d61602e47d103a7c3a7ee61a5df70934b59263868f0bef030d1b5428719548e"),
    ("heart-fresh-08", "heart", "test", 10569, "bad04042c0de5b63946f60a8248a939442bd9c3b83e84ed0d33b8873f968ac20"),
    ("heart-fresh-09", "heart", "test", 10849, "d694b13a8eaf1d70eec41adf33e44d0cd4022e94091fc319ede483c1ee810d36"),
    ("heart-fresh-10", "heart", "test", 11954, "9a2870ba2c0e8c73558e1b4b6762bdb59e058894661cd2c69a0ff4a35e9e3a61"),
    ("heart-fresh-11", "heart", "test", 13324, "f9bdceb93f9f379bddacf242f7138d6e4f203c47e4588095f48f53a20b839bba"),
    ("heart-fresh-12", "heart", "test", 15223, "c7cc204f170f8719a05c37c8e59fff521e99cd46e974956e8d1b90e170b3f6c8"),
    ("heart-fresh-13", "heart", "test", 15614, "b2919a2f61f36f9dbe7918587482ca287805a59b2fbb6528088fe6e3ec0a545e"),
    ("heart-fresh-14", "heart", "test", 15628, "da84f6b8bf47d8725126adfaadaf56876e9ed601e6d27b06093d70979c716990"),
    ("heart-fresh-15", "heart", "test", 17342, "1a40d1876b30a9a2180d46cf977c37ec22245ddd837b2a1dcec9f25076fa9d2a"),
    ("kidney-left-fresh-00", "kidney-left", "test", 28, "5e29a5207a0efd6ba07b962840dde9e7f777333399a9d1767cef054306dd86c2"),
    ("kidney-left-fresh-01", "kidney-left", "test", 390, "0b4335e4f241b7a8177051892c11429548f0b70944dd3706922dc96e453e71ca"),
    ("kidney-left-fresh-02", "kidney-left", "test", 2186, "c94b163467db24a78f1dd5f91c63f2588c4fdfc3ebdfdc98194c44451cf493bd"),
    ("kidney-left-fresh-03", "kidney-left", "test", 2475, "b6f3db3c5ef950b798a6d19f0631fafdda2b6d8593ed7ec54a308842d329e001"),
    ("kidney-left-fresh-04", "kidney-left", "test", 4142, "95107cfcd3ea377e2991c9b3adf11c6ca89aea1f1419ee370966f847a8ab8fcc"),
    ("kidney-left-fresh-05", "kidney-left", "test", 6307, "104caf08d5b723cc254d2a6ee15caaf01c956904f6e2bc590eb421e08bc5509f"),
    ("kidney-left-fresh-06", "kidney-left", "test", 6632, "1f76816f0542a8f7ca2824b1170c0377cfd74abbc7cd5a4dc39c495e19ecdb98"),
    ("kidney-left-fresh-07", "kidney-left", "test", 8582, "45ffb426f785b1365f4e3ccbb84e35f76883922d624bae84cf0801f7776df6d3"),
    ("kidney-left-fresh-08", "kidney-left", "test", 8695, "7e700ccb27e30a9e45789727c1808243a650bcd029584ba69cbc0bfdf95e5bf0"),
    ("kidney-left-fresh-09", "kidney-left", "test", 10921, "dc6698d0db377209d699b9dff18405e31684246cbcdd1877a3226f5c7e8aa45a"),
    ("kidney-left-fresh-10", "kidney-left", "test", 11003, "39f3f0be06b623e44ee0da1557629d6f5a1777081c17b8140bb10bb07858d55c"),
    ("kidney-left-fresh-11", "kidney-left", "test", 13942, "9494efb250632b605d740b5c52bf9a99ee07afdb6de1b6136d8c4ef350044d16"),
    ("kidney-left-fresh-12", "kidney-left", "test", 14520, "178e29afda64023f8bf1a520240184c234c7bf1c24b7e474a686f9549164582b"),
    ("kidney-left-fresh-13", "kidney-left", "test", 15052, "81d81fa46befc0a83e14703a6509b617c41a19e242874f79bdfbdc14cc5769d9"),
    ("kidney-left-fresh-14", "kidney-left", "test", 15353, "b39c4e65d2726d01c5d7f78fecb417ff1bcd7ca26fdde509ff90643ae8af9c4f"),
    ("kidney-left-fresh-15", "kidney-left", "test", 16858, "40ee3b7b6306885731efb137123753863be5b9700586c1f95ffc23753f759067"),
    ("kidney-right-fresh-00", "kidney-right", "test", 64, "68bb6a070f2e071c213cd44dc1d12c079cfb3e532ee51da75b1fa4a96e29eaf4"),
    ("kidney-right-fresh-01", "kidney-right", "test", 904, "843863ae06ede5ca45656f281ea6cdb68475d4b0a7939752ae5184c46a83d217"),
    ("kidney-right-fresh-02", "kidney-right", "test", 2120, "8fa2d28d5c419353e04d2b938df0ca8a1b908542c358c254826f554e9d12cb51"),
    ("kidney-right-fresh-03", "kidney-right", "test", 2846, "73e84f057c9f2a4a24f276719bf5c54afac1d12a7c1a55241ea9fbd3396ebeac"),
    ("kidney-right-fresh-04", "kidney-right", "test", 3174, "e2ff5ae02f9d1905f26bbebda4992b42320f56bd54395c9e8bd570aa400f8a53"),
    ("kidney-right-fresh-05", "kidney-right", "test", 3934, "9f5a2959bfd3da9d5ef3cf3cf1fc8c543a8c3c6f1622f0dd0d554d1eecaa718d"),
    ("kidney-right-fresh-06", "kidney-right", "test", 4537, "0f7b61a4f7f2597ec71ba2c62eed22aa814a7a1cbe7a8d957eb110f881b85c9f"),
    ("kidney-right-fresh-07", "kidney-right", "test", 6618, "7b2fa2f68b7a4430bc92822a312d58e6b6786c7a3be0c21ddee5b7fc188373e3"),
    ("kidney-right-fresh-08", "kidney-right", "test", 8260, "149b91ec11b7b79a2fb30bee0aef65108116078da00c2fc64d392f4bca1582c0"),
    ("kidney-right-fresh-09", "kidney-right", "test", 8614, "104a657da71863b11857a6e00ea66921173995540d743d487a53868721c1b2c9"),
    ("kidney-right-fresh-10", "kidney-right", "test", 9397, "5848829e72613eb0cbadf58cdaa895854da0efc8bc6c235ce8c9d61ce3be1980"),
    ("kidney-right-fresh-11", "kidney-right", "test", 12599, "45e1e78fe02359adb2c5220bdbfacf70868e4321cb6630bf30fd243323afec81"),
    ("kidney-right-fresh-12", "kidney-right", "test", 15138, "497f8b4c7d065583650c8135ca1355bf601dee0013581a6f8b18399c48103c42"),
    ("kidney-right-fresh-13", "kidney-right", "test", 16188, "46b55aa82883c14f99420f7ca736ff5b7c52cf53188868c9e8435fd173a75d90"),
    ("kidney-right-fresh-14", "kidney-right", "test", 16507, "1a67d20a9ac8cf350191dbb47c897a7fcb24cba7845192d437710cbea1daa0c3"),
    ("kidney-right-fresh-15", "kidney-right", "test", 17641, "0e6111be9f6b0a94357ba63220f3adc94230ea177ff1f5dc37c07152f8aba63b"),
    ("liver-fresh-00", "liver", "test", 2423, "0ca31157f3278a4dfe176afbb107014419279dc64eb82fad6e3097fa4ef19194"),
    ("liver-fresh-01", "liver", "test", 3311, "442d57338651071a45bee0f5d5865ae47a4fc3dff46c3b09b138f4b827934178"),
    ("liver-fresh-02", "liver", "test", 4360, "4ef04dbe5a9c9c31d776dc1daae8cf67a582d486ceb9db6e8fe564948f22e27f"),
    ("liver-fresh-03", "liver", "test", 4631, "795c6af1a2d1e7ff93d098b6e63b1b06b99c67b87e6ccb3c467c785366af99e9"),
    ("liver-fresh-04", "liver", "test", 6154, "2492bea561a2b242be14615fb3638f38e562d49a8bedf6f872077cdb0812e029"),
    ("liver-fresh-05", "liver", "test", 7866, "ca6d46e4ec36f066837ebd1d64e2871adc9281919e597b6d59380ce3bfb4c66e"),
    ("liver-fresh-06", "liver", "test", 9912, "77a61abfbe405cb278cbcdf8763e964b380188f734230be9b22b558bb4b383e4"),
    ("liver-fresh-07", "liver", "test", 10195, "12e7fb2e96f6a72590bef40e2152409ef56d5c91471f14fb368ef364dfaf72c0"),
    ("liver-fresh-08", "liver", "test", 10528, "6e5e15877d5df4b0c6e15384d7341e148aa1b1979549dde4d65b1ecdb1384ef8"),
    ("liver-fresh-09", "liver", "test", 13046, "bdb4dffb0dd0f3160d70e27ecd83c421fa3ed71d822898a4c08aca940947288b"),
    ("liver-fresh-10", "liver", "test", 13675, "eef11f415bc026319c43e64030d4ad182d48bc2c056726552f327aed495e4ed4"),
    ("liver-fresh-11", "liver", "test", 13813, "84fd026017643eb91155f3573f0a6a48d11adf6e7e032b825fa0b89300a35124"),
    ("liver-fresh-12", "liver", "test", 15006, "0adce7b0b8e4c4405db7e3f7059216d530e1ea97f6c04e5d3faaa6a8379ae75a"),
    ("liver-fresh-13", "liver", "test", 16538, "91c6ae76e0edaa325d8d040dffa0f6b356c7540ed753ba24d00f914e14c07279"),
    ("liver-fresh-14", "liver", "test", 16853, "2cf503a3fce0cef7afda962609c2cc973a85caacb841403916ba818bfb95d8e5"),
    ("liver-fresh-15", "liver", "test", 17617, "a1bc3fb79df204d460f22bf310efd433fa7098ab2ec1c526859cf9659850ab98"),
    ("lung-left-fresh-00", "lung-left", "test", 1051, "962391ba842f3e4162ebc191e5f00a4cd33a36008f86c7bf315fed6915cd50c9"),
    ("lung-left-fresh-01", "lung-left", "test", 1265, "0d267944e81fb275c38b3b388d9d0d5aa3d3ea3a2f0847bd47830c6966fc006e"),
    ("lung-left-fresh-02", "lung-left", "test", 1714, "48c34af01bd658541c5014b46aa8b07537b855b46f4e32402571ac8ad6a98e63"),
    ("lung-left-fresh-03", "lung-left", "test", 5822, "c069a7e83490d57271129922d4da018270ad44ee2171c031c674d89c28f2a77b"),
    ("lung-left-fresh-04", "lung-left", "test", 6822, "42474dbecb8466797310fca237dc5cef90927f54589fbc358b426139f96960a8"),
    ("lung-left-fresh-05", "lung-left", "test", 9243, "8ed723378dd33e55a1d03ff896626a30673a109a7e8dae3380d0b213c4e99fb2"),
    ("lung-left-fresh-06", "lung-left", "test", 10612, "b65579a827474212d2e503519f05932ed4b4eae336ba56821d757f42793b4f34"),
    ("lung-left-fresh-07", "lung-left", "test", 11079, "d60d6c684a055d7a218fa2d316607f500821de52fac9e35b5b5bb18e93efd3aa"),
    ("lung-left-fresh-08", "lung-left", "test", 11164, "05b995707f44f37b2276a31738978564ea8b3ab7acee9af2423a77e8213a44e7"),
    ("lung-left-fresh-09", "lung-left", "test", 11187, "4bb1eb4f69e19aab725a07006904fde6f87e23ae24bf7eff7c90117dbbc59fef"),
    ("lung-left-fresh-10", "lung-left", "test", 12034, "e67dbdaceb4a68f515333eeb1c36a6bd994646085afa72b37d3b49b325b96be8"),
    ("lung-left-fresh-11", "lung-left", "test", 12930, "74dff9e21c2a17237a36d392c41a41a8e252e87384740ce7df02312129d9dd3e"),
    ("lung-left-fresh-12", "lung-left", "test", 14855, "392c6e973e03a3b11d6e7481546d85c0020b7defd29a3a8132c89711aa8c31f3"),
    ("lung-left-fresh-13", "lung-left", "test", 15186, "4d0eb98acc9876bc10a426fb2e10604fa308f63f14ddc1a1def0763e4dcaae3e"),
    ("lung-left-fresh-14", "lung-left", "test", 16146, "d3c36842ba82a9486fd960ea2288d22ae3a00c7dec3d932a06594d9af5c56529"),
    ("lung-left-fresh-15", "lung-left", "test", 17380, "f6afca679288219abcac34521fd7ee9a314bc8f9e42f0d793afb2c97951996ca"),
    ("lung-right-fresh-00", "lung-right", "test", 689, "04e20d15ceb7f1e247feae17e7abc8140c5ccaad9495a0e7a5bf1265e34792aa"),
    ("lung-right-fresh-01", "lung-right", "test", 1869, "f11591efcdbbb20086dfa5e2cf38dd7ca4f9e6d40382e38d6914b0006e5f0720"),
    ("lung-right-fresh-02", "lung-right", "test", 3553, "1c9b9a28573b5cca4ad8a628196b5b8b3beca4fb8d5805d60c45c270827805fa"),
    ("lung-right-fresh-03", "lung-right", "test", 4133, "afae3b159ac105227a99266e7bdbff7079adcd6372af4b9d158286cffb7b4777"),
    ("lung-right-fresh-04", "lung-right", "test", 4275, "6366b5f39283ad52831986f81ea73ee4905e29af49a1d5ac60bebcc2ba21884b"),
    ("lung-right-fresh-05", "lung-right", "test", 7530, "bf9868fc6254cdf269a77ebddd879491813440c09789006137cefbe49a4dcaf4"),
    ("lung-right-fresh-06", "lung-right", "test", 11302, "094e9223f22f440a64c06288e6d40c856a8e915c624fecb94ae517cb129fce98"),
    ("lung-right-fresh-07", "lung-right", "test", 12627, "dd663621dd7b2ea2105b20d7e59a1d5822ab9931433a713fb704b9a36ecb571f"),
    ("lung-right-fresh-08", "lung-right", "test", 13546, "60b75c8ac0c840cce2b3f85dd5f418b69350a20c3d8a4f0a5f77ce9932847117"),
    ("lung-right-fresh-09", "lung-right", "test", 14324, "9e5f0a624895a92721ff91a83b48cdd957fce6c3ce0da4bd21c9ef3ecfb111e0"),
    ("lung-right-fresh-10", "lung-right", "test", 15009, "2900879077d01095f42a6c94be03ba78f360650483290ca9bf2be113b198fc92"),
    ("lung-right-fresh-11", "lung-right", "test", 16328, "253c1335ba73c731f07837067d51cee96870642279ca9057b6e0005738daacfa"),
    ("lung-right-fresh-12", "lung-right", "test", 16341, "c446aaafc0ee9e668a46af6e99623dcffc5b18c3ac46428fa89191103716dee6"),
    ("lung-right-fresh-13", "lung-right", "test", 16384, "296e9309859641367990e7b0d79c69c26f11df3f1dedfd6b449428b268b16c0f"),
    ("lung-right-fresh-14", "lung-right", "test", 17245, "57731e6ede4a6c18bf0480236ede29b4692666291d3402ff5541b429c927e5ac"),
    ("lung-right-fresh-15", "lung-right", "test", 17254, "0bb3d144a49426776fa67862d5e728fdf221994ad51f2291790d82fd60194754"),
    ("pancreas-fresh-00", "pancreas", "test", 1296, "9f5326b0639929d3196f814f59e11188d5e0ada0eccd40901d9da7fa1af51a58"),
    ("pancreas-fresh-01", "pancreas", "test", 3417, "1bdca8f44a9a00494c8757e63c588b2e2c1a41a010487e953a426ca12b4515d3"),
    ("pancreas-fresh-02", "pancreas", "test", 4054, "9a6b42c240090a99c9b50c0f3109756e6c1b8495fa7a43a310fc99001a8ec23d"),
    ("pancreas-fresh-03", "pancreas", "test", 4419, "2616196905844c4b7f487fcdb02d847cc8a70408620a38b3539a89639bcf8bf1"),
    ("pancreas-fresh-04", "pancreas", "test", 6048, "168c085bc7785d610c87af87136bca1af6cec7a6815a75e7523cb365208bd18b"),
    ("pancreas-fresh-05", "pancreas", "test", 7504, "5fd813852f2bdb1dc9104975e3f122d171208e5987ce61f56d1dd94c0bdf93bb"),
    ("pancreas-fresh-06", "pancreas", "test", 8472, "fdc45f67e1a9758d7c7d202e55aeb40c9e9c9f3d383925f7ccc14aecc0df92d7"),
    ("pancreas-fresh-07", "pancreas", "test", 9684, "bdebf87094dc2c5da1e387024e48feacc504b8dc472a31fb36ad3c93edec43d8"),
    ("pancreas-fresh-08", "pancreas", "test", 10855, "0bcc4a44fc0070ab4aa19a418cdf6f89509dfd32f62e17e3db88f851cdd57acd"),
    ("pancreas-fresh-09", "pancreas", "test", 11642, "a4632670db3703fa2cf92f62aa953f1bdb2f54a56117c33ddfaf3f47fc09b731"),
    ("pancreas-fresh-10", "pancreas", "test", 13622, "95d93ed998c5ce05821226a7705c588b4502b58bc5bcaed076bc92f1b6caaba8"),
    ("pancreas-fresh-11", "pancreas", "test", 15113, "061009f8887eeb081e986b63749822e663beca5d4900d33591fbbb5786ac9c44"),
    ("pancreas-fresh-12", "pancreas", "test", 15432, "27c95de97fd55f770b4498d05d2d454d38d0603be097e548b3144953d179e7c0"),
    ("pancreas-fresh-13", "pancreas", "test", 15538, "2be3096e22c1afdebf2166f99e45ec6e15164f9130aa87cd33d4564b18eccbb9"),
    ("pancreas-fresh-14", "pancreas", "test", 16059, "2370bcd642c2fb50dbca025c9a3dfcad31d51a210ab6e2968f417e86ea855f82"),
    ("pancreas-fresh-15", "pancreas", "test", 16690, "17037da014750c3b10755186f10ae555b64697dc651cc081da8abf9240516581"),
    ("spleen-fresh-00", "spleen", "test", 1544, "2cdc3c359b83773e56abf53e1488ccc6f05637e751fb06a996bd2dc263be788f"),
    ("spleen-fresh-01", "spleen", "test", 4574, "96c23d9a00eb9077ae88e52f7e860b2e8704f6e061edc7c5c85fd36a70fd2e1e"),
    ("spleen-fresh-02", "spleen", "test", 4686, "5889f808eb2b5bd3a7b7e9b1d6e9adca473b40f27e98ba3115f526cb069d736b"),
    ("spleen-fresh-03", "spleen", "test", 6678, "ac3f7a9877b18b2f84e3d4287c720d88ae9266b48d91f399584877728d8b00f4"),
    ("spleen-fresh-04", "spleen", "test", 7769, "a69f66f481c59a54ae69dec77cb83b5a700b67deff95b9399411c79bcb0b5c2c"),
    ("spleen-fresh-05", "spleen", "test", 9224, "7ae843b60c1afcb38fabbe9361514e9b0210a9cbd010172a8739c9059797bb4b"),
    ("spleen-fresh-06", "spleen", "test", 9379, "327347eaae0c2744385103dd73ec31f5cf5bf454d08ad7b20ccc8602c2232c1e"),
    ("spleen-fresh-07", "spleen", "test", 10355, "343c57c80b8b14797796d61157749b825f5aab02fb65afb2f0e56aae888e1eae"),
    ("spleen-fresh-08", "spleen", "test", 11399, "ce51fcf6e40d189f2708539e4cf2c6549cfdbdfb3b2eb56d19823931cf07de8a"),
    ("spleen-fresh-09", "spleen", "test", 11817, "dd775d5736be21fa732875b197fd83c8c6836825d53627d775981d9e54b75961"),
    ("spleen-fresh-10", "spleen", "test", 12024, "7842f117208e0a3efe1735098075089f2a44ff20ad73aa7f0187859fe9781361"),
    ("spleen-fresh-11", "spleen", "test", 12954, "29d07e8789525f9fe91698621936e372519182b1a31639f3f25223714332ec0e"),
    ("spleen-fresh-12", "spleen", "test", 14972, "5a0a7776f2ffe3342dcaf9ba53d509e816cd982111f9dff40392760d3d50ff81"),
    ("spleen-fresh-13", "spleen", "test", 16423, "22ff91cf90eafed1fe4f59941f69f7505335af364eb191f89e1aafeba97ec52b"),
    ("spleen-fresh-14", "spleen", "test", 16942, "c415f14d7be280dcd6026ab85a53fa6e2710598d4daa2583ced3099404090e2c"),
    ("spleen-fresh-15", "spleen", "test", 17589, "23570d2a1befbe839ee78e9ff060812bf4ffd4af5e6632394ebb957b8f0caffc"),
)
VALIDATION_DRAW_SEED = 71_352
FRESH_TEST_DRAW_SEED = 71_176
SAMPLE_SEED = 42
# Per organ, by tutorial role (11 organs -> 396 / 352 / 176 / 176). The 660-slice SAMPLE_RECORDS keep their pins: train and
# the exposed development test (the 176 test slices every earlier record scored) come from them; their 88 val slices retire.
SAMPLE_SPLIT = {"train": 36, "validation": 32, "exposed_dev_test": 16, "test": 16}
ROLE_SOURCE_SPLIT = {"train": "train", "validation": "val", "exposed_dev_test": "test", "test": "test"}


def tutorial_records() -> tuple[tuple[str, str, str, int, str], ...]:
    """The pins the tutorial uses: train + exposed development test from SAMPLE_RECORDS, then the new validation and fresh test."""
    return tuple(r for r in SAMPLE_RECORDS if r[2] != "val") + VALIDATION_RECORDS + FRESH_TEST_RECORDS


def sample_role(rid: str, split: str) -> str:
    """The tutorial role of a pinned slice: the new validation / fresh-test pins by id, the older pins by their split."""
    if any(rid == r[0] for r in FRESH_TEST_RECORDS):
        return "test"
    if any(rid == r[0] for r in VALIDATION_RECORDS):
        return "validation"
    return {"train": "train", "val": "retired_validation", "test": "exposed_dev_test"}[split]


def pin_manifest_digest(records: Sequence[Sequence[Any]] | None = None) -> str:
    """SHA-256 of the pin manifest (id, organ, split, index, slice digest) in order: one value names the whole sample."""
    rows = [list(r) for r in (tutorial_records() if records is None else records)]
    return _sha256_bytes(json.dumps(rows, separators=(",", ":")).encode("utf-8"))


MIN_RECORDS = 8
# The smallest BYOD set the default stratified split accepts: two labels of six images (each label keeps one test
# and one validation image, so 12 leaves the 8 training records MIN_RECORDS asks for; 8 images would leave 4).
BYOD_MIN_RECORDS = 12
BYOD_SPLIT_ROLES = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
MAX_RECORDS = 20_000
MIN_CLASSES = 2
MAX_CLASSES = 100
MAX_LABEL_CHARS = 64
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_ .:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> Path:
    """The pinned MedMNIST archive from the cache or Zenodo, verified by byte size and SHA-256."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / CORPUS_ARCHIVE_NAME
    if not (archive.is_file() and archive.stat().st_size == CORPUS_ARCHIVE_BYTES and _sha256_file(archive) == CORPUS_ARCHIVE_SHA256):
        partial = archive.with_suffix(".npz.part")
        if fetcher is not None:
            partial.write_bytes(fetcher(CORPUS_URL))
        else:
            request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "dimer-pubmedclip-tutorial/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response, open(partial, "wb") as handle:  # noqa: S310 (pinned https URL)
                for chunk in iter(lambda: response.read(1 << 22), b""):
                    handle.write(chunk)
        size = partial.stat().st_size
        digest = _sha256_file(partial)
        if size != CORPUS_ARCHIVE_BYTES or digest != CORPUS_ARCHIVE_SHA256:
            partial.unlink()
            raise ValueError(
                f"{CORPUS_ARCHIVE_NAME}: fetched {size} bytes with sha256 {digest[:16]}…, pinned "
                f"{CORPUS_ARCHIVE_BYTES} / {CORPUS_ARCHIVE_SHA256[:16]}…"
            )
        partial.replace(archive)
    return archive


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None, records: Sequence[Sequence[Any]] | None = None) -> dict[str, bytes]:
    """Return every pinned slice (raw 224 x 224 uint8 bytes keyed by record id): from the per-slice cache when
    every cached file matches its digest, else extracted from the verified archive (the archive is read with
    `numpy.load(allow_pickle=False)`; one split array at a time, released after use). `records` defaults to
    `tutorial_records()`."""
    pins = tuple(tutorial_records() if records is None else records)
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    slices = cache / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    missing = []
    for rid, _label, _split, _index, digest in pins:
        local = slices / f"{rid}.raw"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) == SLICE_SIDE * SLICE_SIDE and _sha256_bytes(data) == digest:
            out[rid] = data
        else:
            missing.append(rid)
    if missing:
        import numpy as np

        archive = fetch_archive(cache_dir=cache, fetcher=fetcher)
        wanted: dict[str, list[tuple[str, int, str]]] = {}
        for rid, _label, split, index, digest in pins:
            if rid in missing:
                wanted.setdefault(split, []).append((rid, index, digest))
        with np.load(archive, allow_pickle=False) as npz:
            for split, items in wanted.items():
                images = npz[f"{split}_images"]
                for rid, index, digest in items:
                    array = images[index]
                    if array.shape != (SLICE_SIDE, SLICE_SIDE) or array.dtype != np.uint8:
                        raise ValueError(f"{rid}: slice {split}[{index}] has shape {array.shape} / {array.dtype}")
                    data = array.tobytes()
                    if _sha256_bytes(data) != digest:
                        raise ValueError(f"{rid}: slice {split}[{index}] sha256 {_sha256_bytes(data)[:16]}… != pinned {digest[:16]}…")
                    (slices / f"{rid}.raw").write_bytes(data)
                    out[rid] = data
                del images
    return {rid: out[rid] for rid, *_ in pins}


def read_corpus(files: Mapping[str, bytes], records: Sequence[Sequence[Any]] | None = None) -> list[dict[str, Any]]:
    """Decode the verified slices into `{id, image, label}` records with their provenance and tutorial role."""
    out = []
    for rid, label, split, index, _digest in (tutorial_records() if records is None else records):
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.frombytes("L", (SLICE_SIDE, SLICE_SIDE), files[rid])
        out.append(
            {
                "id": rid,
                "image": image.convert("RGB"),
                "label": label,
                "display_name": ORGANS[label][0],
                "medmnist_class": ORGANS[label][1],
                "medmnist_split": split,
                "medmnist_index": index,
                "source": f"organamnist_224/{split}[{index}]",
                "role": sample_role(rid, split),
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Roles from the pins (`role`, else the dataset's own split; never a reshuffle across roles): `sizes` are the
    expected counts per organ per role; `seed` only orders the records within a role. Every role must come from its
    official MedMNIST split (`ROLE_SOURCE_SPLIT`), so train shares no CT scan with validation or either test set."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        role = record.get("role") or SPLIT_ROLE.get(str(record.get("medmnist_split")))
        if role == "retired_validation":
            continue
        if role is None or role not in out:
            raise ValueError(f"{record.get('id')}: record carries no known medmnist_split")
        source = ROLE_SOURCE_SPLIT.get(role)
        if source is not None and str(record.get("medmnist_split")) != source:
            raise ValueError(f"{record.get('id')}: role {role} must come from the official {source} split")
        out[role].append(dict(record))
        counts[(role, str(record["label"]))] = counts.get((role, str(record["label"])), 0) + 1
    for (role, label), n in sorted(counts.items()):
        if n != sizes[role]:
            raise ValueError(f"{role}/{label}: {n} records, expected {sizes[role]}")
    for name in out:
        rng.shuffle(out[name])
        out[name] = [
            {**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(out[name])
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image/label")
    for key in ("id", "image", "label"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image, label = record["id"], record["image"], record["label"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label_name}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label_name}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"{label_name}: image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}"
        )
    if not isinstance(label, str) or not _LABEL_RE.match(label.strip()):
        raise ValueError(
            f"{label_name}: label must be a non-empty string of at most {MAX_LABEL_CHARS} plain characters"
        )
    item = {"id": rid, "image": image.convert("RGB"), "label": label.strip()}
    for key in (
        "source_id",
        "source",
        "display_name",
        "medmnist_class",
        "medmnist_split",
        "medmnist_index",
        "group",
        "split",
    ):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    """Structural validation of a labelled-image dataset; raises ValueError before any model import."""
    if (
        isinstance(records, Mapping)
        or not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
    ):
        raise ValueError("records must be a list of {id, image, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        counts[item["label"]] = counts.get(item["label"], 0) + 1
        checked.append(item)
    if not MIN_CLASSES <= len(counts) <= MAX_CLASSES:
        raise ValueError(
            f"{len(counts)} distinct labels; {MIN_CLASSES}..{MAX_CLASSES} are required"
        )
    sides = [max(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "classes": sorted(counts),
        "label_counts": dict(sorted(counts.items())),
        "image_side": {"min": min(sides), "max": max(sides)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def class_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted label vocabulary of a dataset (the `classes` a head is trained for)."""
    names = sorted({str(r["label"]) for r in records})
    if len(names) < MIN_CLASSES:
        raise ValueError(f"a dataset needs at least {MIN_CLASSES} distinct labels")
    return names


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same image matches."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], image_digest(r["image"]), r["label"]] for r in records]
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def source_overlap(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Which MedMNIST splits feed each role (an observation, not an assertion): the sample keeps the dataset's
    own roles, so every role should draw from exactly one MedMNIST split. MedMNIST does not publish the CT scan
    of a slice, so scan-level overlap cannot be counted here; the authors split OrganAMNIST by scan."""
    feeds: dict[str, set[str]] = {}
    for name, records in splits.items():
        for record in records:
            feeds.setdefault(name, set()).add(str(record.get("medmnist_split", "unknown")))
    return {
        "medmnist_splits_per_role": {name: sorted(v) for name, v in feeds.items()},
        "roles_mixing_splits": sum(1 for v in feeds.values() if len(v) > 1),
    }


def byod_split_mode(records: Sequence[Mapping[str, Any]]) -> str:
    """How `split_dataset` assigns roles: 'split column', 'group column' or 'stratified shuffle'."""
    if any(str(r.get("split", "") or "").strip() for r in records):
        return "split column"
    if any(str(r.get("group", "") or "").strip() for r in records):
        return "group column"
    return "stratified shuffle"


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Split a BYOD dataset into train/validation/test after de-duplicating images by decoded pixels.

    - A `split` value (train / validation / val / test) on the records is used as given.
    - Otherwise a `group` value (patient, scan, session...) keeps every group in one role: groups are shuffled with
      the seed and given to test, then validation, until each holds about its fraction of the records.
    - Otherwise a seeded stratified shuffle per label.
    `byod_split_mode` names the mode that applies."""
    if not (
        0.0 <= val_fraction < 1.0
        and 0.0 < test_fraction < 1.0
        and val_fraction + test_fraction < 1.0
    ):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    mode = byod_split_mode(unique)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    if mode == "split column":
        for record in unique:
            value = str(record.get("split", "") or "").strip().lower()
            if value not in BYOD_SPLIT_ROLES:
                raise ValueError(
                    f"record {record['id']!r}: split must be train, validation, val or test when labels.csv "
                    f"has a split column (got {value!r}); fill the column for every row"
                )
            splits[BYOD_SPLIT_ROLES[value]].append(record)
        if not splits["test"]:
            raise ValueError("the split column gives no record the test role; mark some rows test")
    elif mode == "group column":
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in unique:
            group = str(record.get("group", "") or "").strip()
            if not group:
                raise ValueError(
                    f"record {record['id']!r} has an empty group; give every row a group (patient, scan or session)"
                )
            groups.setdefault(group, []).append(record)
        if len(groups) < 3:
            raise ValueError(
                f"the group column names {len(groups)} group(s); at least 3 are needed so train, validation "
                "and test each get their own"
            )
        names = sorted(groups)
        rng.shuffle(names)
        index = 0
        for role, fraction in (("test", test_fraction), ("validation", val_fraction)):
            goal = len(unique) * fraction
            while fraction > 0 and index < len(names) - 1 and (
                not splits[role] or len(splits[role]) + len(groups[names[index]]) / 2 <= goal
            ):
                splits[role].extend(groups[names[index]])
                index += 1
        for name in names[index:]:
            splits["train"].extend(groups[name])
        in_role = {role: {str(r["group"]) for r in part} for role, part in splits.items()}
        shared = (
            (in_role["train"] & in_role["test"])
            | (in_role["train"] & in_role["validation"])
            | (in_role["validation"] & in_role["test"])
        )
        if shared:  # cannot happen by construction; the contract checks it anyway
            raise ValueError(f"groups in two roles: {sorted(shared)[:5]}")
    else:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for record in unique:
            by_label.setdefault(record["label"], []).append(record)
        for label in sorted(by_label):
            pool = by_label[label]
            rng.shuffle(pool)
            n_test = max(1, round(len(pool) * test_fraction))
            n_val = round(len(pool) * val_fraction)
            splits["test"].extend(pool[:n_test])
            splits["validation"].extend(pool[n_test : n_test + n_val])
            splits["train"].extend(pool[n_test + n_val :])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required: add "
            f"images (a stratified split of two labels needs at least {BYOD_MIN_RECORDS}, six per label) or give "
            "more of them the train role"
        )
    return splits


def _byod_error(message: str) -> ValueError:
    return ValueError(f"BYOD: {message}")


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `labels.csv` (columns `id`, `file`,
    `label`, optional `group` or `split`) beside the image files; images are decoded, never extracted to disk.

    `file` is a path relative to the folder holding `labels.csv` (`images/a.png` works). A bare file name is
    accepted when exactly one file has that name; two files with one name in different folders must be named by
    their path. Every failure is a ValueError naming the file and the fix."""
    source = Path(path)
    if source.is_dir():
        root = source.resolve()
        if not (root / "labels.csv").is_file():
            raise _byod_error(f"{source} has no labels.csv; put labels.csv (id, file, label) beside the images")
        table_bytes = (root / "labels.csv").read_bytes()
        files = {q.relative_to(root).as_posix(): q for q in root.rglob("*") if q.is_file()}
        base = ""

        def read(member: str) -> bytes:
            return files[member].read_bytes()

    elif source.is_file():
        if not zipfile.is_zipfile(source):
            raise _byod_error(
                f"{source.name} is not a zip archive; give a .zip (or a folder) holding labels.csv and the images"
            )
        archive = zipfile.ZipFile(source)
        files = {n: n for n in archive.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")}
        tables = [n for n in files if n.rsplit("/", 1)[-1] == "labels.csv"]
        if len(tables) != 1:
            raise _byod_error(
                f"{source.name} must contain exactly one labels.csv (found {len(tables)}); put labels.csv "
                "(id, file, label) beside the images"
            )
        table_bytes = archive.read(tables[0])
        base = tables[0][: -len("labels.csv")]

        def read(member: str) -> bytes:
            return archive.read(member)

    else:
        raise _byod_error(f"{source} does not exist; give a .zip or a folder holding labels.csv and the images")
    try:
        table = table_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _byod_error("labels.csv is not UTF-8 text; save it as UTF-8 CSV") from exc
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise _byod_error(f"labels.csv is missing columns {sorted(missing)} (it needs id, file, label)")
    by_name: dict[str, list[str]] = {}
    for member in files:
        if member.startswith(base):
            by_name.setdefault(member.rsplit("/", 1)[-1], []).append(member)
    out = []
    for line, row in enumerate(rows, start=2):
        name = (row.get("file") or "").strip().replace("\\", "/")
        if not name or name.startswith("/") or ".." in name.split("/"):
            raise _byod_error(
                f"labels.csv line {line}: file {name!r} must be a path inside the dataset (no leading / or ..)"
            )
        member = base + name
        if member not in files:
            candidates = by_name.get(name, []) if "/" not in name else []
            if len(candidates) > 1:
                raise _byod_error(
                    f"labels.csv line {line}: {name!r} matches {len(candidates)} files "
                    f"({', '.join(sorted(candidates)[:3])}); write its path instead, e.g. "
                    f"{sorted(candidates)[0][len(base):]!r}"
                )
            if len(candidates) != 1:
                raise _byod_error(
                    f"labels.csv line {line}: file {name!r} is not in the dataset; check the name and its folder"
                )
            member = candidates[0]
        try:
            image = Image.open(io.BytesIO(read(member)))
            image.load()
        except (OSError, SyntaxError, ValueError) as exc:  # PIL's UnidentifiedImageError is an OSError
            raise _byod_error(
                f"labels.csv line {line}: {name!r} is not an image Pillow can decode ({type(exc).__name__}); "
                "remove the row or replace the file"
            ) from exc
        record = {"id": row["id"], "image": image.convert("RGB"), "label": row["label"]}
        for key in ("group", "split"):
            if (row.get(key) or "").strip():
                record[key] = row[key].strip()
        out.append(record)
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the labels table of a split (id, file, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file", "label", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "file": f"{record['source_id']}.png"
                    if record.get("source")
                    else record["id"],
                    "label": record["label"],
                    "source": record.get("source", ""),
                }
            )
    return out
