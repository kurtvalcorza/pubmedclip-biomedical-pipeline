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
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 36, "validation": 8, "test": 16}  # per organ, from the dataset's own splits; 11 organs -> 396 / 88 / 176
MIN_RECORDS = 8
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


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return every pinned slice (raw 224 x 224 uint8 bytes keyed by record id): from the per-slice cache when
    every cached file matches its digest, else extracted from the verified archive (the archive is read with
    `numpy.load(allow_pickle=False)`; one split array at a time, released after use)."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    slices = cache / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    missing = []
    for rid, _label, _split, _index, digest in SAMPLE_RECORDS:
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
        for rid, _label, split, index, digest in SAMPLE_RECORDS:
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
    return {rid: out[rid] for rid, *_ in SAMPLE_RECORDS}


def read_corpus(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the verified slices into `{id, image, label}` records with their provenance."""
    out = []
    for rid, label, split, index, _digest in SAMPLE_RECORDS:
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
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Roles from the dataset's own splits (`medmnist_split`, never a reshuffle across roles): `sizes` are the
    expected counts per organ per role; `seed` only orders the records within a role."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        role = SPLIT_ROLE.get(str(record.get("medmnist_split")))
        if role is None or role not in out:
            raise ValueError(f"{record.get('id')}: record carries no known medmnist_split")
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


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (
        0.0 <= val_fraction < 1.0
        and 0.0 < test_fraction < 1.0
        and val_fraction + test_fraction < 1.0
    ):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_label.setdefault(record["label"], []).append(record)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
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
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `labels.csv` (columns `id`, `file`,
    `label`) beside the image files; images are decoded, never extracted to disk."""
    source = Path(path)
    if source.is_dir():
        table = (source / "labels.csv").read_text(encoding="utf-8")
        loader = lambda name: Image.open(source / name)  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "labels.csv" not in members:
            raise ValueError("BYOD zip must contain labels.csv")
        table = archive.read(members["labels.csv"]).decode("utf-8")
        loader = lambda name: Image.open(io.BytesIO(archive.read(members[name])))  # noqa: E731
    else:
        raise ValueError(
            "BYOD datasets must be a directory or a .zip holding labels.csv and the image files"
        )
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "file", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"labels.csv is missing columns {sorted(missing)}")
    out = []
    for row in rows:
        image = loader(row["file"])
        image.load()
        out.append({"id": row["id"], "image": image.convert("RGB"), "label": row["label"]})
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
