"""Reproduce the 2026-10-17 draw of the grid-validation and fresh-test pins (pre-registration + amendment 1, review R71P-m3).

Usage: python -I tools/draw_pubmedclip_pins.py <organamnist_224.npz> <repo>/src <out.json>
It (1) checks whether the archive keeps scan order (it does not: adjacent-index slices are no closer than random pairs),
(2) draws 32 val slices per organ (seed 71352) from the val slices never pinned and 16 test slices per organ (seed 71176)
from the test slices never pinned, skipping duplicate-pixel slices and candidates closer to an exposed slice than the
median within-test nearest-neighbour distance (a COARSE screen, not a protection), and (3) writes the pins and the
statistics, including the distribution of fresh-to-exposed nearest distances. It scores nothing.
"""
# ruff: noqa: E501, E401, E702, E731, I001  -- a one-off analysis script kept as run
import hashlib, json, random, sys
import numpy as np
sys.path.insert(0, sys.argv[2])  # repo/src
from pubmedclip_pipeline.samples import ORGANS, SAMPLE_RECORDS, CORPUS_ARCHIVE_SHA256  # noqa: E402

VAL_SEED, FRESH_SEED = 71_352, 71_176
N_VAL, N_FRESH = 32, 16
z = np.load(sys.argv[1], allow_pickle=False)
digests, arrays, labels = {}, {}, {}
for split in ("train", "val", "test"):
    X = z[f"{split}_images"]; arrays[split] = X; labels[split] = z[f"{split}_labels"].ravel()
    digests[split] = [hashlib.sha256(x.tobytes()).hexdigest() for x in X]
count = {}
for s in digests:
    for d in digests[s]:
        count[d] = count.get(d, 0) + 1
dup_slices = {s: sum(1 for d in digests[s] if count[d] > 1) for s in digests}
pinned = {(r[2], r[3]) for r in SAMPLE_RECORDS}
pinned_digests = {r[4] for r in SAMPLE_RECORDS}
for r in SAMPLE_RECORDS:
    assert digests[r[2]][r[3]] == r[4], r
exposed = sorted(r[3] for r in SAMPLE_RECORDS if r[2] == "test")

def small(X):  # 56x56 float32 block means
    return X.reshape(len(X), 56, 4, 56, 4).mean(axis=(2, 4)).reshape(len(X), -1).astype(np.float32)

T = small(arrays["test"]); E = T[exposed]
def nn_dist(A, B, exclude_self=None):
    out = np.empty(len(A), np.float32)
    bn = (B * B).sum(1)
    for s in range(0, len(A), 512):
        a = A[s:s + 512]
        d = (a * a).sum(1)[:, None] + bn[None, :] - 2 * a @ B.T
        if exclude_self is not None:
            for k, i in enumerate(exclude_self[s:s + 512]):
                d[k, i] = np.inf
        out[s:s + 512] = np.sqrt(np.maximum(d.min(1), 0)) / np.sqrt(A.shape[1])
    return out
rng0 = np.random.default_rng(0)
probe = rng0.choice(len(T), 2000, replace=False)
within = nn_dist(T[probe], T, exclude_self=probe)  # nearest other test slice (a sister slice when one exists)
threshold = float(np.median(within))

def draw(split, seed, n_per, exclude, extra_check=None):
    rng = random.Random(seed)
    chosen, rejected = [], {"duplicate_pixels": 0, "near_duplicate_of_exposed": 0}
    for organ, (_name, cls) in ORGANS.items():
        cands = sorted(int(i) for i in np.flatnonzero(labels[split] == cls) if (split, int(i)) not in exclude)
        rng.shuffle(cands)
        picked = []
        for i in cands:
            d = digests[split][i]
            if count[d] > 1 or d in pinned_digests:
                rejected["duplicate_pixels"] += 1; continue
            if extra_check is not None and not extra_check(i):
                rejected["near_duplicate_of_exposed"] += 1; continue
            picked.append(i)
            if len(picked) == n_per:
                break
        assert len(picked) == n_per, organ
        chosen += [(organ, i) for i in sorted(picked)]
    return chosen, rejected

def scan_order(z):
    out = {}
    for split in ("val", "test"):
        X = z[f"{split}_images"]; y = z[f"{split}_labels"].ravel()
        n = len(X)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n - 1, 3000)
        Xf = X.astype(np.int16)
        adj = np.abs(Xf[idx] - Xf[idx + 1]).mean(axis=(1, 2))
        same_lab = y[idx] == y[idx + 1]
        j = rng.integers(0, n, 3000)
        rnd = np.abs(Xf[idx] - Xf[j]).mean(axis=(1, 2))
        rnd_same = rnd[y[idx] == y[j]]
        # gap-k distance profile
        prof = {}
        for k in (1, 2, 3, 5, 10, 20, 50, 200):
            ii = idx[idx + k < n]
            prof[k] = float(np.median(np.abs(Xf[ii] - Xf[ii + k]).mean(axis=(1, 2))))
        runs = int((y[1:] != y[:-1]).sum()) + 1
        out[split] = {"n": n, "adjacent_L1_median": float(np.median(adj)), "adjacent_same_label_L1_median": float(np.median(adj[same_lab])),
                      "adjacent_same_label_frac": float(same_lab.mean()), "random_L1_median": float(np.median(rnd)),
                      "random_same_label_L1_median": float(np.median(rnd_same)), "label_runs": runs, "gap_profile_median_L1": prof,
                      "first_labels": y[:60].tolist()}
    return out


val, val_rej = draw("val", VAL_SEED, N_VAL, pinned)
ex_dist = {}
def not_near(i):
    dist = float(nn_dist(T[i:i + 1], E)[0]); ex_dist[i] = dist
    return dist > threshold
fresh, fresh_rej = draw("test", FRESH_SEED, N_FRESH, pinned, not_near)
def pins(split, chosen, tag):
    out, k = [], {}
    for organ, i in chosen:
        n = k.get(organ, 0); k[organ] = n + 1
        out.append((f"{organ}-{tag}-{n:02d}", organ, split, i, digests[split][i]))
    return out
vp, fp = pins("val", val, "gridval"), pins("test", fresh, "fresh")
all_d = [p[4] for p in vp + fp] + list(pinned_digests)
assert len(set(all_d)) == len(all_d) == 660 + 352 + 176
fresh_ex = [ex_dist[i] for _o, i in fresh]
scan_lines = []
stats = {"archive_sha256": CORPUS_ARCHIVE_SHA256, "val_seed": VAL_SEED, "fresh_seed": FRESH_SEED,
         "duplicate_pixel_slices_per_split": dup_slices, "val_rejected": val_rej, "fresh_rejected": fresh_rej,
         "near_dup_metric": "RMS difference on 56x56 block means (0-255 scale)",
         "within_test_nn_rms_median_threshold": threshold,
         "within_test_nn_rms_quantiles": {q: float(np.quantile(within, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
         "fresh_to_exposed_nn_rms_min_median": [float(min(fresh_ex)), float(np.median(fresh_ex))],
         "fresh_to_exposed_nn_rms_quantiles": {q: float(np.quantile(fresh_ex, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95)},
         "scan_order": scan_order(z),
         "n_val": len(vp), "n_fresh": len(fp)}
json.dump({"stats": stats, "validation": vp, "fresh": fp}, open(sys.argv[3], "w"), indent=1)
print(json.dumps(stats, indent=1))
