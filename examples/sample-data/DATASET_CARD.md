# Deterministic PubMedCLIP Tutorial Samples

This directory defines three deterministic synthetic image assets for smoke tests and tutorials. The generated PPM files are intentionally not committed; `generate_samples.py` recreates them byte-for-byte and `SHA256SUMS` pins the expected outputs.

| File | Synthetic content | SHA-256 |
|---|---|---|
| `red_square.ppm` | red square on a light neutral background | `b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096` |
| `green_circle.ppm` | green circle on a light neutral background | `2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f` |
| `blue_triangle.ppm` | blue triangle on a light neutral background | `f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3` |

## Purpose

The assets make the tutorial self-contained and reproducible without downloading third-party images. They are not a benchmark, validation set, or evidence of model quality. The tutorial deliberately avoids assertions about which label must rank first because zero-shot CLIP scores are prompt- and domain-dependent — and a radiology-caption model has no reason to know drawn shapes at all.

## Generation contract

- format: ASCII PPM (`P3`)
- dimensions: 32 x 32 pixels
- max channel value: 255
- background: `(245, 245, 245)`
- deterministic geometric foregrounds defined in `generate_samples.py`

Generate them into the ignored output directory:

```bash
python examples/sample-data/generate_samples.py --output-dir outputs/sample-data
```

The generator verifies every SHA-256 digest before returning success.
