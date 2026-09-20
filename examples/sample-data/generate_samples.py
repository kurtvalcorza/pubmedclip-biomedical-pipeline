from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

WIDTH = 32
HEIGHT = 32
BACKGROUND = (245, 245, 245)

EXPECTED = {
    "blue_triangle.ppm": "f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3",
    "green_circle.ppm": "2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f",
    "red_square.ppm": "b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096",
}


def _mask(shape: str, x: int, y: int) -> bool:
    if shape == "square":
        return 8 <= x < 24 and 8 <= y < 24
    if shape == "circle":
        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2
    if shape == "triangle":
        if not 7 <= y < 26:
            return False
        half = (y - 7) // 2
        return 16 - half <= x <= 16 + half
    raise ValueError(f"unknown shape: {shape}")


def _render(foreground: tuple[int, int, int], shape: str) -> str:
    lines = ["P3", f"{WIDTH} {HEIGHT}", "255"]
    for y in range(HEIGHT):
        row: list[str] = []
        for x in range(WIDTH):
            pixel = foreground if _mask(shape, x, y) else BACKGROUND
            row.extend(str(value) for value in pixel)
        for offset in range(0, len(row), 24):
            lines.append(" ".join(row[offset : offset + 24]))
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "red_square.ppm": ((220, 40, 40), "square"),
        "green_circle.ppm": ((40, 170, 75), "circle"),
        "blue_triangle.ppm": ((40, 90, 220), "triangle"),
    }
    written: list[Path] = []
    for name, (foreground, shape) in specs.items():
        path = output_dir / name
        path.write_text(_render(foreground, shape), encoding="ascii", newline="\n")
        digest = _sha256(path)
        if digest != EXPECTED[name]:
            raise RuntimeError(f"{name} digest {digest} != expected {EXPECTED[name]}")
        written.append(path)
        print(f"{digest}  {name}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sample-data"),
    )
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
