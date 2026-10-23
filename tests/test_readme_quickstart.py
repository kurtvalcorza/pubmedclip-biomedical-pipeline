"""The documented load path must be the audited one.

The pinned Hub revision ships only the pickle source, so a bare ``load_pipeline()`` falls back to
a snapshot without ``model.safetensors`` and is refused. The README quickstart, the model card
and the example script must load through ``PubMedClipPipeline.from_pretrained(weights_dir=...,
allow_download=True)``, which runs the audited ``convert_source()``. Parsing only; no network.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks(path: Path) -> list[str]:
    return FENCE.findall(path.read_text(encoding="utf-8"))


def _calls(source: str) -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _assert_audited_load(source: str, where: str) -> None:
    loads = [c for c in _calls(source) if _call_name(c) == "PubMedClipPipeline.from_pretrained"]
    assert loads, f"{where}: no PubMedClipPipeline.from_pretrained(...) call"
    keywords = {kw.arg: kw.value for kw in loads[0].keywords}
    assert isinstance(keywords.get("weights_dir"), ast.Name)
    assert keywords["weights_dir"].id == "DEFAULT_WEIGHTS_DIR", f"{where}: weights_dir"
    allow = keywords.get("allow_download")
    assert isinstance(allow, ast.Constant) and allow.value is True, f"{where}: allow_download"


def test_readme_quickstart_uses_the_audited_from_pretrained_path():
    quickstart = _python_blocks(ROOT / "README.md")[0]
    _assert_audited_load(quickstart, "README quickstart")


def test_example_script_uses_the_audited_from_pretrained_path():
    _assert_audited_load(
        (ROOT / "examples" / "quickstart.py").read_text(encoding="utf-8"), "example"
    )


def test_no_doc_snippet_calls_bare_load_pipeline():
    sources = [
        *((f"README.md block {i}", b) for i, b in enumerate(_python_blocks(ROOT / "README.md"))),
        *(
            (f"MODEL_CARD.md block {i}", b)
            for i, b in enumerate(_python_blocks(ROOT / "MODEL_CARD.md"))
        ),
        (
            "examples/quickstart.py",
            (ROOT / "examples" / "quickstart.py").read_text(encoding="utf-8"),
        ),
    ]
    offenders = [
        where
        for where, source in sources
        if any(_call_name(call) == "load_pipeline" for call in _calls(source))
    ]
    assert offenders == []
