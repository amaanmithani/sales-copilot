from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from .conftest import ROOT


def load() -> object:
    spec = importlib.util.spec_from_file_location(
        "render_readme", ROOT / "scripts/render_readme.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_readme"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_splice_and_render() -> None:
    mod = load()
    body = mod.render()  # type: ignore[attr-defined]
    assert body.endswith("\n")
    out = mod.splice("a\n<!-- RESULTS:START -->\nSTALE_BLOCK\n<!-- RESULTS:END -->\nb", body)  # type: ignore[attr-defined]
    assert "STALE_BLOCK" not in out and out.startswith("a\n") and out.endswith("b")
    with pytest.raises(SystemExit):
        mod.splice("no markers", body)  # type: ignore[attr-defined]


def test_committed_readme_is_current() -> None:
    mod = load()
    assert mod.main(["--check"]) == 0  # type: ignore[attr-defined]


def test_tables_cover_all_results(tmp_path: Path) -> None:
    mod = load()
    body = mod.render()  # type: ignore[attr-defined]
    for name in ("cue_eval.json", "asr_wer.json", "latency.json"):
        if (ROOT / "results" / name).exists():
            assert f"results/{name} not generated" not in body
