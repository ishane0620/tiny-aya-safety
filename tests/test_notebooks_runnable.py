"""Static check that every runnable notebook can run, without executing any model.

- every code cell is valid Python (shell `!` and `%` lines aside);
- every file a notebook reads through data_path()/artifact() resolves (locally, or is a
  known Hugging Face file);
- every `from X import Y` in the notebooks names something that exists in the installed
  packages. Imports of packages that are not installed are skipped, so this runs in the
  CPU-only environment too; install requirements-gpu.txt to check the toolkit imports.
"""
import ast
import importlib
import importlib.util
import json
import re

import pytest

from tiny_aya_safety import config
from tiny_aya_safety.config import REPO

NOTEBOOKS = sorted((REPO / "notebooks").glob("0*.ipynb"))
HF_ONLY = {"observations.parquet", "matched_observations.parquet"}
# Written by an earlier notebook in a full retrain; artifact() falls back to the published copy.
ARTIFACTS = {"pipeline1_baseline_mh_stats.json", "pipeline2_results.json", "pipeline3_safety_pairs_v2.jsonl"}


def code_of(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            # IPython-only lines are not Python; replace them so line numbers stay put.
            yield i, "\n".join("pass" if l.lstrip().startswith(("!", "%")) else l for l in src.splitlines())


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_cells_parse(path):
    for i, src in code_of(path):
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{path.name} cell {i}: {e}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_files_resolve(path):
    text = "\n".join(src for _, src in code_of(path))
    names = set(re.findall(r'(?:data_path|artifact)\("([^"]+)"\)', text))
    for name in names:
        if name in HF_ONLY:
            continue
        assert name in ARTIFACTS or config.data_path(name).exists(), f"{path.name}: {name} does not resolve"
        if name in ARTIFACTS:
            assert config.data_path(name).exists(), f"{path.name}: no published fallback for {name}"


def _imports(path):
    for _, src in code_of(path):
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    yield node.module, a.name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_imported_names_exist(path):
    checked, missing = 0, []
    for module, name in _imports(path):
        root = module.split(".")[0]
        if root in ("google",) or importlib.util.find_spec(root) is None:
            continue                     # Colab-only or not installed here
        try:
            mod = importlib.import_module(module)
        except Exception as e:           # noqa: BLE001
            missing.append(f"{module} ({type(e).__name__}: {e})")
            continue
        if name != "*" and not hasattr(mod, name):
            try:
                importlib.import_module(f"{module}.{name}")
            except Exception:            # noqa: BLE001
                missing.append(f"{module}.{name}")
                continue
        checked += 1
    assert not missing, f"{path.name}: {missing}"
