"""Notebooks are valid, use the shared config, and contain no secrets or stale paths."""
import json
import re

import nbformat
import pytest

from tiny_aya_safety.config import REPO

RUNNABLE = sorted((REPO / "notebooks").glob("0*.ipynb"))
EXECUTED = sorted((REPO / "notebooks" / "executed").glob("*.ipynb"))
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}")


def test_runnable_notebooks_numbered():
    assert [p.name[:2] for p in RUNNABLE] == ["01", "02", "03", "04", "05", "06", "07"]


@pytest.mark.parametrize("path", RUNNABLE + EXECUTED, ids=lambda p: p.name)
def test_valid_and_secret_free(path):
    nbformat.validate(nbformat.read(path, as_version=4))
    assert not SECRET.search(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", RUNNABLE, ids=lambda p: p.name)
def test_runnable_notebooks_use_config(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "from tiny_aya_safety.config import" in src
    assert "/content/pipeline" not in src and "COMBINED_DIR" not in src
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "clear outputs before committing"
