"""Fast checks of the installed package and of the test harness itself."""

import ast
import sys
import tomllib
from pathlib import Path

import biomed_retrieval_jax

ROOT = Path(__file__).parents[1]


def test_installed_version_matches_pyproject():
    # Catches a stale install: the package's metadata comes from the last `uv sync`.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert biomed_retrieval_jax.__version__ == pyproject["project"]["version"]


def test_registry_imports_only_the_standard_library():
    # CI's `list checkpoints` job imports the registry with the runner's bare python3.
    tree = ast.parse((ROOT / "tests" / "support" / "registry.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or "").split(".")[0])
    assert imported <= sys.stdlib_module_names, (
        f"non-stdlib imports: {imported - sys.stdlib_module_names}"
    )
