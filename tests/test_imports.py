import importlib

import pytest

MODULES = ["ingest", "rag", "dataset", "agent", "memory"]


def test_package_imports():
    package = importlib.import_module("tt2_agentic_rag")
    assert package.__version__


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"tt2_agentic_rag.{name}")
