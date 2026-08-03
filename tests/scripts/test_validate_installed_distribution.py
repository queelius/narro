"""Unit tests for the installed-distribution artifact validator."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validate_installed_distribution.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_installed_distribution",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDistribution:
    def __init__(self, root: Path):
        self.version = "1.2.3"
        self.metadata = {"Requires-Python": ">=3.10"}
        self.requires = [
            "mcp<2,>=1.8.0",
            "sse-starlette<3.1.0,>=3.0.0",
            "requests",
        ]
        self.entry_points = [
            SimpleNamespace(
                group="console_scripts",
                name="muse",
                value="muse.cli:main",
            ),
        ]
        self._root = root

    def locate_file(self, relative: str) -> Path:
        return self._root / relative


def test_validate_distribution_accepts_complete_artifact(tmp_path):
    validator = _load()
    for relative in validator.REQUIRED_RUNTIME_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data")

    assert validator.validate_distribution(
        expected_version="1.2.3",
        distribution=_FakeDistribution(tmp_path),
    ) == []


def test_validate_distribution_reports_metadata_and_data_errors(tmp_path):
    validator = _load()
    dist = _FakeDistribution(tmp_path)
    dist.version = "0.0.0"
    dist.metadata = {"Requires-Python": ">=3.11"}
    dist.requires = ["mcp>=1.0.0"]
    dist.entry_points = []

    errors = validator.validate_distribution(
        expected_version="1.2.3",
        distribution=dist,
    )
    assert any("installed version" in error for error in errors)
    assert any("curated.yaml" in error for error in errors)
    assert any("console script" in error for error in errors)
    assert any("Requires-Python" in error for error in errors)
    assert any("mcp>=1.8.0,<2" in error for error in errors)
    assert any("sse-starlette>=3.0.0,<3.1.0" in error for error in errors)
