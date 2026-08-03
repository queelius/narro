"""Static guards for failure propagation and release artifact CI."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACTION_REFERENCE = re.compile(r"^\s*uses:\s+([^#\s]+)", re.MULTILINE)


def test_fresh_venv_smoke_propagates_failure_through_tee():
    workflow = (ROOT / ".github/workflows/fresh-venv-smoke.yml").read_text()
    smoke_step = workflow.split("- name: smoke-test fresh venv", 1)[1]
    assert "set -o pipefail" in smoke_step
    assert "| tee smoke-result.json" in smoke_step


def test_release_gate_validates_wheel_and_sdist_built_wheel():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    artifact_job = workflow.split("  artifacts:", 1)[1]
    assert "python -m build --sdist --wheel" in artifact_job
    assert "rebuild wheel from sdist" in artifact_job
    assert artifact_job.count("validate_installed_distribution.py") == 2
    assert "python -m twine check" in artifact_job


def test_external_github_actions_are_pinned_to_full_commit_shas():
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        for reference in ACTION_REFERENCE.findall(path.read_text()):
            if reference.startswith(("./", "docker://")):
                continue
            action, separator, revision = reference.rpartition("@")
            assert action and separator, f"missing action revision in {path}: {reference}"
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"external action must use a full commit SHA in {path}: {reference}"
            )
