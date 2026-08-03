"""Supply-chain inventory for pip requirements installed from VCS."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "muse"

_VCS_REQUIREMENT_START_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*\s*@\s*git\+",
    re.IGNORECASE,
)
_PINNED_VCS_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*@\s*git\+"
    r"(?P<url>(?:https?|ssh|git|file)://.+?\.git)@"
    r"(?P<revision>[0-9a-f]{40})(?:#[^\s]+)?$",
)
_VCS_REQUIREMENT_IN_TEXT_RE = re.compile(
    r"(?P<requirement>[A-Za-z0-9][A-Za-z0-9_.-]*\s*@\s*git\+"
    r"(?:https?|ssh|git|file)://[^\s\"']+)",
    re.IGNORECASE,
)

_REVIEWED_VCS_REQUIREMENTS = {
    (
        "src/muse/modalities/model_3d_generation/hf.py",
        "utils3d",
        "https://github.com/EasternJournalist/utils3d.git",
        "9a4eb15e4021b67b12c460c7057d642626897ec8",
    ),
    (
        "src/muse/modalities/model_3d_generation/hf.py",
        "hy3dgen",
        "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
        "f8db63096c8282cb27354314d896feba5ba6ff8a",
    ),
    (
        "src/muse/models/ace_step_v1_3_5b.py",
        "ace-step",
        "https://github.com/ace-step/ACE-Step.git",
        "1bee4c9f5b43e30995f8d4d33b3919197ce1bd68",
    ),
}

_REVIEWED_STRUCTURED_GIT_SOURCES = {
    (
        "src/muse/modalities/model_3d_generation/hf.py",
        "source:trellis",
        "https://github.com/microsoft/TRELLIS.git",
        "442aa1e1afb9014e80681d3bf604e8d728a86ee7",
    ),
    (
        "src/muse/modalities/model_3d_generation/hf.py",
        "submodule:trellis/representations/mesh/flexicubes",
        "https://github.com/MaxtirError/FlexiCubes.git",
        "815e075a2a400d06c48d94c347674344ed6ae5c5",
    ),
}


def _record_requirement(
    *,
    path: Path,
    lineno: int,
    requirement: str,
    discovered: set[tuple[str, str, str, str]],
) -> None:
    match = _PINNED_VCS_REQUIREMENT_RE.fullmatch(requirement)
    assert match is not None, (
        f"{path.relative_to(REPO_ROOT)}:{lineno} has an unpinned or "
        "malformed VCS requirement; use an official repository URL "
        "with an immutable full lowercase commit SHA"
    )
    discovered.add((
        str(path.relative_to(REPO_ROOT)),
        match.group("name"),
        match.group("url"),
        match.group("revision"),
    ))


def _declarative_dependency_files() -> set[Path]:
    paths = {REPO_ROOT / "pyproject.toml"}
    paths.update(REPO_ROOT.glob("requirements*.txt"))
    for suffix in ("*.toml", "*.yaml", "*.yml", "*.txt"):
        paths.update(SOURCE_ROOT.rglob(suffix))
    return {path for path in paths if path.is_file()}


def test_all_production_vcs_requirements_are_reviewed_commit_pins():
    """Every Git-backed production requirement is immutable and inventoried."""
    discovered: set[tuple[str, str, str, str]] = set()

    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            requirement = node.value.strip()
            if "git+" not in requirement.lower():
                continue
            assert _VCS_REQUIREMENT_START_RE.match(requirement), (
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} has a bare or "
                "malformed VCS requirement; use `distribution @ git+...` "
                "with an immutable reviewed commit"
            )
            _record_requirement(
                path=path,
                lineno=node.lineno,
                requirement=requirement,
                discovered=discovered,
            )

    for path in _declarative_dependency_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if line.lstrip().startswith("#"):
                continue
            candidates = list(_VCS_REQUIREMENT_IN_TEXT_RE.finditer(line))
            if "git+" in line.lower():
                assert candidates, (
                    f"{path.relative_to(REPO_ROOT)}:{lineno} has a bare or "
                    "malformed VCS requirement"
                )
            for candidate in candidates:
                _record_requirement(
                    path=path,
                    lineno=lineno,
                    requirement=candidate.group("requirement"),
                    discovered=discovered,
                )

    assert discovered == _REVIEWED_VCS_REQUIREMENTS


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            constants[target.id] = value.value
    return constants


def _literal_string(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _dict_fields(node: ast.Dict) -> dict[str, ast.AST]:
    fields: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            fields[key.value] = value
    return fields


def test_all_structured_git_sources_and_submodules_are_reviewed_commit_pins():
    """Inventory non-package Git checkouts as strictly as pip VCS URLs."""
    discovered: set[tuple[str, str, str, str]] = set()

    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            fields = _dict_fields(node)
            if _literal_string(fields.get("type"), constants) != "git":
                continue

            name = _literal_string(fields.get("name"), constants)
            url = _literal_string(fields.get("url"), constants)
            revision = _literal_string(fields.get("revision"), constants)
            assert name and url and revision, (
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} must use literal "
                "reviewable source identity fields"
            )
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} source is not pinned"
            )
            discovered.add((
                str(path.relative_to(REPO_ROOT)), f"source:{name}", url, revision,
            ))

            submodules = fields.get("submodules")
            if not isinstance(submodules, (ast.List, ast.Tuple)):
                continue
            for item in submodules.elts:
                assert isinstance(item, ast.Dict), (
                    f"{path.relative_to(REPO_ROOT)}:{item.lineno} submodule must be literal"
                )
                sub_fields = _dict_fields(item)
                sub_path = _literal_string(sub_fields.get("path"), constants)
                sub_url = _literal_string(sub_fields.get("url"), constants)
                sub_revision = _literal_string(
                    sub_fields.get("revision"), constants,
                )
                assert sub_path and sub_url and sub_revision
                assert re.fullmatch(r"[0-9a-f]{40}", sub_revision), (
                    f"{path.relative_to(REPO_ROOT)}:{item.lineno} submodule is not pinned"
                )
                discovered.add((
                    str(path.relative_to(REPO_ROOT)),
                    f"submodule:{sub_path}", sub_url, sub_revision,
                ))

    assert discovered == _REVIEWED_STRUCTURED_GIT_SOURCES
