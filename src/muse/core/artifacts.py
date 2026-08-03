"""Safe local bundles for models composed from multiple HF repositories."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_LEAF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ArtifactBundleError(RuntimeError):
    """Raised when an immutable local artifact bundle is unsafe or corrupt."""


@dataclass(frozen=True)
class HFSnapshotArtifact:
    """One immutable Hugging Face snapshot inside a local bundle."""

    repo_id: str
    revision: str
    subdir: str
    allow_patterns: tuple[str, ...] | None = None
    required_patterns: tuple[str, ...] | None = None


def _normalize_required_patterns(
    raw: Any,
    *,
    index: int,
) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not raw
        or not all(isinstance(pattern, str) and pattern for pattern in raw)
    ):
        raise ArtifactBundleError(
            f"hf_artifacts[{index}].required_patterns must be a non-empty "
            "sequence of non-empty strings"
        )
    patterns = tuple(raw)
    for pattern in patterns:
        parts = pattern.split("/")
        if (
            pattern.startswith(("/", "\\"))
            or "\\" in pattern
            or ":" in pattern
            or "\x00" in pattern
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0] == ".cache"
        ):
            raise ArtifactBundleError(
                f"hf_artifacts[{index}].required_patterns contains an "
                f"unsafe relative pattern: {pattern!r}"
            )
    return patterns


def normalize_hf_artifacts(
    artifacts: Sequence[HFSnapshotArtifact | Mapping[str, Any]],
) -> tuple[HFSnapshotArtifact, ...]:
    """Validate and copy a manifest-provided artifact descriptor."""
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise ArtifactBundleError("hf_artifacts must be a non-string sequence")
    if len(artifacts) < 2:
        raise ArtifactBundleError("an artifact bundle requires at least two snapshots")

    normalized: list[HFSnapshotArtifact] = []
    seen_subdirs: set[str] = set()
    for index, raw in enumerate(artifacts):
        if isinstance(raw, HFSnapshotArtifact):
            artifact = raw
        elif isinstance(raw, Mapping):
            repo_id = raw.get("repo_id")
            revision = raw.get("revision")
            subdir = raw.get("subdir")
            allow_patterns = raw.get("allow_patterns")
            required_patterns = _normalize_required_patterns(
                raw.get("required_patterns"),
                index=index,
            )
            if allow_patterns is not None:
                if (
                    isinstance(allow_patterns, (str, bytes))
                    or not isinstance(allow_patterns, Sequence)
                    or not all(
                        isinstance(pattern, str) and pattern
                        for pattern in allow_patterns
                    )
                ):
                    raise ArtifactBundleError(
                        f"hf_artifacts[{index}].allow_patterns must be "
                        "a sequence of non-empty strings"
                    )
                allow_patterns = tuple(allow_patterns)
            artifact = HFSnapshotArtifact(
                repo_id=repo_id,
                revision=revision,
                subdir=subdir,
                allow_patterns=allow_patterns,
                required_patterns=required_patterns,
            )
        else:
            raise ArtifactBundleError(
                f"hf_artifacts[{index}] must be an object"
            )

        if (
            not isinstance(artifact.repo_id, str)
            or artifact.repo_id.count("/") != 1
            or any(char.isspace() for char in artifact.repo_id)
        ):
            raise ArtifactBundleError(
                f"hf_artifacts[{index}].repo_id is invalid"
            )
        if (
            not isinstance(artifact.revision, str)
            or _COMMIT_RE.fullmatch(artifact.revision) is None
        ):
            raise ArtifactBundleError(
                f"hf_artifacts[{index}].revision must be a 40-character commit"
            )
        if (
            not isinstance(artifact.subdir, str)
            or _SAFE_LEAF_RE.fullmatch(artifact.subdir) is None
            or artifact.subdir in {".", ".."}
        ):
            raise ArtifactBundleError(
                f"hf_artifacts[{index}].subdir must be a safe directory name"
            )
        normalized_allow_patterns = artifact.allow_patterns
        if normalized_allow_patterns is not None:
            if (
                isinstance(normalized_allow_patterns, (str, bytes))
                or not isinstance(normalized_allow_patterns, Sequence)
                or not normalized_allow_patterns
                or not all(
                    isinstance(pattern, str) and pattern
                    for pattern in normalized_allow_patterns
                )
            ):
                raise ArtifactBundleError(
                    f"hf_artifacts[{index}].allow_patterns must be a "
                    "non-empty sequence of non-empty strings"
                )
            normalized_allow_patterns = tuple(normalized_allow_patterns)
        normalized_required_patterns = _normalize_required_patterns(
            artifact.required_patterns,
            index=index,
        )
        artifact = HFSnapshotArtifact(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            subdir=artifact.subdir,
            allow_patterns=normalized_allow_patterns,
            required_patterns=normalized_required_patterns,
        )
        if artifact.subdir in seen_subdirs:
            raise ArtifactBundleError(
                f"duplicate hf_artifacts subdirectory {artifact.subdir!r}"
            )
        seen_subdirs.add(artifact.subdir)
        normalized.append(artifact)
    return tuple(normalized)


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArtifactBundleError(f"{label} directory is missing: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise ArtifactBundleError(f"{label} path is not a real directory: {path}")


def _validate_bundle(
    bundle: Path,
    artifacts: Sequence[HFSnapshotArtifact],
) -> None:
    _require_real_directory(bundle, label="artifact bundle")
    for artifact in artifacts:
        member = bundle / artifact.subdir
        _require_real_directory(member, label=f"artifact {artifact.subdir!r}")
        payload_files: list[PurePosixPath] = []
        for descendant in member.rglob("*"):
            try:
                mode = descendant.lstat().st_mode
            except OSError as exc:
                raise ArtifactBundleError(
                    f"could not inspect artifact {artifact.subdir!r}: "
                    f"{descendant}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ArtifactBundleError(
                    f"artifact {artifact.subdir!r} contains a symlink: "
                    f"{descendant}"
                )
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ArtifactBundleError(
                    f"artifact {artifact.subdir!r} contains a special file: "
                    f"{descendant}"
                )
            # huggingface_hub may create local-dir bookkeeping below
            # .cache even when no requested payload matched.  Metadata alone
            # is not a usable model snapshot.
            relative = descendant.relative_to(member)
            if not relative.parts or relative.parts[0] != ".cache":
                payload_files.append(PurePosixPath(relative.as_posix()))
        if not payload_files:
            raise ArtifactBundleError(
                f"artifact {artifact.subdir!r} contains no payload files"
            )
        for pattern in artifact.required_patterns or ():
            if not any(path.match(pattern) for path in payload_files):
                raise ArtifactBundleError(
                    f"artifact {artifact.subdir!r} is missing required "
                    f"payload pattern {pattern!r}"
                )


def _bundle_target(
    cache_root: Path,
    bundle_name: str,
    artifacts: Sequence[HFSnapshotArtifact],
) -> Path:
    if _SAFE_LEAF_RE.fullmatch(bundle_name) is None or bundle_name in {".", ".."}:
        raise ArtifactBundleError("bundle_name must be a safe directory name")
    identity = json.dumps(
        [
            {
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "subdir": artifact.subdir,
                "allow_patterns": artifact.allow_patterns,
                "required_patterns": artifact.required_patterns,
            }
            for artifact in artifacts
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return cache_root / f"{bundle_name}-{digest}"


def _cleanup_staging(staging: Path, cache_root: Path) -> None:
    if staging.parent != cache_root or not staging.name.startswith("."):
        raise ArtifactBundleError(f"refusing unsafe staging cleanup: {staging}")
    try:
        mode = staging.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        staging.unlink()
        return
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ArtifactBundleError(
            f"refusing recursive bundle cleanup without fd-safe rmtree: {staging}"
        )
    shutil.rmtree(staging)


def download_hf_artifact_bundle(
    cache_root: Path,
    *,
    bundle_name: str,
    artifacts: Sequence[HFSnapshotArtifact | Mapping[str, Any]],
    snapshot_download_fn: Callable[..., str],
) -> Path:
    """Download immutable snapshots and atomically publish one local bundle."""
    normalized = normalize_hf_artifacts(artifacts)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _require_real_directory(cache_root, label="artifact cache root")
    target = _bundle_target(cache_root, bundle_name, normalized)
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ArtifactBundleError(
            f"cannot inspect artifact bundle {target}: {exc}"
        ) from exc
    else:
        _validate_bundle(target, normalized)
        return target

    staging = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-",
        dir=cache_root,
    ))
    try:
        for artifact in normalized:
            kwargs: dict[str, Any] = {
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "local_dir": str(staging / artifact.subdir),
            }
            if artifact.allow_patterns is not None:
                kwargs["allow_patterns"] = list(artifact.allow_patterns)
            snapshot_download_fn(**kwargs)
        _validate_bundle(staging, normalized)
        try:
            staging.rename(target)
        except FileExistsError:
            _validate_bundle(target, normalized)
        return target
    finally:
        preserving_exception = sys.exc_info()[0] is not None
        try:
            _cleanup_staging(staging, cache_root)
        except Exception:  # noqa: BLE001
            if preserving_exception:
                # Preserve the download/cancellation error already in flight.
                logger.warning(
                    "failed to clean artifact staging %s", staging, exc_info=True,
                )
            else:
                raise


def local_artifact_directory(
    local_dir: str,
    subdir: str,
    *,
    label: str,
) -> str:
    """Return one real direct child of a Muse-managed artifact bundle."""
    if (
        not isinstance(subdir, str)
        or _SAFE_LEAF_RE.fullmatch(subdir) is None
        or subdir in {".", ".."}
    ):
        raise ArtifactBundleError(f"invalid {label} bundle subdirectory: {subdir!r}")
    bundle = Path(local_dir)
    _require_real_directory(bundle, label="local artifact bundle")
    member = bundle / subdir
    _require_real_directory(member, label=label)
    return str(member)
