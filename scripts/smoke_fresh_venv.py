"""Fresh-venv smoke-test for a Muse model (#124, v0.32.0).

Catches the production failure mode where a bundled script's
`pip_extras` declares the deps the runtime source-imports but misses
transitive deps that `from_pretrained` (or sentence-transformers, or
diffusers) pulls in at load time.

The host muse install (with broad dev extras: server, audio, images,
embeddings, dev) typically has those transitives already, so the test
suite passes. A fresh per-model venv created via `muse pull` does NOT,
because `pull` installs ONLY `museq[server]` plus the model's declared
`pip_extras`. Transitive holes show up there.

This script reproduces the `muse pull` install path against a clean
venv and then runs the in-venv probe worker. Most models use load-only
coverage; curated audio-analysis models run their short inference probe
as well so decoder dependencies are exercised. A failure surfaces the
missing dep in the worker's stderr and the script exits non-zero with
an informative label.

Usage (local):
    python scripts/smoke_fresh_venv.py --model_id kokoro-82m
    python scripts/smoke_fresh_venv.py --model_id dinov2-small --json

Usage (CI):
    See .github/workflows/fresh-venv-smoke.yml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("muse.smoke")


@dataclasses.dataclass
class SmokeResult:
    """Outcome of a smoke test for one model.

    `label` is the one-line summary CI surfaces in the job log (e.g.,
    "kokoro-82m: OK (12.3s)" or "kokoro-82m: FAIL (missing dep: librosa)").
    """
    model_id: str
    ok: bool
    error: str | None
    duration_s: float
    label: str


def _repo_root() -> Path:
    """Resolve the muse repo root (contains pyproject.toml).

    The script lives at <repo>/scripts/smoke_fresh_venv.py, so two
    parents up is the repo. Defensive: walk up until pyproject.toml
    appears, fall back to the two-parent default if not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def _run_load_only(
    venv_python: Path,
    model_id: str,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run the muse probe worker in load-only mode. Returns (rc, captured).

    The probe worker exists as a hidden CLI subcommand
    (`muse _probe_worker --model <id> --device cpu --no-inference`).
    It calls `load_backend(model_id)`, captures memory, prints a JSON
    record on stdout, and exits 0 on success. The smoke test only cares
    that it loads without ImportError, so we run it on the CPU device.

    `env` overrides the subprocess environment. The smoke pipeline always
    supplies the same isolated `MUSE_CATALOG_DIR` populated by its pull.
    """
    cmd = [
        str(venv_python), "-m", "muse.cli", "_probe_worker",
        "--model", model_id,
        "--device", "cpu",
        "--no-inference",
    ]
    logger.info("running load-only probe: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout + proc.stderr


def _run_inference_probe(
    venv_python: Path,
    model_id: str,
    *,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run representative inference and fail if the worker records an error."""
    cmd = [
        str(venv_python), "-m", "muse.cli", "_probe_worker",
        "--model", model_id,
        "--device", "cpu",
    ]
    logger.info("running inference probe: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    captured = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return proc.returncode, captured
    record = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            record = candidate
            break
    if record is None:
        return 1, captured + "\ninference probe returned no JSON record"
    if record.get("inference_error") or not record.get("ran_inference"):
        reason = record.get("inference_error") or "inference did not run"
        return 1, captured + f"\ninference probe failed: {reason}"
    return 0, captured


_MODULE_NOT_FOUND_RE = re.compile(
    r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"
)


def _extract_failure_reason(captured: str) -> str:
    """Pick the most informative failure line from worker output.

    Recognized patterns (best to least specific):
      - ModuleNotFoundError: 'foo'
      - ImportError: cannot import name 'foo' from 'bar'
      - the `load failed:` prefix the probe worker emits
      - last non-empty line as fallback
    """
    if not captured:
        return "no output"
    m = _MODULE_NOT_FOUND_RE.search(captured)
    if m:
        return f"missing dep: {m.group(1)}"
    for line in captured.splitlines():
        if line.startswith("ImportError:"):
            return line.strip()
        if line.startswith("load failed:"):
            return line.strip()
    # Fall back to the last non-empty line of stderr-style output.
    for line in reversed(captured.splitlines()):
        s = line.strip()
        if s and not s.startswith("{"):
            return s[:200]
    return "unknown failure"


def _smoke_pulled_model(model_id: str, venv_root: Path) -> SmokeResult:
    """Pull and probe any model inside a fresh isolated Muse catalog.

    A probe worker always requires a pulled catalog record. Using the real
    pull path for bundled and resolver-based models exercises venv creation,
    exact dependencies, reviewed sources, weight download, and persisted
    manifest state together instead of constructing an unregistered venv that
    can only fail ``load_backend``'s pulled-model guard.
    """
    t0 = time.monotonic()
    venv_root.mkdir(parents=True, exist_ok=True)
    catalog_dir = Path(tempfile.mkdtemp(prefix="muse-catalog-", dir=venv_root))
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(catalog_dir)

    repo_root = _repo_root()
    cmd = [
        sys.executable, "-m", "muse.cli", "pull", model_id, "--no-probe",
    ]
    logger.info("pulling model in isolated catalog: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    if proc.returncode != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(proc.stdout + proc.stderr)
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"muse pull failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL (pull: {reason})",
        )

    catalog_path = catalog_dir / "catalog.json"
    try:
        entry = json.loads(catalog_path.read_text()).get(model_id, {})
    except (OSError, ValueError) as e:
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"catalog unreadable after pull: {e}",
            duration_s=duration,
            label=f"{model_id}: FAIL (no catalog after pull)",
        )
    python_path = entry.get("python_path")
    if not python_path:
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error="pulled catalog entry has no python_path",
            duration_s=duration,
            label=f"{model_id}: FAIL (no python_path after pull)",
        )

    manifest = entry.get("manifest") or {}
    run_inference = manifest.get("modality") in {
        "audio/alignment", "audio/quality",
    }
    probe = _run_inference_probe if run_inference else _run_load_only
    rc, captured = probe(Path(python_path), model_id, env=env)
    if rc != 0:
        duration = time.monotonic() - t0
        reason = _extract_failure_reason(captured)
        stage = "inference" if run_inference else "load"
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"{stage} failed: {reason}",
            duration_s=duration,
            label=f"{model_id}: FAIL ({reason})",
        )

    duration = time.monotonic() - t0
    return SmokeResult(
        model_id=model_id,
        ok=True,
        error=None,
        duration_s=duration,
        label=f"{model_id}: OK ({duration:.1f}s)",
    )


def _smoke_curated_resolver(
    model_id: str,
    uri: str,
    venv_root: Path,
) -> SmokeResult:
    """Compatibility wrapper for curated-resolver smoke callers."""
    del uri  # Resolution uses the reviewed curated id through `muse pull`.
    return _smoke_pulled_model(model_id, venv_root)


def smoke_one(
    model_id: str,
    venv_root: Path,
) -> SmokeResult:
    """Run the isolated real-pull smoke pipeline for one known model."""
    from muse.core.catalog import known_models
    from muse.core.curated import find_curated

    t0 = time.monotonic()

    catalog_known = known_models()
    if model_id not in catalog_known:
        # Not a discovered bundled script. A curated resolver-based id
        # (has a `uri`, no bundled script) needs an actual `muse pull`
        # to synthesize + persist its manifest before it is loadable at
        # all -- see _smoke_curated_resolver for why.
        curated = find_curated(model_id)
        if curated is not None and curated.uri:
            return _smoke_curated_resolver(model_id, curated.uri, venv_root)
        duration = time.monotonic() - t0
        return SmokeResult(
            model_id=model_id,
            ok=False,
            error=f"unknown model {model_id!r}",
            duration_s=duration,
            label=f"{model_id}: FAIL (unknown model)",
        )
    return _smoke_pulled_model(model_id, venv_root)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="smoke_fresh_venv",
        description=(
            "Smoke-test a Muse model in a fresh per-model venv, "
            "verifying that pip_extras covers the runtime's load-time deps."
        ),
    )
    parser.add_argument("--model_id", required=True, help="model id (e.g., kokoro-82m)")
    parser.add_argument(
        "--venv_root",
        default=None,
        help="directory to create the smoke venv under (default: tempdir)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a JSON record on stdout (default: human-readable label)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.venv_root:
        venv_root = Path(args.venv_root)
        venv_root.mkdir(parents=True, exist_ok=True)
    else:
        venv_root = Path(tempfile.mkdtemp(prefix=f"muse-smoke-{args.model_id}-"))

    result = smoke_one(args.model_id, venv_root)

    if args.as_json:
        record = dataclasses.asdict(result)
        # Ensure stable JSON: drop the human label from the JSON body so
        # CI artifact JSON stays uncluttered.
        print(json.dumps(record, indent=2))
    else:
        print(result.label)

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
