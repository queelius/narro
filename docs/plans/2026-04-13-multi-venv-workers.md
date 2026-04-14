# Muse Multi-Venv Workers + Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `muse serve` into a supervisor that spawns one worker subprocess per venv group and runs a thin HTTP gateway that proxies requests by model-id. Each pulled model gets its own venv so ML library dep conflicts (transformers 4.46 vs 5.x, parler-tts/unsloth clashes) are structurally impossible.

**Architecture:** `muse pull <id>` creates `~/.muse/venvs/<id>/` and installs that model's `pip_extras` into it via `python -m venv` + `pip install`. The catalog records the venv's Python path per model. `muse serve` reads the catalog, groups models by venv (today: always one-per-model; future: auto-grouping via uv), allocates a port per venv (starting at 9001), spawns each worker via `subprocess.Popen([venv_python, "-m", "muse.cli", "_worker", ...])`, polls `/health` until ready, then runs a FastAPI gateway on the user-facing port. The gateway extracts `model` from request body (POSTs) or query (GETs), looks up which worker hosts that model, and proxies the request transparently — SSE streams pass through without buffering. `/v1/models` and `/health` are aggregated in the gateway by parallel httpx calls to each worker. All of `muse.audio.speech.*` and `muse.images.generations.*` is untouched — the worker IS the current `muse serve` logic, renamed.

**Tech Stack:** stdlib `venv` + `subprocess` (no new deps for venv creation), `httpx` for gateway proxying (moved from `[dev]` to `[server]` extras), FastAPI + uvicorn (already in server extras), `asyncio` for parallel aggregation. Python 3.10+.

---

## File Structure (final)

```
src/muse/
├── cli.py                     # Modified: add _worker hidden subcommand; serve now = supervisor
├── cli_impl/
│   ├── serve.py               # Overwritten: now the supervisor entrypoint (was run_serve for single-process)
│   ├── worker.py              # NEW: single-worker mode — extracted from old serve.py
│   ├── gateway.py             # NEW: FastAPI proxy app (~200 lines)
│   ├── supervisor.py          # NEW: subprocess lifecycle, port allocation, shutdown (~150 lines)
│   ├── speak.py               # Unchanged
│   └── imagine.py             # Unchanged
├── core/
│   ├── catalog.py             # Modified: pull() creates venv; CatalogEntry unchanged; catalog.json schema gains venv_path + python_path
│   ├── venv.py                # NEW: create_venv, install_pip_into_venv, find_free_port (~80 lines)
│   ├── registry.py            # Unchanged
│   ├── server.py              # Unchanged (still the worker's app factory)
│   ├── errors.py              # Unchanged
│   └── install.py             # Modified: install_pip_extras grows a venv-scoped variant
├── audio/                     # Unchanged
└── images/                    # Unchanged

tests/
├── core/
│   ├── test_venv.py           # NEW
│   └── test_catalog.py        # Modified: pull() now creates venvs (heavily mocked)
└── cli_impl/
    ├── test_worker.py         # NEW: lean — worker is mostly existing code
    ├── test_gateway.py        # NEW: FastAPI TestClient + mocked httpx backend
    └── test_supervisor.py     # NEW: subprocess.Popen mocked
```

---

## Key design decisions locked in by this file structure

1. **`muse._worker` is a hidden CLI subcommand**, not a separate Python module invokable via `-m`. This keeps all argparse in one place. The supervisor invokes it as `python -m muse.cli _worker ...`.
2. **One venv per pulled model** in this iteration. Auto-grouping by compatible deps is deferred — not impossible, just out of scope. `CatalogEntry.extra` could hold a `venv_group` hint later without schema changes.
3. **Gateway routes by `model` field only**, not by URL path. The rule: POST with JSON body → look up `body["model"]`; GET → look up `query_params["model"]`; special paths (`/v1/models`, `/health`) → aggregate. This pattern works identically for a future `/v1/embeddings` endpoint because routing depends on the model ID, not the URL shape.
4. **Backward-incompat catalog migration**: existing catalog entries without a `python_path` field are logged and skipped by `muse serve` with a warning to re-pull. No silent fallback to in-process mode — keeps the code path singular.
5. **Workers are unaware they're under a supervisor**. Each worker is exactly the current `muse serve` minus the launcher logic. Workers can be run standalone for debugging (`python -m muse.cli _worker --port 9999 --model soprano-80m`).

---

## Part A — Venv Infrastructure

### Task A1: `muse.core.venv` module

**Files:**
- Create: `src/muse/core/venv.py`
- Create: `tests/core/test_venv.py`

- [ ] **Step 1: Write the failing test**

File: `tests/core/test_venv.py`

```python
"""Tests for venv creation + pip install helpers."""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.core.venv import (
    create_venv,
    install_into_venv,
    venv_python,
    find_free_port,
)


class TestVenvPython:
    def test_returns_bin_python_on_posix(self, tmp_path):
        # On POSIX venv layout, python is at <venv>/bin/python
        path = venv_python(tmp_path)
        assert path == tmp_path / "bin" / "python"


class TestCreateVenv:
    @patch("muse.core.venv.subprocess.run")
    def test_calls_python_venv_module(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        target = tmp_path / "myenv"
        create_venv(target)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        # Use sys.executable to guarantee we create the venv with the same
        # Python that muse is running on (matters for ABI compatibility)
        import sys
        assert args[0] == sys.executable
        assert "-m" in args and "venv" in args
        assert str(target) in args

    @patch("muse.core.venv.subprocess.run")
    def test_raises_on_venv_creation_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["python", "-m", "venv"])
        with pytest.raises(subprocess.CalledProcessError):
            create_venv(tmp_path / "doomed")


class TestInstallIntoVenv:
    @patch("muse.core.venv.subprocess.run")
    def test_uses_venvs_pip_not_system_pip(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        # Simulate a venv layout
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        install_into_venv(tmp_path, ["numpy", "scipy"])
        args = mock_run.call_args[0][0]
        # Must be <venv>/bin/python -m pip install <pkgs>
        assert args[0] == str(tmp_path / "bin" / "python")
        assert args[1:4] == ["-m", "pip", "install"]
        assert "numpy" in args
        assert "scipy" in args

    @patch("muse.core.venv.subprocess.run")
    def test_empty_package_list_is_noop(self, mock_run, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        install_into_venv(tmp_path, [])
        mock_run.assert_not_called()

    @patch("muse.core.venv.subprocess.run")
    def test_raises_on_install_failure(self, mock_run, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip"])
        with pytest.raises(subprocess.CalledProcessError):
            install_into_venv(tmp_path, ["bogus"])


class TestFindFreePort:
    def test_returns_an_int_in_range(self):
        p = find_free_port(start=9001, end=9999)
        assert 9001 <= p <= 9999

    def test_skips_bound_ports(self):
        import socket
        # Bind 9001 so find_free_port must skip it
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 9001))
            s.listen(1)
            p = find_free_port(start=9001, end=9003)
            assert p != 9001
        finally:
            s.close()

    def test_raises_when_no_free_port_in_range(self):
        import socket
        sockets = []
        try:
            # Bind every port in a tiny range
            for port in (19001, 19002, 19003):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                s.listen(1)
                sockets.append(s)
            with pytest.raises(RuntimeError, match="no free port"):
                find_free_port(start=19001, end=19003)
        finally:
            for s in sockets:
                s.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/core/test_venv.py -v
```

Expected: `ModuleNotFoundError: No module named 'muse.core.venv'`

- [ ] **Step 3: Implement the module**

File: `src/muse/core/venv.py`

```python
"""Venv management helpers.

Each pulled model gets its own venv under ~/.muse/venvs/<model-id>/.
This module handles creation and pip-install-into-venv; the catalog
records the resulting Python interpreter path per model.
"""
from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def venv_python(venv_path: Path) -> Path:
    """Return the Python interpreter path inside a venv.

    POSIX layout only (bin/python). The Windows layout (Scripts/python.exe)
    is not supported because muse is Linux/macOS-focused.
    """
    return venv_path / "bin" / "python"


def create_venv(target: Path) -> None:
    """Create a fresh venv at `target`, using the same Python that muse runs on.

    Using `sys.executable` guarantees ABI compatibility: the venv's Python
    is the same version as the muse-supervisor's Python, so torch/CUDA
    wheels built for one will load in the other.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("creating venv at %s", target)
    subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        check=True,
    )


def install_into_venv(venv_path: Path, packages: list[str]) -> None:
    """pip-install `packages` using the venv's own pip.

    Shells out to `<venv>/bin/python -m pip install ...` so installs
    land in the target venv, not the supervisor's env.
    """
    if not packages:
        return
    py = venv_python(venv_path)
    logger.info("installing %s into %s", packages, venv_path)
    subprocess.run(
        [str(py), "-m", "pip", "install", *packages],
        check=True,
    )


def find_free_port(start: int = 9001, end: int = 9999) -> int:
    """Find an unbound local port in [start, end]. Raises if exhausted."""
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free port in range [{start}, {end}]")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_venv.py -v
```

Expected: all 9 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/venv.py tests/core/test_venv.py
git commit -m "feat(core): add venv management helpers

create_venv, install_into_venv, venv_python, find_free_port.
Each pulled model will get its own venv via these helpers in Task B1."
```

---

## Part B — Catalog + Pull Integration

### Task B1: Extend catalog to create venvs on pull

**Files:**
- Modify: `src/muse/core/catalog.py` — `pull()` now creates a venv
- Modify: `tests/core/test_catalog.py` — cover the new venv behavior
- Reference: `src/muse/core/venv.py` (from Task A1)

- [ ] **Step 1: Add failing tests to `tests/core/test_catalog.py`**

Insert after the existing `test_pull_warns_on_missing_system_packages` test (or anywhere in the file; ordering doesn't matter):

```python
def test_pull_creates_venv_under_muse_catalog_dir(tmp_catalog):
    """pull() must create a venv at <MUSE_CATALOG_DIR>/venvs/<model-id>/."""
    with patch("muse.core.catalog.create_venv") as mock_create, \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
        mock_create.assert_called_once()
        venv_target = mock_create.call_args[0][0]
        expected = tmp_catalog / "venvs" / "soprano-80m"
        assert venv_target == expected


def test_pull_installs_pip_extras_into_venv_not_system(tmp_catalog):
    """pip_extras go into the venv, never the supervisor's env."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
        mock_install.assert_called_once()
        venv_arg, packages_arg = mock_install.call_args[0]
        assert venv_arg == tmp_catalog / "venvs" / "soprano-80m"
        # transformers and scipy are in soprano-80m's pip_extras
        assert any("transformers" in p for p in packages_arg)


def test_pull_records_venv_path_and_python_in_catalog(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    catalog = _read_catalog()
    entry = catalog["soprano-80m"]
    assert "venv_path" in entry
    assert entry["venv_path"] == str(tmp_catalog / "venvs" / "soprano-80m")
    assert "python_path" in entry
    assert entry["python_path"] == str(tmp_catalog / "venvs" / "soprano-80m" / "bin" / "python")


def test_pull_skips_pip_extras_arg_to_install_pip_extras(tmp_catalog):
    """The old system-wide install_pip_extras must NOT be called — it's venv-scoped now."""
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.install_pip_extras") as mock_system_install, \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    mock_system_install.assert_not_called()
```

Also add an import at the top of the test file (next to existing imports):

```python
from muse.core.catalog import _read_catalog  # already imported in test file
```

(Verify `_read_catalog` is already imported there — if not, add it.)

- [ ] **Step 2: Run — these four tests must fail**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/core/test_catalog.py::test_pull_creates_venv_under_muse_catalog_dir \
       tests/core/test_catalog.py::test_pull_installs_pip_extras_into_venv_not_system \
       tests/core/test_catalog.py::test_pull_records_venv_path_and_python_in_catalog \
       tests/core/test_catalog.py::test_pull_skips_pip_extras_arg_to_install_pip_extras -v
```

Expected: all four FAIL — the current `pull()` calls the system-wide `install_pip_extras`, not `install_into_venv`, and does not create venvs.

- [ ] **Step 3: Update `pull()` in `src/muse/core/catalog.py`**

Add imports at the top (alongside existing imports):

```python
from muse.core.venv import create_venv, install_into_venv, venv_python
```

Replace the body of `pull()` — find the current function and replace it entirely with:

```python
def pull(model_id: str) -> None:
    """Create per-model venv, install deps into it, download weights, record state.

    Each pulled model gets `<MUSE_CATALOG_DIR>/venvs/<model-id>/` with its
    `pip_extras` installed inside. The catalog records the venv's Python path
    so `muse serve` can spawn workers with the right interpreter.
    """
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
    entry = KNOWN_MODELS[model_id]

    # Venv lives under the catalog dir so MUSE_CATALOG_DIR controls everything
    venvs_root = _catalog_dir() / "venvs"
    venv_path = venvs_root / model_id

    # Create venv (idempotent: if it exists, re-use; venv -m skips if present)
    if not venv_path.exists():
        create_venv(venv_path)

    # Install pip_extras INTO the venv, not the supervisor's env
    if entry.pip_extras:
        install_into_venv(venv_path, list(entry.pip_extras))

    if entry.system_packages:
        missing = check_system_packages(list(entry.system_packages))
        if missing:
            logger.warning(
                "model %s needs system packages not found on PATH: %s "
                "(install via apt/brew before running)",
                model_id, missing,
            )

    local_dir = snapshot_download(repo_id=entry.hf_repo)

    catalog = _read_catalog()
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": entry.hf_repo,
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
    }
    _write_catalog(catalog)
```

Note: we keep `install_pip_extras` importable (still used by old tests that we won't break), but `pull()` no longer calls it. The `from muse.core.install import check_system_packages, install_pip_extras` line at the top of catalog.py stays — `check_system_packages` is still used.

- [ ] **Step 4: Run new tests — verify pass**

```bash
pytest tests/core/test_catalog.py::test_pull_creates_venv_under_muse_catalog_dir \
       tests/core/test_catalog.py::test_pull_installs_pip_extras_into_venv_not_system \
       tests/core/test_catalog.py::test_pull_records_venv_path_and_python_in_catalog \
       tests/core/test_catalog.py::test_pull_skips_pip_extras_arg_to_install_pip_extras -v
```

Expected: all 4 pass.

- [ ] **Step 5: Regression check — full catalog tests must still pass**

```bash
pytest tests/core/ -v 2>&1 | tail -5
```

Expected: all pass. The old `test_pull_installs_pip_downloads_and_writes_catalog` test still mocks `install_pip_extras` but now the code doesn't call it — that mock assertion will FAIL. Fix that test in the next step.

- [ ] **Step 6: Fix the stale test in `tests/core/test_catalog.py`**

Find `test_pull_installs_pip_downloads_and_writes_catalog`. Replace its body so it mocks `install_into_venv` + `create_venv` instead of `install_pip_extras`:

```python
def test_pull_installs_pip_downloads_and_writes_catalog(tmp_catalog):
    with patch("muse.core.catalog.create_venv") as mock_create, \
         patch("muse.core.catalog.install_into_venv") as mock_install, \
         patch("muse.core.catalog.snapshot_download") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        mock_download.return_value = "/fake/cache/soprano"
        pull("soprano-80m")
        mock_create.assert_called_once()
        mock_install.assert_called_once()
        mock_download.assert_called_once()
        assert is_pulled("soprano-80m")
```

Also check `test_pull_warns_on_missing_system_packages`, `test_remove_clears_from_catalog`, and `test_load_backend_imports_and_constructs` — any that mock `install_pip_extras` need to mock `create_venv` + `install_into_venv` instead. Update each the same way.

- [ ] **Step 7: Run all catalog tests — pass**

```bash
pytest tests/core/test_catalog.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "feat(core): muse pull now creates per-model venv

pull(<id>) creates \`<MUSE_CATALOG_DIR>/venvs/<id>/\` and installs
that model's pip_extras into it via install_into_venv. Catalog
records venv_path and python_path so muse serve can spawn workers
with the right interpreter.

No longer uses the system-wide install_pip_extras — pip_extras
never pollute the supervisor's environment."
```

---

## Part C — Worker Subcommand

### Task C1: Extract current serve logic into `cli_impl/worker.py`

**Files:**
- Create: `src/muse/cli_impl/worker.py` — single-model-in-venv worker (moved from serve.py)
- Modify: `src/muse/cli_impl/serve.py` — will be rewritten as supervisor in Task E4 (for now, keep as-is so tests don't break mid-plan)
- Create: `tests/cli_impl/__init__.py`
- Create: `tests/cli_impl/test_worker.py`

- [ ] **Step 1: Create `src/muse/cli_impl/worker.py`**

This is nearly verbatim from the current `serve.py`. Copy and adapt. File:

```python
"""`muse _worker` implementation — runs ONE worker (optionally with multiple models from the same venv) and starts uvicorn.

Invoked by the supervisor (`muse serve`) via subprocess:
    <venv>/bin/python -m muse.cli _worker --port 9001 --model soprano-80m

Can also be run standalone for debugging. Not advertised in top-level help.
"""
from __future__ import annotations

import logging

from muse.core.catalog import KNOWN_MODELS, is_pulled, load_backend
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app

log = logging.getLogger(__name__)


def run_worker(*, host: str, port: int, models: list[str], device: str) -> int:
    """Load the specified models into a registry and run uvicorn.

    `models` is the exact set of model-ids to load into this process.
    The supervisor decides which models share a worker; the worker just
    loads what it's told.
    """
    import uvicorn

    registry = ModalityRegistry()
    routers: dict = {}

    to_load = [m for m in models if m in KNOWN_MODELS]
    unknown = [m for m in models if m not in KNOWN_MODELS]
    if unknown:
        log.warning("ignoring unknown models: %s", unknown)

    if not to_load:
        log.warning("worker started with no models; serving empty-registry responses")

    for model_id in to_load:
        if not is_pulled(model_id):
            log.error("model %s not pulled; skipping", model_id)
            continue
        entry = KNOWN_MODELS[model_id]
        log.info("loading %s (%s)", model_id, entry.modality)
        try:
            backend = load_backend(model_id, device=device)
        except Exception as e:
            log.error("failed to load %s: %s", model_id, e)
            continue
        registry.register(entry.modality, backend)

    # Always mount modality routers (even if registry is empty for that modality)
    # so unknown-model requests get the OpenAI envelope rather than FastAPI's
    # default 404.
    from muse.audio.speech.routes import build_router as build_audio
    from muse.images.generations.routes import build_router as build_images

    routers["audio.speech"] = build_audio(registry)
    routers["images.generations"] = build_images(registry)

    app = create_app(registry=registry, routers=routers)
    uvicorn.run(app, host=host, port=port, log_config=None)
    return 0
```

- [ ] **Step 2: Write a smoke test for run_worker (fully mocked — no uvicorn in test)**

File: `tests/cli_impl/test_worker.py`

```python
"""Smoke tests for run_worker (single-worker mode)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.worker import run_worker


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.load_backend")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
def test_worker_loads_requested_models_and_runs_uvicorn(mock_pulled, mock_load, mock_uvicorn):
    fake_backend = MagicMock(model_id="soprano-80m", sample_rate=32000)
    mock_load.return_value = fake_backend

    run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")

    # load_backend was called for the model
    mock_load.assert_called_once_with("soprano-80m", device="cpu")
    # uvicorn was told to serve on the requested port
    mock_uvicorn.run.assert_called_once()
    kwargs = mock_uvicorn.run.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=True)
@patch("muse.cli_impl.worker.load_backend")
def test_worker_skips_load_failures_without_crashing(mock_load, mock_pulled, mock_uvicorn):
    mock_load.side_effect = RuntimeError("diffusers not installed")
    # Should not raise; worker starts empty
    run_worker(host="127.0.0.1", port=9999, models=["sd-turbo"], device="cpu")
    mock_uvicorn.run.assert_called_once()


@patch("muse.cli_impl.worker.uvicorn")
@patch("muse.cli_impl.worker.is_pulled", return_value=False)
def test_worker_skips_unpulled_models_without_crashing(mock_pulled, mock_uvicorn):
    run_worker(host="127.0.0.1", port=9999, models=["soprano-80m"], device="cpu")
    mock_uvicorn.run.assert_called_once()


@patch("muse.cli_impl.worker.uvicorn")
def test_worker_ignores_unknown_model_ids(mock_uvicorn, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    run_worker(host="127.0.0.1", port=9999, models=["bogus-model-xyz"], device="cpu")
    mock_uvicorn.run.assert_called_once()
    assert "ignoring unknown models" in caplog.text
```

Create the test dir:

```bash
mkdir -p tests/cli_impl
touch tests/cli_impl/__init__.py
```

- [ ] **Step 3: Run worker tests — they must pass**

```bash
pytest tests/cli_impl/test_worker.py -v
```

Expected: 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/muse/cli_impl/worker.py tests/cli_impl/
git commit -m "feat(cli): add worker subcommand module (extracted from serve)

cli_impl.worker.run_worker runs a single worker process: loads
specified models from catalog, registers into a ModalityRegistry,
mounts modality routers, starts uvicorn. Always mounts both routers
so empty-registry 404s still use the OpenAI envelope.

Invoked by the supervisor (next task) via subprocess. The old
serve.py is still in place; it'll be repurposed as the supervisor
in Task E4."
```

---

### Task C2: Register `_worker` subcommand in CLI

**Files:**
- Modify: `src/muse/cli.py` — add `_worker` hidden subcommand
- Modify: `tests/test_cli.py` — verify the subcommand exists (but don't test its behavior there)

- [ ] **Step 1: Add failing test to `tests/test_cli.py`**

Add at the end of the file:

```python
def test_worker_subcommand_accepts_port_and_model():
    """`muse _worker --port N --model X` must parse without error."""
    r = _run("_worker", "--port", "9999", "--model", "soprano-80m", "--help")
    # --help short-circuits argparse; exit 0 + help text for the _worker subcommand
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "--port" in combined
    assert "--model" in combined


def test_worker_is_hidden_from_top_level_help():
    """Running `muse --help` should NOT prominently feature _worker.

    We keep it hidden because it's an internal subcommand invoked by
    the supervisor, not a user-facing API.
    """
    r = _run("--help")
    # We don't strictly hide it (argparse makes that hard without hacks),
    # but it must start with an underscore to signal private.
    combined = r.stdout + r.stderr
    # Either _worker isn't in top-level help, or it is but with the
    # underscore prefix (which documents privacy).
    assert "_worker" not in combined or "_worker" in combined  # accept either for now
```

Actually just keep the first test. Delete the second one — argparse doesn't easily hide subcommands without monkey-patching, and the underscore prefix is convention, not enforcement.

Final content to add (just the first test):

```python
def test_worker_subcommand_accepts_port_and_model():
    """`muse _worker --port N --model X` must parse without error."""
    r = _run("_worker", "--port", "9999", "--model", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "--port" in combined
    assert "--model" in combined
```

- [ ] **Step 2: Run — verify it fails**

```bash
pytest tests/test_cli.py::test_worker_subcommand_accepts_port_and_model -v
```

Expected: FAIL because `_worker` subcommand doesn't exist.

- [ ] **Step 3: Add the subcommand to `src/muse/cli.py`**

Open `src/muse/cli.py`. In `build_parser()`, add a new subparser — put it AFTER the existing `serve` subparser definition but BEFORE the alias subparsers (speak/imagine):

```python
    # _worker (internal; invoked by supervisor)
    sp_worker = sub.add_parser("_worker", help="internal: run a single worker (used by muse serve)")
    sp_worker.add_argument("--host", default="127.0.0.1",
                           help="bind address (default: 127.0.0.1 — workers are internal)")
    sp_worker.add_argument("--port", type=int, required=True)
    sp_worker.add_argument("--model", action="append", default=[], required=True,
                           help="model to load (repeatable; one worker can host multiple compatible models)")
    sp_worker.add_argument("--device", default="auto",
                           choices=["auto", "cpu", "cuda", "mps"])
    sp_worker.set_defaults(func=_cmd_worker)
```

Also add the handler function alongside `_cmd_serve`, `_cmd_pull`, etc.:

```python
def _cmd_worker(args):
    from muse.cli_impl.worker import run_worker
    return run_worker(
        host=args.host, port=args.port,
        models=args.model, device=args.device,
    )
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/test_cli.py::test_worker_subcommand_accepts_port_and_model -v
```

Expected: pass.

- [ ] **Step 5: Full CLI tests — no regressions**

```bash
pytest tests/test_cli.py -v 2>&1 | tail -5
```

Expected: all 11 pass (10 existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli.py tests/test_cli.py
git commit -m "feat(cli): add hidden _worker subcommand

Invoked by the supervisor (Task E4) via subprocess:
    <venv>/bin/python -m muse.cli _worker --port 9001 --model X

Underscore prefix signals internal API; runnable standalone for
debugging a specific worker in isolation."
```

---

## Part D — Gateway

### Task D1: Gateway request parsing

**Files:**
- Create: `src/muse/cli_impl/gateway.py` (initial skeleton + request parsing)
- Create: `tests/cli_impl/test_gateway.py`

The gateway is a FastAPI proxy. Its core job: look at incoming request, figure out which worker hosts the requested model, forward the request. This task implements the "figure out which model" part, which is the trickiest piece.

- [ ] **Step 1: Write failing tests for model extraction**

File: `tests/cli_impl/test_gateway.py`

```python
"""Tests for the gateway proxy FastAPI app."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import (
    extract_model_from_request,
    build_gateway,
    WorkerRoute,
)


class TestExtractModel:
    @pytest.mark.asyncio
    async def test_extracts_model_from_json_body(self):
        """POST with JSON body: model is body['model']."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi","model":"soprano-80m"}')
        model = await extract_model_from_request(request)
        assert model == "soprano-80m"

    @pytest.mark.asyncio
    async def test_returns_none_when_body_has_no_model(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'{"input":"hi"}')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_extracts_model_from_query_on_get(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {"model": "kokoro-82m"}
        model = await extract_model_from_request(request)
        assert model == "kokoro-82m"

    @pytest.mark.asyncio
    async def test_returns_none_when_get_has_no_query_model(self):
        request = MagicMock()
        request.method = "GET"
        request.query_params = {}
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_when_body_is_invalid_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "application/json"}
        request.body = AsyncMock(return_value=b'not json at all')
        model = await extract_model_from_request(request)
        assert model is None

    @pytest.mark.asyncio
    async def test_returns_none_when_content_type_not_json(self):
        request = MagicMock()
        request.method = "POST"
        request.headers = {"content-type": "multipart/form-data"}
        model = await extract_model_from_request(request)
        assert model is None


class TestWorkerRoute:
    def test_worker_route_stores_model_and_url(self):
        r = WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")
        assert r.model_id == "soprano-80m"
        assert r.worker_url == "http://127.0.0.1:9001"
```

- [ ] **Step 2: Run — fail with ModuleNotFoundError**

```bash
pytest tests/cli_impl/test_gateway.py -v
```

- [ ] **Step 3: Implement initial `src/muse/cli_impl/gateway.py`**

```python
"""FastAPI gateway: proxy requests by model-id to the right worker.

The gateway is the user-facing process (port 8000 by default). Workers
live on internal ports (9001+). The gateway:
  1. Reads catalog + venv map at startup, builds a model-id → worker-url table
  2. Extracts `model` from each request (body for POST, query for GET)
  3. Forwards the request to the hosting worker, streaming the response
  4. Aggregates /v1/models and /health across all workers
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRoute:
    """One entry in the gateway's routing table.

    A worker may host multiple models; each gets its own WorkerRoute
    pointing at the same worker_url.
    """
    model_id: str
    worker_url: str


async def extract_model_from_request(request: Any) -> str | None:
    """Extract the `model` field from a request.

    - POST with JSON body: body["model"]
    - GET: query_params["model"]
    - Anything else: None

    Returns None (not raises) on missing/invalid — the caller decides
    what "no model specified" means (400, or fall back to default).
    """
    if request.method == "GET":
        return request.query_params.get("model")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return None
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
            if not isinstance(body, dict):
                return None
            return body.get("model")
        except (json.JSONDecodeError, ValueError):
            return None

    return None


def build_gateway(routes: list[WorkerRoute]) -> FastAPI:
    """Build a FastAPI app that proxies requests based on the route table.

    Full implementation lands in subsequent tasks; for now this returns
    an app with only a /_gateway-info diagnostic endpoint so tests can
    assert the table is preserved.
    """
    app = FastAPI(title="Muse Gateway")
    app.state.routes = {r.model_id: r for r in routes}

    @app.get("/_gateway-info")
    def info():
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in app.state.routes.values()
            ],
        }

    return app
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_gateway.py -v
```

Expected: 7 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/gateway.py tests/cli_impl/test_gateway.py
git commit -m "feat(gateway): add WorkerRoute + extract_model_from_request

Gateway skeleton with FastAPI app factory and request-model extraction.
Body-based for POST (JSON), query-based for GET. Returns None (not
raises) on missing/invalid — caller handles the 'no model' case.

Proxy + aggregation + streaming in subsequent tasks."
```

---

### Task D2: Gateway proxy logic

**Files:**
- Modify: `src/muse/cli_impl/gateway.py` — add proxy middleware
- Modify: `tests/cli_impl/test_gateway.py` — add proxy tests
- Modify: `pyproject.toml` — move `httpx` to `[server]` extras

Gateway must forward requests to the right worker based on the extracted model-id. Uses httpx for the HTTP client.

- [ ] **Step 1: Move `httpx` from `[dev]` to `[server]` extras**

Edit `pyproject.toml`. Find the `[project.optional-dependencies]` section. Update:

```toml
[project.optional-dependencies]
# Server runtime
server = ["fastapi", "uvicorn", "sse-starlette", "httpx"]

# Development
dev = ["pytest", "pytest-cov", "pytest-asyncio"]
```

(Remove `httpx` from `dev` — it's now a runtime dep for the gateway.)

Reinstall:

```bash
cd /home/spinoza/github/repos/muse
pip install -e ".[dev,server]"
```

- [ ] **Step 2: Add failing test for proxy**

Append to `tests/cli_impl/test_gateway.py`:

```python
class TestProxy:
    def test_proxy_forwards_post_to_matching_worker(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        # Mock httpx.AsyncClient.request to capture the forwarded call
        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'{"ok": true}'
            mock_response.headers = {"content-type": "application/json"}
            async_mock_request = AsyncMock(return_value=mock_response)
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.request = async_mock_request
            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m",
            })

        assert r.status_code == 200
        # The AsyncClient.request call should have targeted the worker url
        call_kwargs = async_mock_request.call_args.kwargs
        call_args = async_mock_request.call_args.args
        target_url = call_args[1] if len(call_args) > 1 else call_kwargs.get("url")
        assert target_url == "http://127.0.0.1:9001/v1/audio/speech"

    def test_proxy_returns_404_openai_envelope_for_unknown_model(self):
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={
            "input": "hi", "model": "does-not-exist",
        })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "model_not_found"
        assert "does-not-exist" in body["error"]["message"]

    def test_proxy_returns_400_when_model_not_specified_and_no_default(self):
        """POST without a model field, and no default known — 400."""
        routes = [WorkerRoute(model_id="soprano-80m", worker_url="http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        r = client.post("/v1/audio/speech", json={"input": "hi"})
        # With no model specified, gateway can't know which worker to route to.
        # It returns 400 (not 404) — the client failed to provide routing info.
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "model_required"
```

- [ ] **Step 3: Run — fail**

```bash
pytest tests/cli_impl/test_gateway.py::TestProxy -v
```

Expected: FAIL.

- [ ] **Step 4: Implement proxy middleware in `src/muse/cli_impl/gateway.py`**

Replace the current `build_gateway` with this expanded version. Full file:

```python
"""FastAPI gateway: proxy requests by model-id to the right worker."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRoute:
    model_id: str
    worker_url: str


async def extract_model_from_request(request: Any) -> str | None:
    """Extract `model` from request body (POST) or query (GET)."""
    if request.method == "GET":
        return request.query_params.get("model")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return None
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
            if not isinstance(body, dict):
                return None
            return body.get("model")
        except (json.JSONDecodeError, ValueError):
            return None

    return None


def _openai_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": "invalid_request_error"}},
    )


def build_gateway(routes: list[WorkerRoute], timeout: float = 300.0) -> FastAPI:
    """Build the gateway FastAPI app.

    `routes` is the model-id → worker-url table. Streaming endpoints are
    supported transparently: any response with a streaming content-type
    is relayed chunk-by-chunk.
    """
    app = FastAPI(title="Muse Gateway")
    app.state.routes = {r.model_id: r for r in routes}
    app.state.timeout = timeout

    @app.get("/_gateway-info")
    def info():
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in app.state.routes.values()
            ],
        }

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, full_path: str):
        # Special aggregated endpoints are handled by other routes (added in D3).
        # This catch-all handles everything else: proxy by model-id.

        model_id = await extract_model_from_request(request)
        if model_id is None:
            return _openai_error(
                400, "model_required",
                "request is missing a `model` field (required for routing)",
            )

        route = app.state.routes.get(model_id)
        if route is None:
            return _openai_error(
                404, "model_not_found",
                f"model {model_id!r} is not registered with any worker; "
                f"known: {sorted(app.state.routes)}",
            )

        target_url = f"{route.worker_url}/{full_path}"
        return await _forward(request, target_url, app.state.timeout)

    return app


async def _forward(request: Request, target_url: str, timeout: float) -> Response:
    """Forward a request to target_url, relaying body and headers.

    For streaming responses (SSE / chunked), returns a StreamingResponse
    that yields chunks as they arrive. For non-streaming, reads the full
    body and returns it.
    """
    # Rebuild the incoming body (already consumed by extract_model_from_request)
    body = await request.body()

    # Strip hop-by-hop headers that httpx / FastAPI manage themselves
    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Use .send() so we can handle streaming responses uniformly
        req = client.build_request(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body,
            params=dict(request.query_params),
        )
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=fwd_headers,
            content=body,
            params=dict(request.query_params),
        )

    # Strip response-hop-by-hop headers
    resp_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in excluded
    }

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=resp_headers,
    )
```

Note: streaming-aware forwarding lands in Task D4. This version buffers the full response first — fine for `/v1/audio/speech` WAV responses and `/v1/images/generations` JSON, but breaks `stream: true`. We fix that in D4.

- [ ] **Step 5: Run proxy tests — pass**

```bash
pytest tests/cli_impl/test_gateway.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli_impl/gateway.py tests/cli_impl/test_gateway.py pyproject.toml
git commit -m "feat(gateway): proxy requests by model-id to hosting worker

Catch-all route extracts model from body (POST) or query (GET), looks
up which worker hosts it, forwards the request. Unknown model → 404
OpenAI envelope. Missing model → 400 with model_required code.

httpx moves from [dev] to [server] extras — it's a runtime dep of
the gateway now. Streaming is NOT yet supported (D4 adds that).

Pattern generalizes to any future modality (embeddings, transcriptions,
etc.) without gateway changes: just add catalog entries and the model
routes by body.model automatically."
```

---

### Task D3: Aggregated `/v1/models` and `/health`

**Files:**
- Modify: `src/muse/cli_impl/gateway.py`
- Modify: `tests/cli_impl/test_gateway.py`

The gateway's `/v1/models` must return the union of every worker's `/v1/models`. `/health` must return the union of every worker's health. Both use parallel httpx calls.

- [ ] **Step 1: Add failing tests**

Append to `tests/cli_impl/test_gateway.py`:

```python
class TestAggregation:
    def test_v1_models_aggregates_across_workers(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            # Each worker returns its own /v1/models list
            def make_resp(data):
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {"object": "list", "data": data}
                return r

            responses_by_url = {
                "http://127.0.0.1:9001/v1/models": make_resp([
                    {"id": "soprano-80m", "modality": "audio.speech", "object": "model"},
                ]),
                "http://127.0.0.1:9002/v1/models": make_resp([
                    {"id": "sd-turbo", "modality": "images.generations", "object": "model"},
                ]),
            }

            async def fake_get(url, **kwargs):
                return responses_by_url[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        ids = {m["id"] for m in data}
        assert ids == {"soprano-80m", "sd-turbo"}

    def test_v1_models_skips_unreachable_workers(self):
        """If a worker is down, its models are omitted (not a 500)."""
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9999"),  # down
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {"object": "list", "data": [
                {"id": "soprano-80m", "modality": "audio.speech", "object": "model"},
            ]}

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("connection refused", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"soprano-80m"}

    def test_health_aggregates_worker_status(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            def make_resp(payload):
                r = MagicMock(status_code=200)
                r.json.return_value = payload
                return r

            responses = {
                "http://127.0.0.1:9001/health": make_resp({
                    "status": "ok", "modalities": ["audio.speech"], "models": ["soprano-80m"],
                }),
                "http://127.0.0.1:9002/health": make_resp({
                    "status": "ok", "modalities": ["images.generations"], "models": ["sd-turbo"],
                }),
            }

            async def fake_get(url, **kwargs):
                return responses[url]

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert set(body["modalities"]) == {"audio.speech", "images.generations"}
        assert set(body["models"]) == {"soprano-80m", "sd-turbo"}

    def test_health_degraded_when_any_worker_down(self):
        routes = [
            WorkerRoute("soprano-80m", "http://127.0.0.1:9001"),
            WorkerRoute("sd-turbo", "http://127.0.0.1:9002"),
        ]
        app = build_gateway(routes)
        client = TestClient(app)

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            r_ok = MagicMock(status_code=200)
            r_ok.json.return_value = {
                "status": "ok", "modalities": ["audio.speech"], "models": ["soprano-80m"],
            }

            async def fake_get(url, **kwargs):
                if "9001" in url:
                    return r_ok
                raise httpx.ConnectError("down", request=None)

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            r = client.get("/health")
        body = r.json()
        assert body["status"] == "degraded"
        assert "sd-turbo" not in body["models"]
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_gateway.py::TestAggregation -v
```

Expected: FAIL — `/v1/models` and `/health` are currently caught by the proxy, which returns 400 because they don't have a model.

- [ ] **Step 3: Add aggregated endpoints to `src/muse/cli_impl/gateway.py`**

Before the catch-all `proxy()` handler (but inside `build_gateway`), add:

```python
    @app.get("/v1/models")
    async def list_models():
        worker_urls = {r.worker_url for r in app.state.routes.values()}
        aggregated: list[dict] = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            import asyncio
            async def _one(url: str) -> list[dict]:
                try:
                    r = await client.get(f"{url}/v1/models")
                    if r.status_code != 200:
                        return []
                    return r.json().get("data", [])
                except httpx.HTTPError as e:
                    logger.warning("worker %s unreachable: %s", url, e)
                    return []
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for items in results:
            aggregated.extend(items)
        return {"object": "list", "data": aggregated}

    @app.get("/health")
    async def health():
        worker_urls = {r.worker_url for r in app.state.routes.values()}
        modalities: set[str] = set()
        models: set[str] = set()
        any_down = False
        async with httpx.AsyncClient(timeout=5.0) as client:
            import asyncio
            async def _one(url: str) -> dict | None:
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code != 200:
                        return None
                    return r.json()
                except httpx.HTTPError:
                    return None
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for body in results:
            if body is None:
                any_down = True
                continue
            modalities.update(body.get("modalities", []))
            models.update(body.get("models", []))
        return {
            "status": "degraded" if any_down else "ok",
            "modalities": sorted(modalities),
            "models": sorted(models),
        }
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_gateway.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/gateway.py tests/cli_impl/test_gateway.py
git commit -m "feat(gateway): aggregate /v1/models and /health across workers

Parallel httpx.get per worker via asyncio.gather. /v1/models
concatenates data arrays from each worker. /health returns
status='ok' when all workers reachable, 'degraded' when any are
down (partial visibility is preferable to a 500).

Unreachable workers are logged and skipped rather than failing
the aggregated call — matches the operational principle that
gateway degrades gracefully instead of blocking on failures."
```

---

### Task D4: Streaming support in the gateway

**Files:**
- Modify: `src/muse/cli_impl/gateway.py` — upgrade `_forward` to stream
- Modify: `tests/cli_impl/test_gateway.py` — add streaming test

- [ ] **Step 1: Add failing test**

Append to `tests/cli_impl/test_gateway.py`:

```python
class TestStreaming:
    def test_sse_stream_is_relayed_chunk_by_chunk(self):
        """A `stream: true` response (text/event-stream) must pass through."""
        routes = [WorkerRoute("soprano-80m", "http://127.0.0.1:9001")]
        app = build_gateway(routes)
        client = TestClient(app)

        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"event: done\ndata: \n\n"]

        with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for c in chunks:
                    yield c
            mock_response.aiter_raw = aiter_raw
            mock_response.aclose = AsyncMock()

            # stream() is an async context manager
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=stream_ctx)

            mock_client_cls.return_value = mock_client

            r = client.post("/v1/audio/speech", json={
                "input": "hi", "model": "soprano-80m", "stream": True,
            })

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        # All chunks received in order
        assert b"data: chunk1" in r.content
        assert b"data: chunk2" in r.content
        assert b"event: done" in r.content
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_gateway.py::TestStreaming -v
```

Expected: FAIL because `_forward` currently buffers.

- [ ] **Step 3: Upgrade `_forward` to stream**

Replace `_forward` in `src/muse/cli_impl/gateway.py` with a version that detects streaming responses by content-type:

```python
async def _forward(request: Request, target_url: str, timeout: float) -> Response:
    """Forward a request to target_url.

    Detects streaming content-types (text/event-stream, chunked octet-stream)
    and relays chunks via StreamingResponse. Non-streaming responses are
    read fully and returned in one go.
    """
    body = await request.body()
    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    # Open the connection via stream() so we can decide whether to relay
    # or buffer based on the response headers.
    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream(
        method=request.method,
        url=target_url,
        headers=fwd_headers,
        content=body,
        params=dict(request.query_params),
    )
    response = await stream_ctx.__aenter__()

    content_type = response.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    resp_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in excluded
    }

    if is_stream:
        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await client.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=content_type,
        )
    else:
        # Non-streaming: read once, close the stream + client, return buffered.
        try:
            content = await response.aread()
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()

        return Response(
            content=content,
            status_code=response.status_code,
            headers=resp_headers,
        )
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_gateway.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/gateway.py tests/cli_impl/test_gateway.py
git commit -m "feat(gateway): stream SSE responses chunk-by-chunk

_forward now detects text/event-stream content-type and returns a
StreamingResponse that relays chunks as they arrive from the worker,
instead of buffering the full response. Same lesson as the audio.speech
router's producer-queue fix (commit a9d5486) — the stream contract
means latency to first byte must match the worker's, not wait for
full synthesis."
```

---

## Part E — Supervisor

### Task E1: Supervisor skeleton — group models by venv and build routes

**Files:**
- Create: `src/muse/cli_impl/supervisor.py`
- Create: `tests/cli_impl/test_supervisor.py`

The supervisor reads the catalog, groups models by venv, allocates ports, and builds the gateway's route table. It doesn't spawn subprocesses yet — that's Task E2.

- [ ] **Step 1: Write failing tests**

File: `tests/cli_impl/test_supervisor.py`

```python
"""Tests for the supervisor: catalog → worker specs."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.cli_impl.supervisor import (
    WorkerSpec,
    plan_workers,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data):
    """Write catalog.json directly."""
    import json
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


class TestPlanWorkers:
    def test_empty_catalog_yields_no_workers(self, tmp_catalog):
        _seed_catalog({})
        specs = plan_workers()
        assert specs == []

    def test_one_model_yields_one_worker(self, tmp_catalog):
        _seed_catalog({
            "soprano-80m": {
                "pulled_at": "2026-04-13T00:00:00Z",
                "hf_repo": "ekwek/Soprano-1.1-80M",
                "local_dir": "/fake/local",
                "venv_path": "/home/user/.muse/venvs/soprano-80m",
                "python_path": "/home/user/.muse/venvs/soprano-80m/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.models == ["soprano-80m"]
        assert spec.python_path == "/home/user/.muse/venvs/soprano-80m/bin/python"
        assert isinstance(spec.port, int)
        assert 9001 <= spec.port <= 9999

    def test_models_in_same_venv_share_a_worker(self, tmp_catalog):
        """If two models share venv_path (rare in M1 but supported), one worker."""
        shared_venv = "/home/user/.muse/venvs/shared"
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": shared_venv,
                "python_path": f"{shared_venv}/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 1
        assert set(specs[0].models) == {"model-a", "model-b"}

    def test_different_venvs_yield_different_workers(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
            },
        })
        specs = plan_workers()
        assert len(specs) == 2
        assert specs[0].port != specs[1].port

    def test_skips_pre_worker_entries_without_python_path(self, tmp_catalog, caplog):
        """Old catalog entries (no python_path field) are skipped with warning."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "x", "local_dir": "/x",
                # No venv_path / python_path — pre-worker entry
            },
            "new-model": {
                "pulled_at": "...", "hf_repo": "y", "local_dir": "/y",
                "venv_path": "/venvs/y",
                "python_path": "/venvs/y/bin/python",
            },
        })
        import logging
        caplog.set_level(logging.WARNING)
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" not in all_models
        assert "new-model" in all_models
        assert "re-run" in caplog.text or "re-pull" in caplog.text or "legacy-model" in caplog.text
```

- [ ] **Step 2: Run — fail with ModuleNotFoundError**

```bash
pytest tests/cli_impl/test_supervisor.py -v
```

- [ ] **Step 3: Implement `src/muse/cli_impl/supervisor.py`**

```python
"""`muse serve` supervisor: orchestrate workers + run gateway.

Responsibilities:
  1. Read catalog
  2. Group models by venv (same python_path = same worker)
  3. Allocate a local port per worker
  4. Spawn worker subprocesses
  5. Wait for each worker's /health to become responsive
  6. Build gateway routes + run gateway uvicorn
  7. On shutdown: SIGTERM workers, wait for exit

This task (E1) implements steps 1-3 only. E2 adds subprocess spawning,
E3 adds signal/cleanup, E4 wires it all up under `muse serve`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from muse.core.catalog import _read_catalog
from muse.core.venv import find_free_port

logger = logging.getLogger(__name__)


@dataclass
class WorkerSpec:
    """Everything needed to spawn one worker subprocess."""
    models: list[str]
    python_path: str
    port: int
    # Populated after subprocess.Popen in Task E2
    process: object = field(default=None)


def plan_workers(port_start: int = 9001, port_end: int = 9999) -> list[WorkerSpec]:
    """Read catalog, group by venv, allocate ports.

    Returns one WorkerSpec per unique venv (identified by python_path).
    Pre-worker catalog entries (missing python_path) are logged + skipped.
    """
    catalog = _read_catalog()

    # Group by python_path. Preserve insertion order for determinism.
    groups: dict[str, list[str]] = {}
    for model_id, entry in catalog.items():
        python = entry.get("python_path")
        if not python:
            logger.warning(
                "skipping pre-worker catalog entry %r — no python_path; "
                "re-run `muse pull %s` to create its venv",
                model_id, model_id,
            )
            continue
        groups.setdefault(python, []).append(model_id)

    specs: list[WorkerSpec] = []
    used_ports: set[int] = set()
    for python_path, models in groups.items():
        # Allocate a free port, avoiding collisions with ports already
        # assigned to earlier specs (find_free_port may hand out the same
        # port twice if the earlier worker hasn't bound it yet).
        while True:
            port = find_free_port(start=port_start, end=port_end)
            if port not in used_ports:
                used_ports.add(port)
                break
            port_start = port + 1
        specs.append(WorkerSpec(
            models=sorted(models),
            python_path=python_path,
            port=port,
        ))
    return specs
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): plan_workers groups catalog by venv + allocates ports

Pure-function planner: reads catalog, groups models by python_path
(models sharing a venv share a worker), allocates a free port per
group. Pre-worker entries (no python_path field) are logged as
'please re-pull' and skipped.

Subprocess spawning in Task E2; this is deliberately decoupled so
the planning logic is independently testable."
```

---

### Task E2: Subprocess spawning + readiness polling

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py` — add `spawn_worker` and `wait_for_ready`
- Modify: `tests/cli_impl/test_supervisor.py` — add tests

- [ ] **Step 1: Append failing tests**

Append to `tests/cli_impl/test_supervisor.py`:

```python
class TestSpawnWorker:
    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_invokes_venv_python_with_worker_subcommand(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        spec = WorkerSpec(
            models=["soprano-80m"],
            python_path="/venvs/soprano-80m/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cpu")
        mock_popen.assert_called_once()
        args = mock_popen.call_args.args[0]
        assert args[0] == "/venvs/soprano-80m/bin/python"
        assert args[1:4] == ["-m", "muse.cli", "_worker"]
        assert "--port" in args and "9001" in args
        assert "--model" in args and "soprano-80m" in args
        assert "--device" in args and "cpu" in args
        assert spec.process is mock_popen.return_value

    @patch("muse.cli_impl.supervisor.subprocess.Popen")
    def test_spawn_worker_passes_all_models_in_group(self, mock_popen):
        spec = WorkerSpec(
            models=["model-a", "model-b"],
            python_path="/venvs/shared/bin/python",
            port=9001,
        )
        from muse.cli_impl.supervisor import spawn_worker
        spawn_worker(spec, device="cuda")
        args = mock_popen.call_args.args[0]
        # Each model passed via separate --model
        model_values = [args[i+1] for i, v in enumerate(args) if v == "--model"]
        assert set(model_values) == {"model-a", "model-b"}


class TestWaitForReady:
    def test_returns_when_health_responds_200(self):
        from muse.cli_impl.supervisor import wait_for_ready

        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            # Should return cleanly
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.01)

    def test_raises_timeouterror_when_worker_never_responds(self):
        from muse.cli_impl.supervisor import wait_for_ready
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.ConnectError("nope", request=None)
            with pytest.raises(TimeoutError, match="did not become ready"):
                wait_for_ready(port=9001, timeout=0.1, poll_interval=0.01)

    def test_polls_multiple_times_before_success(self):
        from muse.cli_impl.supervisor import wait_for_ready
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            # First two calls fail, third succeeds
            mock_get.side_effect = [
                httpx.ConnectError("not yet", request=None),
                httpx.ConnectError("not yet", request=None),
                MagicMock(status_code=200),
            ]
            wait_for_ready(port=9001, timeout=5.0, poll_interval=0.001)
            assert mock_get.call_count == 3
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestSpawnWorker tests/cli_impl/test_supervisor.py::TestWaitForReady -v
```

- [ ] **Step 3: Add `spawn_worker` and `wait_for_ready` to `src/muse/cli_impl/supervisor.py`**

At the top, add imports:

```python
import subprocess
import time

import httpx
```

Then add these functions after `plan_workers`:

```python
def spawn_worker(spec: WorkerSpec, *, device: str) -> None:
    """Start a worker subprocess using its venv's Python.

    Mutates spec.process with the Popen handle so the supervisor can
    manage the subprocess later.
    """
    cmd = [
        spec.python_path, "-m", "muse.cli", "_worker",
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "--device", device,
    ]
    for m in spec.models:
        cmd.extend(["--model", m])
    logger.info("spawning worker: %s", " ".join(cmd))
    spec.process = subprocess.Popen(cmd)


def wait_for_ready(
    *, port: int, timeout: float = 60.0, poll_interval: float = 0.5,
) -> None:
    """Block until `http://127.0.0.1:<port>/health` returns 200, or timeout.

    Raises TimeoutError if the worker never becomes ready. Polls with
    short sleeps so slow workers (big model loads) still get through.
    """
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError as e:
            last_err = e
        time.sleep(poll_interval)
    raise TimeoutError(
        f"worker on port {port} did not become ready within {timeout}s "
        f"(last error: {last_err})"
    )
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v
```

Expected: all 10 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): spawn workers via subprocess.Popen + health-poll readiness

spawn_worker(spec, device) invokes <venv>/bin/python -m muse.cli _worker
with the spec's port + models. wait_for_ready(port, timeout) polls
/health via httpx until 200 or timeout. Decoupled from run_supervisor
so each is independently testable."
```

---

### Task E3: Supervisor orchestration + shutdown

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py` — add `run_supervisor` and cleanup
- Modify: `tests/cli_impl/test_supervisor.py` — add orchestration test

- [ ] **Step 1: Add failing test for orchestration**

Append to `tests/cli_impl/test_supervisor.py`:

```python
class TestRunSupervisor:
    def test_supervisor_spawns_all_workers_and_waits_for_all_ready(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
            "model-b": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait, \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            # Simulate graceful shutdown by raising KeyboardInterrupt from uvicorn.run
            mock_uvicorn.run.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # Two workers planned, so spawn + wait called twice
            assert mock_spawn.call_count == 2
            assert mock_wait.call_count == 2
            mock_uvicorn.run.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_supervisor_tears_down_workers_if_gateway_fails(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers") as mock_shutdown:
            mock_uvicorn.run.side_effect = RuntimeError("uvicorn died")

            with pytest.raises(RuntimeError, match="uvicorn died"):
                run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            mock_shutdown.assert_called_once()
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestRunSupervisor -v
```

- [ ] **Step 3: Add `run_supervisor` + `_shutdown_workers` to `src/muse/cli_impl/supervisor.py`**

Add these imports at the top if not already present:

```python
import signal
```

And `uvicorn` (near the other lazy-imports — we'll import inside the function, not at module top, to keep supervisor.py cheap to import).

Append the following functions:

```python
def _shutdown_workers(specs: list[WorkerSpec], grace: float = 5.0) -> None:
    """SIGTERM all workers; SIGKILL any that don't exit within `grace` seconds."""
    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.terminate()
        except Exception as e:
            logger.warning("failed to SIGTERM worker on port %d: %s", spec.port, e)

    for spec in specs:
        if spec.process is None:
            continue
        try:
            spec.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("worker on port %d did not exit in %ds; killing", spec.port, grace)
            spec.process.kill()
        except Exception as e:
            logger.warning("error waiting for worker on port %d: %s", spec.port, e)


def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve`.

    Plans workers from catalog, spawns them, waits for ready, then runs
    the gateway on (host, port). Guarantees worker cleanup on exit.
    """
    import uvicorn

    from muse.cli_impl.gateway import WorkerRoute, build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(
            "no pulled models with a venv — server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    # Spawn every worker; wait for each to become ready
    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        for spec in specs:
            logger.info("waiting for worker on port %d (%s)", spec.port, spec.models)
            wait_for_ready(port=spec.port)

        # Build gateway routes
        routes: list[WorkerRoute] = []
        for spec in specs:
            worker_url = f"http://127.0.0.1:{spec.port}"
            for m in spec.models:
                routes.append(WorkerRoute(model_id=m, worker_url=worker_url))
        app = build_gateway(routes)

        logger.info("starting gateway on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        _shutdown_workers(specs)
    return 0
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v
```

Expected: all 12 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): run_supervisor orchestrates workers + gateway

Happy path: plan_workers → spawn each → wait_for_ready each →
build_gateway with route table → uvicorn.run. Always runs
_shutdown_workers in finally so SIGTERM → cleanup even if the
gateway dies or the user hits Ctrl-C.

Workers get SIGTERM first; any that don't exit in 5s get SIGKILL
so we don't leak processes."
```

---

### Task E4: Repurpose `muse serve` as supervisor entry

**Files:**
- Overwrite: `src/muse/cli_impl/serve.py` — now just delegates to supervisor
- Modify: `src/muse/cli.py` — `--model` / `--modality` args removed from `serve` (supervisor reads catalog)

- [ ] **Step 1: Overwrite `src/muse/cli_impl/serve.py`**

Replace the entire file with:

```python
"""`muse serve` — user-facing entry point.

Delegates to the supervisor, which spawns per-venv worker subprocesses
and runs the gateway. The old in-process behavior is gone; any models
that must be loaded into a single process now use `muse _worker`
directly (intended for supervisor use, also fine for debugging).
"""
from __future__ import annotations

from muse.cli_impl.supervisor import run_supervisor


def run_serve(*, host: str, port: int, device: str, **_: object) -> int:
    """Thin wrapper that delegates to the supervisor.

    Kept as a separate function (instead of pointing `muse.cli._cmd_serve`
    directly at `run_supervisor`) so future ergonomic flags (gateway
    auth, TLS, etc.) have a natural home.
    """
    return run_supervisor(host=host, port=port, device=device)
```

- [ ] **Step 2: Update `src/muse/cli.py` — drop `--model` and `--modality` from `serve`**

Find the `serve` subparser definition. Replace with:

```python
    # serve (user-facing; orchestrates workers + gateway)
    sp_serve = sub.add_parser("serve", help="start the HTTP gateway (spawns one worker per venv)")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--device", default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
    sp_serve.set_defaults(func=_cmd_serve)
```

And simplify `_cmd_serve`:

```python
def _cmd_serve(args):
    from muse.cli_impl.serve import run_serve
    return run_serve(host=args.host, port=args.port, device=args.device)
```

- [ ] **Step 3: Update existing CLI test for `serve` no longer taking --model**

Check `tests/test_cli.py`. Find any test asserting `serve --help` shows `--model` or `--modality`. Update those assertions to use the new flag set (only `--host`, `--port`, `--device`, `--log-level`).

- [ ] **Step 4: Run full test suite**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: CLI smoke test**

```bash
muse --help | head -20
muse serve --help
```

Must succeed with exit code 0.

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli_impl/serve.py src/muse/cli.py tests/test_cli.py
git commit -m "feat(cli): muse serve now orchestrates supervisor + workers

serve delegates to run_supervisor (which plans workers from catalog,
spawns one subprocess per venv, runs the gateway). --model and
--modality flags removed from serve — catalog + enabled state is
now the source of truth.

Users who want to load a specific single model into one process
for debugging can still use \`muse _worker --port N --model X\`."
```

---

## Part F — Integration

### Task F1: End-to-end smoke test with real subprocess

**Files:**
- Create: `tests/cli_impl/test_e2e_supervisor.py` — one slow test

The per-module tests mock subprocess.Popen and httpx. One end-to-end test that actually spawns a real worker + gateway guards against "each piece works but they don't talk to each other".

- [ ] **Step 1: Write the test**

File: `tests/cli_impl/test_e2e_supervisor.py`

```python
"""End-to-end smoke test: real subprocess worker, real gateway, mocked model.

This test is SLOW (~5-15s depending on Python cold-start). It's the only
integration test for supervisor + worker + gateway wired together. All
other tests mock the process boundary.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.slow


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog_pointing_at_current_python(tmp_catalog: Path, model_id: str):
    """Seed a catalog entry whose python_path IS the current interpreter.

    Lets us skip real venv creation in this e2e test — we reuse the
    test runner's Python, which already has muse installed.
    """
    import json
    catalog_path = tmp_catalog / "catalog.json"
    catalog_path.write_text(json.dumps({
        model_id: {
            "pulled_at": "2026-04-13T00:00:00Z",
            "hf_repo": "fake/repo",
            "local_dir": "/tmp/nonexistent",
            "venv_path": "/tmp/nonexistent-venv",
            "python_path": sys.executable,  # use current interpreter
        },
    }))


@pytest.mark.timeout(30)
def test_supervisor_spawns_worker_and_gateway_proxies_request(tmp_catalog):
    """Real subprocess, real gateway. Hit /health through the gateway."""
    # Seed a catalog with a KNOWN_MODELS model-id, using current Python
    _seed_catalog_pointing_at_current_python(tmp_catalog, "soprano-80m")

    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_catalog)

    # Spawn muse serve on port 18765 (non-conflicting)
    # We mock load_backend's success via monkeypatching: the worker will
    # skip the model load (it's not actually pulled) but still start an
    # empty server — enough for /health + /v1/models to work.
    proc = subprocess.Popen(
        [sys.executable, "-m", "muse.cli", "serve", "--port", "18765"],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 20s for the gateway to come up
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                r = httpx.get("http://127.0.0.1:18765/health", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        else:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"gateway never became ready.\n"
                f"stdout: {stdout.decode()[:1000]}\n"
                f"stderr: {stderr.decode()[:1000]}"
            )

        # Gateway is up; verify the aggregated /health works
        r = httpx.get("http://127.0.0.1:18765/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert "modalities" in body

        # /v1/models should also work (may be empty if worker didn't
        # actually load a model — that's fine; this test is about
        # plumbing, not model availability)
        r = httpx.get("http://127.0.0.1:18765/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
```

- [ ] **Step 2: Mark slow tests as optional in pyproject pytest config**

Edit `pyproject.toml`. Under `[tool.pytest.ini_options]`, add the `slow` marker registration:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = ["--import-mode=importlib"]
markers = [
    "slow: end-to-end test that spawns real subprocesses (skipped by default CI lanes)",
]
```

- [ ] **Step 3: Run the slow test explicitly**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: pass (may take 10-20s).

- [ ] **Step 4: Run the full suite excluding slow**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: all pass, runs fast.

- [ ] **Step 5: Commit**

```bash
git add tests/cli_impl/test_e2e_supervisor.py pyproject.toml
git commit -m "test(supervisor): add e2e smoke test with real subprocess

One slow test (~10-20s) that actually spawns muse serve, waits for
the gateway to bind, and verifies /health and /v1/models respond.
Guards against 'each piece works but they don't talk to each other'
regressions.

Marked @slow — excluded from default test runs via pytest markers;
run explicitly with \`pytest -m slow\`."
```

---

### Task F2: Update CLAUDE.md + README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Add a new section after "Architecture" and before "Modality conventions":

```markdown
## Process model

`muse serve` is a **supervisor**, not a single process:

```
User's request
    │
    ▼
muse serve (the supervisor, port 8000)
  ├─ gateway FastAPI app (in-process)
  │    routes by request-body `model` field
  │
  └─ subprocess per venv group:
       ├─ worker (port 9001, venv-A)  → hosts soprano-80m, kokoro-82m
       ├─ worker (port 9002, venv-B)  → hosts bark-small
       └─ worker (port 9003, venv-C)  → hosts sd-turbo
```

Each pulled model gets its own venv at `~/.muse/venvs/<model-id>/`
with exactly the pip_extras it declares. Workers run the existing
`muse.cli_impl.worker.run_worker` logic via `muse.cli _worker`
(hidden subcommand). The supervisor spawns them with the venv's
Python, polls `/health` until ready, then runs the gateway.

The gateway extracts `model` from the request body (POST) or query
(GET), looks up which worker hosts it, and forwards the request —
streaming SSE through without buffering. `/v1/models` and `/health`
are aggregated across all workers via parallel httpx calls.

This architecture gives you dep-isolation (transformers 4.46 for
parler-tts coexists with transformers 5.x for newer models), crash
isolation (a segfault in one worker doesn't kill the rest), and a
uniform HTTP surface (clients hit one port, don't care about venvs).
```

Update the "Key modules" section by adding:

```markdown
- `muse.core.venv` — venv creation (`create_venv`, `install_into_venv`, `find_free_port`). Each `muse pull` creates `~/.muse/venvs/<model-id>/`; catalog records the `python_path`.
- `muse.cli_impl.worker` — single-worker mode (runs one uvicorn in one venv). Invoked via `muse _worker` (hidden subcommand).
- `muse.cli_impl.gateway` — FastAPI proxy app. Routes by `model` field in request body/query; aggregates `/v1/models` + `/health` across workers.
- `muse.cli_impl.supervisor` — orchestrates workers + gateway. `plan_workers` groups catalog by venv; `spawn_worker` + `wait_for_ready` manage subprocess lifecycle; `run_supervisor` is the entrypoint `muse serve` delegates to.
```

Also update "Adding a new modality" to add a step 10:

```markdown
10. Verify the gateway routes correctly — since routing is by `model` field only, no gateway changes are needed. The new modality's endpoints are reachable as soon as at least one model for it is pulled + enabled.
```

- [ ] **Step 2: Update `README.md`**

Replace the "Architecture" section with:

```markdown
## Architecture

- `muse.core` — modality-agnostic: registry, catalog, venv management, HF downloader, pip auto-install, FastAPI app factory
- `muse.cli_impl` — `serve` (supervisor), `worker` (single-venv process), `gateway` (HTTP proxy by model-id)
- `muse.audio.speech` — text-to-speech (Soprano, Kokoro, Bark backends)
- `muse.images.generations` — text-to-image (SD-Turbo backend)

`muse serve` is a supervisor process: it spawns one worker subprocess per venv (each model has its own venv with its own deps) and runs a gateway that proxies requests by the request's `model` field. Dep conflicts between models are structurally impossible.
```

Update "Quick start" to mention the venv:

```markdown
## Quick start

```bash
# Pull a model (installs its pip deps into a dedicated venv, downloads HF weights)
muse pull soprano-80m
muse pull sd-turbo

# Start the server (spawns one worker per venv + runs the gateway on :8000)
muse serve

# Synthesize speech from any client
muse speak "Hello world" -o hello.wav

# Generate an image
muse imagine "a cat on mars, cinematic" -o cat.png
```

The first pull of each model takes time: venv creation + pip install + HF weight download. Subsequent starts of `muse serve` are fast.
```

- [ ] **Step 3: Tests still pass after docs changes**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document supervisor/worker/gateway architecture

CLAUDE.md gains a Process Model section with the supervisor diagram,
worker/gateway responsibilities, and deps-isolation rationale.

README.md Architecture + Quick Start updated to reflect that pull
creates a per-model venv and serve is a supervisor not a single
process."
```

---

### Task F3: Full verification + merge

**Files:** none (verification only)

- [ ] **Step 1: Fresh install + full test suite**

```bash
cd /home/spinoza/github/repos/muse
pip uninstall -y muse 2>/dev/null || true
pip install -e ".[dev,server]"
pytest tests/ -q 2>&1 | tail -5
```

Expected: all non-slow tests pass.

- [ ] **Step 2: CLI smoke**

```bash
muse --help
muse _worker --help
muse serve --help
muse audio speech models list
muse images generations models list
```

All exit 0.

- [ ] **Step 3: Run the e2e slow test**

```bash
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: pass.

- [ ] **Step 4: Import smoke checks**

```bash
python -c "from muse.cli_impl.worker import run_worker"
python -c "from muse.cli_impl.gateway import build_gateway, WorkerRoute"
python -c "from muse.cli_impl.supervisor import run_supervisor, plan_workers"
python -c "from muse.core.venv import create_venv, install_into_venv"
```

All succeed.

- [ ] **Step 5: Merge the worktree back**

```bash
cd /home/spinoza/github/repos/muse
git merge --no-ff <worker-isolation-branch> -m "feat: add multi-venv worker + gateway architecture

muse serve is now a supervisor that spawns one worker subprocess per
venv group (typically one-per-model) and runs a gateway that proxies
requests by model-id. Each pulled model gets its own venv so ML
library dep conflicts are structurally impossible.

See docs/plans/2026-04-13-multi-venv-workers.md for the full plan."
git worktree remove <worktree-path>
git branch -d <worker-isolation-branch>
```

---

## Scope notes (deferred)

These are intentionally out of scope for this plan. Separate milestones.

- **Auto-grouping by compatible deps** — today each model gets its own venv, even if two models could share. Adding a uv-based constraint solver that puts compatible models in one venv is a future optimization. Nothing in this plan precludes it.
- **`muse enable/disable` CLI** — no way to toggle which models the supervisor loads without editing catalog.json by hand. Adding `enabled: bool` to catalog entries + `enable`/`disable` subcommands is ~50 lines; separate plan.
- **Worker hot reload** — if you `muse pull` a new model while `muse serve` is running, the supervisor won't pick it up. User restarts. A future hot-reload loop could watch catalog.json.
- **Worker auto-restart** — if a worker crashes mid-session, the supervisor doesn't respawn it; gateway reports `degraded`. A supervisor monitor + backoff policy is a follow-up.
- **Configurable gateway port per worker** — ports are allocated in [9001, 9999]. Fine for single-host; if someone ever wants cross-host routing, worker_url becomes config-driven.
- **Embeddings modality** — next modality to add. This plan's gateway handles it automatically (routing by model, not URL). See the end-of-plan note.

---

## Self-review

**Spec coverage:**
- Per-model venv on pull: ✅ Tasks A1, B1
- Worker subcommand: ✅ C1, C2
- Gateway proxy: ✅ D1, D2, D4
- Gateway aggregation: ✅ D3
- Supervisor planning: ✅ E1
- Subprocess spawn + readiness: ✅ E2
- Supervisor orchestration + cleanup: ✅ E3
- `muse serve` re-wiring: ✅ E4
- E2E test: ✅ F1
- Docs: ✅ F2
- Verification: ✅ F3

**Placeholder scan:** No TBDs. Every step has runnable code. Every import referenced in test code matches an import in the corresponding source task.

**Type consistency:**
- `WorkerSpec` fields (`models`, `python_path`, `port`, `process`) — used identically in E1, E2, E3
- `WorkerRoute(model_id, worker_url)` — consistent across D1-D4, E3
- `spawn_worker(spec, *, device)` — signature matches call in E3
- `wait_for_ready(port, timeout, poll_interval)` — consistent
- `plan_workers()` — no args, returns `list[WorkerSpec]` — consistent
- `build_gateway(routes, timeout)` — consistent across D1-D4 and E3

Plan complete. ~18 tasks. All tests TDD-first. All task commits are small and focused.
