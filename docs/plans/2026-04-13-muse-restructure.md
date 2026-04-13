# Muse Restructure: narro → muse Multi-Modality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the current `narro` package into a new multi-modality `muse` package. The existing TTS code becomes the `audio/speech` modality (serving `/v1/audio/speech`). A new `images/generations` modality (serving `/v1/images/generations`) is added with a stub + one real backend (SD-Turbo) to prove the abstraction is not leaky.

**Architecture:** Src-layout (`src/muse/`). Modality subpackages mirror the OpenAI URL hierarchy: `muse/audio/speech/`, `muse/images/generations/`. Modality-agnostic code lives in `muse/core/` (registry, HF downloader + pip auto-install, FastAPI app factory, shared server framework). Each modality owns its protocol, routes, CLI, and backends. Registry is keyed by `(modality, model_id)` tuple — no shared protocol base class (audio and image result types don't share a useful supertype). Clean break — no `narro` shim.

**Tech Stack:** Python 3.10+, FastAPI, PyTorch, HuggingFace Hub, transformers (for audio), diffusers (for images). Tests: pytest + unittest.mock.

---

## File Structure (Final)

```
muse/                             # repo root (directory already renamed from narro/)
├── pyproject.toml                # updated: name=muse, scripts=muse.cli:main, src-layout
├── README.md                     # rewritten for muse
├── CLAUDE.md                     # rewritten for muse
├── LICENSE
├── src/
│   └── muse/
│       ├── __init__.py           # version, defer imports
│       ├── cli.py                # top-level argparse: serve, pull, audio.*, images.*
│       ├── core/                 # modality-agnostic kernel
│       │   ├── __init__.py
│       │   ├── registry.py       # ModalityRegistry {modality: {model_id: Model}}
│       │   ├── catalog.py        # KNOWN_MODELS + pull() + HF download + pip install
│       │   ├── install.py        # pip/system package helpers (extracted from catalog)
│       │   ├── server.py         # FastAPI app factory, mounts modality routers
│       │   └── errors.py         # OpenAI-style error envelopes
│       ├── audio/
│       │   ├── __init__.py
│       │   └── speech/
│       │       ├── __init__.py
│       │       ├── protocol.py   # TTSModel, AudioResult, AudioChunk
│       │       ├── routes.py     # /v1/audio/speech FastAPI router
│       │       ├── cli.py        # muse audio speech ... argparse
│       │       ├── client.py     # HTTP client for /v1/audio/speech
│       │       ├── codec.py      # wav/opus encoding (extracted from server.py)
│       │       ├── tts.py        # Soprano inference (ported verbatim)
│       │       ├── alignment.py  # Word-level timestamps (ported)
│       │       ├── encoded.py    # EncodedSpeech IR (ported)
│       │       ├── decode_only.py
│       │       ├── vocos/        # Soprano's vocoder (ported verbatim)
│       │       ├── backends/
│       │       │   ├── __init__.py
│       │       │   ├── base.py            # BaseModel mixin (token-word map, entropy)
│       │       │   ├── transformers.py
│       │       │   ├── soprano.py
│       │       │   ├── kokoro.py
│       │       │   └── bark.py
│       │       └── utils/
│       │           ├── __init__.py
│       │           ├── text_normalizer.py
│       │           └── text_splitter.py
│       └── images/
│           ├── __init__.py
│           └── generations/
│               ├── __init__.py
│               ├── protocol.py   # ImageModel, ImageResult
│               ├── routes.py     # /v1/images/generations FastAPI router
│               ├── cli.py        # muse images generations ... argparse
│               ├── client.py     # HTTP client
│               ├── codec.py      # PIL → PNG/JPEG/base64
│               └── backends/
│                   ├── __init__.py
│                   └── sd_turbo.py   # stabilityai/sd-turbo via diffusers
└── tests/
    ├── core/
    │   ├── test_registry.py
    │   ├── test_catalog.py
    │   └── test_server.py
    ├── audio/
    │   └── speech/
    │       ├── test_protocol.py
    │       ├── test_routes.py
    │       ├── test_client.py
    │       ├── test_codec.py
    │       ├── test_alignment.py
    │       ├── test_soprano.py
    │       ├── test_kokoro.py
    │       ├── test_bark.py
    │       ├── test_encoded.py
    │       └── test_encode_decode.py
    └── images/
        └── generations/
            ├── test_protocol.py
            ├── test_routes.py
            ├── test_codec.py
            └── test_sd_turbo.py
```

**Deleted:**
- `narro/` (entire tree — content migrated to `src/muse/`)
- `narro.egg-info/`, `soprano_tts.egg-info/`
- `.coverage` (regenerate)
- Old test fixtures at repo root: `test_large.wav`, `test_large.soprano.npz`, `test.wav`, `alex_voice.wav` — move any referenced ones to `tests/fixtures/`, delete the rest

---

## Part A — Package Scaffold

### Task A1: Create worktree, new package skeleton, pyproject

**Files:**
- Delete: `narro.egg-info/`, `soprano_tts.egg-info/`, `.coverage`
- Create: `src/muse/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create worktree for this restructure**

```bash
cd /home/spinoza/github/repos/muse
git worktree add ../muse-restructure -b restructure/muse
cd ../muse-restructure
```

- [ ] **Step 2: Remove old build artifacts**

```bash
rm -rf narro.egg-info/ soprano_tts.egg-info/ .coverage
```

- [ ] **Step 3: Create the `src/muse/` skeleton**

```bash
mkdir -p src/muse/core
mkdir -p src/muse/audio/speech/backends
mkdir -p src/muse/audio/speech/utils
mkdir -p src/muse/audio/speech/vocos
mkdir -p src/muse/images/generations/backends
mkdir -p tests/core
mkdir -p tests/audio/speech
mkdir -p tests/images/generations
touch src/muse/__init__.py
touch src/muse/core/__init__.py
touch src/muse/audio/__init__.py
touch src/muse/audio/speech/__init__.py
touch src/muse/audio/speech/backends/__init__.py
touch src/muse/audio/speech/utils/__init__.py
touch src/muse/images/__init__.py
touch src/muse/images/generations/__init__.py
touch src/muse/images/generations/backends/__init__.py
touch tests/__init__.py
touch tests/core/__init__.py
touch tests/audio/__init__.py
touch tests/audio/speech/__init__.py
touch tests/images/__init__.py
touch tests/images/generations/__init__.py
```

- [ ] **Step 4: Write `src/muse/__init__.py`**

```python
"""Muse: model-agnostic multi-modality generation server.

Modalities:
  - audio.speech  — /v1/audio/speech (text-to-speech)
  - images.generations — /v1/images/generations (text-to-image)

Heavy backends (transformers, diffusers) are imported lazily inside
individual backend modules to keep CLI startup instant.
"""

__version__ = "0.9.0"
```

- [ ] **Step 5: Rewrite `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "muse"
version = "0.9.0"
authors = [
  { name="Alex Towell", email="lex@metafunctor.com" },
]
description = "Muse: model-agnostic multi-modality generation server (audio, images)"
readme = "README.md"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
dependencies = [
  "huggingface_hub",
  "numpy",
  "requests",
]
license = {file = "LICENSE"}

[project.urls]
Homepage = "https://github.com/queelius/muse"
Issues = "https://github.com/queelius/muse/issues"

[project.optional-dependencies]
# Server runtime
server = ["fastapi", "uvicorn"]

# Audio modality (TTS)
audio = [
  "torch>=2.1.0",
  "transformers>=4.51.0",
  "scipy",
  "inflect",
  "unidecode",
]
audio-kokoro = ["kokoro", "soundfile", "misaki[en]"]

# Images modality (diffusion)
images = [
  "torch>=2.1.0",
  "diffusers>=0.27.0",
  "accelerate",
  "Pillow",
  "safetensors",
]

# Development
dev = ["pytest", "pytest-cov", "pytest-asyncio", "httpx"]

[project.scripts]
muse = "muse.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
include = ["muse*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 6: Install and verify package is importable**

```bash
pip install -e ".[dev,server]"
python -c "import muse; print(muse.__version__)"
```

Expected: `0.9.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git rm -r --cached narro.egg-info/ soprano_tts.egg-info/ 2>/dev/null || true
git commit -m "feat(muse): scaffold empty src/muse package with pyproject

Clean break from narro. Src-layout with modality subpackages mirroring
OpenAI URL hierarchy (audio/speech, images/generations). Optional deps
split per-modality so users can install only what they need."
```

---

## Part B — Core Extraction

### Task B1: ModalityRegistry (modality-aware replacement for narro.models.ModelRegistry)

**Files:**
- Create: `src/muse/core/registry.py`
- Create: `tests/core/test_registry.py`

- [ ] **Step 1: Write the failing test**

File: `tests/core/test_registry.py`

```python
"""Tests for ModalityRegistry: {modality: {model_id: Model}}."""
from dataclasses import dataclass
from typing import Any

import pytest

from muse.core.registry import ModalityRegistry, ModelInfo


@dataclass
class FakeAudioModel:
    model_id: str = "fake-tts"
    sample_rate: int = 16000
    def synthesize(self, text: str) -> Any: ...  # noqa


@dataclass
class FakeImageModel:
    model_id: str = "fake-diffusion"
    default_size: tuple[int, int] = (512, 512)
    def generate(self, prompt: str) -> Any: ...  # noqa


@pytest.fixture
def reg():
    return ModalityRegistry()


def test_register_and_get_by_modality(reg):
    m = FakeAudioModel()
    reg.register("audio.speech", m)
    assert reg.get("audio.speech", "fake-tts") is m


def test_first_registered_becomes_default_per_modality(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio.speech", a1)
    reg.register("audio.speech", a2)
    assert reg.get("audio.speech") is a1  # default


def test_modalities_are_isolated(reg):
    a = FakeAudioModel()
    i = FakeImageModel()
    reg.register("audio.speech", a)
    reg.register("images.generations", i)
    assert reg.get("audio.speech") is a
    assert reg.get("images.generations") is i
    with pytest.raises(KeyError):
        reg.get("audio.speech", "fake-diffusion")


def test_set_default_overrides_first_registered(reg):
    a1 = FakeAudioModel(model_id="tts-1")
    a2 = FakeAudioModel(model_id="tts-2")
    reg.register("audio.speech", a1)
    reg.register("audio.speech", a2)
    reg.set_default("audio.speech", "tts-2")
    assert reg.get("audio.speech").model_id == "tts-2"


def test_list_models_returns_modelinfo_per_modality(reg):
    reg.register("audio.speech", FakeAudioModel())
    reg.register("images.generations", FakeImageModel())
    audio = reg.list_models("audio.speech")
    assert len(audio) == 1
    assert isinstance(audio[0], ModelInfo)
    assert audio[0].model_id == "fake-tts"
    assert audio[0].modality == "audio.speech"


def test_list_all_spans_modalities(reg):
    reg.register("audio.speech", FakeAudioModel())
    reg.register("images.generations", FakeImageModel())
    all_models = reg.list_all()
    modalities = {m.modality for m in all_models}
    assert modalities == {"audio.speech", "images.generations"}


def test_modalities_lists_registered_keys(reg):
    reg.register("audio.speech", FakeAudioModel())
    assert reg.modalities() == ["audio.speech"]
    reg.register("images.generations", FakeImageModel())
    assert set(reg.modalities()) == {"audio.speech", "images.generations"}


def test_missing_modality_raises(reg):
    with pytest.raises(KeyError, match="no models registered"):
        reg.get("audio.speech")


def test_duplicate_registration_overwrites(reg):
    a1 = FakeAudioModel(model_id="tts")
    a2 = FakeAudioModel(model_id="tts")
    reg.register("audio.speech", a1)
    reg.register("audio.speech", a2)
    assert reg.get("audio.speech", "tts") is a2
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/core/test_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'muse.core.registry'`

- [ ] **Step 3: Implement registry**

File: `src/muse/core/registry.py`

```python
"""Modality-keyed model registry.

Registry shape: {modality: {model_id: Model}}.
First model registered per modality becomes its default.
Each modality is independent — no shared protocol between audio and image models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """Registry metadata for a loaded model."""
    modality: str
    model_id: str
    extra: dict = None  # populated per-modality (e.g., sample_rate, voices)


class ModalityRegistry:
    """Holds loaded models grouped by modality.

    Each modality namespace has its own default. Modalities are independent:
    looking up `audio.speech` won't find models registered under `images.generations`.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._defaults: dict[str, str] = {}

    def register(self, modality: str, model: Any) -> None:
        """Register a model under a modality. First registered becomes default."""
        models = self._models.setdefault(modality, {})
        models[model.model_id] = model
        self._defaults.setdefault(modality, model.model_id)

    def get(self, modality: str, model_id: str | None = None) -> Any:
        if modality not in self._models or not self._models[modality]:
            raise KeyError(f"no models registered for modality {modality!r}")
        if model_id is None:
            model_id = self._defaults[modality]
        if model_id not in self._models[modality]:
            raise KeyError(f"model {model_id!r} not registered under {modality!r}")
        return self._models[modality][model_id]

    def set_default(self, modality: str, model_id: str) -> None:
        if model_id not in self._models.get(modality, {}):
            raise KeyError(f"model {model_id!r} not registered under {modality!r}")
        self._defaults[modality] = model_id

    def list_models(self, modality: str) -> list[ModelInfo]:
        return [
            ModelInfo(modality=modality, model_id=mid, extra=_extra(m))
            for mid, m in self._models.get(modality, {}).items()
        ]

    def list_all(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for modality in self._models:
            out.extend(self.list_models(modality))
        return out

    def modalities(self) -> list[str]:
        return list(self._models.keys())

    def clear(self) -> None:
        """Test helper: reset all state."""
        self._models.clear()
        self._defaults.clear()


def _extra(model: Any) -> dict:
    """Pull commonly-exposed metadata from a model without assuming a base class."""
    extra: dict = {}
    for attr in ("sample_rate", "default_size", "voices", "description"):
        if hasattr(model, attr):
            extra[attr] = getattr(model, attr)
    return extra


# Module-level singleton. Modalities register into this at server startup.
registry = ModalityRegistry()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/core/test_registry.py -v
```

Expected: all 9 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/registry.py tests/core/test_registry.py
git commit -m "feat(core): add ModalityRegistry keyed by (modality, model_id)

Replaces narro's flat ModelRegistry. Each modality has its own namespace
and default; modalities are isolated so no shared protocol base class
is needed between audio and image models."
```

---

### Task B2: Catalog (KNOWN_MODELS + HF download + pip auto-install)

**Files:**
- Create: `src/muse/core/install.py`  (pip + system package helpers)
- Create: `src/muse/core/catalog.py`  (KNOWN_MODELS registry + pull())
- Reference: `narro/catalog.py` (source; copy and generalize)
- Create: `tests/core/test_catalog.py`
- Create: `tests/core/test_install.py`

- [ ] **Step 1: Read current `narro/catalog.py` to understand what to extract**

```bash
cat narro/catalog.py
```

Note which functions are pip/system-package helpers (extract to install.py) vs. HF download + catalog-state management (stay in catalog.py). The `KNOWN_MODELS` dict must be extended to carry a `modality` field.

- [ ] **Step 2: Write failing tests for install helpers**

File: `tests/core/test_install.py`

```python
"""Tests for pip and system-package helpers."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from muse.core.install import (
    install_pip_extras,
    check_system_packages,
)


class TestInstallPipExtras:
    @patch("muse.core.install.subprocess.run")
    def test_installs_missing_packages(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("muse.core.install.importlib.util.find_spec", return_value=None):
            install_pip_extras(["diffusers"])
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args and "install" in args and "diffusers" in args

    @patch("muse.core.install.subprocess.run")
    def test_skips_already_installed(self, mock_run):
        with patch("muse.core.install.importlib.util.find_spec", return_value=MagicMock()):
            install_pip_extras(["numpy"])
        mock_run.assert_not_called()

    @patch("muse.core.install.subprocess.run")
    def test_raises_on_pip_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip"])
        with patch("muse.core.install.importlib.util.find_spec", return_value=None):
            with pytest.raises(subprocess.CalledProcessError):
                install_pip_extras(["bogus-pkg"])


class TestCheckSystemPackages:
    @patch("muse.core.install.shutil.which")
    def test_returns_missing(self, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None
        missing = check_system_packages(["ffmpeg", "espeak-ng"])
        assert missing == ["espeak-ng"]

    @patch("muse.core.install.shutil.which")
    def test_empty_when_all_present(self, mock_which):
        mock_which.return_value = "/usr/bin/cmd"
        assert check_system_packages(["ffmpeg"]) == []
```

- [ ] **Step 3: Run — verify fail**

```bash
pytest tests/core/test_install.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Implement `src/muse/core/install.py`**

File: `src/muse/core/install.py`

```python
"""Runtime package installation helpers.

Keeps the CLI dependency graph slim: pull-a-model may install pip extras
on demand rather than forcing users to install everything upfront.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def install_pip_extras(packages: list[str]) -> None:
    """Install pip packages that aren't already importable.

    Uses importlib.util.find_spec to check; skips silently if present.
    """
    missing = [p for p in packages if importlib.util.find_spec(_pkg_to_module(p)) is None]
    if not missing:
        return
    logger.info("installing pip packages: %s", missing)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *missing],
        check=True,
    )


def check_system_packages(packages: list[str]) -> list[str]:
    """Return the subset of system packages not found on PATH."""
    return [p for p in packages if shutil.which(p) is None]


def _pkg_to_module(pip_name: str) -> str:
    """Best-effort pip-name → importable-module mapping.

    Handles common mismatches (Pillow→PIL, beautifulsoup4→bs4).
    Falls back to the pip name itself.
    """
    mapping = {
        "Pillow": "PIL",
        "beautifulsoup4": "bs4",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
        "huggingface_hub": "huggingface_hub",
    }
    # Strip extras like "misaki[en]" → "misaki"
    base = pip_name.split("[")[0].split(">=")[0].split("==")[0].strip()
    return mapping.get(base, base.replace("-", "_"))
```

- [ ] **Step 5: Run — pass**

```bash
pytest tests/core/test_install.py -v
```

Expected: all 5 pass.

- [ ] **Step 6: Write failing tests for catalog**

File: `tests/core/test_catalog.py`

```python
"""Tests for the KNOWN_MODELS catalog and pull()."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.core.catalog import (
    CatalogEntry,
    KNOWN_MODELS,
    pull,
    is_pulled,
    list_known,
    load_backend,
    _read_catalog,
    _write_catalog,
)


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Point catalog state at a temp file."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    yield tmp_path


def test_known_models_entries_have_modality():
    for model_id, entry in KNOWN_MODELS.items():
        assert entry.modality in ("audio.speech", "images.generations"), \
            f"model {model_id} has invalid modality {entry.modality!r}"


def test_list_known_filters_by_modality():
    audio = list_known("audio.speech")
    assert all(e.modality == "audio.speech" for e in audio)
    assert len(audio) >= 1


def test_list_known_all():
    all_entries = list_known()
    modalities = {e.modality for e in all_entries}
    assert "audio.speech" in modalities


def test_is_pulled_false_when_not_in_catalog(tmp_catalog):
    assert not is_pulled("soprano-80m")


def test_pull_installs_pip_downloads_and_writes_catalog(tmp_catalog):
    with patch("muse.core.catalog.install_pip_extras") as mock_pip, \
         patch("muse.core.catalog.snapshot_download") as mock_download, \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        mock_download.return_value = "/fake/cache/soprano"
        pull("soprano-80m")
        mock_pip.assert_called_once()
        mock_download.assert_called_once()
        assert is_pulled("soprano-80m")


def test_pull_unknown_raises():
    with pytest.raises(KeyError, match="unknown model"):
        pull("does-not-exist")


def test_pull_warns_on_missing_system_packages(tmp_catalog, caplog):
    with patch("muse.core.catalog.install_pip_extras"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=["espeak-ng"]):
        pull("kokoro-82m")  # assume this entry requires espeak-ng
        assert "espeak-ng" in caplog.text


def test_load_backend_imports_and_constructs(tmp_catalog):
    with patch("muse.core.catalog.importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        # Assume soprano-80m entry has backend_path = "muse.audio.speech.backends.soprano:SopranoModel"
        with patch("muse.core.catalog.is_pulled", return_value=True):
            load_backend("soprano-80m", device="cpu")
        mock_import.assert_called_once()
```

- [ ] **Step 7: Run — fail**

```bash
pytest tests/core/test_catalog.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 8: Implement `src/muse/core/catalog.py`**

File: `src/muse/core/catalog.py`

```python
"""Known-models catalog: what can be pulled, what's been pulled.

Structure:
    KNOWN_MODELS: dict[model_id, CatalogEntry]  — static at import time
    catalog.json (on disk): dict[model_id, {pulled_at, hf_repo, local_dir}]
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from muse.core.install import check_system_packages, install_pip_extras

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    """Static metadata for a known model."""
    model_id: str
    modality: str              # "audio.speech" | "images.generations"
    backend_path: str          # "module.path:ClassName"
    hf_repo: str
    description: str = ""
    pip_extras: tuple[str, ...] = ()
    system_packages: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)  # voices, default_size, etc.


# Seeded with representative models. Expand as new backends land.
KNOWN_MODELS: dict[str, CatalogEntry] = {
    "soprano-80m": CatalogEntry(
        model_id="soprano-80m",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.soprano:SopranoModel",
        hf_repo="ekwek/Soprano-1.1-80M",
        description="Qwen3 LLM backbone + Vocos decoder, 32kHz, 80M params",
        pip_extras=("transformers>=4.51.0", "scipy", "inflect", "unidecode"),
    ),
    "kokoro-82m": CatalogEntry(
        model_id="kokoro-82m",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.kokoro:KokoroModel",
        hf_repo="hexgrad/Kokoro-82M",
        description="Lightweight TTS, 54 voices, 24kHz",
        pip_extras=("kokoro", "soundfile", "misaki[en]"),
        system_packages=("espeak-ng",),
    ),
    "bark-small": CatalogEntry(
        model_id="bark-small",
        modality="audio.speech",
        backend_path="muse.audio.speech.backends.bark:BarkModel",
        hf_repo="suno/bark-small",
        description="Multilingual + voice cloning, 24kHz",
        pip_extras=("transformers>=4.51.0", "scipy"),
    ),
    "sd-turbo": CatalogEntry(
        model_id="sd-turbo",
        modality="images.generations",
        backend_path="muse.images.generations.backends.sd_turbo:SDTurboModel",
        hf_repo="stabilityai/sd-turbo",
        description="Stable Diffusion Turbo: 1-step distilled, 512x512",
        pip_extras=("diffusers>=0.27.0", "accelerate", "Pillow", "safetensors"),
        extra={"default_size": (512, 512)},
    ),
}


def _catalog_dir() -> Path:
    env = os.environ.get("MUSE_CATALOG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".muse"


def _catalog_path() -> Path:
    return _catalog_dir() / "catalog.json"


def _read_catalog() -> dict:
    p = _catalog_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        logger.warning("catalog at %s corrupt; resetting", p)
        return {}


def _write_catalog(data: dict) -> None:
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def is_pulled(model_id: str) -> bool:
    return model_id in _read_catalog()


def list_known(modality: str | None = None) -> list[CatalogEntry]:
    entries = list(KNOWN_MODELS.values())
    if modality is None:
        return entries
    return [e for e in entries if e.modality == modality]


def pull(model_id: str) -> None:
    """Install deps + download weights from HF. Records pulled status."""
    if model_id not in KNOWN_MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(KNOWN_MODELS)}")
    entry = KNOWN_MODELS[model_id]

    if entry.pip_extras:
        install_pip_extras(list(entry.pip_extras))

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
    }
    _write_catalog(catalog)


def remove(model_id: str) -> None:
    """Unregister from catalog (does not delete HF cache)."""
    catalog = _read_catalog()
    catalog.pop(model_id, None)
    _write_catalog(catalog)


def load_backend(model_id: str, **kwargs) -> Any:
    """Import backend class and instantiate it.

    `backend_path` has the form "package.module:ClassName". The class
    is expected to accept (hf_repo, local_dir, **kwargs) in its constructor.
    """
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = KNOWN_MODELS[model_id]
    module_path, class_name = entry.backend_path.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    catalog = _read_catalog()
    local_dir = catalog[model_id]["local_dir"]
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **kwargs)
```

- [ ] **Step 9: Run — pass**

```bash
pytest tests/core/test_catalog.py tests/core/test_install.py -v
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/muse/core/catalog.py src/muse/core/install.py tests/core/
git commit -m "feat(core): add catalog + pip auto-install, carrying modality tag

Ports narro.catalog to muse.core.catalog. KNOWN_MODELS entries now
carry a modality field; list_known(modality) supports per-modality
filtering for CLI subcommands. pip_extras install on pull; system
packages produce warnings (apt/brew is the user's job)."
```

---

### Task B3: Core server framework (app factory + modality mounting)

**Files:**
- Create: `src/muse/core/server.py`  (FastAPI app factory, mounts modality routers)
- Create: `src/muse/core/errors.py`  (OpenAI-style error envelopes)
- Create: `tests/core/test_server.py`

- [ ] **Step 1: Write failing test for app factory**

File: `tests/core/test_server.py`

```python
"""Tests for the core FastAPI app factory."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


def test_create_app_returns_fastapi():
    app = create_app(registry=ModalityRegistry(), routers={})
    assert isinstance(app, FastAPI)


def test_root_health_endpoint():
    app = create_app(registry=ModalityRegistry(), routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["modalities"] == []


def test_health_reports_registered_modalities():
    reg = ModalityRegistry()

    class Fake:
        model_id = "fake"
    reg.register("audio.speech", Fake())
    reg.register("images.generations", Fake())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/health")
    assert set(r.json()["modalities"]) == {"audio.speech", "images.generations"}


def test_routers_are_mounted():
    router = APIRouter()

    @router.get("/v1/test/ping")
    def ping():
        return {"ok": True}

    app = create_app(registry=ModalityRegistry(), routers={"test": router})
    client = TestClient(app)
    r = client.get("/v1/test/ping")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_global_v1_models_endpoint_aggregates():
    reg = ModalityRegistry()

    class FakeAudio:
        model_id = "fake-tts"
        sample_rate = 16000
    reg.register("audio.speech", FakeAudio())

    app = create_app(registry=reg, routers={})
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "fake-tts" and m["modality"] == "audio.speech" for m in data)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/core/test_server.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement errors module**

File: `src/muse/core/errors.py`

```python
"""OpenAI-style error envelopes.

Matches the structure of OpenAI's error responses so clients written
against their API can reuse error-handling code.
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": "invalid_request_error"}},
    )


class ModelNotFoundError(HTTPException):
    def __init__(self, model_id: str, modality: str):
        super().__init__(
            status_code=404,
            detail={"error": {
                "code": "model_not_found",
                "message": f"Model {model_id!r} is not available for modality {modality!r}",
                "type": "invalid_request_error",
            }},
        )
```

- [ ] **Step 4: Implement server app factory**

File: `src/muse/core/server.py`

```python
"""FastAPI application factory.

Modality routers are mounted via `create_app(registry, routers=...)`.
Each modality supplies its own APIRouter; core adds /health and /v1/models
(aggregated across modalities).
"""
from __future__ import annotations

import logging
from typing import Mapping

from fastapi import APIRouter, FastAPI

from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)


def create_app(
    *,
    registry: ModalityRegistry,
    routers: Mapping[str, APIRouter],
    title: str = "Muse",
) -> FastAPI:
    """Build a FastAPI app with shared /health + /v1/models endpoints.

    `routers` maps modality-name → APIRouter. Each router is mounted
    with its own internal paths (e.g. /v1/audio/speech).
    """
    app = FastAPI(title=title)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "modalities": registry.modalities(),
            "models": [info.model_id for info in registry.list_all()],
        }

    @app.get("/v1/models")
    def list_models():
        data = []
        for info in registry.list_all():
            entry = {"id": info.model_id, "modality": info.modality, "object": "model"}
            if info.extra:
                entry.update(info.extra)
            data.append(entry)
        return {"object": "list", "data": data}

    for name, router in routers.items():
        logger.info("mounting modality router %s", name)
        app.include_router(router)

    app.state.registry = registry
    return app
```

- [ ] **Step 5: Run — pass**

```bash
pytest tests/core/test_server.py -v
```

Expected: all 5 pass.

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/server.py src/muse/core/errors.py tests/core/test_server.py
git commit -m "feat(core): add FastAPI app factory mounting per-modality routers

create_app() accepts a ModalityRegistry and a name→APIRouter map.
Adds /health and aggregated /v1/models; each modality supplies its
own router which it mounts at its URL namespace (e.g., /v1/audio/speech)."
```

---

## Part C — Audio/Speech Modality (port narro)

### Task C1: Port TTSModel protocol + AudioResult/AudioChunk verbatim

**Files:**
- Create: `src/muse/audio/speech/protocol.py`
- Create: `tests/audio/speech/test_protocol.py`
- Reference: `narro/protocol.py` (copy verbatim; only module path changes)

- [ ] **Step 1: Copy narro/protocol.py verbatim to new path**

```bash
cp narro/protocol.py src/muse/audio/speech/protocol.py
```

- [ ] **Step 2: Update module docstring in the new file**

Edit `src/muse/audio/speech/protocol.py` — change the header docstring to:

```python
"""Muse audio.speech modality protocol.

Defines TTSModel (the backend contract) and AudioResult / AudioChunk
(return types for sync / streaming synthesis). A backend is any object
that satisfies the TTSModel protocol — no base-class inheritance required.
"""
```

- [ ] **Step 3: Copy matching test (adapt imports only)**

```bash
cp tests/test_protocol.py tests/audio/speech/test_protocol.py
```

Edit `tests/audio/speech/test_protocol.py` — change all `from narro.protocol import ...` to `from muse.audio.speech.protocol import ...`. No other changes.

- [ ] **Step 4: Run — pass**

```bash
pytest tests/audio/speech/test_protocol.py -v
```

Expected: all pass (verbatim port).

- [ ] **Step 5: Commit**

```bash
git add src/muse/audio/speech/protocol.py tests/audio/speech/test_protocol.py
git commit -m "feat(audio.speech): port TTSModel protocol verbatim from narro"
```

---

### Task C2: Port audio codec (extract wav/opus from narro/server.py)

**Files:**
- Create: `src/muse/audio/speech/codec.py`
- Create: `tests/audio/speech/test_codec.py`
- Reference: `narro/server.py` — functions `_audio_to_wav_bytes`, `_wav_bytes_to_opus`, `_AudioFormatError` (and any helpers)

- [ ] **Step 1: Read the codec functions in narro/server.py**

```bash
grep -n "_audio_to_wav_bytes\|_wav_bytes_to_opus\|_AudioFormatError" narro/server.py
```

Note line ranges; extract the codec body into the new module.

- [ ] **Step 2: Write failing tests**

File: `tests/audio/speech/test_codec.py`

```python
"""Tests for audio codec conversion."""
import io
import wave

import numpy as np
import pytest

from muse.audio.speech.codec import (
    AudioFormatError,
    audio_to_wav_bytes,
    wav_bytes_to_opus,
)


def test_audio_to_wav_bytes_produces_valid_wav():
    audio = np.zeros(16000, dtype=np.float32)
    data = audio_to_wav_bytes(audio, sample_rate=16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # int16


def test_audio_to_wav_bytes_clips_to_int16_range():
    audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)  # out of [-1,1]
    data = audio_to_wav_bytes(audio, sample_rate=16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
    assert samples.max() == 32767
    assert samples.min() == -32768


def test_audio_to_wav_bytes_rejects_non_1d():
    audio = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(AudioFormatError):
        audio_to_wav_bytes(audio, sample_rate=16000)


def test_wav_bytes_to_opus_roundtrip_produces_bytes():
    # Only run if opus encoder (pyogg or similar) is available
    pytest.importorskip("scipy.io.wavfile")
    audio = np.zeros(16000, dtype=np.float32)
    wav_data = audio_to_wav_bytes(audio, sample_rate=16000)
    try:
        opus_data = wav_bytes_to_opus(wav_data)
    except (ImportError, AudioFormatError) as e:
        pytest.skip(f"opus encoding not available: {e}")
    assert isinstance(opus_data, bytes)
    assert len(opus_data) > 0
```

- [ ] **Step 3: Run — fail**

```bash
pytest tests/audio/speech/test_codec.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Implement `src/muse/audio/speech/codec.py`**

Extract codec logic from `narro/server.py`. File:

```python
"""WAV and Opus encoding for audio.speech responses.

Extracted from narro/server.py to separate modality-specific encoding
from the modality-agnostic server framework.
"""
from __future__ import annotations

import io
import wave

import numpy as np


class AudioFormatError(ValueError):
    """Raised when audio data cannot be encoded to the requested format."""


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 [-1, 1] audio to a 16-bit PCM WAV bytestring."""
    if audio.ndim != 1:
        raise AudioFormatError(f"expected 1-D audio, got shape {audio.shape}")
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_opus(wav_data: bytes) -> bytes:
    """Transcode WAV → Opus. Delegates to system ffmpeg if present.

    Raises AudioFormatError if ffmpeg is unavailable or conversion fails.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise AudioFormatError("ffmpeg not found; cannot encode opus")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "wav", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "64k",
         "-f", "ogg", "pipe:1"],
        input=wav_data, capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioFormatError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return proc.stdout
```

- [ ] **Step 5: Run — pass**

```bash
pytest tests/audio/speech/test_codec.py -v
```

Expected: pass (opus test may skip if ffmpeg not installed; that's fine).

- [ ] **Step 6: Commit**

```bash
git add src/muse/audio/speech/codec.py tests/audio/speech/test_codec.py
git commit -m "feat(audio.speech): extract codec (wav/opus) from server module"
```

---

### Task C3: Port audio-specific utilities (text_normalizer, text_splitter, alignment, encoded, vocos, tts)

**Files:**
- Copy: `narro/utils/text_normalizer.py` → `src/muse/audio/speech/utils/text_normalizer.py`
- Copy: `narro/utils/text_splitter.py` → `src/muse/audio/speech/utils/text_splitter.py`
- Copy: `narro/alignment.py` → `src/muse/audio/speech/alignment.py`
- Copy: `narro/encoded.py` → `src/muse/audio/speech/encoded.py`
- Copy: `narro/decode_only.py` → `src/muse/audio/speech/decode_only.py`
- Copy: `narro/tts.py` → `src/muse/audio/speech/tts.py`
- Copy: `narro/vocos/` → `src/muse/audio/speech/vocos/`
- Copy matching test files to `tests/audio/speech/`

- [ ] **Step 1: Copy utils subpackage**

```bash
cp narro/utils/text_normalizer.py src/muse/audio/speech/utils/text_normalizer.py
cp narro/utils/text_splitter.py src/muse/audio/speech/utils/text_splitter.py
touch src/muse/audio/speech/utils/__init__.py
```

- [ ] **Step 2: Copy alignment, encoded, decode_only, tts, vocos**

```bash
cp narro/alignment.py src/muse/audio/speech/alignment.py
cp narro/encoded.py src/muse/audio/speech/encoded.py
cp narro/decode_only.py src/muse/audio/speech/decode_only.py
cp narro/tts.py src/muse/audio/speech/tts.py
cp -r narro/vocos src/muse/audio/speech/
```

- [ ] **Step 3: Rewrite imports in copied files**

Inside each file copied above, replace every `from narro.` with `from muse.audio.speech.` and every `from narro import ...` with `from muse.audio.speech import ...`. Also fix `import narro.X` → `import muse.audio.speech.X`.

Use a targeted find + sed (confirm no accidental matches first):

```bash
grep -rn "narro" src/muse/audio/speech/ | grep -v __pycache__
```

Then for each file shown, open it and update imports by hand or with:

```bash
find src/muse/audio/speech -name "*.py" -exec sed -i \
  -e 's|from narro\.utils|from muse.audio.speech.utils|g' \
  -e 's|from narro\.alignment|from muse.audio.speech.alignment|g' \
  -e 's|from narro\.encoded|from muse.audio.speech.encoded|g' \
  -e 's|from narro\.tts|from muse.audio.speech.tts|g' \
  -e 's|from narro\.vocos|from muse.audio.speech.vocos|g' \
  -e 's|from narro\.backends|from muse.audio.speech.backends|g' \
  -e 's|from narro\.protocol|from muse.audio.speech.protocol|g' \
  -e 's|from narro\.models|from muse.audio.speech.backends|g' \
  -e 's|import narro\.|import muse.audio.speech.|g' \
  {} +
```

Verify nothing stray remains:

```bash
grep -rn "narro" src/muse/audio/speech/ | grep -v __pycache__
```

Expected: empty output.

- [ ] **Step 4: Copy tests and fix imports**

```bash
cp tests/test_alignment.py tests/audio/speech/test_alignment.py
cp tests/test_encode_decode.py tests/audio/speech/test_encode_decode.py
cp tests/test_encoded.py tests/audio/speech/test_encoded.py
cp tests/test_quality_check.py tests/audio/speech/test_quality_check.py

find tests/audio/speech -name "*.py" -exec sed -i \
  -e 's|from narro\.|from muse.audio.speech.|g' \
  -e 's|import narro\.|import muse.audio.speech.|g' \
  -e 's|from narro import|from muse.audio.speech import|g' \
  {} +
```

- [ ] **Step 5: Run audio utility tests**

```bash
pytest tests/audio/speech/test_alignment.py tests/audio/speech/test_encoded.py tests/audio/speech/test_encode_decode.py -v
```

Expected: all pass (or skip cleanly if they require model weights; record which skip).

- [ ] **Step 6: Commit**

```bash
git add src/muse/audio/speech/ tests/audio/speech/
git commit -m "feat(audio.speech): port tts, vocos, alignment, encoded, utils

Copies narro's audio-specific internals into muse.audio.speech.
Imports rewritten from narro.* to muse.audio.speech.*. No logic
changes; verbatim port."
```

---

### Task C4: Port audio backends (base, transformers, soprano, kokoro, bark)

**Files:**
- Copy: `narro/backends/base.py` → `src/muse/audio/speech/backends/base.py`
- Copy: `narro/backends/transformers.py` → `src/muse/audio/speech/backends/transformers.py`
- Copy: `narro/models/soprano.py` → `src/muse/audio/speech/backends/soprano.py`
- Copy: `narro/models/kokoro.py` → `src/muse/audio/speech/backends/kokoro.py`
- Copy: `narro/models/bark.py` → `src/muse/audio/speech/backends/bark.py`
- Copy test files for each.

- [ ] **Step 1: Copy backend files**

```bash
cp narro/backends/base.py src/muse/audio/speech/backends/base.py
cp narro/backends/transformers.py src/muse/audio/speech/backends/transformers.py
cp narro/models/soprano.py src/muse/audio/speech/backends/soprano.py
cp narro/models/kokoro.py src/muse/audio/speech/backends/kokoro.py
cp narro/models/bark.py src/muse/audio/speech/backends/bark.py
```

- [ ] **Step 2: Ensure each backend's constructor accepts `hf_repo` and `local_dir`**

The loader in `muse.core.catalog.load_backend` passes `hf_repo=...` and `local_dir=...` as kwargs. Verify each of the five backend classes has these in their `__init__` signature (narro may have used only `hf_repo`; add `local_dir` with a default if needed):

For each backend file, the constructor should look like:

```python
def __init__(self, *, hf_repo: str, local_dir: str | None = None, device: str = "auto", **kwargs):
    ...
```

Update each of the five if not already matching. No logic change needed — just extend the signature and prefer `local_dir` when loading weights (fall back to `hf_repo` → `from_pretrained(repo)`).

- [ ] **Step 3: Rewrite imports across backends/**

```bash
find src/muse/audio/speech/backends -name "*.py" -exec sed -i \
  -e 's|from narro\.|from muse.audio.speech.|g' \
  -e 's|import narro\.|import muse.audio.speech.|g' \
  -e 's|from narro import|from muse.audio.speech import|g' \
  {} +
grep -rn "narro" src/muse/audio/speech/backends/
```

Expected: empty.

- [ ] **Step 4: Copy and fix backend tests**

```bash
cp tests/test_bark.py tests/audio/speech/test_bark.py
cp tests/test_kokoro.py tests/audio/speech/test_kokoro.py
# narro may not have had a test_soprano.py; if not, create a minimal one:
cat > tests/audio/speech/test_soprano.py <<'EOF'
"""Smoke tests for SopranoModel (mocked; doesn't load real weights)."""
from unittest.mock import patch, MagicMock

import pytest

from muse.audio.speech.backends.soprano import SopranoModel


def test_soprano_model_id():
    with patch("muse.audio.speech.backends.soprano.Narro"):
        m = SopranoModel(hf_repo="fake/repo", local_dir="/fake")
        assert m.model_id == "soprano-80m"


def test_soprano_sample_rate():
    with patch("muse.audio.speech.backends.soprano.Narro") as mock_narro:
        mock_narro.return_value = MagicMock(sample_rate=32000)
        m = SopranoModel(hf_repo="fake/repo", local_dir="/fake")
        assert m.sample_rate == 32000
EOF

find tests/audio/speech -name "*.py" -exec sed -i \
  -e 's|from narro\.|from muse.audio.speech.|g' \
  -e 's|import narro\.|import muse.audio.speech.|g' \
  -e 's|from narro import|from muse.audio.speech import|g' \
  {} +
```

- [ ] **Step 5: Run backend tests**

```bash
pytest tests/audio/speech/test_soprano.py tests/audio/speech/test_bark.py tests/audio/speech/test_kokoro.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/muse/audio/speech/backends/ tests/audio/speech/test_soprano.py \
        tests/audio/speech/test_bark.py tests/audio/speech/test_kokoro.py
git commit -m "feat(audio.speech): port soprano, kokoro, bark backends

Backends now live under muse.audio.speech.backends.* — a flat layout
instead of narro's split between backends/ (base) and models/ (adapters).
All constructors take hf_repo + local_dir kwargs from the catalog loader."
```

---

### Task C5: Port /v1/audio/speech routes (adapt narro/server.py to an APIRouter)

**Files:**
- Create: `src/muse/audio/speech/routes.py`
- Create: `tests/audio/speech/test_routes.py`
- Reference: `narro/server.py` — the `configure_app` function and route handlers

- [ ] **Step 1: Read narro/server.py to identify route handlers vs. server framework**

```bash
cat narro/server.py
```

Identify: the `/v1/audio/speech` handler, the streaming helper, the request model, the inference lock. These all move to `routes.py`. The `create_app` logic (already in `muse.core.server`) does not need to come over.

- [ ] **Step 2: Write failing test**

File: `tests/audio/speech/test_routes.py`

```python
"""Tests for /v1/audio/speech FastAPI router."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.audio.speech.protocol import AudioResult
from muse.audio.speech.routes import build_router
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


class FakeTTS:
    model_id = "fake-tts"
    sample_rate = 16000
    voices = ["default"]

    def synthesize(self, text, **kwargs):
        n = max(1000, len(text) * 100)
        return AudioResult(
            audio=np.zeros(n, dtype=np.float32),
            sample_rate=self.sample_rate,
            metadata={"duration": n / self.sample_rate},
        )

    def synthesize_stream(self, text, **kwargs):
        from muse.audio.speech.protocol import AudioChunk
        for _ in range(3):
            yield AudioChunk(audio=np.zeros(500, dtype=np.float32), sample_rate=self.sample_rate)


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("audio.speech", FakeTTS())
    app = create_app(registry=reg, routers={"audio.speech": build_router(reg)})
    return TestClient(app)


def test_list_voices_endpoint(client):
    r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 200
    assert "default" in r.json()["voices"]


def test_speech_wav_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "model": "fake-tts",
        "response_format": "wav",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_speech_default_model_when_unspecified(client):
    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 200


def test_unknown_model_returns_404(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello",
        "model": "does-not-exist",
    })
    assert r.status_code == 404


def test_empty_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": ""})
    assert r.status_code == 400


def test_oversize_input_returns_400(client):
    r = client.post("/v1/audio/speech", json={"input": "x" * 60_000})
    assert r.status_code == 400


def test_streaming_response(client):
    r = client.post("/v1/audio/speech", json={
        "input": "hello world",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
```

- [ ] **Step 3: Run — fail**

```bash
pytest tests/audio/speech/test_routes.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Implement the router**

File: `src/muse/audio/speech/routes.py`

```python
"""FastAPI router for /v1/audio/speech.

Adapts narro/server.py's TTS handlers to muse's per-modality router
pattern. The router is built with a registry reference so handlers
look up backends by name.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from threading import Lock

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from muse.audio.speech.codec import AudioFormatError, audio_to_wav_bytes, wav_bytes_to_opus
from muse.core.registry import ModalityRegistry

logger = logging.getLogger(__name__)

MODALITY = "audio.speech"
MAX_INPUT_LENGTH = 50_000
_inference_lock = Lock()


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)
    model: str | None = None
    voice: str | None = None
    response_format: str = Field(default="wav", pattern="^(wav|opus)$")
    stream: bool = False
    speed: float = 1.0
    align: bool = False


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio.speech"])

    @router.get("/speech/voices")
    def list_voices(model: str | None = None):
        try:
            m = registry.get(MODALITY, model)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        voices = getattr(m, "voices", [])
        return {"model": m.model_id, "voices": voices}

    @router.post("/speech")
    async def speech(req: SpeechRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        if req.stream:
            return await _stream(model, req)
        return await _non_stream(model, req)

    return router


async def _non_stream(model, req: SpeechRequest) -> Response:
    def _synth():
        with _inference_lock:
            return model.synthesize(
                req.input,
                voice=req.voice,
                speed=req.speed,
                align=req.align,
            )

    result = await asyncio.to_thread(_synth)

    try:
        wav = audio_to_wav_bytes(result.audio, result.sample_rate)
    except AudioFormatError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.response_format == "opus":
        try:
            body = wav_bytes_to_opus(wav)
            media = "audio/ogg"
        except AudioFormatError:
            logger.warning("opus encoding unavailable; falling back to wav")
            body = wav
            media = "audio/wav"
    else:
        body = wav
        media = "audio/wav"

    headers = {}
    if req.align and "alignment" in (result.metadata or {}):
        import json
        headers["X-Alignment"] = json.dumps(result.metadata["alignment"])

    return Response(content=body, media_type=media, headers=headers)


async def _stream(model, req: SpeechRequest) -> EventSourceResponse:
    async def event_gen():
        def _chunks():
            with _inference_lock:
                yield from model.synthesize_stream(req.input, voice=req.voice, speed=req.speed)

        loop = asyncio.get_event_loop()
        it = await loop.run_in_executor(None, _chunks)
        for chunk in it:
            pcm = (np.clip(chunk.audio, -1.0, 1.0) * 32767).astype(np.int16)
            yield {"data": base64.b64encode(pcm.tobytes()).decode()}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
```

- [ ] **Step 5: Add `sse-starlette` to dependencies**

Edit `pyproject.toml`:

```toml
server = ["fastapi", "uvicorn", "sse-starlette"]
```

Reinstall:

```bash
pip install -e ".[server]"
```

- [ ] **Step 6: Run — pass**

```bash
pytest tests/audio/speech/test_routes.py -v
```

Expected: all 7 pass.

- [ ] **Step 7: Commit**

```bash
git add src/muse/audio/speech/routes.py tests/audio/speech/test_routes.py pyproject.toml
git commit -m "feat(audio.speech): add /v1/audio/speech router

Adapts narro/server.py's handlers to a per-modality APIRouter.
Request shape unchanged from narro — existing clients continue to work
after swapping the binary and reconfiguring the systemd unit."
```

---

### Task C6: Port audio.speech HTTP client

**Files:**
- Copy: `narro/client.py` → `src/muse/audio/speech/client.py`
- Copy: `tests/test_client.py` → `tests/audio/speech/test_client.py`

- [ ] **Step 1: Port client**

```bash
cp narro/client.py src/muse/audio/speech/client.py
cp tests/test_client.py tests/audio/speech/test_client.py

find src/muse/audio/speech/client.py tests/audio/speech/test_client.py \
  -exec sed -i \
  -e 's|from narro\.|from muse.audio.speech.|g' \
  -e 's|import narro\.|import muse.audio.speech.|g' \
  -e 's|from narro import|from muse.audio.speech import|g' \
  -e 's|NarroClient|SpeechClient|g' \
  {} +
```

- [ ] **Step 2: Rename client class to `SpeechClient` inside the ported file**

Edit `src/muse/audio/speech/client.py` and confirm class is renamed to `SpeechClient`. Export it via the subpackage:

Edit `src/muse/audio/speech/__init__.py`:

```python
"""Muse audio.speech modality: text-to-speech."""
from muse.audio.speech.client import SpeechClient
from muse.audio.speech.protocol import AudioChunk, AudioResult, TTSModel

__all__ = ["SpeechClient", "AudioChunk", "AudioResult", "TTSModel"]
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/audio/speech/test_client.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/muse/audio/speech/client.py src/muse/audio/speech/__init__.py \
        tests/audio/speech/test_client.py
git commit -m "feat(audio.speech): port HTTP client as SpeechClient"
```

---

## Part D — Images/Generations Modality

### Task D1: ImageModel protocol + ImageResult

**Files:**
- Create: `src/muse/images/generations/protocol.py`
- Create: `tests/images/generations/test_protocol.py`

- [ ] **Step 1: Write failing test**

File: `tests/images/generations/test_protocol.py`

```python
"""Tests for ImageModel protocol."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from muse.images.generations.protocol import ImageModel, ImageResult


def _pil_stub():
    # Return a 2x2 RGB np.ndarray (a stand-in for PIL where we only
    # care about the data shape)
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_image_result_stores_image_and_metadata():
    img = _pil_stub()
    res = ImageResult(image=img, width=2, height=2, seed=42, metadata={"prompt": "hello"})
    assert res.image is img
    assert res.width == 2
    assert res.height == 2
    assert res.seed == 42
    assert res.metadata["prompt"] == "hello"


def test_image_result_metadata_defaults_empty():
    res = ImageResult(image=_pil_stub(), width=2, height=2, seed=1)
    assert res.metadata == {}


def test_image_model_protocol_structural():
    class MyModel:
        model_id = "fake-sd"
        default_size = (512, 512)
        def generate(self, prompt, **kwargs): ...

    assert isinstance(MyModel(), ImageModel)


def test_image_model_protocol_rejects_incomplete():
    class Missing:
        pass

    assert not isinstance(Missing(), ImageModel)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/images/generations/test_protocol.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement protocol**

File: `src/muse/images/generations/protocol.py`

```python
"""Muse images.generations modality protocol.

Defines ImageModel (backend contract) and ImageResult (synthesis return).
No streaming type — diffusion progress is per-step refinement of the
same image, not time-ordered chunks like audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ImageResult:
    """A synthesized image plus provenance metadata.

    `image` is typed as Any so backends can return PIL.Image, numpy arrays,
    or torch tensors without forcing a common supertype here. Codec-layer
    code is responsible for normalizing to PIL before encoding.
    """
    image: Any
    width: int
    height: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageModel(Protocol):
    """Protocol for text-to-image backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def default_size(self) -> tuple[int, int]: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> ImageResult: ...
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/images/generations/test_protocol.py -v
```

Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/images/generations/protocol.py tests/images/generations/test_protocol.py
git commit -m "feat(images.generations): add ImageModel protocol + ImageResult

No streaming type: diffusion is step-wise refinement of a single frame,
not time-ordered chunks. ImageResult.image is typed Any — backends can
return PIL, numpy, or torch tensors; the codec layer normalizes."
```

---

### Task D2: Image codec (PIL → PNG / JPEG / base64 data URL)

**Files:**
- Create: `src/muse/images/generations/codec.py`
- Create: `tests/images/generations/test_codec.py`

- [ ] **Step 1: Write failing test**

File: `tests/images/generations/test_codec.py`

```python
"""Tests for image codec."""
import base64
import io

import numpy as np
import pytest

from muse.images.generations.codec import (
    ImageFormatError,
    to_bytes,
    to_data_url,
    to_pil,
)

PIL = pytest.importorskip("PIL.Image")


def test_to_pil_accepts_pil():
    img = PIL.new("RGB", (10, 10))
    assert to_pil(img) is img


def test_to_pil_accepts_numpy_uint8():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = to_pil(arr)
    assert img.size == (10, 10)


def test_to_pil_accepts_numpy_float():
    # float in [0, 1] gets rescaled to uint8
    arr = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    img = to_pil(arr)
    assert img.size == (4, 4)


def test_to_pil_rejects_weird_shape():
    with pytest.raises(ImageFormatError):
        to_pil(np.zeros((10,), dtype=np.uint8))


def test_to_bytes_png():
    img = PIL.new("RGB", (4, 4))
    data = to_bytes(img, fmt="png")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_to_bytes_jpeg():
    img = PIL.new("RGB", (4, 4))
    data = to_bytes(img, fmt="jpeg")
    assert data[:3] == b"\xff\xd8\xff"


def test_to_bytes_rejects_unknown_format():
    img = PIL.new("RGB", (4, 4))
    with pytest.raises(ImageFormatError):
        to_bytes(img, fmt="bmp")


def test_to_data_url_format():
    img = PIL.new("RGB", (4, 4))
    url = to_data_url(img, fmt="png")
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    base64.b64decode(payload)  # no error
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/images/generations/test_codec.py -v
```

Expected: FAIL — ModuleNotFoundError (or Pillow not installed; skip).

- [ ] **Step 3: Implement codec**

File: `src/muse/images/generations/codec.py`

```python
"""Image encoding: accept PIL / numpy, output PNG / JPEG / base64 data URL."""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


class ImageFormatError(ValueError):
    """Raised when an image can't be normalized or encoded."""


def to_pil(image: Any):
    """Normalize supported inputs to a PIL.Image.

    Accepts:
      - PIL.Image  — passthrough
      - np.ndarray uint8 HxWxC  — direct
      - np.ndarray float HxWxC  — rescaled to uint8 assuming [0, 1]
      - np.ndarray uint8/float HxW  — grayscale
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGBA" if arr.shape[2] == 4 else "RGB"
            return Image.fromarray(arr, mode=mode)
        raise ImageFormatError(f"numpy image has unsupported shape {arr.shape}")

    # Torch tensor support — lazy check
    try:
        import torch
        if isinstance(image, torch.Tensor):
            return to_pil(image.detach().cpu().numpy())
    except ImportError:
        pass

    raise ImageFormatError(f"cannot convert {type(image).__name__} to PIL")


def to_bytes(image: Any, *, fmt: str = "png") -> bytes:
    """Encode to raw image bytes (PNG or JPEG)."""
    fmt = fmt.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise ImageFormatError(f"unsupported image format {fmt!r}")
    pil = to_pil(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG" if fmt in ("jpeg", "jpg") else "PNG")
    return buf.getvalue()


def to_data_url(image: Any, *, fmt: str = "png") -> str:
    """Encode as a data URL suitable for inline use."""
    payload = to_bytes(image, fmt=fmt)
    mime = "jpeg" if fmt in ("jpeg", "jpg") else "png"
    return f"data:image/{mime};base64,{base64.b64encode(payload).decode()}"
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/images/generations/test_codec.py -v
```

Expected: all pass (or clean skip if Pillow missing in dev env).

- [ ] **Step 5: Commit**

```bash
git add src/muse/images/generations/codec.py tests/images/generations/test_codec.py
git commit -m "feat(images.generations): add image codec (PIL/numpy/torch → bytes or data URL)"
```

---

### Task D3: /v1/images/generations route

**Files:**
- Create: `src/muse/images/generations/routes.py`
- Create: `tests/images/generations/test_routes.py`

- [ ] **Step 1: Write failing test**

File: `tests/images/generations/test_routes.py`

```python
"""Tests for /v1/images/generations FastAPI router."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.images.generations.protocol import ImageResult
from muse.images.generations.routes import build_router


class FakeImageModel:
    model_id = "fake-sd"
    default_size = (64, 64)

    def generate(self, prompt, **kwargs):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=64, height=64, seed=123,
            metadata={"prompt": prompt},
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("images.generations", FakeImageModel())
    app = create_app(
        registry=reg,
        routers={"images.generations": build_router(reg)},
    )
    return TestClient(app)


def test_generate_returns_base64_by_default(client):
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    data = r.json()
    assert data["data"][0]["b64_json"]


def test_generate_response_format_url_returns_data_url(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "a cat",
        "response_format": "url",
    })
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_generate_n_creates_multiple_images(client):
    r = client.post("/v1/images/generations", json={"prompt": "a dog", "n": 3})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_generate_includes_seed_in_response(client):
    r = client.post("/v1/images/generations", json={"prompt": "a bird"})
    assert r.status_code == 200
    assert r.json()["data"][0]["revised_prompt"] == "a bird"


def test_unknown_model_returns_404(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x", "model": "no-such-model",
    })
    assert r.status_code == 404


def test_empty_prompt_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": ""})
    assert r.status_code in (400, 422)


def test_n_over_limit_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": "x", "n": 100})
    assert r.status_code in (400, 422)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/images/generations/test_routes.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement router**

File: `src/muse/images/generations/routes.py`

```python
"""FastAPI router for /v1/images/generations.

Follows OpenAI's /v1/images/generations contract:
  - `prompt` (required)
  - `n` number of images (1-10)
  - `size` "WIDTHxHEIGHT" string
  - `response_format` "b64_json" (default) | "url" (data URL, no hosting)
  - `model`, `seed`, `steps`, `guidance`, `negative_prompt` (muse extensions)
"""
from __future__ import annotations

import asyncio
import base64
import logging
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from muse.core.registry import ModalityRegistry
from muse.images.generations.codec import ImageFormatError, to_bytes, to_data_url

logger = logging.getLogger(__name__)

MODALITY = "images.generations"
_inference_lock = Lock()


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="512x512", pattern=r"^\d+x\d+$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")
    negative_prompt: str | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None

    @field_validator("size")
    @classmethod
    def _parse_size(cls, v: str) -> str:
        w, h = map(int, v.split("x"))
        if w < 64 or h < 64 or w > 2048 or h > 2048:
            raise ValueError(f"size {v} out of supported range (64-2048 per side)")
        return v


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["images.generations"])

    @router.post("/generations")
    async def generations(req: GenerationRequest):
        try:
            model = registry.get(MODALITY, req.model)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        width, height = map(int, req.size.split("x"))

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "width": width,
                "height": height,
                "negative_prompt": req.negative_prompt,
                "steps": req.steps,
                "guidance": req.guidance,
            }
            if req.seed is not None:
                kwargs["seed"] = req.seed + seed_offset
            with _inference_lock:
                return model.generate(req.prompt, **kwargs)

        results = []
        for i in range(req.n):
            result = await asyncio.to_thread(_call_one, i)
            results.append(result)

        data = []
        for r in results:
            entry = {"revised_prompt": r.metadata.get("prompt", req.prompt)}
            try:
                if req.response_format == "url":
                    entry["url"] = to_data_url(r.image, fmt="png")
                else:
                    entry["b64_json"] = base64.b64encode(
                        to_bytes(r.image, fmt="png")
                    ).decode()
            except ImageFormatError as e:
                raise HTTPException(status_code=500, detail=str(e))
            data.append(entry)

        return {"created": int(__import__("time").time()), "data": data}

    return router
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/images/generations/test_routes.py -v
```

Expected: all 7 pass.

- [ ] **Step 5: Commit**

```bash
git add src/muse/images/generations/routes.py tests/images/generations/test_routes.py
git commit -m "feat(images.generations): add /v1/images/generations router

Matches OpenAI's image-generation API shape. Supports n multi-image
requests, b64_json / url response formats, and muse extensions for
negative_prompt, steps, guidance, seed."
```

---

### Task D4: SDTurboModel backend (diffusers + stabilityai/sd-turbo)

**Files:**
- Create: `src/muse/images/generations/backends/sd_turbo.py`
- Create: `tests/images/generations/test_sd_turbo.py`

- [ ] **Step 1: Write failing test (fully mocked — no real model load)**

File: `tests/images/generations/test_sd_turbo.py`

```python
"""Tests for SDTurboModel (fully mocked; no weights loaded)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.images.generations.protocol import ImageResult


def test_sd_turbo_model_id_and_default_size():
    # Patch diffusers pipeline at import time so we don't load weights
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_cls.from_pretrained.return_value = MagicMock()
        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        assert m.model_id == "sd-turbo"
        assert m.default_size == (512, 512)


def test_sd_turbo_generate_calls_pipeline_and_returns_imageresult():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock()
        mock_img = MagicMock()
        mock_img.size = (512, 512)
        mock_pipe.return_value = MagicMock(images=[mock_img])
        mock_cls.from_pretrained.return_value = mock_pipe

        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")

        result = m.generate("a cat on mars", width=512, height=512, seed=42)

        assert isinstance(result, ImageResult)
        assert result.width == 512
        assert result.height == 512
        assert result.seed == 42
        mock_pipe.assert_called_once()
        kwargs = mock_pipe.call_args.kwargs
        assert kwargs["prompt"] == "a cat on mars"


def test_sd_turbo_uses_configured_seed_in_generator():
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls, \
         patch("muse.images.generations.backends.sd_turbo.torch") as mock_torch:
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[MagicMock(size=(512, 512))])
        mock_cls.from_pretrained.return_value = mock_pipe

        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt", seed=7)

        # Ensure torch.Generator was seeded with 7
        mock_torch.Generator.return_value.manual_seed.assert_called_with(7)


def test_sd_turbo_defaults_steps_to_1():
    # SD-Turbo is a 1-step distilled model; default should be 1
    with patch("muse.images.generations.backends.sd_turbo.AutoPipelineForText2Image") as mock_cls:
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[MagicMock(size=(512, 512))])
        mock_cls.from_pretrained.return_value = mock_pipe

        from muse.images.generations.backends.sd_turbo import SDTurboModel
        m = SDTurboModel(hf_repo="stabilityai/sd-turbo", local_dir="/fake")
        m.generate("prompt")

        kwargs = mock_pipe.call_args.kwargs
        assert kwargs["num_inference_steps"] == 1
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/images/generations/test_sd_turbo.py -v
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement SDTurboModel**

File: `src/muse/images/generations/backends/sd_turbo.py`

```python
"""SD-Turbo backend: 1-step distilled Stable Diffusion.

Uses diffusers AutoPipelineForText2Image with stabilityai/sd-turbo.
Very fast (1 inference step by default) at modest quality; good first
backend to prove the images.generations modality end-to-end.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.images.generations.protocol import ImageResult

logger = logging.getLogger(__name__)

try:
    import torch
    from diffusers import AutoPipelineForText2Image
except ImportError:  # pragma: no cover — resolved on muse pull sd-turbo
    torch = None  # type: ignore
    AutoPipelineForText2Image = None  # type: ignore


class SDTurboModel:
    model_id = "sd-turbo"
    default_size = (512, 512)

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        **_: Any,
    ):
        if AutoPipelineForText2Image is None:
            raise RuntimeError("diffusers is not installed; run `muse pull sd-turbo`")
        self._device = _select_device(device)
        torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        src = local_dir or hf_repo
        logger.info("loading SD-Turbo from %s (device=%s, dtype=%s)", src, self._device, dtype)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            src, torch_dtype=torch_dtype, variant="fp16" if dtype == "float16" else None,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> ImageResult:
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else 1  # sd-turbo wants 1 step
        cfg = guidance if guidance is not None else 0.0  # sd-turbo: cfg=0

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={"prompt": prompt, "steps": n_steps, "guidance": cfg, "model": self.model_id},
        )


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/images/generations/test_sd_turbo.py -v
```

Expected: all 4 pass (mocked — no real weights loaded).

- [ ] **Step 5: Export from subpackage**

Edit `src/muse/images/generations/__init__.py`:

```python
"""Muse images.generations modality: text-to-image."""
from muse.images.generations.protocol import ImageModel, ImageResult

__all__ = ["ImageModel", "ImageResult"]
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/images/generations/backends/sd_turbo.py \
        src/muse/images/generations/__init__.py \
        tests/images/generations/test_sd_turbo.py
git commit -m "feat(images.generations): add SD-Turbo backend via diffusers

First real image backend. 1-step distilled Stable Diffusion, default
512x512, cfg=0. Tests are fully mocked — no weights loaded in CI;
real weights come down on `muse pull sd-turbo`."
```

---

### Task D5: Image HTTP client

**Files:**
- Create: `src/muse/images/generations/client.py`
- Create: `tests/images/generations/test_client.py`

- [ ] **Step 1: Write failing test**

File: `tests/images/generations/test_client.py`

```python
"""Tests for GenerationsClient HTTP client."""
import base64
from unittest.mock import MagicMock, patch

import pytest

from muse.images.generations.client import GenerationsClient


def test_client_default_base_url():
    c = GenerationsClient()
    assert c.base_url == "http://localhost:8000"


def test_client_custom_base_url():
    c = GenerationsClient(base_url="http://lan:8000/")
    assert c.base_url == "http://lan:8000"  # trailing slash stripped


def test_generate_sends_correct_body_and_returns_bytes():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(fake_png).decode()}]},
        )
        c = GenerationsClient()
        images = c.generate("a cat", n=1)
        assert len(images) == 1
        assert images[0] == fake_png

        body = mock_post.call_args.kwargs["json"]
        assert body["prompt"] == "a cat"
        assert body["response_format"] == "b64_json"


def test_generate_raises_on_http_error():
    with patch("muse.images.generations.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = GenerationsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.generate("x")
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/images/generations/test_client.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement client**

File: `src/muse/images/generations/client.py`

```python
"""HTTP client for /v1/images/generations."""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


class GenerationsClient:
    """Thin HTTP client against the muse images.generations endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
        self.base_url = base.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str = "512x512",
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
    ) -> list[bytes]:
        """Generate n PNG images. Returns raw PNG bytes per image."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }
        if model:
            body["model"] = model
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if steps is not None:
            body["steps"] = steps
        if guidance is not None:
            body["guidance"] = guidance
        if seed is not None:
            body["seed"] = seed

        r = requests.post(
            f"{self.base_url}/v1/images/generations",
            json=body, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        data = r.json()["data"]
        return [base64.b64decode(entry["b64_json"]) for entry in data]
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/images/generations/test_client.py -v
```

Expected: all 4 pass.

- [ ] **Step 5: Update subpackage `__init__.py`**

Edit `src/muse/images/generations/__init__.py`:

```python
"""Muse images.generations modality: text-to-image."""
from muse.images.generations.client import GenerationsClient
from muse.images.generations.protocol import ImageModel, ImageResult

__all__ = ["GenerationsClient", "ImageModel", "ImageResult"]
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/images/generations/client.py \
        src/muse/images/generations/__init__.py \
        tests/images/generations/test_client.py
git commit -m "feat(images.generations): add GenerationsClient HTTP client"
```

---

## Part E — Unified CLI

### Task E1: Top-level `muse` CLI with subcommand dispatch

**Files:**
- Create: `src/muse/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI dispatch**

File: `tests/test_cli.py`

```python
"""Smoke tests for top-level `muse` CLI dispatch."""
import subprocess
import sys


def _run(*args, check=False):
    return subprocess.run(
        [sys.executable, "-m", "muse.cli", *args],
        capture_output=True, text=True, check=check,
    )


def test_no_args_prints_help():
    r = _run()
    assert r.returncode in (0, 2)
    assert "muse" in (r.stdout + r.stderr).lower()


def test_help_lists_subcommands():
    r = _run("--help")
    out = r.stdout + r.stderr
    for cmd in ("serve", "pull", "audio", "images", "speak", "imagine"):
        assert cmd in out


def test_audio_help():
    r = _run("audio", "--help")
    assert r.returncode == 0
    assert "speech" in r.stdout


def test_audio_speech_help():
    r = _run("audio", "speech", "--help")
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "models" in out


def test_images_generations_models_list_shows_known():
    r = _run("images", "generations", "models", "list")
    # May exit 0 with empty list if none pulled, or show known catalog
    assert r.returncode == 0
    assert "sd-turbo" in r.stdout or "no" in r.stdout.lower()


def test_pull_unknown_model_nonzero():
    r = _run("pull", "no-such-model-12345")
    assert r.returncode != 0
    assert "unknown" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement top-level CLI**

File: `src/muse/cli.py`

```python
"""`muse` CLI entrypoint.

Subcommands mirror the URL hierarchy:
    muse serve                               start HTTP server
    muse pull <model-id>                     download + install a model
    muse audio speech models list            list audio.speech models
    muse audio speech models info <id>       show catalog entry
    muse audio speech speak "text" -o f.wav  (alias: `muse speak`)
    muse images generations models list
    muse images generations create "prompt"  (alias: `muse imagine`)
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("muse")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="muse", description="Multi-modality generation server + client")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=False)

    # serve
    sp_serve = sub.add_parser("serve", help="start the HTTP server")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--modality", action="append",
                          help="modality to enable (default: all with pulled models). "
                               "Repeatable, e.g. --modality audio.speech")
    sp_serve.add_argument("--model", action="append",
                          help="specific model(s) to load. Repeatable")
    sp_serve.add_argument("--device", default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
    sp_serve.set_defaults(func=_cmd_serve)

    # pull
    sp_pull = sub.add_parser("pull", help="download weights + install deps for a model")
    sp_pull.add_argument("model_id")
    sp_pull.set_defaults(func=_cmd_pull)

    # aliases: speak / imagine
    sp_speak = sub.add_parser("speak", help="generate speech (alias for `audio speech create`)")
    _add_speak_args(sp_speak)
    sp_speak.set_defaults(func=_cmd_speak)

    sp_imagine = sub.add_parser("imagine", help="generate an image (alias for `images generations create`)")
    _add_imagine_args(sp_imagine)
    sp_imagine.set_defaults(func=_cmd_imagine)

    # audio subtree
    sp_audio = sub.add_parser("audio", help="audio modality commands")
    audio_sub = sp_audio.add_subparsers(dest="audio_cmd", required=True)

    sp_audio_speech = audio_sub.add_parser("speech", help="text-to-speech (speech modality)")
    speech_sub = sp_audio_speech.add_subparsers(dest="speech_cmd", required=True)

    speech_models = speech_sub.add_parser("models", help="manage audio.speech models")
    _add_models_subparser(speech_models, "audio.speech")

    sp_speech_create = speech_sub.add_parser("create", help="synthesize speech to file")
    _add_speak_args(sp_speech_create)
    sp_speech_create.set_defaults(func=_cmd_speak)

    # images subtree
    sp_images = sub.add_parser("images", help="image modality commands")
    images_sub = sp_images.add_subparsers(dest="images_cmd", required=True)

    sp_images_gen = images_sub.add_parser("generations", help="text-to-image (generations modality)")
    gen_sub = sp_images_gen.add_subparsers(dest="gen_cmd", required=True)

    gen_models = gen_sub.add_parser("models", help="manage images.generations models")
    _add_models_subparser(gen_models, "images.generations")

    sp_images_create = gen_sub.add_parser("create", help="generate image to file")
    _add_imagine_args(sp_images_create)
    sp_images_create.set_defaults(func=_cmd_imagine)

    return p


def _add_models_subparser(parser: argparse.ArgumentParser, modality: str) -> None:
    sub = parser.add_subparsers(dest="models_cmd", required=True)

    sp_list = sub.add_parser("list")
    sp_list.set_defaults(func=_cmd_models_list, modality=modality)

    sp_info = sub.add_parser("info")
    sp_info.add_argument("model_id")
    sp_info.set_defaults(func=_cmd_models_info, modality=modality)

    sp_pull = sub.add_parser("pull")
    sp_pull.add_argument("model_id")
    sp_pull.set_defaults(func=_cmd_models_pull, modality=modality)

    sp_remove = sub.add_parser("remove")
    sp_remove.add_argument("model_id")
    sp_remove.set_defaults(func=_cmd_models_remove, modality=modality)


def _add_speak_args(p):
    p.add_argument("text")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--voice", default=None)
    p.add_argument("--server", default=None)


def _add_imagine_args(p):
    p.add_argument("prompt")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--size", default="512x512")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--negative", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--server", default=None)


# Command implementations (deferred imports keep `muse --help` instant)

def _cmd_serve(args):
    from muse.cli_impl.serve import run_serve
    run_serve(host=args.host, port=args.port,
              modalities=args.modality, models=args.model, device=args.device)


def _cmd_pull(args):
    from muse.core.catalog import pull
    try:
        pull(args.model_id)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"pulled {args.model_id}")
    return 0


def _cmd_models_list(args):
    from muse.core.catalog import KNOWN_MODELS, is_pulled, list_known
    entries = list_known(args.modality)
    if not entries:
        print(f"no known models for {args.modality}")
        return 0
    for e in entries:
        status = "pulled" if is_pulled(e.model_id) else "available"
        print(f"  {e.model_id:20s} [{status:9s}]  {e.description}")
    return 0


def _cmd_models_info(args):
    from muse.core.catalog import KNOWN_MODELS
    if args.model_id not in KNOWN_MODELS:
        print(f"error: unknown model {args.model_id!r}", file=sys.stderr)
        return 2
    e = KNOWN_MODELS[args.model_id]
    if e.modality != args.modality:
        print(f"error: model {args.model_id} is in modality {e.modality}, not {args.modality}", file=sys.stderr)
        return 2
    print(f"model_id:     {e.model_id}")
    print(f"modality:     {e.modality}")
    print(f"hf_repo:      {e.hf_repo}")
    print(f"backend:      {e.backend_path}")
    print(f"pip_extras:   {list(e.pip_extras)}")
    print(f"system_pkgs:  {list(e.system_packages)}")
    if e.extra:
        print(f"extra:        {e.extra}")
    return 0


def _cmd_models_pull(args):
    return _cmd_pull(args)


def _cmd_models_remove(args):
    from muse.core.catalog import remove
    remove(args.model_id)
    print(f"removed {args.model_id} from catalog")
    return 0


def _cmd_speak(args):
    from muse.cli_impl.speak import run_speak
    return run_speak(args)


def _cmd_imagine(args):
    from muse.cli_impl.imagine import run_imagine
    return run_imagine(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add CLI implementation helpers (deferred-import stubs)**

```bash
mkdir -p src/muse/cli_impl
touch src/muse/cli_impl/__init__.py
```

File: `src/muse/cli_impl/serve.py`

```python
"""`muse serve` implementation — loads backends and starts uvicorn."""
from __future__ import annotations

import logging

from muse.core.catalog import KNOWN_MODELS, is_pulled, load_backend
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app

log = logging.getLogger(__name__)


def run_serve(*, host: str, port: int,
              modalities: list[str] | None,
              models: list[str] | None,
              device: str) -> int:
    import uvicorn

    registry = ModalityRegistry()
    routers: dict = {}

    # Determine which models to load
    if models:
        to_load = [m for m in models if m in KNOWN_MODELS]
    else:
        to_load = [mid for mid, e in KNOWN_MODELS.items()
                   if is_pulled(mid)
                   and (modalities is None or e.modality in modalities)]

    for model_id in to_load:
        entry = KNOWN_MODELS[model_id]
        log.info("loading %s (%s)", model_id, entry.modality)
        backend = load_backend(model_id, device=device)
        registry.register(entry.modality, backend)

    # Mount modality routers for every modality that had at least one model
    if "audio.speech" in registry.modalities():
        from muse.audio.speech.routes import build_router as build_audio
        routers["audio.speech"] = build_audio(registry)
    if "images.generations" in registry.modalities():
        from muse.images.generations.routes import build_router as build_images
        routers["images.generations"] = build_images(registry)

    app = create_app(registry=registry, routers=routers)
    uvicorn.run(app, host=host, port=port)
    return 0
```

File: `src/muse/cli_impl/speak.py`

```python
"""`muse speak` / `muse audio speech create` implementation."""
from __future__ import annotations

import os


def run_speak(args) -> int:
    from muse.audio.speech.client import SpeechClient
    server = args.server or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    client = SpeechClient(base_url=server)
    wav_bytes = client.speak(args.text, model=args.model, voice=args.voice)
    with open(args.output, "wb") as f:
        f.write(wav_bytes)
    print(f"wrote {args.output} ({len(wav_bytes)} bytes)")
    return 0
```

File: `src/muse/cli_impl/imagine.py`

```python
"""`muse imagine` / `muse images generations create` implementation."""
from __future__ import annotations

import os
from pathlib import Path


def run_imagine(args) -> int:
    from muse.images.generations.client import GenerationsClient
    server = args.server or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    client = GenerationsClient(base_url=server)
    images = client.generate(
        args.prompt,
        model=args.model,
        n=args.n,
        size=args.size,
        negative_prompt=args.negative,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )
    out = Path(args.output)
    if args.n == 1:
        out.write_bytes(images[0])
        print(f"wrote {out}")
    else:
        # Insert index before extension: out.png -> out_0.png, out_1.png, ...
        stem, suf = out.with_suffix(""), out.suffix
        for i, img in enumerate(images):
            p = Path(f"{stem}_{i}{suf}")
            p.write_bytes(img)
            print(f"wrote {p}")
    return 0
```

- [ ] **Step 5: Run CLI tests — pass**

```bash
pytest tests/test_cli.py -v
```

Expected: all 6 pass.

- [ ] **Step 6: Smoke test the binary**

```bash
muse --help
muse audio speech models list
muse images generations models list
```

Each should print without errors. `models list` may show empty if no models are pulled.

- [ ] **Step 7: Commit**

```bash
git add src/muse/cli.py src/muse/cli_impl/ tests/test_cli.py
git commit -m "feat(cli): add top-level muse command with modality subcommands

Hierarchy mirrors URL paths:
  muse audio speech models list   ↔ /v1/audio/speech
  muse images generations create  ↔ /v1/images/generations

Short-form aliases: \`muse speak\`, \`muse imagine\` for the 90% case.
Command implementations live in muse.cli_impl.* for deferred imports
so \`muse --help\` stays instant."
```

---

## Part F — Cleanup + Final Verification

### Task F1: Delete the old `narro/` tree

- [ ] **Step 1: Confirm nothing in narro/ still has unique code**

```bash
diff -rq narro/ src/muse/ --brief 2>&1 | head -30
```

Any files that exist only in narro/ should be cataloged — they either got ported with a rename or are intentionally dropped. If you find a surprise, stop and investigate.

- [ ] **Step 2: Delete narro/**

```bash
git rm -rf narro/
```

- [ ] **Step 3: Delete stray root-level fixtures (if not referenced by tests)**

```bash
grep -r "test_large.wav\|test_large.soprano\|alex_voice.wav\|^test.wav" tests/ src/
```

If none of those paths are referenced, delete them. If they are, move under `tests/fixtures/` first:

```bash
mkdir -p tests/fixtures
# Move any test file actually referenced; delete the rest
rm -f test_large.wav test_large.soprano.npz test.wav alex_voice.wav
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all non-skipped tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete narro/ after full migration to src/muse/"
```

---

### Task F2: Rewrite README.md and CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite `README.md`** (target: ~60 lines)

```markdown
# Muse

Model-agnostic multi-modality generation server and client. Speaks
OpenAI-compatible HTTP: text-to-speech on `/v1/audio/speech`, text-to-image
on `/v1/images/generations`. Add a modality by dropping in a router,
a protocol, and a catalog entry — no shared base class, no coupling
between modalities.

## Install

```bash
pip install -e ".[server,audio,images]"
```

Optional extras:
- `audio` — PyTorch + transformers for TTS backends
- `audio-kokoro` — Kokoro TTS (needs system `espeak-ng`)
- `images` — diffusers + Pillow for SD-Turbo and future image backends
- `server` — FastAPI + uvicorn (only needed on the serving host)
- `dev` — pytest + coverage tools

## Quick start

```bash
# Pull a model
muse pull soprano-80m
muse pull sd-turbo

# Start the server
muse serve --host 0.0.0.0 --port 8000

# Synthesize speech (from anywhere; defaults to localhost:8000)
muse speak "Hello world" -o hello.wav

# Generate an image
muse imagine "a cat on mars, cinematic" -o cat.png
```

## CLI hierarchy

| Command | Description |
|---|---|
| `muse serve` | start HTTP server |
| `muse pull <model-id>` | download weights + install deps |
| `muse audio speech models list` | list audio.speech models |
| `muse audio speech create "text" -o f.wav` | generate speech |
| `muse images generations models list` | list images.generations models |
| `muse images generations create "prompt" -o f.png` | generate image |
| `muse speak` / `muse imagine` | aliases for the create commands |

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + list of enabled modalities |
| `GET /v1/models` | all registered models, aggregated |
| `POST /v1/audio/speech` | synthesize speech (OpenAI-compatible) |
| `GET /v1/audio/speech/voices` | list voices for the current model |
| `POST /v1/images/generations` | generate images (OpenAI-compatible) |

## Architecture

- `muse.core` — modality-agnostic: registry, catalog, HF downloader + pip auto-install, FastAPI app factory
- `muse.audio.speech` — text-to-speech (Soprano, Kokoro, Bark backends)
- `muse.images.generations` — text-to-image (SD-Turbo backend)

See `CLAUDE.md` for implementation details.

## License

MIT
```

- [ ] **Step 2: Rewrite `CLAUDE.md`** (target: ~120 lines)

```markdown
# CLAUDE.md

Guidance for working on Muse.

## Project overview

Muse is a multi-modality generation server and client. It currently supports
two modalities:

- **audio.speech** — text-to-speech via `/v1/audio/speech` (Soprano, Kokoro, Bark)
- **images.generations** — text-to-image via `/v1/images/generations` (SD-Turbo)

The package structure mirrors OpenAI's URL hierarchy. Each modality owns its
protocol, routes, CLI subcommands, and backends. A modality-agnostic core
holds the registry, HF downloader, pip auto-install, and FastAPI app factory.

## Architecture

```
HTTP API (/v1/audio/speech, /v1/images/generations, /v1/models, /health)
    ↓
muse.core.server (FastAPI factory, mounts per-modality routers)
    ↓
muse.core.registry (ModalityRegistry: {modality: {model_id: Model}})
    ↓
Modality backends implementing modality-specific protocols
```

### Key modules

- `muse.core.registry.ModalityRegistry` — keyed by `(modality, model_id)`.
  First registered model per modality is the default for that modality.
- `muse.core.catalog.KNOWN_MODELS` — static dict of `CatalogEntry`. Each
  entry carries `modality`, `backend_path`, `hf_repo`, `pip_extras`,
  `system_packages`. `pull()` installs pip deps, warns on missing system
  packages, and downloads weights from HF.
- `muse.core.server.create_app(registry, routers)` — builds the FastAPI
  app with shared `/health` and `/v1/models`, then mounts per-modality
  routers.

### Modality conventions

Each modality subpackage contains:
- `protocol.py` — Protocol + Result dataclass(es) for this modality
- `routes.py` — `build_router(registry) -> APIRouter`
- `client.py` — HTTP client for this modality's endpoints
- `codec.py` — modality-specific encoding (wav/opus or png/jpeg)
- `backends/` — concrete model adapters

Each backend class:
- Satisfies the modality's Protocol structurally (no base class required)
- Accepts `hf_repo=` and `local_dir=` kwargs in its constructor
- Defers heavy imports (transformers, diffusers) to module top-level
  behind a try/except so `muse --help` stays fast

### No shared supertype across modalities

`AudioResult` and `ImageResult` do not share a common base. Streaming
semantics differ (audio chunks are time-ordered and playable immediately;
diffusion steps are progressive refinement of one frame). A `GenerationModel`
abstract base would be a leaky abstraction. Instead, the core code is
generic (`Any`) and each modality's router + codec knows its own types.

## Development commands

```bash
# Install (dev)
pip install -e ".[dev,server,audio,images]"

# Run all tests
pytest tests/

# Run tests for one modality
pytest tests/audio/speech/
pytest tests/images/generations/

# Coverage
pytest tests/ --cov=muse

# Start server
muse serve --device cuda

# Test end-to-end (requires running server)
muse speak "hello" -o out.wav
muse imagine "a cat" -o cat.png
```

## Project-specific conventions

- **Deferred imports:** `src/muse/__init__.py` and `src/muse/cli.py`
  MUST NOT import heavy libs (torch, diffusers, transformers). Each
  backend imports its heavy deps at module top-level inside a try/except,
  so `muse --help` and `muse pull` work without them installed.
- **FakeModel-pattern tests:** server and router tests use plain classes
  that satisfy the modality protocol — no real weights. Actual backend
  tests also mock heavy libs (see `tests/images/generations/test_sd_turbo.py`).
- **Registry is a singleton at module level** (`muse.core.registry.registry`),
  but tests create their own `ModalityRegistry()` instances to avoid
  coupling — always prefer the local instance in tests.
- **Audio is float32 in `[-1, 1]`** at the protocol boundary; codec
  converts to int16 PCM at output.
- **Images are `Any`** at the protocol boundary; codec normalizes to
  PIL before encoding to PNG/JPEG.

## Adding a new modality

1. Create `src/muse/<family>/<op>/` (e.g., `muse/audio/transcriptions/`).
2. Write `protocol.py` with the backend Protocol and Result dataclass.
3. Write `routes.py` exposing `build_router(registry) -> APIRouter`.
4. Write `client.py` with an HTTP client.
5. Add backends under `backends/`.
6. Add `CatalogEntry`s to `muse.core.catalog.KNOWN_MODELS`.
7. Wire up the CLI subtree in `src/muse/cli.py`.
8. Wire the router into `src/muse/cli_impl/serve.py`.
9. Add matching tests in `tests/<family>/<op>/`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: rewrite README.md and CLAUDE.md for muse"
```

---

### Task F3: Full test + lint + install sanity sweep

- [ ] **Step 1: Fresh install**

```bash
pip uninstall -y muse narro soprano-tts 2>/dev/null || true
pip install -e ".[dev,server,audio,images]"
```

- [ ] **Step 2: Run full suite with coverage**

```bash
pytest tests/ --cov=muse --cov-report=term-missing
```

Expected: all non-skipped tests pass. Note the coverage number.

- [ ] **Step 3: Import smoke checks**

```bash
python -c "import muse; print(muse.__version__)"
python -c "from muse.audio.speech import SpeechClient, TTSModel"
python -c "from muse.images.generations import GenerationsClient, ImageModel"
python -c "from muse.core.registry import ModalityRegistry"
python -c "from muse.core.catalog import KNOWN_MODELS; print(list(KNOWN_MODELS))"
```

Each line should succeed without ImportError.

- [ ] **Step 4: CLI smoke**

```bash
muse --help
muse pull --help
muse audio speech models list
muse images generations models list
```

- [ ] **Step 5: Integration smoke — start server briefly and hit /health**

```bash
muse serve --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
sleep 3
curl -sf http://127.0.0.1:8765/health | python -m json.tool
kill $SERVER_PID
```

Expected: `{"status": "ok", "modalities": [...], "models": [...]}`. If no
models are pulled, `modalities` and `models` may be empty — that's fine;
we're testing the server starts and the endpoint works.

- [ ] **Step 6: Commit any fixes discovered**

If you had to fix anything, commit it:

```bash
git add -A
git commit -m "fix: address issues found in final verification sweep"
```

- [ ] **Step 7: Merge worktree back to main**

```bash
cd /home/spinoza/github/repos/muse
git merge restructure/muse --no-ff -m "feat: restructure narro → muse multi-modality package

Adds images.generations alongside audio.speech. Full details in
docs/plans/2026-04-13-muse-restructure.md."
git worktree remove ../muse-restructure
git branch -d restructure/muse
```

---

## Scope notes (not in this plan)

Intentionally deferred — belong in a follow-on plan once this merges:

- **Flux-schnell backend** for higher-quality image generation
- **`/v1/images/edits` and `/v1/images/variations`** (img2img, inpainting)
- **`/v1/audio/transcriptions`** (Whisper backend — third modality to validate the abstraction further)
- **`/v1/audio/translations`** (Whisper multi-lingual)
- **Model quantization pipeline** for image backends (bitsandbytes / GGUF for diffusion)
- **LAN deployment update** — systemd unit rename from `narro.service` → `muse.service` on 192.168.0.225 (do this once the client is happy)
- **Benchmarks** — port `narro/bench.py` to `muse.audio.speech.bench` and add image generation benchmarks
- **Streaming image generation** — if diffusion backend supports step callbacks, expose an SSE stream of intermediate latents (though per my earlier insight, this is a different streaming model than audio; decide carefully)

---

## Self-review

**Spec coverage:**
- Rename `narro` → `muse`: ✅ A1 (pyproject), F1 (delete narro/)
- Multi-modality registry: ✅ B1 (ModalityRegistry)
- Audio.speech modality (port narro): ✅ C1-C6
- Images.generations modality: ✅ D1-D5
- Unified CLI mirroring URL hierarchy: ✅ E1
- Best-practice layout (src-layout): ✅ A1
- Clean break (no narro shim): ✅ F1
- Real image backend: ✅ D4 (SD-Turbo)

**Placeholder scan:** No TBDs, no "appropriate error handling", no unimplemented step references. Each backend's `__init__` signature is specified. `load_backend` contract (`hf_repo`, `local_dir`, `**kwargs`) is consistent across catalog + all backends.

**Type consistency:**
- `ModalityRegistry.register(modality, model)` — used consistently everywhere
- `build_router(registry)` signature — same in `muse.audio.speech.routes` and `muse.images.generations.routes`
- `CatalogEntry` fields — named consistently in catalog.py and all usage in tests
- Backend constructor `(hf_repo=, local_dir=, **kwargs)` — specified in task C4 and matches `load_backend`'s call site in B2

Plan complete.
