"""Central settings registry for muse.

This module is the single source of truth for every environment-variable
knob muse reads: what its dotted config key is, what env var backs it,
what type it coerces to, what its default is, which group it belongs to,
and a human-readable help string. Later tasks build a layered Config
object (env > config.yaml > defaults) on top of this registry; this task
only establishes the registry, the coercion function, and the two
bootstrap path helpers that the config file itself needs before it can
be loaded.

Import-light by design: stdlib + pathlib + yaml only (yaml is a core
muse dependency). No torch, no fastapi, no transformers. `muse --help`
and `muse pull` must work without any ML deps installed, and this
module is imported early enough that it must not pull in anything
heavy.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger("muse.config")

_MB = 1024 * 1024


class ConfigError(ValueError):
    """A config value could not be coerced to its declared type."""


@dataclass(frozen=True)
class Setting:
    key: str          # dotted "group.leaf"
    env: str          # "MUSE_*"
    type: str         # int|float|str|bool|opt_int|opt_float|opt_str
    default: Any
    group: str
    help: str


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def coerce(setting: Setting, raw: str) -> Any:
    """Parse a raw string per setting.type. Raises ConfigError on failure.
    Callers choose lenient (Config.get) vs strict (config set)."""
    t = setting.type
    if t.startswith("opt_"):
        if raw is None or raw.strip() == "":
            return None
        t = t[len("opt_"):]
    try:
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            low = raw.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError(f"not a boolean: {raw!r}")
        if t == "str":
            return raw
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{setting.env} / {setting.key} must be {setting.type}, got {raw!r}: {e}"
        ) from e
    raise ConfigError(f"unknown type {setting.type!r} for {setting.key}")


SETTINGS: list[Setting] = [
    # --- server ---
    Setting("server.idle_sweep_interval_seconds", "MUSE_IDLE_SWEEP_INTERVAL_SECONDS",
            "float", 30.0, "server", "Seconds between idle-eviction sweeps."),
    Setting("server.idle_timeout_seconds", "MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS",
            "opt_float", 600.0, "server",
            "Global default idle timeout (s) before an untouched model is evicted; 0/negative disables."),
    Setting("server.shutdown_grace_seconds", "MUSE_SHUTDOWN_GRACE_SECONDS",
            "opt_float", None, "server",
            "Grace period (s) for workers to exit on shutdown; None uses the built-in default."),
    Setting("server.default_max_concurrency", "MUSE_DEFAULT_MAX_CONCURRENCY",
            "int", 0, "server",
            "Default per-model concurrent-request cap for models without "
            "capabilities.max_concurrency; 0 = unlimited."),
    Setting("server.queue_timeout_seconds", "MUSE_QUEUE_TIMEOUT_SECONDS",
            "float", 300.0, "server",
            "Max seconds a request is held waiting for a concurrency slot "
            "and/or capacity before a 503 queue_timeout; 0 disables waiting."),
    Setting("server.max_queue_depth", "MUSE_MAX_QUEUE_DEPTH",
            "int", 0, "server",
            "Per-model bound on parked waiters; exceeded requests fail fast "
            "503 queue_full; 0 = unbounded."),
    Setting("server.gpu_budget_gb", "MUSE_GPU_BUDGET_GB",
            "opt_float", None, "server",
            "Declared GPU memory cap (GB); muse uses min(declared, live)."),
    Setting("server.cpu_budget_gb", "MUSE_CPU_BUDGET_GB",
            "opt_float", None, "server", "Declared host-RAM cap (GB)."),
    Setting("server.gpu_headroom_gb", "MUSE_GPU_HEADROOM_GB",
            "float", 1.0, "server",
            "GB subtracted from live free VRAM before deciding fit."),
    Setting("server.cpu_headroom_gb", "MUSE_CPU_HEADROOM_GB",
            "float", 2.0, "server",
            "GB subtracted from live free RAM before deciding fit."),
    Setting("server.aggregation_timeout_seconds", "MUSE_AGGREGATION_TIMEOUT_SECONDS",
            "float", 5.0, "server",
            "Per-worker httpx timeout (s) for /v1/models and /health fan-out."),
    Setting("server.device", "MUSE_DEVICE",
            "str", "auto", "server",
            "Default device for models (auto|cpu|cuda|mps); `muse serve --device` overrides."),
    Setting("server.video_cpu_offload", "MUSE_VIDEO_CPU_OFFLOAD",
            "opt_str", None, "server",
            "Global override for video CPU offload mode (model|sequential|off); unset uses the per-model capability."),
    # --- telemetry ---
    Setting("telemetry.enabled", "MUSE_TELEMETRY_ENABLED",
            "bool", True, "telemetry",
            "Record telemetry events + serve the /dashboard observability UI."),
    Setting("telemetry.retention_days", "MUSE_TELEMETRY_RETENTION_DAYS",
            "int", 7, "telemetry", "Rolling retention window for telemetry events."),
    Setting("telemetry.log_buffer_kb", "MUSE_TELEMETRY_LOG_BUFFER_KB",
            "int", 64, "telemetry", "Per-model recent-log ring-buffer size (KB)."),
    Setting("telemetry.sample_interval_seconds", "MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS",
            "float", 10.0, "telemetry", "Seconds between VRAM/RAM/loaded samples."),
    Setting("telemetry.log_ticket_ttl_seconds", "MUSE_TELEMETRY_LOG_TICKET_TTL_SECONDS",
            "float", 60.0, "telemetry",
            "Seconds a dashboard SSE log-stream ticket stays valid."),
    # --- federation ---
    Setting("federation.refresh_interval_seconds", "MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS",
            "float", 3.0, "federation",
            "Seconds between coordinator polls of each node's state."),
    Setting("federation.forward_timeout_seconds", "MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS",
            "float", 300.0, "federation",
            "Per-request timeout when the coordinator forwards to a node."),
    Setting("federation.poll_timeout_seconds", "MUSE_FEDERATION_POLL_TIMEOUT_SECONDS",
            "float", 10.0, "federation",
            "Per-node timeout for the coordinator's /v1/models + /health poll. "
            "Keep above a node's server.aggregation_timeout_seconds (5s) so a "
            "briefly-slow node is not falsely dropped from routing."),
    Setting("federation.config_file", "MUSE_FEDERATION_CONFIG",
            "opt_str", None, "federation",
            "Path to the coordinator node-list yaml (default <catalog_dir>/federation.yaml)."),
    # --- admin ---
    Setting("admin.token", "MUSE_ADMIN_TOKEN",
            "opt_str", None, "admin",
            "Bearer token that unlocks /v1/admin/*; unset keeps admin closed."),
    # --- client ---
    Setting("client.server_url", "MUSE_SERVER",
            "str", "http://localhost:8000", "client",
            "Base URL muse clients + CLI target."),
    # --- paths (bootstrap: catalog_dir/config_file resolve env+default only) ---
    Setting("paths.catalog_dir", "MUSE_CATALOG_DIR",
            "str", "~/.muse", "paths", "Directory for catalog.json, venvs, config.yaml."),
    Setting("paths.home", "MUSE_HOME",
            "str", "~/.muse", "paths", "Base dir for bundled voices/assets."),
    Setting("paths.models_dir", "MUSE_MODELS_DIR",
            "opt_str", None, "paths", "Extra directory scanned for model scripts."),
    Setting("paths.modalities_dir", "MUSE_MODALITIES_DIR",
            "opt_str", None, "paths", "Extra directory scanned for modality packages."),
    Setting("paths.config_file", "MUSE_CONFIG",
            "opt_str", None, "paths",
            "Explicit config.yaml path; overrides <catalog_dir>/config.yaml."),
    # --- fetch ---
    Setting("fetch.allow_private", "MUSE_ALLOW_PRIVATE_FETCH",
            "bool", False, "fetch",
            "Allow image/URL fetches to non-public IPs (SSRF guard off)."),
    Setting("fetch.mcp_allowed_path_prefixes", "MUSE_MCP_ALLOWED_PATH_PREFIXES",
            "str", "", "fetch",
            "Colon-separated dir prefixes MCP *_path inputs may read from."),
    # --- limits (per-modality request caps) ---
    Setting("limits.image_input_max_bytes", "MUSE_IMAGE_INPUT_MAX_BYTES",
            "opt_int", 10 * _MB, "limits", "Max bytes per image upload / data URL."),
    Setting("limits.audio_cls_max_bytes", "MUSE_AUDIO_CLS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-classification upload."),
    Setting("limits.audio_quality_max_bytes", "MUSE_AUDIO_QUALITY_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-quality upload."),
    Setting("limits.audio_quality_max_duration_seconds",
            "MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS",
            "opt_float", 600.0, "limits",
            "Max decoded seconds per audio-quality request."),
    Setting("limits.audio_alignment_max_bytes",
            "MUSE_AUDIO_ALIGNMENT_MAX_BYTES",
            "opt_int", 50 * _MB, "limits",
            "Max bytes per audio-alignment upload."),
    Setting("limits.audio_alignment_max_duration_seconds",
            "MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS",
            "opt_float", 300.0, "limits",
            "Max decoded seconds per audio-alignment request."),
    Setting("limits.audio_alignment_max_text_chars",
            "MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS",
            "opt_int", 50000, "limits",
            "Max reference-text characters per audio-alignment request."),
    Setting("limits.audio_embeddings_max_bytes", "MUSE_AUDIO_EMBEDDINGS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-embedding upload."),
    Setting("limits.asr_max_mb", "MUSE_ASR_MAX_MB",
            "int", 100, "limits", "Max MB per transcription/translation upload."),
    Setting("limits.embeddings_max_batch", "MUSE_EMBEDDINGS_MAX_BATCH",
            "int", 2048, "limits", "Max inputs per /v1/embeddings request."),
    Setting("limits.embeddings_max_chars_per_item", "MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per embedding input."),
    Setting("limits.image_embeddings_max_batch", "MUSE_IMAGE_EMBEDDINGS_MAX_BATCH",
            "int", 64, "limits", "Max inputs per /v1/images/embeddings request."),
    Setting("limits.segmentation_max_input_side", "MUSE_SEGMENTATION_MAX_INPUT_SIDE",
            "int", 2048, "limits", "Max px on the long side of a segmentation input."),
    Setting("limits.upscale_max_input_side", "MUSE_UPSCALE_MAX_INPUT_SIDE",
            "int", 1024, "limits", "Max px on the long side of an upscale input."),
    Setting("limits.vectorization_max_input_side",
            "MUSE_VECTORIZATION_MAX_INPUT_SIDE",
            "int", 2048, "limits",
            "Max px on the long side of a vectorization input."),
    Setting("limits.model_3d_input_max_bytes", "MUSE_3D_INPUT_MAX_BYTES",
            "opt_int", 20 * _MB, "limits", "Max bytes per image-to-3D upload."),
    Setting("limits.moderations_max_batch", "MUSE_MODERATIONS_MAX_BATCH",
            "int", 1024, "limits", "Max inputs per /v1/moderations request."),
    Setting("limits.moderations_max_chars_per_item", "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per moderation input."),
    Setting("limits.classifications_max_labels", "MUSE_CLASSIFICATIONS_MAX_LABELS",
            "int", 200, "limits", "Max candidate labels per zero-shot classification."),
    Setting("limits.rerank_max_documents", "MUSE_RERANK_MAX_DOCUMENTS",
            "int", 1000, "limits", "Max documents per /v1/rerank request."),
    Setting("limits.rerank_max_query_chars", "MUSE_RERANK_MAX_QUERY_CHARS",
            "int", 4000, "limits", "Max chars in a rerank query."),
    Setting("limits.rerank_max_doc_chars", "MUSE_RERANK_MAX_DOC_CHARS",
            "int", 100000, "limits", "Max chars per rerank document."),
    Setting("limits.summarize_max_text_chars", "MUSE_SUMMARIZE_MAX_TEXT_CHARS",
            "int", 100000, "limits", "Max chars per /v1/summarize request."),
    Setting("limits.translate_max_chars", "MUSE_TRANSLATE_MAX_CHARS",
            "int", 20000, "limits", "Max total chars across q for /v1/translate."),
    Setting("limits.video_max_frames_b64", "MUSE_VIDEO_MAX_FRAMES_B64",
            "int", 240, "limits", "Max frames returned as base64 from /v1/video/generations."),
]

SETTINGS_BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}
SETTINGS_BY_ENV: dict[str, Setting] = {s.env: s for s in SETTINGS}

# Bootstrap keys: env+default ONLY. config.yaml lives at
# <paths.catalog_dir>/config.yaml (or the explicit paths.config_file
# override), so the file that would carry a config.yaml value for
# EITHER of these two keys is, by construction, not yet known when they
# are resolved -- a value in the file can never redirect the path used
# to find that same file. Config.get / .source skip the file layer for
# these keys entirely (env or default only); `set_value` refuses to
# write them (a value that can never take effect should not be
# writable); `unset_value` stays allowed (harmless cleanup of a stale,
# already-inert value).
BOOTSTRAP_KEYS: frozenset[str] = frozenset({"paths.catalog_dir", "paths.config_file"})


def _catalog_dir() -> pathlib.Path:
    """Resolve the catalog/config directory from env+default only.
    Standalone by design: must NOT import catalog.py (import cycle)."""
    raw = os.environ.get("MUSE_CATALOG_DIR")
    base = raw if raw else "~/.muse"
    return pathlib.Path(base).expanduser()


def config_path() -> pathlib.Path:
    """Resolve the config.yaml path from env+default only (bootstrap)."""
    raw = os.environ.get("MUSE_CONFIG")
    if raw:
        return pathlib.Path(raw).expanduser()
    return _catalog_dir() / "config.yaml"


_GROUPS = {s.group for s in SETTINGS}

_MISSING = object()


class Config:
    """Layered config resolver: override > env (live) > file (cached) > default.

    The env var is re-read live on every `get` call so tests (and operators)
    that change the environment after construction see the new value. The
    yaml file is parsed once and cached on first access.
    """

    def __init__(self, *, path: pathlib.Path | None = None,
                 overrides: dict[str, Any] | None = None):
        self._path = path if path is not None else config_path()
        self._overrides = dict(overrides or {})
        self._file: dict | None = None  # lazy

    def file_values(self) -> dict:
        if self._file is None:
            self._file = self._load_file()
        return self._file

    def _load_file(self) -> dict:
        try:
            text = self._path.read_text()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return {}
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            logger.warning("could not parse config file %s: %s", self._path, e)
            return {}
        if not isinstance(data, dict):
            logger.warning("config file %s is not a mapping; ignoring", self._path)
            return {}
        cleaned: dict = {}
        for group, leaves in data.items():
            if group not in _GROUPS or not isinstance(leaves, dict):
                logger.warning("unknown config section %r in %s; ignoring", group, self._path)
                continue
            for leaf, val in leaves.items():
                if f"{group}.{leaf}" not in SETTINGS_BY_KEY:
                    logger.warning("unknown config key %r in %s; ignoring",
                                   f"{group}.{leaf}", self._path)
                    continue
                cleaned.setdefault(group, {})[leaf] = val
        return cleaned

    def _file_raw(self, setting: Setting):
        group, leaf = setting.key.split(".", 1)
        return self.file_values().get(group, {}).get(leaf, _MISSING)

    def get(self, key: str, override: Any | None = None) -> Any:
        setting = SETTINGS_BY_KEY[key]  # KeyError on unknown key (programmer error)
        if override is not None:
            return override
        if key in self._overrides:
            return self._overrides[key]
        env_raw = os.environ.get(setting.env)
        if env_raw is not None:
            return self._coerce_lenient(setting, env_raw, "env")
        if key in BOOTSTRAP_KEYS:
            # The file cannot redirect the path used to locate itself;
            # skip straight to the default (see BOOTSTRAP_KEYS docstring).
            return setting.default
        file_raw = self._file_raw(setting)
        if file_raw is not _MISSING:
            if file_raw is None:
                # yaml null: str(None) would be the literal "None", which
                # coerce would fail to parse as int/float. Handle natively.
                if setting.type.startswith("opt_"):
                    return None
                logger.warning(
                    "%s / %s cannot be null; using default %r",
                    setting.env, setting.key, setting.default,
                )
                return setting.default
            # file values come from yaml already typed; still route through
            # coerce via str() so a yaml "true"/"5" and a python bool/int both work
            return self._coerce_lenient(setting, str(file_raw), "file")
        return setting.default

    def _coerce_lenient(self, setting: Setting, raw: str, origin: str) -> Any:
        try:
            return coerce(setting, raw)
        except ConfigError as e:
            logger.warning("%s; using default %r", e, setting.default)
            return setting.default

    def source(self, key: str) -> str:
        setting = SETTINGS_BY_KEY[key]
        if key in self._overrides:
            return "override"
        if os.environ.get(setting.env) is not None:
            return "env"
        if key in BOOTSTRAP_KEYS:
            return "default"
        if self._file_raw(setting) is not _MISSING:
            return "file"
        return "default"


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def reset_config() -> None:
    global _CONFIG
    _CONFIG = None


def get(key: str, override: Any | None = None) -> Any:
    return get_config().get(key, override=override)


def source(key: str) -> str:
    return get_config().source(key)


def render_template() -> str:
    """Produce a commented config.yaml body listing every registered setting.

    Each leaf is preceded by a `# <help> (env: MUSE_X)` comment. Bootstrap
    paths (catalog_dir, config_file) are included commented-out, since the
    file itself cannot resolve the path used to find it.
    """
    lines = ["# muse configuration (~/.muse/config.yaml)",
             "# Precedence: MUSE_* env var > this file > built-in default.",
             "# Generated by `muse config generate`; edit freely.", ""]
    for group in sorted({s.group for s in SETTINGS}):
        lines.append(f"{group}:")
        for s in [x for x in SETTINGS if x.group == group]:
            leaf = s.key.split(".", 1)[1]
            default_yaml = yaml.safe_dump(s.default, default_flow_style=True).strip().splitlines()[0]
            lines.append(f"  # {s.help} (env: {s.env})")
            if s.key in BOOTSTRAP_KEYS:
                lines.append(f"  # {leaf}: {default_yaml}   # resolved from env/default; file cannot set its own path")
            else:
                lines.append(f"  {leaf}: {default_yaml}")
        lines.append("")
    return "\n".join(lines)


def set_value(key: str, raw: str, *, path: pathlib.Path | None = None) -> Any:
    """Strict-validate one dotted key, then atomically write it into a yaml file.

    Raises KeyError for an unknown key and ConfigError for an un-coercible
    raw value; in both cases nothing is written. Preserves other keys
    already present in the target file. Returns the coerced value written.

    When `path` is omitted (or explicitly equals the resolved active
    `config_path()`), the process-wide Config singleton is reset so a
    subsequent `config.get()` call in THIS process sees the new value
    instead of a stale cached parse. A write to some other explicit path
    (the common test pattern) does not touch the singleton, since that
    path isn't the one `get_config()` reads from anyway.
    """
    setting = SETTINGS_BY_KEY[key]           # KeyError on unknown
    if key in BOOTSTRAP_KEYS:
        raise ConfigError(
            f"{key} is a bootstrap path resolved from env/default only "
            f"(config.yaml cannot redirect the path used to locate "
            f"itself); it cannot be set via config file. Set the "
            f"{setting.env} environment variable instead."
        )
    value = coerce(setting, raw)             # strict: raises ConfigError
    target = path if path is not None else config_path()
    data = {}
    if target.exists():
        data = yaml.safe_load(target.read_text()) or {}
    group, leaf = key.split(".", 1)
    data.setdefault(group, {})[leaf] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=True))
    tmp.replace(target)                      # atomic write-then-rename
    if path is None or path == config_path():
        reset_config()
    return value


def unset_value(key: str, *, path: pathlib.Path | None = None) -> bool:
    """Remove one dotted key from a yaml file so it falls back to env/default.

    Raises KeyError for a key not in the registry. Returns True if the key
    was present and removed, False if it was absent (a no-op). Preserves
    other keys and prunes a group that becomes empty. This is the counterpart
    to `set_value`: there is no override value that means "use the lower
    -precedence default", so reverting a key requires removing it.

    Mirrors `set_value`'s singleton-reset guard: when the write actually
    happens against the resolved active `config_path()` (path omitted or
    explicitly equal to it), the process-wide Config singleton is reset
    so a subsequent `config.get()` in this process reflects the removal.
    A no-op (key/file absent) or a write to some other explicit path
    never touches the singleton.
    """
    SETTINGS_BY_KEY[key]                     # KeyError on unknown key
    target = path if path is not None else config_path()
    if not target.exists():
        return False
    data = yaml.safe_load(target.read_text()) or {}
    if not isinstance(data, dict):
        return False
    group, leaf = key.split(".", 1)
    if group not in data or not isinstance(data[group], dict) or leaf not in data[group]:
        return False
    del data[group][leaf]
    if not data[group]:                      # prune a now-empty group
        del data[group]
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=True))
    tmp.replace(target)                      # atomic write-then-rename
    if path is None or path == config_path():
        reset_config()
    return True
