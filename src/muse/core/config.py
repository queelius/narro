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

import contextlib
import logging
import math
import os
import pathlib
import stat
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger("muse.config")

_MB = 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024


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
    minimum: int | float | None = None
    maximum: int | float | None = None
    minimum_exclusive: bool = False
    maximum_exclusive: bool = False
    choices: tuple[str, ...] | None = None


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _validate_value(setting: Setting, value: Any) -> Any:
    """Validate and canonicalize a value that already has its declared type."""
    optional = setting.type.startswith("opt_")
    declared_type = setting.type[len("opt_"):] if optional else setting.type
    label = f"{setting.env} / {setting.key}"

    if value is None:
        if optional:
            return None
        raise ConfigError(f"{label} cannot be null")

    if declared_type == "float" and not math.isfinite(value):
        raise ConfigError(f"{label} must be finite, got {value!r}")

    if declared_type in {"int", "float"}:
        if setting.minimum is not None:
            below = (
                value <= setting.minimum
                if setting.minimum_exclusive
                else value < setting.minimum
            )
            if below:
                op = ">" if setting.minimum_exclusive else ">="
                raise ConfigError(
                    f"{label} must be {op} {setting.minimum}, got {value!r}"
                )
        if setting.maximum is not None:
            above = (
                value >= setting.maximum
                if setting.maximum_exclusive
                else value > setting.maximum
            )
            if above:
                op = "<" if setting.maximum_exclusive else "<="
                raise ConfigError(
                    f"{label} must be {op} {setting.maximum}, got {value!r}"
                )

    if setting.choices is not None:
        if declared_type != "str":
            raise ConfigError(f"{label} has choices but is not a string setting")
        canonical = value.strip().lower()
        if canonical not in setting.choices:
            allowed = "|".join(setting.choices)
            raise ConfigError(
                f"{label} must be one of {allowed}, got {value!r}"
            )
        return canonical

    return value


def coerce(setting: Setting, raw: Any) -> Any:
    """Parse a raw string per setting.type. Raises ConfigError on failure.
    Callers choose lenient (Config.get) vs strict (config set)."""
    t = setting.type
    if t.startswith("opt_"):
        if raw is None or str(raw).strip() == "":
            return None
        t = t[len("opt_"):]
    elif raw is None:
        raise ConfigError(
            f"{setting.env} / {setting.key} must be {setting.type}, got null"
        )
    raw_text = raw if isinstance(raw, str) else str(raw)
    try:
        if t == "int":
            value = int(raw_text)
        elif t == "float":
            value = float(raw_text)
        elif t == "bool":
            low = raw_text.strip().lower()
            if low in _TRUE:
                value = True
            elif low in _FALSE:
                value = False
            else:
                raise ValueError(f"not a boolean: {raw!r}")
        elif t == "str":
            value = raw_text
        else:
            raise ConfigError(f"unknown type {setting.type!r} for {setting.key}")
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{setting.env} / {setting.key} must be {setting.type}, got {raw!r}: {e}"
        ) from e
    return _validate_value(setting, value)


SETTINGS: list[Setting] = [
    # --- server ---
    Setting("server.idle_sweep_interval_seconds", "MUSE_IDLE_SWEEP_INTERVAL_SECONDS",
            "float", 30.0, "server", "Seconds between idle-eviction sweeps.",
            minimum=0, minimum_exclusive=True),
    Setting("server.idle_timeout_seconds", "MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS",
            "opt_float", 600.0, "server",
            "Global default idle timeout (s) before an untouched model is evicted; 0/negative disables."),
    Setting("server.shutdown_grace_seconds", "MUSE_SHUTDOWN_GRACE_SECONDS",
            "opt_float", None, "server",
            "Grace period (s) for workers to exit on shutdown; None uses the built-in default.",
            minimum=0),
    Setting("server.default_max_concurrency", "MUSE_DEFAULT_MAX_CONCURRENCY",
            "int", 0, "server",
            "Default per-model concurrent-request cap for models without "
            "capabilities.max_concurrency; 0 = unlimited.", minimum=0),
    Setting("server.queue_timeout_seconds", "MUSE_QUEUE_TIMEOUT_SECONDS",
            "float", 300.0, "server",
            "Max seconds a request is held waiting for a concurrency slot "
            "and/or capacity before a 503 queue_timeout; 0 disables waiting.",
            minimum=0),
    Setting("server.max_queue_depth", "MUSE_MAX_QUEUE_DEPTH",
            "int", 256, "server",
            "Per-model bound on parked waiters; exceeded requests fail fast "
            "503 queue_full; 0 explicitly opts into unbounded queueing.", minimum=0),
    Setting("server.gpu_budget_gb", "MUSE_GPU_BUDGET_GB",
            "opt_float", None, "server",
            "Total Muse GPU working-set cap (GB); physical capacity still wins.",
            minimum=0),
    Setting("server.cpu_budget_gb", "MUSE_CPU_BUDGET_GB",
            "opt_float", None, "server", "Total Muse host-RAM working-set cap (GB).",
            minimum=0),
    Setting("server.gpu_headroom_gb", "MUSE_GPU_HEADROOM_GB",
            "float", 1.0, "server",
            "VRAM reserved from hard capacity and live admission (GB).", minimum=0),
    Setting("server.cpu_headroom_gb", "MUSE_CPU_HEADROOM_GB",
            "float", 2.0, "server",
            "Host RAM reserved from hard capacity and live admission (GB).", minimum=0),
    Setting("server.aggregation_timeout_seconds", "MUSE_AGGREGATION_TIMEOUT_SECONDS",
            "float", 5.0, "server",
            "Per-worker httpx timeout (s) for /v1/models and /health fan-out.",
            minimum=0, minimum_exclusive=True),
    Setting("server.max_request_body_mb", "MUSE_MAX_REQUEST_BODY_MB",
            "int", 64, "server",
            "Global maximum request-body size (MB) accepted by the gateway.",
            minimum=0, minimum_exclusive=True),
    Setting("server.device", "MUSE_DEVICE",
            "str", "auto", "server",
            "Default device for models (auto|cpu|cuda|mps); `muse serve --device` overrides.",
            choices=("auto", "cpu", "cuda", "mps")),
    Setting("server.video_cpu_offload", "MUSE_VIDEO_CPU_OFFLOAD",
            "opt_str", None, "server",
            "Global override for video CPU offload mode (model|sequential|off); unset uses the per-model capability.",
            choices=("model", "sequential", "off", "false", "none", "no", "0")),
    # --- telemetry ---
    Setting("telemetry.enabled", "MUSE_TELEMETRY_ENABLED",
            "bool", True, "telemetry",
            "Record telemetry events + serve the /dashboard observability UI."),
    Setting("telemetry.require_auth", "MUSE_TELEMETRY_REQUIRE_AUTH",
            "bool", True, "telemetry",
            "Require admin-token authentication for dashboard telemetry APIs."),
    Setting("telemetry.retention_days", "MUSE_TELEMETRY_RETENTION_DAYS",
            "int", 7, "telemetry", "Rolling retention window for telemetry events.",
            minimum=0),
    Setting("telemetry.log_buffer_kb", "MUSE_TELEMETRY_LOG_BUFFER_KB",
            "int", 64, "telemetry", "Per-model recent-log ring-buffer size (KB).",
            minimum=0, minimum_exclusive=True),
    Setting("telemetry.sample_interval_seconds", "MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS",
            "float", 10.0, "telemetry", "Seconds between VRAM/RAM/loaded samples.",
            minimum=0, minimum_exclusive=True),
    Setting("telemetry.trace_sample_interval_seconds", "MUSE_TELEMETRY_TRACE_SAMPLE_INTERVAL_SECONDS",
            "float", 0.25, "telemetry",
            "VRAM sampling cadence while one or more requests are active.",
            minimum=0, minimum_exclusive=True),
    Setting("telemetry.log_ticket_ttl_seconds", "MUSE_TELEMETRY_LOG_TICKET_TTL_SECONDS",
            "float", 60.0, "telemetry",
            "Seconds a dashboard SSE log-stream ticket stays valid.",
            minimum=0, minimum_exclusive=True),
    # --- federation ---
    Setting("federation.refresh_interval_seconds", "MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS",
            "float", 3.0, "federation",
            "Seconds between coordinator polls of each node's state.",
            minimum=0, minimum_exclusive=True),
    Setting("federation.forward_timeout_seconds", "MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS",
            "float", 300.0, "federation",
            "Per-request timeout when the coordinator forwards to a node.",
            minimum=0, minimum_exclusive=True),
    Setting("federation.poll_timeout_seconds", "MUSE_FEDERATION_POLL_TIMEOUT_SECONDS",
            "float", 10.0, "federation",
            "Per-node timeout for the coordinator's /v1/models + /health poll. "
            "Keep above a node's server.aggregation_timeout_seconds (5s) so a "
            "briefly-slow node is not falsely dropped from routing.",
            minimum=0, minimum_exclusive=True),
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
    # --- storage ---
    Setting("storage.auto_prune_before_pull", "MUSE_STORAGE_AUTO_PRUNE_BEFORE_PULL",
            "bool", True, "storage",
            "Before pulls under low disk headroom, delete only aged partial downloads and abandoned staging data."),
    Setting("storage.auto_prune_grace_hours", "MUSE_STORAGE_AUTO_PRUNE_GRACE_HOURS",
            "float", 24.0, "storage",
            "Minimum inactive age (hours) for automatic transient cleanup.",
            minimum=0),
    Setting("storage.auto_prune_min_free_gb", "MUSE_STORAGE_AUTO_PRUNE_MIN_FREE_GB",
            "float", 50.0, "storage",
            "Automatic cleanup threshold in free GiB; crossing either this or the percentage threshold triggers maintenance.",
            minimum=0),
    Setting("storage.auto_prune_min_free_percent", "MUSE_STORAGE_AUTO_PRUNE_MIN_FREE_PERCENT",
            "float", 5.0, "storage",
            "Automatic cleanup threshold in filesystem free percent; crossing either low-space threshold triggers maintenance.",
            minimum=0, maximum=100),
    # --- fetch ---
    Setting("fetch.allow_private", "MUSE_ALLOW_PRIVATE_FETCH",
            "bool", False, "fetch",
            "Allow image/URL fetches to non-public IPs (SSRF guard off)."),
    Setting("fetch.mcp_allowed_path_prefixes", "MUSE_MCP_ALLOWED_PATH_PREFIXES",
            "str", "", "fetch",
            "Colon-separated dir prefixes MCP *_path inputs may read from."),
    # --- limits (per-modality request caps) ---
    Setting("limits.image_input_max_bytes", "MUSE_IMAGE_INPUT_MAX_BYTES",
            "opt_int", 10 * _MB, "limits", "Max bytes per image upload / data URL.",
            minimum=0),
    Setting("limits.image_input_max_pixels", "MUSE_IMAGE_INPUT_MAX_PIXELS",
            "int", 4096 * 4096, "limits",
            "Max decoded pixels per input image (checked before raster decode).",
            minimum=0, minimum_exclusive=True),
    Setting("limits.image_input_max_total_pixels", "MUSE_IMAGE_INPUT_MAX_TOTAL_PIXELS",
            "int", 32 * 1024 * 1024, "limits",
            "Max aggregate decoded pixels retained by one image request.",
            minimum=0, minimum_exclusive=True),
    Setting("limits.audio_cls_max_bytes", "MUSE_AUDIO_CLS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-classification upload.",
            minimum=0),
    Setting("limits.audio_quality_max_bytes", "MUSE_AUDIO_QUALITY_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-quality upload.",
            minimum=0),
    Setting("limits.audio_quality_max_duration_seconds",
            "MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS",
            "opt_float", 600.0, "limits",
            "Max decoded seconds per audio-quality request.", minimum=0),
    Setting("limits.audio_alignment_max_bytes",
            "MUSE_AUDIO_ALIGNMENT_MAX_BYTES",
            "opt_int", 50 * _MB, "limits",
            "Max bytes per audio-alignment upload.", minimum=0),
    Setting("limits.audio_alignment_max_duration_seconds",
            "MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS",
            "opt_float", 300.0, "limits",
            "Max decoded seconds per audio-alignment request.", minimum=0),
    Setting("limits.audio_alignment_max_text_chars",
            "MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS",
            "opt_int", 50000, "limits",
            "Max reference-text characters per audio-alignment request.", minimum=0),
    Setting("limits.audio_embeddings_max_bytes", "MUSE_AUDIO_EMBEDDINGS_MAX_BYTES",
            "opt_int", 50 * _MB, "limits", "Max bytes per audio-embedding upload.",
            minimum=0),
    Setting("limits.asr_max_mb", "MUSE_ASR_MAX_MB",
            "int", 100, "limits", "Max MB per transcription/translation upload.",
            minimum=0),
    Setting("limits.embeddings_max_batch", "MUSE_EMBEDDINGS_MAX_BATCH",
            "int", 2048, "limits", "Max inputs per /v1/embeddings request.",
            minimum=0),
    Setting("limits.embeddings_max_chars_per_item", "MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per embedding input.", minimum=0),
    Setting("limits.image_embeddings_max_batch", "MUSE_IMAGE_EMBEDDINGS_MAX_BATCH",
            "int", 64, "limits", "Max inputs per /v1/images/embeddings request.",
            minimum=0),
    Setting("limits.segmentation_max_input_side", "MUSE_SEGMENTATION_MAX_INPUT_SIDE",
            "int", 2048, "limits", "Max px on the long side of a segmentation input.",
            minimum=0),
    Setting("limits.upscale_max_input_side", "MUSE_UPSCALE_MAX_INPUT_SIDE",
            "int", 1024, "limits", "Max px on the long side of an upscale input.",
            minimum=0),
    Setting("limits.vectorization_max_input_side",
            "MUSE_VECTORIZATION_MAX_INPUT_SIDE",
            "int", 2048, "limits",
            "Max px on the long side of a vectorization input.", minimum=0),
    Setting("limits.model_3d_input_max_bytes", "MUSE_3D_INPUT_MAX_BYTES",
            "opt_int", 20 * _MB, "limits", "Max bytes per image-to-3D upload.",
            minimum=0),
    Setting("limits.moderations_max_batch", "MUSE_MODERATIONS_MAX_BATCH",
            "int", 1024, "limits", "Max inputs per /v1/moderations request.",
            minimum=0),
    Setting("limits.moderations_max_chars_per_item", "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM",
            "int", 100000, "limits", "Max chars per moderation input.",
            minimum=0),
    Setting("limits.classifications_max_labels", "MUSE_CLASSIFICATIONS_MAX_LABELS",
            "int", 200, "limits", "Max candidate labels per zero-shot classification.",
            minimum=0),
    Setting("limits.rerank_max_documents", "MUSE_RERANK_MAX_DOCUMENTS",
            "int", 1000, "limits", "Max documents per /v1/rerank request.", minimum=0),
    Setting("limits.rerank_max_query_chars", "MUSE_RERANK_MAX_QUERY_CHARS",
            "int", 4000, "limits", "Max chars in a rerank query.", minimum=0),
    Setting("limits.rerank_max_doc_chars", "MUSE_RERANK_MAX_DOC_CHARS",
            "int", 100000, "limits", "Max chars per rerank document.", minimum=0),
    Setting("limits.summarize_max_text_chars", "MUSE_SUMMARIZE_MAX_TEXT_CHARS",
            "int", 100000, "limits", "Max chars per /v1/summarize request.",
            minimum=0),
    Setting("limits.translate_max_chars", "MUSE_TRANSLATE_MAX_CHARS",
            "int", 20000, "limits", "Max total chars across q for /v1/translate.",
            minimum=0),
    Setting("limits.video_max_frames_b64", "MUSE_VIDEO_MAX_FRAMES_B64",
            "int", 240, "limits", "Max frames returned as base64 from /v1/video/generations.",
            minimum=0),
]

for _setting in SETTINGS:
    _validate_value(_setting, _setting.default)

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
                 overrides: dict[str, Any] | None = None,
                 strict: bool = False):
        self._path = path if path is not None else config_path()
        self._overrides = dict(overrides or {})
        self._strict = strict
        self._file: dict | None = None  # lazy

    def file_values(self) -> dict:
        if self._file is None:
            self._file = self._load_file()
        return self._file

    def _load_file(self) -> dict:
        try:
            data = _read_config_for_update(self._path)
        except ConfigError as e:
            if self._strict:
                raise
            logger.warning("could not load config file %s: %s", self._path, e)
            return {}
        if not isinstance(data, dict):
            if self._strict:
                raise ConfigError(f"config file {self._path} is not a mapping")
            logger.warning("config file %s is not a mapping; ignoring", self._path)
            return {}
        cleaned: dict = {}
        for group, leaves in data.items():
            if group not in _GROUPS:
                if self._strict:
                    raise ConfigError(
                        f"unknown config section {group!r} in {self._path}"
                    )
                logger.warning("unknown config section %r in %s; ignoring", group, self._path)
                continue
            if not isinstance(leaves, dict):
                if self._strict:
                    raise ConfigError(
                        f"config section {group!r} in {self._path} is not a mapping"
                    )
                logger.warning("unknown config section %r in %s; ignoring", group, self._path)
                continue
            for leaf, val in leaves.items():
                if f"{group}.{leaf}" not in SETTINGS_BY_KEY:
                    if self._strict:
                        raise ConfigError(
                            f"unknown config key {group + '.' + str(leaf)!r} "
                            f"in {self._path}"
                        )
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
            return self._coerce_lenient(setting, override, "override")
        if key in self._overrides:
            return self._coerce_lenient(
                setting, self._overrides[key], "override"
            )
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
                if self._strict:
                    raise ConfigError(
                        f"{setting.env} / {setting.key} cannot be null"
                    )
                logger.warning(
                    "%s / %s cannot be null; using default %r",
                    setting.env, setting.key, setting.default,
                )
                return setting.default
            # file values come from yaml already typed; still route through
            # coerce via str() so a yaml "true"/"5" and a python bool/int both work
            return self._coerce_lenient(setting, str(file_raw), "file")
        return setting.default

    def _coerce_lenient(self, setting: Setting, raw: Any, origin: str) -> Any:
        try:
            return coerce(setting, raw)
        except ConfigError as e:
            if self._strict:
                raise
            logger.warning("%s; using default %r", e, setting.default)
            return setting.default

    def validate(self) -> "Config":
        """Resolve every setting, raising on any strict-mode defect."""
        self.file_values()
        for setting in SETTINGS:
            self.get(setting.key)
        return self

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


def validated_config() -> Config:
    """Return a fresh, fully validated strict view for process startup."""
    return Config(strict=True).validate()


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


_CONFIG_DIR_MODE = 0o700
_CONFIG_FILE_MODE = 0o600
_CONFIG_THREAD_LOCK = threading.RLock()


def _same_path(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Compare paths lexically without requiring either target to exist."""
    return left.expanduser().absolute() == right.expanduser().absolute()


def _prepare_config_parent(target: pathlib.Path) -> None:
    """Create and validate the parent without following its leaf."""
    parent = target.parent
    try:
        before = parent.lstat()
    except FileNotFoundError:
        parent_existed = False
        try:
            parent.mkdir(mode=_CONFIG_DIR_MODE, parents=True, exist_ok=False)
        except OSError as exc:
            raise ConfigError(f"cannot create config directory {parent}: {exc}") from exc
    else:
        parent_existed = True
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ConfigError(f"config parent is not a regular directory: {parent}")

    private = not parent_existed or _same_path(parent, _catalog_dir())
    if os.name != "posix":  # pragma: no cover - Windows fallback
        info = parent.lstat()
        is_junction = getattr(parent, "is_junction", lambda: False)
        if stat.S_ISLNK(info.st_mode) or is_junction() or not parent.is_dir():
            raise ConfigError(f"config parent is not a safe directory: {parent}")
        if private:
            parent.chmod(_CONFIG_DIR_MODE)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise ConfigError(f"config parent is not a safe directory: {parent}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ConfigError(f"config parent is not a directory: {parent}")
        if private:
            if info.st_uid != os.geteuid():
                raise ConfigError(f"config directory is not owned by this user: {parent}")
            os.fchmod(descriptor, _CONFIG_DIR_MODE)
        elif info.st_mode & 0o022:
            raise ConfigError(
                f"config parent is group/other writable: {parent}"
            )
    finally:
        os.close(descriptor)


def _open_config_lock(path: pathlib.Path):
    """Open a private regular lock file without following its leaf."""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    else:  # pragma: no cover - Windows fallback
        try:
            before = path.lstat()
        except FileNotFoundError:
            before = None
        if before is not None and stat.S_ISLNK(before.st_mode):
            raise ConfigError(f"refusing symlink config lock {path}")
        flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, _CONFIG_FILE_MODE)
    except OSError as exc:
        raise ConfigError(f"cannot safely open config lock {path}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ConfigError(f"config lock is not a regular file: {path}")
        if os.name == "posix":
            if info.st_uid != os.geteuid():
                raise ConfigError(f"config lock is not owned by this user: {path}")
            if info.st_nlink != 1:
                raise ConfigError(f"config lock has multiple links: {path}")
            os.fchmod(descriptor, _CONFIG_FILE_MODE)
    except BaseException:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "a+b", buffering=0)


def _lock_config_handle(handle) -> None:
    if os.name == "nt":  # pragma: no cover - Windows fallback
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_config_handle(handle) -> None:
    if os.name == "nt":  # pragma: no cover - Windows fallback
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _config_write_transaction(target: pathlib.Path):
    """Serialize one config transaction across threads and processes."""
    with _CONFIG_THREAD_LOCK:
        _prepare_config_parent(target)
        lock_path = target.parent / f".{target.name}.lock"
        handle = _open_config_lock(lock_path)
        try:
            _lock_config_handle(handle)
        except (ImportError, OSError) as exc:
            handle.close()
            raise ConfigError(f"cannot lock config file {target}: {exc}") from exc
        try:
            yield
        finally:
            try:
                _unlock_config_handle(handle)
            except (ImportError, OSError):
                logger.warning(
                    "could not unlock config file %s", target, exc_info=True,
                )
            handle.close()


def _fsync_config_directory(path: pathlib.Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_config_text_unlocked(text: str, *, target: pathlib.Path) -> None:
    """Durably replace ``target``; caller owns its transaction lock."""
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    tmp = pathlib.Path(raw_tmp)
    open_fd: int | None = fd
    try:
        if os.name == "posix":
            os.fchmod(fd, _CONFIG_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            open_fd = None
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        _fsync_config_directory(target.parent)
    except BaseException:
        if open_fd is not None:
            os.close(open_fd)
        tmp.unlink(missing_ok=True)
        raise


def _create_config_text_unlocked(text: str, *, target: pathlib.Path) -> bool:
    """Durably create ``target`` without replacing any existing leaf."""
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    tmp = pathlib.Path(raw_tmp)
    open_fd: int | None = fd
    published = False
    try:
        if os.name == "posix":
            os.fchmod(fd, _CONFIG_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            open_fd = None
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "posix":
                # Same-directory hard-link publication is atomic and has no
                # replacement mode: any existing file, symlink, or directory
                # makes this fail with FileExistsError.
                os.link(tmp, target)
                published = True
                try:
                    tmp.unlink()
                except BaseException:
                    # Avoid leaving the just-created config with two links;
                    # the secure reader deliberately rejects that topology.
                    target.unlink(missing_ok=True)
                    published = False
                    raise
            else:  # pragma: no cover - Windows no-replace rename semantics
                os.rename(tmp, target)
                published = True
        except FileExistsError:
            return False
        _fsync_config_directory(target.parent)
        return True
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        if open_fd is not None:
            os.close(open_fd)
        tmp.unlink(missing_ok=True)


def _read_config_for_update(
    target: pathlib.Path, *, require_owned: bool = False,
) -> Any:
    """Read one bounded regular config file without following its leaf."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if os.name == "posix":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    else:  # pragma: no cover - Windows fallback
        try:
            before = target.lstat()
        except FileNotFoundError:
            return {}
        if stat.S_ISLNK(before.st_mode):
            raise ConfigError(f"refusing symlink config file {target}")
        flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(target, flags)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot safely read config file {target}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ConfigError(f"config path is not a regular file: {target}")
        if os.name == "posix":
            if info.st_nlink != 1:
                raise ConfigError(f"config file has multiple links: {target}")
            if info.st_uid not in {os.geteuid(), 0}:
                raise ConfigError(f"config file has an untrusted owner: {target}")
            if require_owned and info.st_uid != os.geteuid():
                raise ConfigError(f"config file is not owned by this user: {target}")
            if info.st_uid == os.geteuid():
                if stat.S_IMODE(info.st_mode) != _CONFIG_FILE_MODE:
                    os.fchmod(descriptor, _CONFIG_FILE_MODE)
            elif info.st_mode & 0o022:
                raise ConfigError(f"config file is group/other writable: {target}")
        if info.st_size > _MAX_CONFIG_BYTES:
            raise ConfigError(
                f"config file exceeds {_MAX_CONFIG_BYTES} bytes: {target}"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            text = stream.read(_MAX_CONFIG_BYTES + 1)
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"cannot read config file {target}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(text.encode("utf-8")) > _MAX_CONFIG_BYTES:
        raise ConfigError(f"config file exceeds {_MAX_CONFIG_BYTES} bytes: {target}")
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config file {target}: {exc}") from exc


def write_config_text(text: str, *, path: pathlib.Path | None = None) -> None:
    """Atomically write config text without exposing secrets through umask.

    The temporary file is mode 0600 before content is written and remains
    0600 after replacement. A newly created config directory is mode 0700;
    the standard catalog directory is also hardened when it already exists.
    Existing arbitrary parents of an explicit MUSE_CONFIG path are not
    chmodded because they may be shared system/user directories.
    """
    target = path if path is not None else config_path()
    with _config_write_transaction(target):
        _write_config_text_unlocked(text, target=target)


def create_config_text(text: str, *, path: pathlib.Path | None = None) -> bool:
    """Atomically create a secure config, returning false if it exists.

    Unlike an ``exists()`` check followed by :func:`write_config_text`, the
    final publication is itself no-replace, so even a non-cooperating creator
    cannot be overwritten between the check and write.
    """
    target = path if path is not None else config_path()
    with _config_write_transaction(target):
        return _create_config_text_unlocked(text, target=target)


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
    with _config_write_transaction(target):
        data = _read_config_for_update(target, require_owned=True)
        if not isinstance(data, dict):
            raise ConfigError(f"config file {target} is not a mapping")
        group, leaf = key.split(".", 1)
        group_values = data.setdefault(group, {})
        if not isinstance(group_values, dict):
            raise ConfigError(f"config section {group!r} is not a mapping")
        group_values[leaf] = value
        _write_config_text_unlocked(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=True),
            target=target,
        )
    if path is None or _same_path(path, config_path()):
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
    changed = False
    with _config_write_transaction(target):
        data = _read_config_for_update(target, require_owned=True)
        if not isinstance(data, dict):
            return False
        group, leaf = key.split(".", 1)
        if (
            group not in data
            or not isinstance(data[group], dict)
            or leaf not in data[group]
        ):
            return False
        del data[group][leaf]
        if not data[group]:                  # prune a now-empty group
            del data[group]
        _write_config_text_unlocked(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=True),
            target=target,
        )
        changed = True
    if changed and (path is None or _same_path(path, config_path())):
        reset_config()
    return changed
