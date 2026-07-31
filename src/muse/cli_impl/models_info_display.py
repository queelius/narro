"""Organized `muse models info <id>` output.

The display is split into sections:
  - Header: id + status (enabled/disabled, loaded on worker port N)
  - Basics: modality, hf_repo, description, license, source
  - Storage: weights size on disk, venv path
  - Memory: annotated and measured (per device)
  - Capabilities: per-modality known-flag table; unknown keys roll up
    to a single "(other capabilities: ...)" line
  - Worker status: pid, uptime, status, restart count, last error
    (from live admin data when available; "not running" otherwise)

The capability table is data-driven. Each modality has a small dict
mapping manifest keys to (label, formatter) pairs. Unknown keys do not
crash; they aggregate into a tail line. This matches the codebase's
working-style note about deriving behavior from structure.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _yes_no(v: Any) -> str:
    return "yes" if bool(v) else "no"


def _join_list(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _str(v: Any) -> str:
    return str(v)


# Per-modality known-capability table. Each entry is:
#   manifest_key -> (display_label, formatter)
# The formatter must turn arbitrary values into a human-readable string.
KNOWN_CAPABILITIES: dict[str, dict[str, tuple[str, Callable[[Any], str]]]] = {
    "image/generation": {
        "supports_text_to_image": ("text-to-image", _yes_no),
        "supports_img2img": ("img2img", _yes_no),
        "supports_inpainting": ("inpainting", _yes_no),
        "supports_variations": ("variations", _yes_no),
        "default_size": ("default size", _str),
        "default_steps": ("default steps", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "audio/speech": {
        "voices": ("voices", _join_list),
        "sample_rate": ("sample rate", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "audio/transcription": {
        "compute_type": ("compute type", _str),
        "default_language": ("default language", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "audio/embedding": {
        "embedding_dim": ("embedding dim", _str),
        "sample_rate": ("sample rate", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "audio/generation": {
        "sample_rate": ("sample rate", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "chat/completion": {
        "context_length": ("context length", _str),
        "supports_tools": ("tools", _yes_no),
        "chat_format": ("chat format", _str),
        "gguf_file": ("gguf file", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "embedding/text": {
        "embedding_dim": ("embedding dim", _str),
        "max_seq_length": ("max seq length", _str),
        "trust_remote_code": ("trust remote code", _yes_no),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "image/embedding": {
        "embedding_dim": ("embedding dim", _str),
        "image_size": ("image size", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "image/segmentation": {
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "image/upscale": {
        "scale_factor": ("scale factor", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "image/vectorization": {
        "supports_image_to_svg": ("image-to-SVG", _yes_no),
        "output_mime_type": ("output MIME", _str),
        "static_svg_only": ("static SVG only", _yes_no),
        "max_new_tokens": ("max tokens", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "image/animation": {
        "default_size": ("default size", _str),
        "default_frames": ("default frames", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "video/generation": {
        "default_size": ("default size", _str),
        "default_frames": ("default frames", _str),
        "default_fps": ("default fps", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "text/classification": {
        "labels": ("labels", _join_list),
        "safe_labels": ("safe labels", _join_list),
        "default_threshold": ("default threshold", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "text/rerank": {
        "max_seq_length": ("max seq length", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
    "text/summarization": {
        "max_input_tokens": ("max input tokens", _str),
        "max_output_tokens": ("max output tokens", _str),
        "device": ("device pref", _str),
        "memory_gb": ("annotated memory", lambda v: f"{v} GB"),
        "idle_timeout_seconds": ("idle timeout", lambda v: f"{v}s"),
    },
}


def _format_uptime(seconds: float) -> str:
    """Render a duration as compact h/m/s string."""
    s = int(max(0, seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _render_capability_value(
    modality: str, key: str, value: Any,
) -> tuple[str, str] | None:
    """Resolve (label, formatted_value) for one capability entry.

    Returns None when the modality has no known table or the key is
    not in the table; caller groups those into a single "(other)" line.
    """
    table = KNOWN_CAPABILITIES.get(modality)
    if table is None:
        return None
    entry = table.get(key)
    if entry is None:
        return None
    label, fmt = entry
    try:
        return label, fmt(value)
    except Exception as e:  # noqa: BLE001
        logger.debug("formatter for %s.%s failed: %s", modality, key, e)
        return label, str(value)


def format_info(
    model_id: str,
    *,
    catalog_known: dict,
    catalog_data: dict,
    online_status: dict | None = None,
) -> str:
    """Return the multi-line info text for a model.

    `catalog_known` is `known_models()`; `catalog_data` is one entry's
    raw dict from `_read_catalog()` (or {} if not pulled).

    `online_status` is a dict from `AdminClient.workers()` lookups, or
    None when the supervisor isn't reachable.
    """
    if model_id not in catalog_known:
        return f"error: unknown model {model_id!r}"
    entry = catalog_known[model_id]

    is_pulled = bool(catalog_data)
    enabled = catalog_data.get("enabled", True) if catalog_data else False
    loaded_worker = (
        online_status if online_status and online_status.get("loaded") else None
    )

    lines: list[str] = []

    # Header
    status_bits: list[str] = []
    if not is_pulled:
        status_bits.append("not pulled")
    elif loaded_worker is not None:
        port = loaded_worker.get("worker_port")
        status_bits.append(
            f"enabled, loaded on worker port {port}" if port else "enabled, loaded"
        )
    elif enabled:
        status_bits.append("enabled, not loaded")
    else:
        status_bits.append("disabled")
    header_status = ", ".join(status_bits)
    lines.append(f"{model_id}".ljust(36) + f"  [{header_status}]")
    lines.append("")

    # Basics
    lines.append("Basics:")
    lines.append(f"  modality:        {entry.modality}")
    lines.append(f"  hf_repo:         {entry.hf_repo}")
    if entry.description:
        lines.append(f"  description:     {entry.description}")
    if catalog_data.get("source"):
        lines.append(f"  source:          {catalog_data['source']}")
    override = catalog_data.get("device_override") if is_pulled else None
    if override:
        lines.append(
            f"  device override: {override} "
            "(operator pin via `muse models set-device`; overrides manifest device)"
        )
    layers_pin = catalog_data.get("gpu_layers_override") if is_pulled else None
    if layers_pin is not None:
        lines.append(
            f"  gpu layers pin:  {layers_pin} "
            "(operator pin via `muse models set-gpu-layers`)"
        )

    # Storage
    if is_pulled:
        lines.append("")
        lines.append("Storage:")
        local_dir = catalog_data.get("local_dir")
        venv_path = catalog_data.get("venv_path")
        weights = catalog_data.get("local_dir_size_bytes")
        if weights:
            lines.append(f"  weights:         {_format_bytes(weights)}")
        elif local_dir:
            lines.append(f"  weights:         {local_dir}")
        if venv_path:
            lines.append(f"  venv:            {venv_path}")

    # Memory section
    measurements = (catalog_data.get("measurements") or {}) if is_pulled else {}
    annotation = (entry.extra or {}).get("memory_gb")
    if measurements or annotation is not None:
        lines.append("")
        lines.append("Memory:")
        if annotation is not None:
            try:
                ann_gb = float(annotation)
                lines.append(
                    f"  annotated peak:  {ann_gb:.1f} GB (architecture estimate)"
                )
            except (TypeError, ValueError):
                lines.append(
                    f"  annotated peak:  {annotation!r} (architecture estimate)"
                )
        for device, m in measurements.items():
            weights_gb = (m.get("weights_bytes", 0) or 0) / (1024**3)
            peak_gb = (m.get("peak_bytes", 0) or 0) / (1024**3)
            line = f"  measured ({device}): weights {weights_gb:.2f} GB"
            if m.get("ran_inference"):
                shape = m.get("shape", "?")
                line += f", peak {peak_gb:.2f} GB at {shape}"
            elif peak_gb > 0:
                line += f", peak {peak_gb:.2f} GB (load only)"
            probed_at = m.get("probed_at") or ""
            if probed_at:
                line += f" (probed {probed_at[:10]})"
            lines.append(line)

    # Capabilities
    extra = entry.extra or {}
    if extra:
        # Memory annotation already shown above; don't repeat in caps.
        skip = {"memory_gb", "allow_patterns"}
        known_lines: list[str] = []
        unknown: list[tuple[str, Any]] = []
        for k, v in extra.items():
            if k in skip:
                continue
            # idle_timeout_seconds is cross-modality (lazy-load related).
            # Null / 0 means "never idle-evict" per v0.40.1 contract; render
            # nothing in either case so a missing value doesn't surface as
            # "idle timeout: 0s" (misleading) or get dumped into the
            # "(other capabilities: ...)" tail.
            if k == "idle_timeout_seconds" and (v is None or v == 0):
                continue
            rendered = _render_capability_value(entry.modality, k, v)
            if rendered is not None:
                known_lines.append(rendered)
            else:
                unknown.append((k, v))
        if known_lines or unknown:
            lines.append("")
            lines.append("Capabilities:")
            for label, value in known_lines:
                lines.append(f"  {label + ':':17s}{value}")
            if unknown:
                tail = ", ".join(f"{k}={v!r}" for k, v in unknown)
                lines.append(f"  (other capabilities: {tail})")

    # Worker status
    lines.append("")
    lines.append("Worker status:")
    if loaded_worker is not None and loaded_worker.get("detail_source") == "public":
        # Loaded state came from the public /v1/models endpoint (no admin
        # token): we know it's resident but not pid/uptime/restarts.
        lines.append(
            "  loaded (set MUSE_ADMIN_TOKEN for worker pid / uptime / restarts)"
        )
    elif loaded_worker is not None:
        pid = loaded_worker.get("worker_pid")
        uptime = loaded_worker.get("worker_uptime_seconds")
        status_value = loaded_worker.get("worker_status") or "running"
        restart = loaded_worker.get("restart_count", 0)
        last_err = loaded_worker.get("last_error")
        lines.append(f"  pid:             {pid if pid is not None else '-'}")
        lines.append(
            f"  uptime:          {_format_uptime(uptime) if uptime is not None else '-'}"
        )
        lines.append(f"  status:          {status_value}")
        lines.append(f"  restart count:   {restart}")
        lines.append(f"  last error:      {last_err if last_err else 'none'}")
    elif online_status is None:
        lines.append("  not running (supervisor unreachable; run `muse serve`)")
    else:
        lines.append("  not loaded")

    return "\n".join(lines)


def _format_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    if b > 0:
        return f"{b / 1024:.0f} KB"
    return "-"
