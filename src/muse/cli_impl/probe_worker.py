"""muse _probe_worker: in-venv probe execution.

Runs in the per-model venv (where the model's heavy deps are installed).
Loads the model, captures pre/post memory, runs representative inference
(unless --no-inference), captures peak. Prints one final line of JSON
on stdout for the parent process to parse and persist.

Progress logs go to stderr so the final stdout line is unambiguous.

psutil is imported defensively because older venvs (created before
psutil was added to museq[server] in v0.18.2) won't have it. We fall
back to /proc/self/statm on Linux when needed.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone


log = logging.getLogger("muse.probe_worker")


def _exception_summary(exc: BaseException) -> str:
    """Return a non-empty, type-qualified exception description."""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_rss_bytes() -> int:
    """Process resident set size in bytes. psutil if present; /proc fallback."""
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        pass
    # Linux fallback: /proc/self/statm reports pages; convert via PAGESIZE.
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
        rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        return rss_pages * page_size
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return 0



def _gpu_free_gb():
    """Indirection over memory_probe.gpu_free_gb so tests can patch it."""
    from muse.core.memory_probe import gpu_free_gb
    return gpu_free_gb()


class PynvmlVramMeter:
    """VRAM measurement via pynvml free-memory deltas (#330, v0.57.1).

    Fallback for venvs WITHOUT torch (GGUF/llama.cpp, ONNX) whose resolved
    device is cuda: torch.cuda peak counters are unavailable there, and
    falling back to RSS mis-records the measurement under `cpu`, which
    mis-pools the model for admission (a GGUF split got sized against the
    wrong pool on 2026-07-08). pynvml ships in muse[server] in EVERY venv.

    Approximate by design: free-delta tracks reserved memory, not the true
    allocation peak, and is monotonic-max across delta_bytes() calls so a
    post-inference sample cannot shrink the recorded figure.
    """

    def __init__(self) -> None:
        self._baseline: float | None = None
        self._max_delta_gb = 0.0

    def start(self) -> bool:
        free = _gpu_free_gb()
        if free is None:
            return False
        self._baseline = free
        return True

    def delta_bytes(self) -> int:
        if self._baseline is None:
            return 0
        free = _gpu_free_gb()
        if free is None:
            return 0
        self._max_delta_gb = max(self._max_delta_gb, self._baseline - free)
        return int(max(0.0, self._max_delta_gb) * 1024**3)


def _inference_peak_bytes(reported_peak: int, weights_bytes: int,
                          nvml_meter) -> int:
    """Pick the recorded inference peak for the resolved device pool.

    In the torch-less cuda branch (nvml_meter active) _run_inference's
    reported peak is host RSS; a host-RAM number must never contaminate
    the cuda-pool measurement (v0.57.3 review finding: a split GGUF's
    CPU-side layers made RSS dwarf the real VRAM delta, inflating the
    recorded cuda peak until admission 503'd a fittable model). The VRAM
    delta REPLACES the reported peak there rather than being maxed in.
    """
    if nvml_meter is not None:
        return max(nvml_meter.delta_bytes(), weights_bytes)
    return max(reported_peak, weights_bytes)


def run_probe_worker(*, model_id: str, device: str, run_inference: bool) -> int:
    """In-venv probe entry. Prints JSON record on stdout's last line."""
    from muse.core.catalog import (  # noqa: PLC0415
        _read_catalog,
        known_models,
        load_backend,
    )

    entry = known_models()[model_id]
    # device_override is a top-level catalog field (operator set-device pin),
    # not part of the manifest; read it live so the probe resolves the same
    # device load_backend will, keeping the memory measurement on the right
    # pool.
    override = _read_catalog().get(model_id, {}).get("device_override")
    actual_device = _resolve_device(device, entry, override)

    # Pre-load baseline. CUDA peak counter must be reset BEFORE loading
    # so reset_peak_memory_stats and the load both see the same epoch.
    ram_baseline = _process_rss_bytes()
    vram_baseline = 0
    cuda = None
    if actual_device.startswith("cuda"):
        try:
            import torch  # noqa: PLC0415
            cuda = torch
            cuda.cuda.reset_peak_memory_stats()
            cuda.cuda.empty_cache()
            vram_baseline = cuda.cuda.memory_allocated()
        except ImportError:
            cuda = None
    # #330: cuda device but no torch in this venv (GGUF, ONNX): measure
    # VRAM via pynvml free-deltas so the measurement stays on the cuda
    # pool instead of degrading to RSS-as-cpu.
    nvml_meter = None
    if actual_device.startswith("cuda") and cuda is None:
        meter = PynvmlVramMeter()
        if meter.start():
            nvml_meter = meter

    print(
        f"baseline: RAM={ram_baseline/1024**3:.2f} GB, "
        f"VRAM={vram_baseline/1024**3:.2f} GB",
        file=sys.stderr,
    )

    # Load
    t0 = time.time()
    try:
        backend = load_backend(model_id, device=device)
    except Exception as e:  # noqa: BLE001
        detail = _exception_summary(e)
        print(f"load failed: {detail}", file=sys.stderr)
        record = {
            "model_id": model_id,
            "device": actual_device,
            "error": f"load failed: {detail}",
            "probed_at": _utcnow_iso(),
            "ran_inference": False,
        }
        print(json.dumps(record))
        return 2
    load_seconds = time.time() - t0

    ram_post_load = _process_rss_bytes()
    vram_post_load = 0
    if cuda is not None:
        vram_post_load = cuda.cuda.memory_allocated()

    if actual_device.startswith("cuda") and cuda is not None:
        weights_bytes = max(0, vram_post_load - vram_baseline)
    elif nvml_meter is not None:
        weights_bytes = nvml_meter.delta_bytes()
    else:
        weights_bytes = max(0, ram_post_load - ram_baseline)

    print(
        f"load: {load_seconds:.1f}s, weights={weights_bytes/1024**3:.2f} GB",
        file=sys.stderr,
    )

    record: dict = {
        "model_id": model_id,
        "modality": entry.modality,
        "device": actual_device,
        "weights_bytes": weights_bytes,
        "peak_bytes": weights_bytes,
        "load_seconds": load_seconds,
        "ran_inference": False,
        "probed_at": _utcnow_iso(),
    }

    if run_inference:
        try:
            shape, peak_bytes = _run_inference(backend, entry, actual_device, cuda)
            record["ran_inference"] = True
            record["shape"] = shape
            record["peak_bytes"] = _inference_peak_bytes(
                peak_bytes, weights_bytes, nvml_meter)
        except Exception as e:  # noqa: BLE001
            detail = _exception_summary(e)
            print(f"inference probe failed: {detail}", file=sys.stderr)
            record["inference_error"] = detail

    # Drop references so any GC happens before exit.
    del backend
    gc.collect()

    print(json.dumps(record))
    return 0


def _resolve_device(requested: str, entry, override: str | None = None) -> str:
    """Mirror catalog.load_backend's device resolution, most authoritative first.

      1. catalog device_override (operator `muse models set-device` pin)
      2. manifest capabilities.device pin (!= "auto")
      3. requested (--device flag)
      4. torch auto-detect
      5. "cpu"

    Tier 1 mirrors load_backend exactly: a truthy override (including
    "auto") clobbers BOTH the capability pin and the requested flag, so an
    operator who forces a cpu-pinned model onto cuda (or un-pins with
    "auto") gets probed on the device the model will actually load on. Must
    stay in lockstep with load_backend or the probe measures the wrong
    memory pool and records a bogus peak.
    """
    cap = entry.extra.get("device")
    if override:
        chosen = override
    elif cap and cap != "auto":
        chosen = cap
    else:
        chosen = requested
    if chosen and chosen != "auto":
        return chosen
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _run_inference(backend, entry, device: str, cuda) -> tuple[str, int]:
    """Run representative inference, return (shape_label, peak_bytes).

    Resets the CUDA peak counter immediately before the call so the
    measurement isolates the inference activations from the load step.
    """
    if device.startswith("cuda") and cuda is not None:
        cuda.cuda.reset_peak_memory_stats()

    defaults = _read_probe_defaults(entry.modality)
    shape = defaults["shape"]
    fn = defaults["call"]
    fn(backend)

    if device.startswith("cuda") and cuda is not None:
        peak_bytes = cuda.cuda.max_memory_allocated()
    else:
        # CPU: process RSS now is a poor proxy for activation peak (it
        # does not capture transient peaks that get freed). Documented
        # in CLAUDE.md memory accounting section.
        peak_bytes = _process_rss_bytes()

    return shape, peak_bytes


def _read_probe_defaults(modality: str) -> dict:
    """Look up PROBE_DEFAULTS on the modality package; fallback if missing."""
    try:
        import importlib  # noqa: PLC0415
        module_name = "muse.modalities." + modality.replace("/", "_")
        mod = importlib.import_module(module_name)
        defaults = getattr(mod, "PROBE_DEFAULTS", None)
        if defaults is not None:
            return defaults
    except Exception:  # noqa: BLE001
        pass
    return _hardcoded_defaults_for(modality)


def _hardcoded_defaults_for(modality: str) -> dict:
    """Fallback when a modality doesn't declare PROBE_DEFAULTS yet.

    Keeps the probe useful even for external modality plugins dropped
    into $MUSE_MODALITIES_DIR that haven't been updated for v0.18.2.
    """
    if modality == "audio/speech":
        return {
            "shape": "5s synthesis",
            "call": lambda m: m.synthesize("Hello, this is a probe."),
        }
    if modality == "audio/transcription":
        return {
            "shape": "5s 16k mono",
            "call": _transcribe_short_silence,
        }
    if modality == "embedding/text":
        return {
            "shape": "1 short string",
            "call": lambda m: m.embed(["probe text"]),
        }
    if modality == "image/generation":
        return {
            "shape": "default size, 1 image",
            "call": lambda m: m.generate("probe scene"),
        }
    if modality == "image/animation":
        return {
            "shape": "default frames @ default size",
            "call": lambda m: m.generate("probe motion"),
        }
    if modality == "text/classification":
        return {
            "shape": "1 short string",
            "call": lambda m: m.classify(["probe text"]),
        }
    if modality == "chat/completion":
        return {
            "shape": "8-token completion",
            "call": lambda m: m.chat(
                messages=[{"role": "user", "content": "probe"}],
                max_tokens=8,
            ),
        }
    raise NotImplementedError(f"no probe default for modality {modality!r}")


def _transcribe_short_silence(model):
    """Generate 5s of silent wav and pass to backend.transcribe."""
    import io  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import wave  # noqa: PLC0415

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 5)
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(buf.read())
        path = f.name
    try:
        return model.transcribe(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
