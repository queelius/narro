"""ACE-Step v1 3.5B: text-to-music + full songs, 48kHz stereo.

Model by the ACE-Step team (Apache 2.0). Generates music from a
genre/style prompt, optionally with structured lyrics (`[verse]`,
`[chorus]`, ...) for full sung songs; empty/None lyrics produce an
instrumental. ~3.5B params; bf16 fits comfortably on 12GB VRAM (~8GB
tight with the cpu_offload / overlapped_decode knobs).

The script aliases the shared AceStepRuntime as `Model` (the bundled-VLM
pattern): discovery binds `Model` here, but `load_backend` resolves the
backend via the runtime class's real module path, and `get_manifest`
falls back to this script's MANIFEST. The class is named exactly `Model`
per the discovery convention.
"""
from muse.modalities.audio_generation.runtimes.ace_step import (
    AceStepRuntime as Model,
)


MANIFEST = {
    "model_id": "ace-step-v1-3.5b",
    "modality": "audio/generation",
    "hf_repo": "ACE-Step/ACE-Step-v1-3.5B",
    "revision": "82cd0d7b6322bd28cd4e830fe675ddb6180ce36c",
    "description": (
        "ACE-Step v1 3.5B: text-to-music + full songs (genre/style "
        "prompt + optional structured lyrics), 48kHz stereo. Apache 2.0."
    ),
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        # ACE-Step's pipeline. The PyPI `ace-step` is a stale v0.1.0, so
        # pin the git source; it pulls ACE-Step's own deps (transformers,
        # diffusers, etc.) transitively. NOTE: the distribution name is
        # `ace-step` (hyphen) even though the import name is `acestep`;
        # `acestep @ git+...` fails pip's name-consistency check. Upstream has
        # no tags/releases; Muse's adapter was reviewed against this immutable
        # commit (including the upstream extend-mode shape-mismatch fix).
        "ace-step @ git+https://github.com/ace-step/ACE-Step.git@1bee4c9f5b43e30995f8d4d33b3919197ce1bd68",
        # ACE-Step's save_wav_file calls torchaudio.save, which on modern
        # torchaudio (>=2.8) delegates encoding to torchcodec. Without it
        # generation fails at save time with "TorchCodec is required".
        # Verified on the GPU box (Step B1). torchcodec uses ffmpeg's
        # libav (declared in system_packages below).
        "torchcodec",
        # The pipeline writes WAV to disk; the runtime reads it back.
        "soundfile",
        # numpy is pulled by torch but the runtime imports it directly
        # for waveform shaping (#110 audit).
        "numpy",
    ),
    # ffmpeg is optional; only needed for mp3/opus response_format. The
    # codec returns a clean 400 if it's missing, so the server doesn't
    # crash without it.
    "system_packages": ("ffmpeg",),
    "capabilities": {
        # 3.5B; GPU-required. Heavy GPU-only models pin "cuda" (CPU
        # inference is impractical, minutes per clip).
        "device": "cuda",
        "supports_music": True,
        # Music / song only. `/v1/audio/sfx` returns 400 for this model.
        "supports_sfx": False,
        "default_duration": 60.0,
        "min_duration": 1.0,
        # ACE-Step does long clips; the route ceiling matches at 240s.
        "max_duration": 240.0,
        "default_sample_rate": 48000,
        "default_steps": 60,
        "default_guidance": 15.0,
        # Conservative VRAM peak (bf16). Probed 2026-07-08 on an RTX 3060:
        # 7.46 GB peak (5s clip). 8.0 leaves margin for longer songs
        # (latents grow with duration). Declared memory_gb outranks BOTH
        # probe measurements and the observed-peak writeback in the sizing
        # ladder, so this value is authoritative and does NOT self-heal: if
        # long songs are ever measured above 8.0, this number must be
        # raised by hand. The previous hand-estimate of 10.0 made the model
        # unservable on 12 GB cards once the v0.52 gpu_headroom_gb wiring
        # went live (default 1.0; ops boxes commonly configure 1.5):
        # 10.0 > (12 - headroom - driver overhead).
        "memory_gb": 8.0,
        # Optional low-VRAM / speed knobs forwarded to the pipeline ctor
        # (default off). Operators flip these via a curated overlay.
        "cpu_offload": False,
        "overlapped_decode": False,
        "torch_compile": False,
    },
}
