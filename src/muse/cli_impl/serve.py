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

    if models:
        to_load = [m for m in models if m in KNOWN_MODELS]
        unknown = [m for m in models if m not in KNOWN_MODELS]
        if unknown:
            log.warning("ignoring unknown models: %s", unknown)
    else:
        to_load = [
            mid for mid, e in KNOWN_MODELS.items()
            if is_pulled(mid) and (modalities is None or e.modality in modalities)
        ]

    if not to_load:
        log.warning(
            "no models to load — server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    for model_id in to_load:
        entry = KNOWN_MODELS[model_id]
        log.info("loading %s (%s)", model_id, entry.modality)
        try:
            backend = load_backend(model_id, device=device)
        except Exception as e:
            log.error("failed to load %s: %s", model_id, e)
            continue
        registry.register(entry.modality, backend)

    if "audio.speech" in registry.modalities():
        from muse.audio.speech.routes import build_router as build_audio
        routers["audio.speech"] = build_audio(registry)
    if "images.generations" in registry.modalities():
        from muse.images.generations.routes import build_router as build_images
        routers["images.generations"] = build_images(registry)

    app = create_app(registry=registry, routers=routers)
    uvicorn.run(app, host=host, port=port)
    return 0
