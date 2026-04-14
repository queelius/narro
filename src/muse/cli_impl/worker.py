"""`muse _worker` implementation — runs ONE worker (optionally hosting
multiple models from the same venv) and starts uvicorn.

Invoked by the supervisor (`muse serve`) via subprocess:
    <venv>/bin/python -m muse.cli _worker --port 9001 --model soprano-80m

Can also be run standalone for debugging. Not advertised in top-level help.
"""
from __future__ import annotations

import logging

import uvicorn

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

    # Always mount all modality routers so empty-registry requests get
    # the OpenAI envelope rather than FastAPI's default {"detail": "Not Found"}.
    from muse.audio.speech.routes import build_router as build_audio
    from muse.embeddings.routes import build_router as build_embeddings
    from muse.images.generations.routes import build_router as build_images

    routers["audio.speech"] = build_audio(registry)
    routers["embeddings"] = build_embeddings(registry)
    routers["images.generations"] = build_images(registry)

    app = create_app(registry=registry, routers=routers)
    uvicorn.run(app, host=host, port=port, log_config=None)
    return 0
