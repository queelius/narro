"""FastAPI application factory.

Modality routers are mounted via `create_app(registry, routers=...)`.
Each modality supplies its own APIRouter; core adds /health and /v1/models
(aggregated across modalities).
"""
from __future__ import annotations

import logging
from typing import Mapping

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from muse.core.errors import ModelNotFoundError
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

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": details or str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

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
            entry = {**info.extra, "id": info.model_id, "modality": info.modality, "object": "model"}
            data.append(entry)
        return {"object": "list", "data": data}

    for name, router in routers.items():
        logger.info("mounting modality router %s", name)
        app.include_router(router)

    app.state.registry = registry
    return app
