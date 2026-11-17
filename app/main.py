"""FastAPI application factory."""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import DISCLAIMER, get_settings
from app.core.exceptions import AppError
from app.core.logging import request_id_var, setup_logging

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DESCRIPTION = f"""Hybrid retrieval (BM25 + dense + RRF + cross-encoder) over PubMed abstracts
with citation-enforced answers and abstention.

**{DISCLAIMER}**"""


def _error(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def create_app() -> FastAPI:
    """Build the API. No models or indexes are loaded here."""
    setup_logging(get_settings().log_level)
    app = FastAPI(title="Medical Evidence RAG", version="0.1.0", description=_DESCRIPTION)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return _error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}"
            for err in exc.errors()
        )
        return _error(request, 422, "validation_error", problems)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return _error(request, 500, "internal_error", "Internal server error.")

    app.include_router(router)
    return app


app = create_app()
