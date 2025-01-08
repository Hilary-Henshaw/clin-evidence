"""Request tracing middleware for ClinEvidence API."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"
_PROCESSING_TIME_HEADER = "X-Processing-Time"


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Adds request IDs and processing time to every response."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and inject tracing headers."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(
            uuid.uuid4()
        )
        t0 = time.perf_counter()

        response: Response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        response.headers[_REQUEST_ID_HEADER] = request_id
        response.headers[_PROCESSING_TIME_HEADER] = str(elapsed_ms)

        logger.info(
            "Request processed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
                "request_id": request_id,
            },
        )
        return response
