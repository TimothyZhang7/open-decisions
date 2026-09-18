"""Model-independent HTTP service. Demos are separate consumers of this API."""
import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schema import Evaluation, EvaluationRequest


class BodyLimit:
    """Bound streamed bodies, including requests without Content-Length."""
    def __init__(self, app, limit=24 * 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.limit:
                response = JSONResponse({"detail": "Request exceeds 24 MiB"}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(model=None, *, backend="mlx-vlm", backend_options=None, client_factory=None):
    if client_factory is None:
        from .client import LocalClient
        client_factory = lambda: LocalClient(model, backend=backend, backend_options=backend_options)

    @asynccontextmanager
    async def lifespan(app):
        app.state.client = await asyncio.to_thread(client_factory)
        app.state.pending = None
        yield
        await asyncio.to_thread(app.state.client.close)

    app = FastAPI(title="Open Decisions", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    @app.get("/health")
    async def health():
        pending = app.state.pending
        return {"status": "ready", "busy": pending is not None and not pending.done(),
                "model": app.state.client.model}

    @app.get("/api/info")
    async def info():
        return {**app.state.client.info, "primitives": ["choice", "noul", "score", "gates"],
                "generated_tokens": 0, "license": "MIT",
                "probability_kind": "uncalibrated_label_probability",
                "confidence_kind": "normalized_entropy_concentration"}

    @app.post("/v1/evaluate", response_model=Evaluation)
    async def evaluate(payload: EvaluationRequest, request: Request):
        # CORS stays off. Browser cross-site POSTs cannot use JSON without a preflight.
        if request.headers.get("origin") and request.headers["origin"] != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Use a same-origin request")
        pending = app.state.pending
        if pending is not None and not pending.done():
            raise HTTPException(429, "The model is busy; retry after the current evaluation", headers={"Retry-After": "1"})
        task = asyncio.create_task(asyncio.to_thread(app.state.client.evaluate_request, payload))
        app.state.pending = task
        # Consume exceptions even if a disconnected caller abandons its result.
        task.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
        try:
            return await asyncio.shield(task)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logging.exception("Model evaluation failed")
            raise HTTPException(500, "Model evaluation failed; see the server log") from exc

    return app
