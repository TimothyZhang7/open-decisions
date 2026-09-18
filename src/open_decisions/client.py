"""Local and HTTP clients with the same evaluate interface."""
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request

from .schema import Evaluation, EvaluationRequest


class LocalClient:
    """One model runtime per client. All backend work owns one dedicated thread."""
    def __init__(self, model=None, *, backend="mlx-vlm", backend_options=None, backend_factory=None):
        if backend_factory is None and model is None:
            raise ValueError("Supply a model identifier/path, or backend_factory")
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-decisions")
        self._closed = False

        def load():
            from .engine import Engine
            from .backends.base import create_backend
            runtime = backend_factory() if backend_factory else create_backend(backend, model, **(backend_options or {}))
            return Engine(runtime)

        try:
            self._engine = self._worker.submit(load).result()
        except BaseException:
            self._worker.shutdown(wait=True)
            raise
        self.model = self._engine.backend.model_id
        runtime = self._engine.backend
        self.backend = runtime.name
        self.info = {"model": self.model, "backend": self.backend, "label_capacity": len(runtime.labels),
                     "images": runtime.capabilities.images, "prefix_cache": runtime.capabilities.prefix_cache}

    def evaluate(self, *, state, questions, images=(), temperature=1.0) -> Evaluation:
        request = EvaluationRequest(state=state, questions=questions, images=list(images), temperature=temperature)
        return self.evaluate_request(request)

    def evaluate_request(self, request: EvaluationRequest) -> Evaluation:
        if self._closed:
            raise RuntimeError("Client is closed")
        # Snapshot mutable caller input before queueing it.
        snapshot = request.model_copy(deep=True)
        return self._worker.submit(self._engine.evaluate, snapshot).result()

    def close(self):
        if not self._closed:
            self._closed = True
            def release():
                self._engine = None
            self._worker.submit(release).result()
            self._worker.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Client:
    """Connect to the local HTTP service without importing MLX."""
    def __init__(self, base_url="http://127.0.0.1:8772", *, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate(self, *, state, questions, images=(), temperature=1.0) -> Evaluation:
        payload = EvaluationRequest(state=state, questions=questions, images=list(images), temperature=temperature)
        request = urllib.request.Request(
            self.base_url + "/v1/evaluate", data=payload.model_dump_json().encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return Evaluation.model_validate_json(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Decision service returned HTTP {exc.code}: {detail}") from exc
