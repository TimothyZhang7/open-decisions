from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient

from open_decisions.engine import Engine, LogitResult
from open_decisions.server import create_app
from open_decisions import Capabilities


class StubClient:
    name = "fake"
    capabilities = Capabilities()
    info = {"model": "test", "backend": "fake", "images": False, "label_capacity": 4}
    model = "test"
    model_id = "test"
    labels = list("ABCD")

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.wait = False

    def score(self, *args):
        return LogitResult([0, 1], .9, 30)

    def evaluate_request(self, request):
        self.entered.set()
        if self.wait:
            assert self.release.wait(5)
        return Engine(self).evaluate(request)

    def close(self):
        pass


REQUEST = {"state": "hello", "questions": {"yes": {"type": "noul", "instructions": "Is this a greeting?"}}}


def test_api_playground_and_validation():
    with TestClient(create_app(client_factory=StubClient)) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/api/info").json()["backend"] == "fake"
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/openapi.json").status_code == 200
        r = client.post("/v1/evaluate", json=REQUEST)
        assert r.status_code == 200
        assert r.json()["usage"]["output_tokens"] == 0
        assert client.post("/v1/evaluate", json={**REQUEST, "unknown": True}).status_code == 422
        assert client.post("/v1/evaluate", json=REQUEST, headers={"Origin": "https://example.com"}).status_code == 403


def test_busy_does_not_block_health_or_queue_another_inference():
    stub = StubClient()
    stub.wait = True
    with TestClient(create_app(client_factory=lambda: stub)) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, "/v1/evaluate", json=REQUEST)
        assert stub.entered.wait(2)
        assert client.get("/health").json()["busy"] is True
        assert client.get("/docs").status_code == 200
        busy = client.post("/v1/evaluate", json=REQUEST)
        assert busy.status_code == 429
        stub.release.set()
        assert pending.result(timeout=3).status_code == 200
        assert client.get("/health").json()["busy"] is False


def test_oversized_body_rejected():
    with TestClient(create_app(client_factory=StubClient)) as client:
        response = client.post("/v1/evaluate", content=b"x" * (24 * 1024 * 1024 + 1))
        assert response.status_code == 413
