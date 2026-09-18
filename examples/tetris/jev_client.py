"""Minimal hosted Jev transport used only by the optional Tetris baseline."""
import http.client
import json
import math
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parent


class TypeSafeError(RuntimeError):
    """A safe, user-facing provider failure, never containing headers or bodies."""

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def normalized_probabilities(values):
    """Accept the API's observed percentage rounding, preserving its ordering.

    Live Jev responses can round each option to 0.01, e.g. a total of 0.99.
    Only permit that quantization's mathematical rounding bound; never normalize
    arbitrary malformed distributions. Callers retain the raw provider answer.
    """
    if not values or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid probability")
    total = math.fsum(values)
    rounded = (all(math.isclose(p * 100, round(p * 100), rel_tol=0, abs_tol=1e-7) for p in values)
               and abs(total - 1) <= .005 * len(values) + 1e-9)
    if total <= 0 or not (math.isclose(total, 1, rel_tol=0, abs_tol=1e-6) or rounded):
        raise ValueError("Invalid probability sum")
    return [p / total for p in values], total


class TypeSafeClient:
    """Reusable, single-worker v1 transport for typed judgments."""
    def __init__(self, model=None):
        self._api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError("Set TYPESAFE_API_KEY in the environment.")
        self.model = model or os.environ.get("TYPESAFE_MODEL") or "jev-latest"
        self._connection = None
        self.provider_verified = False

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def evaluate(self, state, questions):
        payload = {"model": self.model, "state": state, "questions": questions}
        if self._connection is None:
            self._connection = http.client.HTTPSConnection("api.typesafe.ai", timeout=20)
        try:
            self._connection.request("POST", "/v1/systemone", json.dumps(payload).encode(),
                                     {"Authorization": f"Bearer {self._api_key}",
                                      "Content-Type": "application/json"})
            response = self._connection.getresponse()
            data = response.read()
            if response.status != 200:
                messages = {401: "TypeSafe rejected the API key. Check TYPESAFE_API_KEY.",
                            403: "This TypeSafe account cannot access the requested model.",
                            422: "TypeSafe rejected the question or input.",
                            429: "TypeSafe rate limit reached. Wait before trying again.",
                            529: "TypeSafe is overloaded. Try again shortly."}
                status = 429 if response.status == 429 else 503 if response.status == 529 else 502
                raise TypeSafeError(messages.get(response.status, f"TypeSafe returned HTTP {response.status}."), status)
        except (TimeoutError, socket.timeout):
            self.close()
            raise TypeSafeError("TypeSafe did not respond within 20 seconds. Try again.", 504) from None
        except (OSError, http.client.HTTPException):
            self.close()
            raise TypeSafeError("Could not reach TypeSafe. Check network access and try again.") from None
        except TypeSafeError:
            self.close()
            raise
        try:
            return json.loads(data)
        except (ValueError, TypeError):
            raise TypeSafeError("TypeSafe returned invalid JSON; no decision was made.") from None
