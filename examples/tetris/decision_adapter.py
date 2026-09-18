"""Compatibility boundary for the existing game; inference is entirely in the SDK."""
import math
from pathlib import Path
import time

from open_decisions import LocalClient

ROOT = Path(__file__).parent


class LocalChoiceError(RuntimeError):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def normalized_probabilities(values):
    if not values or any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid probabilities")
    total = math.fsum(values)
    if not math.isclose(total, 1, abs_tol=1e-6):
        raise ValueError("Invalid probability sum")
    return [p / total for p in values], total


class LocalChoiceClient:
    def __init__(self, model, backend="mlx-vlm"):
        started = time.perf_counter()
        self.client = LocalClient(model, backend=backend)
        self.model, self.info = self.client.model, self.client.info
        self.load_ms = (time.perf_counter() - started) * 1000

    def evaluate(self, state, questions):
        result = self.client.evaluate(state=state, questions=questions)
        body = result.model_dump()
        key = next(iter(questions))
        trace = result.traces[key]
        body["inference"] = {
            "package": "open-decisions", "backend": result.backend, "question": key,
            "generated_tokens": result.usage.output_tokens, "latency_ms": result.latency_ms,
            "cached_prefix_tokens": trace.cached_tokens, "label_logits": trace.logits,
            "label_to_choice": trace.labels, "label_mass_before_normalization": trace.label_mass,
            "confidence_kind": result.confidence_kind, "score_type": result.probability_kind,
        }
        return body

    def close(self):
        self.client.close()
