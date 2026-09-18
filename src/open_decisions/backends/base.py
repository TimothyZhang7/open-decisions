"""The entire contract between decision semantics and model runtimes."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Capabilities:
    images: bool = False
    prefix_cache: bool = False


@dataclass
class LogitResult:
    logits: list[float]
    label_mass: float
    prompt_tokens: int
    image_tokens: int = 0
    cached_tokens: int = 0


class Backend(Protocol):
    model_id: str
    name: str
    labels: list[str]
    capabilities: Capabilities

    def score(self, system: str, state: str, images: list, count: int) -> LogitResult:
        """Return next-token logits for labels[:count], without decoding tokens."""
        ...


def create_backend(name: str, model: str, **options) -> Backend:
    # Imports stay lazy: the core and HTTP client require no model runtime.
    if name == "mlx-vlm":
        from .mlx_vlm import MLXVLMBackend
        return MLXVLMBackend(model, **options)
    if name == "mlx-lm":
        from .mlx_lm import MLXLMBackend
        return MLXLMBackend(model, **options)
    if name == "transformers":
        from .transformers import TransformersBackend
        return TransformersBackend(model, **options)
    raise ValueError(f"Unknown backend {name!r}; choose mlx-vlm, mlx-lm, or transformers, or supply backend_factory")
