"""Public request and result types. Importing these never loads a model."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Content = str | dict[str, JsonValue] | list[JsonValue]
Key = Annotated[str, Field(min_length=1, max_length=128)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class QuestionBase(StrictModel):
    instructions: Content

    @field_validator("instructions")
    @classmethod
    def meaningful_instructions(cls, value):
        if not value or isinstance(value, str) and not value.strip():
            raise ValueError("Instructions must be nonempty")
        return value


class Choice(QuestionBase):
    type: Literal["choice"] = "choice"
    criteria: dict[Key, Content | None] = Field(min_length=1, max_length=255)


class Noul(QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], Content] | None = None

    @field_validator("criteria")
    @classmethod
    def both_outcomes(cls, value):
        if value is not None and set(value) != {"true", "false"}:
            raise ValueError("Noul criteria must describe both true and false")
        return value


class Score(QuestionBase):
    type: Literal["score"] = "score"
    criteria: list[Content] = Field(min_length=2, max_length=10)


class Gates(QuestionBase):
    """Joint boolean judgments: one label per combination, then marginalize."""
    type: Literal["gates"] = "gates"
    criteria: dict[Key, Content] = Field(min_length=1, max_length=4)


Question = Annotated[Choice | Noul | Score | Gates, Field(discriminator="type")]


class ImageInput(StrictModel):
    # API never fetches URLs or opens caller-specified paths.
    data_url: str = Field(max_length=5_600_000)

    @field_validator("data_url")
    @classmethod
    def supported_data_url(cls, value):
        prefix = value.partition(",")[0]
        if prefix not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
            raise ValueError("Use a base64 PNG, JPEG, or WebP data URL")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> ImageInput:
        path = Path(path)
        mime = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError("Expected a PNG, JPEG, or WebP file")
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Image exceeds 4 MiB")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return cls(data_url=f"data:image/{mime};base64,{data}")


class EvaluationRequest(StrictModel):
    state: JsonValue
    questions: dict[Key, Question] = Field(min_length=1, max_length=16)
    images: list[ImageInput] = Field(default_factory=list, max_length=4)
    temperature: float = Field(default=1.0, ge=0.05, le=10, allow_inf_nan=False)


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: Probability


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, le=9, allow_inf_nan=False)
    normalized_score: Probability
    probabilities: dict[str, Probability]
    legend: dict[str, Content]
    confidence: Probability


class GatesAnswer(StrictModel):
    type: Literal["gates"] = "gates"
    gates: dict[str, Probability]
    joint_probabilities: dict[str, Probability]
    outcomes: dict[str, dict[str, bool]]
    confidence: Probability


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer | GatesAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    processed_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    image_tokens: int = Field(ge=0)
    output_tokens: Literal[0] = 0
    model_evaluations: int = Field(ge=1)


class QuestionTrace(StrictModel):
    labels: dict[str, str]
    logits: dict[str, float]
    label_mass: Probability
    prompt_tokens: int
    cached_tokens: int
    image_tokens: int
    latency_ms: float


class Evaluation(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float
    traces: dict[str, QuestionTrace]
    backend: str
    probability_kind: Literal["uncalibrated_label_probability"] = "uncalibrated_label_probability"
    confidence_kind: Literal["normalized_entropy_concentration"] = "normalized_entropy_concentration"
