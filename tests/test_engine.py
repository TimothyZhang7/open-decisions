import math

import pytest
from pydantic import ValidationError

from open_decisions import Choice, Noul, Score, Gates, EvaluationRequest, ImageInput
from open_decisions.engine import Engine, LogitResult, compile_question, probabilities, concentration
from open_decisions.images import decode_images
from open_decisions import Capabilities


class FakeBackend:
    name = "fake"
    capabilities = Capabilities(images=True, prefix_cache=True)
    model_id = "test"
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self, rows):
        self.rows = iter(rows)
        self.calls = []

    def score(self, system, state, images, count):
        self.calls.append((system, state, images, count))
        return LogitResult(next(self.rows), .8, 100, 0, 20)


def test_all_three_primitives_have_exact_semantics():
    backend = FakeBackend([[0, math.log(3)], [0, math.log(4)], [0, math.log(2), 0]])
    response = Engine(backend).evaluate(EvaluationRequest(state={"message": "test"}, questions={
        "team": Choice(instructions="Pick", criteria={"a": None, "b": {"description": "B"}}),
        "flag": Noul(instructions="True?"),
        "level": Score(instructions="Rate", criteria=["low", "medium", "high"]),
    }))
    assert response.answers["team"].choice == "b"
    assert response.answers["team"].probabilities == pytest.approx({"a": .25, "b": .75})
    assert response.answers["flag"].noul == pytest.approx(.8)
    assert "confidence" not in response.answers["flag"].model_dump()
    assert response.answers["level"].score == pytest.approx(1)
    assert response.answers["level"].normalized_score == pytest.approx(.5)
    assert response.usage.model_dump() == dict(input_tokens=300, processed_tokens=240, cached_tokens=60,
                                               image_tokens=0, output_tokens=0, model_evaluations=3)


def test_joint_gates_return_two_marginals_from_one_evaluation():
    backend = FakeBackend([[math.log(p) for p in [.1, .2, .3, .4]]])
    response = Engine(backend).evaluate(EvaluationRequest(state="x", questions={
        "gates": Gates(instructions="Decide", criteria={"delegate": "Work", "ask": "Context"})}))
    answer = response.answers["gates"]
    assert answer.gates == pytest.approx({"delegate": .7, "ask": .6})
    assert answer.outcomes["10"] == {"delegate": True, "ask": False}
    assert len(backend.calls) == 1
    assert backend.calls[0][-1] == 4


def test_question_id_is_not_in_model_prompt():
    q = Choice(instructions="Pick", criteria={"a": "A"})
    backend = FakeBackend([[2]])
    response = Engine(backend).evaluate(EvaluationRequest(state="input", questions={"private_internal_id": q}))
    assert "private_internal_id" not in backend.calls[0][0]
    assert response.answers["private_internal_id"].confidence == 1


@pytest.mark.parametrize("payload", [
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"x": {"type": "choice", "instructions": "Pick", "criteria": {}}}},
    {"state": "x", "questions": {"x": {"type": "score", "instructions": "Rate", "criteria": ["one"]}}},
    {"state": "x", "questions": {"x": {"type": "noul", "instructions": "", "criteria": None}}},
    {"state": "x", "questions": {"x": {"type": "noul", "instructions": "?", "criteria": {"true": "Y"}}}},
    {"state": "x", "questions": {"x": {"type": "gates", "instructions": "?", "criteria": {str(i): "?" for i in range(5)}}}},
    {"state": "x", "questions": {"x": {"type": "noul", "instructions": "?"}}, "temperature": float("nan")},
    {"state": "x", "questions": {"x": {"type": "noul", "instructions": "?"}}, "typo": 1},
])
def test_invalid_requests_rejected(payload):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(payload)


def test_capacity_rejection_preserves_all_options():
    q = Choice(instructions="Pick", criteria={str(i): "item" for i in range(27)})
    with pytest.raises(ValueError, match="Too many"):
        compile_question(q, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def test_stable_softmax_and_entropy():
    assert probabilities([1e9, 1e9], 1) == [.5, .5]
    assert probabilities([-1e9, 1e9], .05) == [0, 1]
    assert concentration([.5, .5]) == pytest.approx(0)
    assert concentration([1, 0]) == pytest.approx(1)
    for logits in [[], [float("nan")], [float("inf")]]:
        with pytest.raises(RuntimeError):
            probabilities(logits, 1)


def test_image_cannot_reference_server_file_or_remote_url():
    for value in ["file:///etc/passwd", "https://example.com/a.png", "data:image/svg+xml;base64,AAAA"]:
        with pytest.raises(ValidationError):
            ImageInput(data_url=value)


def test_bad_image_rejected_before_inference():
    backend = FakeBackend([])
    request = EvaluationRequest(state="x", images=[ImageInput(data_url="data:image/png;base64,!!!")],
                                questions={"x": Noul(instructions="Yes?")})
    with pytest.raises(ValueError):
        Engine(backend).evaluate(request)
    assert not backend.calls


def test_large_state_rejected_before_inference():
    backend = FakeBackend([])
    with pytest.raises(ValueError, match="State exceeds"):
        Engine(backend).evaluate(EvaluationRequest(state="x" * 64001, questions={"x": Noul(instructions="?")}))
    assert not backend.calls
