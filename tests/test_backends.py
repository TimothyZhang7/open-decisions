import threading

import pytest

from open_decisions import Capabilities, LogitResult, LocalClient, Choice, ImageInput
from open_decisions.backends.base import create_backend
from open_decisions.backends.labels import label_vocabulary


class CustomBackend:
    name = "third-party"
    model_id = "custom-open-model"
    labels = ["A", "B"]
    capabilities = Capabilities()

    def __init__(self):
        self.owner = threading.get_ident()

    def score(self, system, state, images, count):
        assert threading.get_ident() == self.owner
        return LogitResult([1, 2][:count], .7, 20)


def test_custom_backend_factory_runs_on_its_own_thread_without_core_changes():
    caller = threading.get_ident()
    owners = []
    def factory():
        owners.append(threading.get_ident())
        return CustomBackend()
    with LocalClient(backend_factory=factory) as client:
        response = client.evaluate(state="hello", questions={"x": Choice(instructions="Pick", criteria={"one": None, "two": None})})
        assert response.backend == "third-party"
        assert response.model == "custom-open-model"
        assert response.answers["x"].choice == "two"
        assert client.info["images"] is False
    assert owners[0] != caller
    with pytest.raises(RuntimeError, match="closed"):
        client.evaluate(state="x", questions={"x": Choice(instructions="Pick", criteria={"one": None})})


def test_text_backend_rejects_images_before_decode_or_inference():
    with LocalClient(backend_factory=CustomBackend) as client:
        with pytest.raises(ValueError, match="does not support images"):
            client.evaluate(state="x", images=[ImageInput(data_url="data:image/png;base64,!!!")],
                            questions={"x": Choice(instructions="Pick", criteria={"one": None})})


def test_explicit_model_and_known_runtime_are_required():
    with pytest.raises(ValueError, match="Supply a model"):
        LocalClient()
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("missing", "test")


def test_labels_filter_duplicates_and_context_dependent_tokens():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            vocabulary = {"prefix": [100], "A": [1], "B": [2], "C": [3], "D": [1],
                          "prefixA": [100, 1], "prefixB": [200], "prefixC": [100, 3], "prefixD": [100, 1]}
            return vocabulary.get(text, [8, 9])
    labels, tokens = label_vocabulary(Tokenizer(), "prefix")
    assert labels == ["A", "C"]
    assert tokens == [1, 3]
