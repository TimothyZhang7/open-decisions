"""General text-model adapter for MLX-LM's load/forward/cache interface."""
from collections import OrderedDict
import copy

from .base import Capabilities, LogitResult
from .labels import label_vocabulary, reject_control_tokens


class MLXLMBackend:
    name = "mlx-lm"
    capabilities = Capabilities(prefix_cache=True)

    def __init__(self, model, *, revision=None, cache=True, max_tokens=8192):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.mx, self.make_cache = mx, make_prompt_cache
        self.model_id = str(model)
        self.network, self.tokenizer = load(str(model), revision=revision)
        self.network.eval()
        mx.eval(self.network.parameters())
        cue = self._format([{"role": "user", "content": "Select a label."}], True, tokenize=False)
        self.labels, self.label_ids = label_vocabulary(self.tokenizer, cue)
        self.prefixes = OrderedDict()
        self.cache_enabled, self.max_tokens = cache, max_tokens

    def _format(self, messages, generation, tokenize=True):
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize,
            add_generation_prompt=generation, enable_thinking=False)

    def _prefill(self, tokens, cache):
        for start in range(0, len(tokens), 512):
            self.network(self.mx.array([tokens[start:start+512]]), cache=cache)
            self.mx.eval([layer.state for layer in cache])

    def score(self, system, state, images, count):
        if images:
            raise ValueError("mlx-lm is a text-only backend")
        reject_control_tokens(self.tokenizer, system, state)
        mx = self.mx
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "State: " + state}]
        tokens = self._format(messages, True)
        if len(tokens) > self.max_tokens:
            raise ValueError(f"Prompt exceeds {self.max_tokens} tokens")
        cache = self.make_cache(self.network)
        offset, cached = 0, 0
        if self.cache_enabled:
            if system in self.prefixes:
                self.prefixes.move_to_end(system)
                prefix, stored = self.prefixes[system]
                if tokens[:len(prefix)] != prefix:
                    raise RuntimeError("Cached policy differs from the request prefix")
                cache, offset, cached = copy.deepcopy(stored), len(prefix), len(prefix)
            else:
                prefix = self._format(messages[:1], False)
                if len(prefix) <= 2048 and tokens[:len(prefix)] == prefix:
                    self._prefill(prefix, cache)
                    self.prefixes[system] = (prefix, copy.deepcopy(cache))
                    offset = len(prefix)
                    while len(self.prefixes) > 8 or sum(len(p[0]) for p in self.prefixes.values()) > 4096:
                        self.prefixes.popitem(last=False)
        remaining = tokens[offset:]
        # Prefill all but the last token so the vocabulary head sees one position.
        self._prefill(remaining[:-1], cache)
        logits = self.network(mx.array([remaining[-1:]]), cache=cache)[0, -1].astype(mx.float32)
        selected = logits[mx.array(self.label_ids[:count])]
        mass = mx.exp(mx.logsumexp(selected)-mx.logsumexp(logits))
        mx.eval(selected, mass)
        result = LogitResult(selected.tolist(), min(1., max(0., mass.item())), len(tokens), 0, cached)
        mx.clear_cache()
        return result
