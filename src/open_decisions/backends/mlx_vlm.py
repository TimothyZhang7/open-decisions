"""Qwen3-VL on Apple Silicon. Prefill only; never calls generate or samples a token.

The version-pinned MLX-VLM decoder preserves Qwen's vision encoder, DeepStack
features and multimodal positions. Only the final position goes through lm_head.
"""
from collections import OrderedDict
import copy
from .base import Capabilities, LogitResult
from .labels import label_vocabulary, reject_control_tokens

DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_REVISION = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b"


class MLXVLMBackend:
    name = "mlx-vlm"
    capabilities = Capabilities(images=True, prefix_cache=True)
    def __init__(self, model=DEFAULT_MODEL, *, revision=None, cache=True, max_tokens=8192):
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_vlm.utils import prepare_inputs

        self.mx = mx
        self.prepare_inputs = prepare_inputs
        self.make_cache = make_prompt_cache
        self.model_id = str(model)
        self.network, self.processor = load(str(model), revision=revision, trust_remote_code=False)
        if self.network.config.model_type != "qwen3_vl":
            raise ValueError("This backend supports Qwen3-VL models only")
        self.tokenizer = self.processor.tokenizer
        cue = self._format([{"role": "user", "content": "Select a label."}], True)
        self.labels, self.label_ids = label_vocabulary(self.tokenizer, cue)
        self.cache_enabled = cache
        self.prefixes = OrderedDict()
        self.max_tokens = max_tokens
        self.special_tokens = tuple(self.tokenizer.all_special_tokens)
        self.network.eval()
        mx.eval(self.network.parameters())

    def _format(self, messages, generation):
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)

    def _prefix(self, system, token_ids):
        mx = self.mx
        if system in self.prefixes:
            self.prefixes.move_to_end(system)
            entry = self.prefixes[system]
            return copy.deepcopy(entry), len(entry[0])
        prefix_text = self._format([{"role": "system", "content": system}], False)
        tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(tokens) > 2048 or token_ids[:len(tokens)] != tokens:
            return None, 0
        cache = self.make_cache(self.network.language_model)
        ids = mx.array([tokens])
        features = self.network.get_input_embeddings(ids)
        self.network.language_model.model(ids, inputs_embeds=features.inputs_embeds,
                                          position_ids=features.position_ids, cache=cache)
        mx.eval([layer.state for layer in cache])
        self.prefixes[system] = (tokens, cache)
        # Bound both the number and combined length of stored policies.
        while len(self.prefixes) > 8 or sum(len(p[0]) for p in self.prefixes.values()) > 4096:
            self.prefixes.popitem(last=False)
        return copy.deepcopy((tokens, cache)), 0

    def score(self, system, state, images, count):
        mx = self.mx
        reject_control_tokens(self.tokenizer, system, state)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": ([{"type": "image"} for _ in images]
                                         + [{"type": "text", "text": "State: " + state}])},
        ]
        prompt = self._format(messages, True)
        inputs = self.prepare_inputs(
            self.processor, images=images or None, prompts=prompt,
            image_token_index=self.network.config.image_token_index,
            max_pixels=1024 * 1024, min_pixels=32 * 32 * 4,
        )
        ids = inputs.pop("input_ids")
        mask = inputs.pop("attention_mask", None)
        pixels = inputs.pop("pixel_values", None)
        tokens = ids[0].tolist()
        if len(tokens) > self.max_tokens:
            raise ValueError(f"Prompt exceeds {self.max_tokens} tokens after image expansion")
        image_tokens = tokens.count(self.network.config.image_token_index)
        prefix, cached = self._prefix(system, tokens) if self.cache_enabled else (None, 0)
        offset, cache = (len(prefix[0]), prefix[1]) if prefix else (0, None)
        features = self.network.get_input_embeddings(ids, pixels, mask=mask, **inputs)
        visual_mask = features.visual_pos_masks
        hidden = self.network.language_model.model(
            ids[:, offset:], inputs_embeds=features.inputs_embeds[:, offset:],
            position_ids=features.position_ids[..., offset:], cache=cache,
            visual_pos_masks=visual_mask[:, offset:] if visual_mask is not None else None,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
        )[:, -1:, :]
        language = self.network.language_model
        logits = (language.model.embed_tokens.as_linear(hidden) if language.args.tie_word_embeddings
                  else language.lm_head(hidden))[0, 0].astype(mx.float32)
        selected = logits[mx.array(self.label_ids[:count])]
        mass = mx.exp(mx.logsumexp(selected) - mx.logsumexp(logits))
        mx.eval(selected, mass)
        result = LogitResult(selected.tolist(), min(1.0, max(0.0, mass.item())),
                             len(tokens), image_tokens, cached)
        mx.clear_cache()
        return result
