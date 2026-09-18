"""Portable PyTorch adapter for standard Hugging Face causal/vision models.

Uses one forward pass and reads final-position logits. No generation method.
Cross-request KV caching is intentionally left to runtime-specific adapters.
"""
import inspect

from .base import Capabilities, LogitResult
from .labels import label_vocabulary, reject_control_tokens


class TransformersBackend:
    name = "transformers"

    def __init__(self, model, *, revision=None, device="auto", vision=False, max_tokens=8192, dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        self.torch, self.model_id = torch, str(model)
        self.capabilities = Capabilities(images=vision)
        self.max_tokens = max_tokens
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        model_class = AutoModelForImageTextToText if vision else AutoModelForCausalLM
        self.network = model_class.from_pretrained(str(model), revision=revision, dtype=dtype,
                                                  trust_remote_code=False).to(device).eval()
        if vision:
            self.processor = AutoProcessor.from_pretrained(str(model), revision=revision, trust_remote_code=False)
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(str(model), revision=revision, trust_remote_code=False)
        cue = self._format([{"role": "user", "content": "Select a label."}])
        self.labels, self.label_ids = label_vocabulary(self.tokenizer, cue)
        self.last_logits_only = "logits_to_keep" in inspect.signature(self.network.forward).parameters

    def _format(self, messages):
        formatter = self.processor or self.tokenizer
        return formatter.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def score(self, system, state, images, count):
        if images and not self.capabilities.images:
            raise ValueError("This Transformers model was loaded in text-only mode")
        reject_control_tokens(self.tokenizer, system, state)
        content = ([{"type": "image", "image": im} for im in images]
                   + [{"type": "text", "text": "State: " + state}]) if images else "State: " + state
        prompt = self._format([{"role": "system", "content": system}, {"role": "user", "content": content}])
        if images:
            inputs = self.processor(text=prompt, images=images, return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count_tokens = inputs["input_ids"].shape[-1]
        if count_tokens > self.max_tokens:
            raise ValueError(f"Prompt exceeds {self.max_tokens} tokens")
        image_id = getattr(self.network.config, "image_token_id", None)
        image_tokens = int((inputs["input_ids"] == image_id).sum()) if image_id is not None else 0
        inputs = inputs.to(self.device)
        if self.last_logits_only:
            inputs["logits_to_keep"] = 1
        with self.torch.inference_mode():
            logits = self.network(**inputs, use_cache=False).logits[0, -1].float()
            selected = logits[self.label_ids[:count]]
            mass = (selected.logsumexp(0)-logits.logsumexp(0)).exp().item()
            values = selected.cpu().tolist()
        return LogitResult(values, min(1., max(0., mass)), count_tokens, image_tokens)
