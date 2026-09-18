# Evaluation

Local verification on September 18, 2026, on an Apple M4 Pro / 48 GB Mac.
These are implementation and synthetic smoke checks. They are not a comparison
with Jev or a general accuracy/calibration benchmark.

## Package and runtime checks

- 23 Python tests passed: primitive math, structured input validation, custom
  backend injection, worker-thread ownership, unsupported modality rejection,
  tokenizer context checks, HTTP validation, body limits, and concurrent-request handling.
- 14 Tetris JavaScript tests passed: independent gravity, lock timing, retained
  targets, stale responses, error handling, and stopping conditions.
- Three real backends returned all four primitives using the same public SDK:

| Backend | Model | Device | Generated tokens |
| --- | --- | --- | --- |
| MLX-VLM 0.7.1 | Qwen3-VL-4B-Instruct, 4-bit | Apple GPU | 0 |
| MLX-LM 0.31.3 | Qwen3-4B-Instruct-2507, 4-bit | Apple GPU | 0 |
| Transformers 5.17.0 / PyTorch 2.14.0 | SmolLM2-135M-Instruct | CPU | 0 |

Raw responses: [backend-matrix.json](results/backend-matrix.json). Both Qwen
models changed their routing choice from billing to technical when the state
changed. The small SmolLM model incorrectly kept billing on the technical
request. That establishes working inference through the adapter, not useful
quality from every model.

The Transformers vision path and CUDA/MPS devices remain untested. MLX-VLM
currently supports the Qwen3-VL family; other architectures need an adapter.

## Qwen vision smoke evaluation

[smoke.json](results/smoke.json) records 28 cases and full outputs:

- 7 team-routing cases, 4 yes/no cases, 3 rubric scores, and 5 dual-gate cases.
- 6 single-image color/shape cases, 1 receipt-total case, and 2 image-order cases.
- **27/28 correct** using argmax or a 0.5 gate threshold (nearest level for Score).
- Zero generated tokens across every evaluation.
- Median text latency with an existing policy cache: **62 ms** (15 cases).
- Median single-image latency: **302 ms** (7 cases, including first-use overhead).
- Median across all image cases: **454 ms** (9 cases).
- Model load in this run: **3.40 seconds**; first image evaluation: **1.87 seconds**.

These timings measure the engine call, not model download, client network time,
or browser rendering. They mix small synthetic inputs and should not be used
as service-level latency promises.

Seven cached/full-prefill comparisons retained the same choices; the largest
probability difference was 0.0000317. Four comparisons with MLX-VLM's ordinary
forward pass differed by at most 0.00000844 in normalized probability. These
include text, single images, a receipt and the failing image-order case.

### Known failure

With a red shape first and a blue shape second, Qwen chose the reversed order
with very high probability. The ordinary uncached forward agreed with the
optimized path. Adding image numbering in a separate diagnostic did not correct
it. The failure remains in the suite; no prompt was tuned to hide it. High
concentration is therefore visibly not a correctness guarantee.

Model revision is recorded in [model-manifest.json](model-manifest.json).
The Qwen weights are Apache-2.0; the MLX files occupy approximately 2.9 GiB.

## Examples

The router browser test returned both delegate and ask probabilities from one
Gates evaluation, with zero generated tokens. The label-logit display was also
checked. Browser file-upload interaction is not covered by these checks.
Image inference was exercised directly by the SDK tests.

The rebuilt Tetris example ran for 20.14 seconds, seed 42, 500 ms gravity:

- 27 decisions through the public Open Decisions SDK.
- 24 applied inputs, 2 no-action decisions, 1 discarded stale response.
- 0 inference errors, 0 blocked inputs, 0 generated tokens.
- 1 locked piece and 0 cleared lines; the game was paused at the test limit.

The game physics and target/control prompts match the earlier demo. Only the
inference integration and branding changed. This test confirms the application
uses the package; it does not establish game-playing quality.
Raw trace: [tetris-smoke.json](results/tetris-smoke.json).

## Reproduce

```sh
uv run pytest -q
node examples/tetris/test_tetris_ui.cjs
uv run python examples/evaluate_model.py --model .models/qwen3-vl-4b-instruct-4bit
uv run python examples/check_backends.py \
  --qwen-vl .models/qwen3-vl-4b-instruct-4bit \
  --qwen-text /path/to/qwen-text-model \
  --hf-model .models/smollm2-135m-instruct
```

The fixtures are synthetic and committed. No private concierge prompt or user
data is part of the package. Evaluation failures are reported in JSON and
printed; backend-contract and cache-parity failures raise an exception.
