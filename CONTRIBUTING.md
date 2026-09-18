# Contributing

Install `uv sync --extra dev --extra server` to run the core and API tests without a model runtime. `uv run pytest -q` covers primitive semantics, validation, backend injection, worker ownership, and service behavior.

The boundary is deliberate: the core must not import a model runtime, and examples must not implement tokenization or model scoring. New runtimes implement the protocol in `src/open_decisions/backends/base.py`. Keep unsupported-model and input errors explicit. Do not silently substitute token generation for direct logit scoring.

For runtime changes, install the corresponding optional extra and run real-model checks. `examples/check_backends.py` exercises the same four primitives across three runtimes. `examples/evaluate_model.py` includes Qwen vision checks, cache comparisons, and reference-forward parity. Record incorrect model answers as failures; use separate held-out data when tuning on these fixtures.

Do not commit weights, environments, credentials or private input data. Contributions use the project's MIT license.
