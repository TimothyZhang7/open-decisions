# Tetris comparison: Qwen and Jev

Recorded September 18, 2026. This is a short integration demonstration of two decision backends in the same game/controller, with no claim of general model parity.

[Video](https://github.com/TimothyZhang7/open-decisions/releases/download/demo-2026-09-18/tetris-qwen-vs-jev.mp4)

## Setup

- Qwen3-VL-4B-Instruct, MLX 4-bit, through Open Decisions on an Apple M4 Pro / 48 GB Mac. The model was already loaded; its existing policy cache was not reset.
- Hosted Jev, requested as `jev-latest`; successful responses reported `jev-1.13.0`. Remote hardware and service cache state are unknown.
- One run per backend, sequentially, seed 42, empty starting board, 500 ms per gravity row, and 500 ms lock delay. No manual inputs, hard drop, or hold.
- The same `tetris.js` controller and `tetris_game.js` physics drove both runs. Each run was stopped after about 60 seconds; pending requests were allowed to settle for final accounting.
- Task instructions, option descriptions, state construction, shapes, rotations, and goal matched. Planner code differs only in provider attribution and confidence metadata, which do not enter the model's control state. [Parity checks](../../docs/media/tetris-parity.json).
- Qwen receives Open Decisions' compiled chat prompt; Jev receives its typed API request. Those inference interfaces and internal model processing differ.

The model first chooses a target from reachable landings. Application code has already calculated holes, height, row clears, and travel distances. The model then selects individual movement buttons. A target is retained until reached, invalidated, or replaced with the next piece. Gravity continues during requests, and stale responses are discarded.

## Observed results

| Measure | Qwen local | Jev hosted |
| --- | ---: | ---: |
| Elapsed game run | 60.07 s | 60.07 s |
| Lines / score | 0 / 0 | 0 / 0 |
| Locked pieces | 6 | 6 |
| Game over | No | No |
| Controller requests | 62 | 103 |
| Successful model evaluations | 69 | 109 |
| Median controller request time | 419 ms | 135 ms |
| Request range, successful requests | 397–4,196 ms | 87–489 ms |
| Movement inputs applied | 40 | 22 |
| No-input choices applied | 18 | 79 |
| Blocked movement inputs | 0 | 0 |
| Stale responses discarded | 4 | 1 |
| Failed requests | 0 | 1 |

Jev's first request failed with a connection error before returning a model answer. The controller continued its normal loop. It is retained in the trace and the error counter; it is not evidence of a reasoning failure. The Qwen run had one pending request at the time limit; its eventual response is included in the final stale count.

The reported timing spans server-side state/planner work and the model calls for that controller request. A new target can require two serial evaluations; a retained target usually requires one. Jev's timing includes its remote API connection. These are workload-dependent request timings, not an isolated model-speed benchmark. Failure duration is excluded from the successful-request median and remains in the trace.

Qwen generated zero answer tokens. Jev reported 7,548 output tokens in its API usage fields. That counter is preserved as `reported_output_tokens`; it is **not** interpreted as autoregressive text generation or compared as an equivalent token-generation metric.

## Limits of the demonstration

Neither backend cleared a line in this run. The ten-line challenge was not run to completion; only six pieces locked before the time limit. Six pieces and one seed are insufficient to assess useful Tetris-playing strength, long-run survival, or an improvement over a simple deterministic policy. No such baseline was tested here.

The agent consumes structured game state and calculated candidate outcomes. It does not read the board through vision. The game supplies exact geometry, collision rules, reachability, and distances. This assistance is a large part of the system. The model can still choose poor placements, waste inputs, or lose a reachable target while waiting for inference. Qwen's label probabilities are uncalibrated; Jev confidence and Qwen concentration are not interchangeable metrics.

The runs took place separately on different inference infrastructure. Input states diverge as soon as choices or timing differ. Serving load, network conditions, prior cache state, and first-use costs were not controlled. This does not establish Jev parity, a causal speed comparison, or a ranking of the underlying models.

## Recording and reproduction

The video reconstructs the actual recorded game state at 1× elapsed time. States were sampled approximately every 100 ms; the rendered MP4 is 20 fps. It is a replay rather than a browser screen capture. All 60 seconds are shown, followed by a five-second results card. The README GIF shows the same minute at 1× with fewer sampled frames. No moves or outcomes are replaced.

Raw state and request traces: [Qwen](../../docs/media/tetris-qwen.json), [Jev](../../docs/media/tetris-jev.json). The traces include the backend's instructions, per-frame board state, requests, timings, and response data. Inputs are synthetic; no credentials are included.

Run the local server as described in [README.md](README.md). For the optional Jev baseline, provide your own TypeSafe API key through `TYPESAFE_API_KEY`, then from the package root:

```sh
uv run python examples/tetris/jev_server.py --port 8767 --model jev-1.13.0

# Run sequentially to match this setup; each writes one complete recording.
node examples/tetris/record_demo.cjs --port=8773 --seed=42 --gravity=500 --seconds=60 --output=tetris-qwen.json
node examples/tetris/record_demo.cjs --port=8767 --seed=42 --gravity=500 --seconds=60 --output=tetris-jev.json
```

The Jev baseline is an optional example and does not add a hosted-service dependency to Open Decisions. Hosted inference uses the account associated with the supplied key. Future runs may differ; a pinned model name does not control service load, API availability, or sampling behavior.

To render the saved recordings, install Pillow and OpenCV with an H.264 encoder in a separate environment, then run:

```sh
python examples/tetris/render_comparison.py --data-dir docs/media --output-dir tetris-video
```

The renderer was exercised with Pillow 12.3.0 and OpenCV 5.0.0 on this Mac. H.264 encoding availability depends on the OpenCV build. It uses only the saved traces and makes no inference requests.
