# Tetris example

[Watch Qwen and Jev side by side](https://github.com/TimothyZhang7/open-decisions/releases/download/demo-2026-09-18/tetris-qwen-vs-jev.mp4) · [Recorded comparison and limitations](COMPARISON.md)

**This is an experimental controller.** In the published 60-second comparison, both backends locked six pieces and cleared zero lines. Neither reached the ten-line goal. The models receive structured state and legal landings with outcomes and movement distances computed by the game. The example does not establish strong gameplay, vision-based control, or parity between models. Qwen's scores are uncalibrated.

Run from the package root:

```sh
uv run python examples/tetris/server.py --backend mlx-vlm --model .models/qwen3-vl-4b-instruct-4bit
```

Open http://127.0.0.1:8773/. `--backend mlx-lm --model /path/to/text/model` switches runtimes without changing the game.

`decision_adapter.py` calls the public `LocalClient.evaluate` method and translates its result into the game's existing HTTP response. The game supplies legal landing candidates and distances; the package chooses one target and then individual movement inputs. There is no model inference implementation in this example.

The target remains active until it is reached, becomes unreachable, or the piece changes. Gravity, lock delay, stale-response rejection, and game rules are deterministic application code. Ten cleared lines is the example's goal, not a promised model capability.

Run a short integration check:

```sh
cd examples/tetris
node run_tetris_live.cjs --port=8773 --seconds=20 --output=../../results/tetris-smoke.json
```

For an actual state-by-state recording, run from the package root:

```sh
node examples/tetris/record_demo.cjs --port=8773 --seconds=60 --seed=42 --output=tetris-qwen.json
```

The recorder uses the same JavaScript controller and physics at wall-clock speed. It samples board state and retains API responses, including failures. See [COMPARISON.md](COMPARISON.md) for the optional Jev baseline and comparison setup.
