# Tetris example

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
