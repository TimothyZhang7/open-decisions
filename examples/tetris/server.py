"""Local Open Decisions persistent-target controller; gravity runs in the client."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.parse import urlsplit

from tetris_engine import SHAPES
from tetris_realtime import ACTIONS, ORIENTATIONS, model_state, validate_snapshot
from tetris_planner import TARGET_INSTRUCTIONS, CONTROL_INSTRUCTIONS, board_key, reachable, retained_target, control_state, target_criteria
from decision_adapter import ROOT, LocalChoiceClient, LocalChoiceError, normalized_probabilities

INSTRUCTIONS = {
    "task": "Choose the single next action that best advances the Tetris goal using the supplied board and falling piece.",
    "goal": "Clear as many complete horizontal rows as possible while keeping room for future pieces. Clearing rows earns points. Merely surviving a few seconds, dropping pieces, or pressing buttons is not the objective. Avoid game over so you can keep clearing rows.",
    "rules": [
        "The board is 10 columns wide and 20 rows tall. Every piece has four occupied cells. Pieces cannot overlap locked cells or move through the walls or floor.",
        "Gravity automatically moves the active piece downward. Once supported by the floor or locked cells, it locks after 500 ms of accumulated ground contact and becomes part of the fixed board. Inputs and none do not reset that timer.",
        "After a piece locks, any horizontal row containing all 10 occupied cells is removed. Rows above it move downward to close the gap. Incomplete rows remain. Filling a column does not clear it. One piece can clear up to four rows at once.",
        "You lose when a new piece cannot spawn because its cells overlap the stack, or when a piece locks with any cells above the top of the board. Avoid building tall piles that block the next piece.",
    ],
    "controls": "Five choices: left, right, clockwise, counterclockwise, none. A movement choice presses one button once. none sends no input and lets gravity continue. There is no drop, hold, placement target, or action sequence.",
    "timing": "Gravity keeps running while you decide. The same piece may be lower when your button is applied. A response for an already locked piece is discarded. Act quickly toward a useful landing.",
    "read_state": [
        "`board_top_to_bottom` contains locked cells only: # means occupied and . means empty. Overlay `occupied_active_cells` to see the falling piece. `active` gives its position and rotation; `next_piece` names the following piece.",
        "`current_gravity_landing` describes the result if you send no more inputs. `one_action_consequences` describes the result after each single action followed by gravity without further inputs. These are predictions, not commands to drop the piece.",
        "In each predicted result, `lines` counts rows cleared and should be larger; `top_out` means losing. `holes` counts empty cells with a filled cell above them in the same column; smaller is better. `max_height` and `total_height` measure pile height after clears; smaller leaves more room. `bumpiness` measures differences between adjacent column heights.",
    ],
    "strategy": [
        "First avoid a losing landing when a safe route exists. Prefer filling gaps that complete rows. If no row can be completed now, build a low stack with few covered holes and leave useful space for the next piece.",
        "Compare the consequences of all five choices against that goal. Shift or rotate toward a useful gap before the piece falls below it. Several future sideways inputs may be needed; choose one step toward the gap now instead of abandoning it because one step alone does not clear a row.",
        "Choose none when the current column and orientation already lead to a useful landing and no movement is needed now. Do not choose none simply because the piece can fall: that can bury holes, miss a clear, or cause game over. If a button improves the landing, use it; if all movement would worsen or merely duplicate a good landing, let it fall.",
        "Use `recent_actions_oldest_first` to avoid aimless left-right reversals and repeated rotations. Coordinates are zero-based: x increases rightward and y downward. Rotation 0/1/2/3 counts clockwise quarter-turns. Collision checks and basic wall kicks apply to every movement.",
    ],
}
CRITERIA = {
    "left": "Shift one column left. Choose when this is the best next step toward completing rows or a safer landing.",
    "right": "Shift one column right. Choose when this is the best next step toward completing rows or a safer landing.",
    "clockwise": "Rotate 90 degrees clockwise. Choose when this orientation best advances a row clear or safer landing.",
    "counterclockwise": "Rotate 90 degrees counterclockwise. Choose when this orientation best advances a row clear or safer landing.",
    "none": "Send no input while gravity and locking continue. Choose when already aligned for a useful landing and no movement now would improve it; do not idle through an avoidable bad landing.",
}


def read_choice(body, question, options):
    try:
        answer = body["answers"][question]
        probabilities = answer["probabilities"]
        choice, confidence = answer["choice"], answer["confidence"]
        if answer["type"] != "choice" or set(probabilities) != set(options) or choice not in options:
            raise ValueError("Unknown choice")
        values = list(probabilities.values())
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        normalized, total = normalized_probabilities(values)
        if max(values) - probabilities[choice] > 1e-9:
            raise ValueError("Invalid distribution")
        usage, model = body["usage"], body["model"]
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
            raise ValueError("Invalid usage")
        if usage['output_tokens'] != 0:
            raise ValueError('Local scorer must not generate tokens')
        if not isinstance(model, str) or not model:
            raise ValueError("Invalid model")
    except (ValueError, KeyError, TypeError, AttributeError):
        raise LocalChoiceError("Open Decisions returned an invalid decision; gravity continues without applying it.") from None
    return answer, dict(zip(probabilities, normalized)), total


def decide_action(client, snapshot):
    state = model_state(snapshot)
    started = time.perf_counter()
    graph = reachable(snapshot)
    target = retained_target(snapshot, graph)
    planned = target is None and bool(graph['options'])
    usage = {'input_tokens': 0, 'output_tokens': 0}
    plan_answer = None
    plan_inference = None
    if planned:
        question = {'target': {'type': 'choice', 'instructions': TARGET_INSTRUCTIONS,
                               'criteria': target_criteria(graph)}}
        planning_state = {key: state[key] for key in ('board_top_to_bottom', 'next_piece', 'board_features')}
        planning_state['current_piece'] = snapshot['active']['piece']
        body = client.evaluate(planning_state, question)
        plan_inference = body.get('inference')
        plan_answer, _, _ = read_choice(body, 'target', graph['options'])
        target = {**graph['options'][plan_answer['choice']], 'piece_id': snapshot['active']['id'],
                  'board_key': board_key(snapshot['board']), 'selected_by': 'open-decisions', 'selection_confidence_kind': 'normalized_entropy_concentration',
                  'selection_confidence': plan_answer['confidence']}
        for key in usage:
            usage[key] += body['usage'][key]
    action_state = control_state(snapshot, graph, target) if target else state
    instructions = CONTROL_INSTRUCTIONS if target else INSTRUCTIONS
    questions = {'action': {'type': 'choice', 'instructions': instructions,
                            'criteria': {action: ('Send no input and let gravity continue.' if action == 'none'
                                                 else 'Press ' + action + ' once.') for action in ACTIONS}}}
    body = client.evaluate(action_state, questions)
    answer, probabilities, total = read_choice(body, 'action', ACTIONS)
    for key in usage:
        usage[key] += body['usage'][key]
    return {'action': answer['choice'], 'piece_id': snapshot['active']['id'],
            'generation': snapshot['generation'], 'request_id': snapshot['request_id'],
            'probabilities': probabilities, 'confidence': answer['confidence'],
            'latency_ms': (time.perf_counter() - started) * 1000,
            'model': body['model'], 'usage': usage, 'api_requests': 2 if planned else 1,
            'snapshot_y': snapshot['active']['y'], 'provider_answer': answer,
            'provider_probability_sum': total, 'target': target,
            'target_status': 'selected' if planned else 'retained' if target else 'unavailable',
            'target_answer': plan_answer, 'steps_to_target': action_state.get('current_steps_to_target'),
            'inference': body.get('inference'), 'target_inference': plan_inference,
            'backend': 'open-decisions', 'confidence_kind': 'normalized_entropy_concentration'}


# Compatibility for local callers of the previous single-stage service.
decide = decide_action


def create_server(client, worker, host="127.0.0.1", port=8773):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def send(self, status, value, content_type="application/json"):
            data = json.dumps(value).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {"/": ("tetris.html", "text/html; charset=utf-8"),
                      "/tetris.js": ("tetris.js", "text/javascript; charset=utf-8"),
                      "/tetris_game.js": ("tetris_game.js", "text/javascript; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                return self.send(200, (ROOT / name).read_bytes(), mime)
            if path == "/api/info":
                return self.send(200, {"model": client.model, "shapes": SHAPES,
                                       "orientations": ORIENTATIONS, "actions": ACTIONS,
                                       "instructions": INSTRUCTIONS, "target_instructions": TARGET_INSTRUCTIONS,
                                       "control_instructions": CONTROL_INSTRUCTIONS, "goal_lines": 10, "backend": "local",
                                       "workflow": "persistent_target", "local_only": True,
                                       "decision_package": "open-decisions", "runtime": client.info,
                                       "target_label_capacity": client.info['label_capacity'],
                                       "load_ms": client.load_ms})
            if path == "/health":
                return self.send(200, {"status": "ready", "model": client.model})
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/action":
                return self.send(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8192:
                    raise ValueError("Send a JSON board smaller than 8 KB.")
                body = json.loads(self.rfile.read(size))
                validate_snapshot(body)
                if not inference_lock.acquire(blocking=False):
                    return self.send(429, {"error": "Model is still deciding; gravity continues."})
                try:
                    result = worker.submit(decide_action, client, body).result()
                finally:
                    inference_lock.release()
                self.send(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.send(400, {"error": "Invalid board request." if isinstance(error, (KeyError, TypeError)) else str(error)})
            except LocalChoiceError as error:
                self.send(error.status, {"error": str(error)})
            except TimeoutError:
                self.send(408, {"error": "Timed out reading the board."})
            except Exception:
                self.send(500, {"error": "Local inference failed. Gravity continues."})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["mlx-vlm", "mlx-lm", "transformers"], default="mlx-vlm")
    args = parser.parse_args()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-decisions-tetris")
    client = worker.submit(LocalChoiceClient, args.model, args.backend).result()
    server = create_server(client, worker, args.host, args.port)
    print(f"Persistent-target Open Decisions Tetris ready at http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.shutdown(wait=True, cancel_futures=True)
        client.close()


if __name__ == "__main__":
    main()
