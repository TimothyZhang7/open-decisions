"""Geometry for Tetris snapshots: four buttons or no input. Never chooses input."""
from tetris_engine import SHAPES, WIDTH, HEIGHT, features, validate_state

ACTIONS = ("left", "right", "clockwise", "counterclockwise", "none")
KICKS = ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, -2))


def orientations():
    result = {}
    for piece, original in SHAPES.items():
        cells = [(x, y + 1) for x, y in original] if piece == "I" else list(original)
        size = 4 if piece == "I" else 3
        result[piece] = []
        for _ in range(4):
            result[piece].append(cells)
            if piece != "O":
                cells = [(size - 1 - y, x) for x, y in cells]
    return result


ORIENTATIONS = orientations()


def collision(board, active):
    return any(active["x"] + x < 0 or active["x"] + x >= WIDTH or active["y"] + y >= HEIGHT
               or (active["y"] + y >= 0 and board[active["y"] + y][active["x"] + x])
               for x, y in ORIENTATIONS[active["piece"]][active["rotation"]])


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("active"), dict):
        raise ValueError("A board and active piece are required.")
    active = snapshot["active"]
    validate_state(snapshot.get("board"), active.get("piece"), snapshot.get("next_piece"))
    for key, low, high in (("x", -4, 10), ("y", -4, 19), ("rotation", 0, 3), ("id", 1, 1000000)):
        if type(active.get(key)) is not int or not low <= active[key] <= high:
            raise ValueError(f"Invalid active piece {key}.")
    for key in ("generation", "request_id"):
        if type(snapshot.get(key)) is not int or not 0 <= snapshot[key] <= 1000000000:
            raise ValueError(f"Invalid {key}.")
    if collision(snapshot["board"], active):
        raise ValueError("Active piece collides with the board.")
    gravity = snapshot.get("gravity_ms")
    if type(gravity) not in (int, float) or not 50 <= gravity <= 2000:
        raise ValueError("Invalid gravity interval.")


def apply_button(board, active, action):
    if action not in ACTIONS:
        raise ValueError("Choose left, right, clockwise, counterclockwise, or none.")
    if action == "none":
        return active.copy(), False
    if action in ("left", "right"):
        moved = {**active, "x": active["x"] + (-1 if action == "left" else 1)}
        return (active.copy(), False) if collision(board, moved) else (moved, True)
    rotation = (active["rotation"] + (1 if action == "clockwise" else -1)) % 4
    if active["piece"] == "O":
        return active.copy(), False  # A square rotation is a legal input with no effect.
    for dx, dy in KICKS:
        moved = {**active, "rotation": rotation, "x": active["x"] + dx, "y": active["y"] + dy}
        if not collision(board, moved):
            return moved, True
    return active.copy(), False


def projected_landing(board, active):
    """A counterfactual feature calculation, never a drop command."""
    landed = active.copy()
    while not collision(board, {**landed, "y": landed["y"] + 1}):
        landed["y"] += 1
    cells = [(landed["x"] + x, landed["y"] + y) for x, y in ORIENTATIONS[landed["piece"]][landed["rotation"]]]
    if any(y < 0 for x, y in cells):
        return {"top_out": True}
    after = [row[:] for row in board]
    for x, y in cells:
        after[y][x] = 1
    rows = [row for row in after if not all(row)]
    cleared = HEIGHT - len(rows)
    after = [[0] * WIDTH for _ in range(cleared)] + rows
    return {"top_out": False, "landing_y": landed["y"], "lines": cleared, **features(after)}


def model_state(snapshot):
    validate_snapshot(snapshot)
    board, active = snapshot["board"], snapshot["active"]
    consequences = {}
    for action in ACTIONS:
        moved, changed = apply_button(board, active, action)
        consequences[action] = {"changes_piece": changed, "x": moved["x"], "y": moved["y"],
                                "rotation": moved["rotation"], "if_gravity_continues_without_more_inputs": projected_landing(board, moved)}
    previous = snapshot.get("recent_actions", [])
    if not isinstance(previous, list) or len(previous) > 8 or any(action not in ACTIONS for action in previous):
        raise ValueError("Invalid recent action history.")
    return {"board_top_to_bottom": ["".join("#" if cell else "." for cell in row) for row in board],
            "active": {key: active[key] for key in ("piece", "x", "y", "rotation")},
            "occupied_active_cells": [[active["x"] + x, active["y"] + y] for x, y in ORIENTATIONS[active["piece"]][active["rotation"]]],
            "next_piece": snapshot["next_piece"], "gravity_ms_per_row": snapshot["gravity_ms"],
            "recent_actions_oldest_first": previous, "board_features": features(board),
            "current_gravity_landing": projected_landing(board, active),
            "one_action_consequences": consequences,
            "coordinates": "x=0 is left, y=0 is top; board is 10x20. Rotation 0/1/2/3 is clockwise quarter-turn count."}
