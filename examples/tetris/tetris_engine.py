"""Deterministic hard-drop Tetris. Geometry and features, never move selection."""
import hashlib
import json

WIDTH, HEIGHT = 10, 20
SHAPES = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}
COLORS = {name: i + 1 for i, name in enumerate(SHAPES)}


def validate_state(board, piece, next_piece):
    if (not isinstance(board, list) or len(board) != HEIGHT
            or any(not isinstance(row, list) or len(row) != WIDTH
                   or any(type(cell) is not int or not 0 <= cell <= 7 for cell in row) for row in board)):
        raise ValueError("Board must be 20 rows of 10 integers from 0 to 7.")
    if not isinstance(piece, str) or piece not in SHAPES:
        raise ValueError("Unknown current piece.")
    if not isinstance(next_piece, str) or next_piece not in SHAPES:
        raise ValueError("Unknown next piece.")


def fingerprint(board, piece, next_piece):
    return hashlib.sha256(json.dumps([board, piece, next_piece], separators=(",", ":")).encode()).hexdigest()


def rotations(piece):
    cells = SHAPES[piece]
    seen = set()
    result = []
    for turn in range(4):
        left, top = min(x for x, y in cells), min(y for x, y in cells)
        cells = sorted((x - left, y - top) for x, y in cells)
        key = tuple(cells)
        if key not in seen:
            seen.add(key)
            result.append((turn * 90, cells))
        cells = [(-y, x) for x, y in cells]
    return result


def collides(board, cells, x, y):
    return any(x + dx < 0 or x + dx >= WIDTH or y + dy >= HEIGHT
               or (y + dy >= 0 and board[y + dy][x + dx]) for dx, dy in cells)


def features(board):
    heights = []
    holes = 0
    for x in range(WIDTH):
        top = next((y for y in range(HEIGHT) if board[y][x]), HEIGHT)
        heights.append(HEIGHT - top)
        holes += sum(not board[y][x] for y in range(top, HEIGHT))
    return {"holes": holes, "max_height": max(heights), "total_height": sum(heights),
            "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:])),
            "column_heights": heights}


def placements(board, piece):
    """Every rotation/column reachable by a vertical drop from above the board.

    This placement-mode demo intentionally has no hold, wall kicks, or tucks.
    """
    result = []
    for rotation, cells in rotations(piece):
        width, height = max(x for x, y in cells) + 1, max(y for x, y in cells) + 1
        for x in range(WIDTH - width + 1):
            y = -height
            while not collides(board, cells, x, y + 1):
                y += 1
            if y < 0:  # Cannot lock a piece above the ceiling.
                continue
            landed = [[x + dx, y + dy] for dx, dy in cells]
            locked = [row[:] for row in board]
            for cx, cy in landed:
                locked[cy][cx] = COLORS[piece]
            cleared_rows = [i for i, row in enumerate(locked) if all(row)]
            remaining = [row for row in locked if not all(row)]
            after = [[0] * WIDTH for _ in cleared_rows] + remaining
            result.append({"id": f"r{rotation}_c{x + 1}", "rotation": rotation, "column": x + 1,
                           "x": x, "y": y, "shape": cells, "cells": landed,
                           "cleared_rows": cleared_rows, "lines": len(cleared_rows),
                           "board_after": after, "features": features(after)})
    return result


def summarize(move):
    f = move["features"]
    return (f"Rotate {move['rotation']} degrees clockwise; leftmost column {move['column']}. "
            f"After dropping and clearing: {move['lines']} lines cleared, {f['holes']} holes, "
            f"max height {f['max_height']}, sum of heights {f['total_height']}, "
            f"surface bumpiness {f['bumpiness']}.")
