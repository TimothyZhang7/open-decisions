"""Reachable landing candidates and progress measurements. Never selects an input."""
from collections import deque
import hashlib
import json

from tetris_realtime import ACTIONS, ORIENTATIONS, apply_button, collision, projected_landing

MOVEMENTS = ACTIONS[:4]
TARGET_INSTRUCTIONS = {
    'task': 'Choose the best reachable final landing for this falling Tetris piece. This is a persistent goal; separate decisions will press individual buttons to reach it while gravity continues.',
    'goal': 'Clear 10 horizontal rows before the stack blocks a new piece. A row clears only when all 10 cells are filled; cleared rows disappear and higher rows shift down.',
    'strategy': [
        'Prefer clearing rows without creating covered holes. A hole is an empty cell below an occupied cell in the same column and is costly to repair.',
        'Avoid tall piles. When holes and clears are comparable, prefer lower total height and a flatter surface. Keep room for the next piece.',
        'All candidates have exact outcomes AFTER locking and clearing. Compare the complete landings, not just how close they are to the current position.',
        'Judge the resulting board. Prefer fewer holes over ease of execution. The controller will keep this target until the piece locks or it becomes unreachable.',
    ],
    'coordinates': 'Columns are zero-based from the left. Heights count upward from the floor. Landing cells are [x,y], with y=0 at the top.',
}
CONTROL_INSTRUCTIONS = {
    'task': 'Choose one button to reach the already selected landing target. Keep pursuing this same target; do not choose a different landing here.',
    'rules': [
        'The geometry table gives minimum remaining button presses to the target AFTER each possible action. Smaller is closer; null means the target is unreachable after that action.',
        'If current_steps_to_target is zero, choose none and let the aligned piece fall. Otherwise choose a movement that reduces remaining presses. Do not choose none while useful movement is still needed.',
        'Prefer an action that changes the piece. Avoid blocked inputs and reversals. If multiple actions make equal progress, either is acceptable.',
        'Gravity and the 500 ms lock timer keep running during inference. Press exactly one of left, right, clockwise, counterclockwise, none. No drop, teleport, or action sequence.',
    ],
}


def board_key(board):
    return hashlib.sha256(json.dumps(board, separators=(',', ':')).encode()).hexdigest()[:20]


def key(active):
    return active['x'], active['y'], active['rotation']


def landed_cells(board, active):
    landed = dict(active)
    while not collision(board, {**landed, 'y': landed['y'] + 1}):
        landed['y'] += 1
    return tuple(sorted((landed['x'] + x, landed['y'] + y)
                        for x, y in ORIENTATIONS[landed['piece']][landed['rotation']]))


def reachable(snapshot):
    """Movement graph at this live height, including the game's basic wall kicks."""
    board, start = snapshot['board'], snapshot['active']
    states = {key(start): dict(start)}
    steps = {key(start): 0}
    edges = {}
    queue = deque([key(start)])
    while queue:
        at = queue.popleft()
        active = states[at]
        edges[at] = {}
        for action in MOVEMENTS:
            moved, changed = apply_button(board, active, action)
            # Bound upward wall kicks; no route through hidden space above spawn.
            if not changed or moved['y'] < max(-4, start['y'] - 2):
                continue
            next_key = key(moved)
            edges[at][action] = next_key
            if next_key not in states:
                states[next_key] = moved
                steps[next_key] = steps[at] + 1
                queue.append(next_key)
    landings = {at: landed_cells(board, active) for at, active in states.items()}
    candidates = {}
    for at, cells in landings.items():
        if any(y < 0 for x, y in cells):
            continue
        active = states[at]
        if cells in candidates and candidates[cells]['steps'] <= steps[at]:
            continue
        outcome = projected_landing(board, active)
        candidates[cells] = {'x': active['x'], 'rotation': active['rotation'],
                             'cells': [list(cell) for cell in cells], 'steps': steps[at],
                             'outcome': outcome}
    ordered = sorted(candidates.items(), key=lambda pair: pair[0])
    options = {f'p{index}': value for index, (_, value) in enumerate(ordered)}
    return {'states': states, 'steps': steps, 'edges': edges, 'landings': landings, 'options': options}


def retained_target(snapshot, graph):
    target = snapshot.get('target')
    if not isinstance(target, dict) or target.get('piece_id') != snapshot['active']['id'] or target.get('board_key') != board_key(snapshot['board']):
        return None
    for option in graph['options'].values():
        if option['cells'] == target.get('cells'):
            return {**option, 'piece_id': snapshot['active']['id'], 'board_key': board_key(snapshot['board']),
                    'selected_by': 'open-decisions', 'selection_confidence_kind': 'normalized_entropy_concentration', 'selection_confidence': target.get('selection_confidence')}
    return None


def control_state(snapshot, graph, target):
    cells = tuple(map(tuple, target['cells']))
    distances = {}
    reverse = {at: [] for at in graph['states']}
    for at, edges in graph['edges'].items():
        for dest in edges.values():
            reverse[dest].append(at)
    queue = deque()
    for at, landed in graph['landings'].items():
        if landed == cells:
            distances[at] = 0
            queue.append(at)
    while queue:
        at = queue.popleft()
        for prev in reverse[at]:
            if prev not in distances:
                distances[prev] = distances[at] + 1
                queue.append(prev)
    current = key(snapshot['active'])
    effects = {}
    for action in ACTIONS:
        dest = graph['edges'][current].get(action, current)
        effects[action] = {'changes_piece': dest != current,
                           'remaining_presses': distances.get(dest),
                           'target_reachable': dest in distances}
    return {'piece': snapshot['active']['piece'], 'active': snapshot['active'],
            'target_cells': target['cells'], 'target_outcome': target['outcome'],
            'current_steps_to_target': distances.get(current), 'after_one_action': effects,
            'recent_actions': snapshot.get('recent_actions', [])}


def target_criteria(graph):
    return {name: (f"Rotate {candidate['rotation'] * 90} degrees clockwise; leftmost column "
                   f"{min(x for x, y in candidate['cells']) + 1}. After landing and clearing: "
                   f"{candidate['outcome']['lines']} lines cleared, {candidate['outcome']['holes']} holes, "
                   f"max height {candidate['outcome']['max_height']}, sum of heights {candidate['outcome']['total_height']}, "
                   f"surface bumpiness {candidate['outcome']['bumpiness']}.")
            for name, candidate in graph['options'].items()}
