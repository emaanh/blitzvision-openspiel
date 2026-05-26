#!/usr/bin/env python3
"""
Replay a recording through both the Python and C++ engines and compare state
at every step. Prints a diff if the engines disagree.

Usage:
    python3 test_consistency.py                  # all recordings
    python3 test_consistency.py game_01          # one recording by name
    python3 test_consistency.py /path/to/foo.json
"""

import argparse
import json
import os
import sys

BOARD_DIM = 13
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")

sys.path.insert(0, os.path.join(REPO_ROOT, "python_blitzgo"))
sys.path.insert(0, os.path.join(REPO_ROOT, "build", "python"))
sys.path.insert(0, REPO_ROOT)

from game import Game
from player import User
import pyspiel


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def py_extract(game):
    board = game.board
    black = next(p for p in game.players if p.isBlack)
    white = next(p for p in game.players if not p.isBlack)

    stones = []
    territory = []
    for y in range(BOARD_DIM):
        for x in range(BOARD_DIM):
            s = board.stones[y][x]
            stones.append(1 if s is black else 2 if s is white else 0)
            t = board.territory[y][x]
            territory.append(1 if t is black else 2 if t is white else 0)

    return {
        "stones":    stones,
        "territory": territory,
        "black":     board.territory_counts[black],
        "white":     board.territory_counts[white],
        "player":    0 if game.currPlayer().isBlack else 1,
    }


def cpp_extract(state):
    import numpy as np
    # Query from Black's perspective so planes are absolute, not player-relative.
    tensor = np.array(state.observation_tensor(0)).reshape(5, BOARD_DIM, BOARD_DIM)

    stones = []
    territory = []
    for y in range(BOARD_DIM):
        for x in range(BOARD_DIM):
            stones.append(
                1 if tensor[0, y, x] > 0.5 else
                2 if tensor[1, y, x] > 0.5 else 0
            )
            territory.append(
                1 if tensor[2, y, x] > 0.5 else
                2 if tensor[3, y, x] > 0.5 else 0
            )

    black = sum(1 for v in stones if v == 1) + sum(1 for v in territory if v == 1)
    white = sum(1 for v in stones if v == 2) + sum(1 for v in territory if v == 2)

    cp = state.current_player()
    return {
        "stones":    stones,
        "territory": territory,
        "black":     black,
        "white":     white,
        "player":    cp,
    }


# ---------------------------------------------------------------------------
# Diff display
# ---------------------------------------------------------------------------

SYMS = {0: "·", 1: "○", 2: "●"}


def show_grid_diff(label, py_vals, cpp_vals):
    header = "   " + "".join(f"{c:2}" for c in range(1, BOARD_DIM + 1))
    py_lines  = [header]
    cpp_lines = [header]
    diff_cells = []

    for y in range(BOARD_DIM):
        row_label = f"{BOARD_DIM - y:2}  "
        py_row = cpp_row = row_label
        for x in range(BOARD_DIM):
            i = y * BOARD_DIM + x
            pv, cv = py_vals[i], cpp_vals[i]
            py_row  += SYMS[pv] + " "
            cpp_row += SYMS[cv] + " "
            if pv != cv:
                diff_cells.append((BOARD_DIM - y, x + 1, pv, cv))
        py_lines.append(py_row)
        cpp_lines.append(cpp_row)

    if not diff_cells:
        return

    print(f"  {label} mismatch at {len(diff_cells)} cell(s):")
    for row, col, pv, cv in diff_cells:
        print(f"    ({row:2},{col:2})  py={SYMS[pv]}  cpp={SYMS[cv]}")
    print(f"  Python {label}:\n" + "\n".join("    " + l for l in py_lines))
    print(f"  C++    {label}:\n" + "\n".join("    " + l for l in cpp_lines))


def show_full_board(label, state):
    header = "   " + "".join(f"{c:2}" for c in range(1, BOARD_DIM + 1))
    print(f"  {label} stones:")
    print("    " + header)
    for y in range(BOARD_DIM):
        row = f"{BOARD_DIM - y:2}  "
        for x in range(BOARD_DIM):
            row += SYMS[state["stones"][y * BOARD_DIM + x]] + " "
        print("    " + row)
    print("    " + header)
    print(f"  {label} territory:")
    print("    " + header)
    for y in range(BOARD_DIM):
        row = f"{BOARD_DIM - y:2}  "
        for x in range(BOARD_DIM):
            row += SYMS[state["territory"][y * BOARD_DIM + x]] + " "
        print("    " + row)
    print("    " + header)
    print(f"  {label} counts: ○{state['black']} ●{state['white']}  player={state['player']}")


def compare(move_idx, action, py, cpp, verbose=False):
    x, y = action % BOARD_DIM, action // BOARD_DIM
    row, col = BOARD_DIM - y, x + 1
    header = f"  after move {move_idx} action={action} ({row},{col})"
    diffs = []

    if py["stones"] != cpp["stones"]:
        diffs.append(("stones", py["stones"], cpp["stones"]))
    if py["black"] != cpp["black"] or py["white"] != cpp["white"]:
        diffs.append(("counts", None, None))

    if not diffs:
        return True

    if verbose:
        print(header)
        show_full_board("Python", py)
        show_full_board("C++   ", cpp)
        for name, pv, cv in diffs:
            if name in ("stones", "territory"):
                show_grid_diff(name, pv, cv)
            elif name == "counts":
                print(f"  counts  py=○{py['black']} ●{py['white']}  cpp=○{cpp['black']} ●{cpp['white']}")

    return False


# ---------------------------------------------------------------------------
# Per-recording test
# ---------------------------------------------------------------------------

def test_recording(path, verbose=False):
    with open(path) as f:
        rec = json.load(f)

    name = rec.get("name", os.path.basename(path))

    if rec.get("board_dim") != BOARD_DIM:
        print(f"SKIP [{name}]  board_dim={rec['board_dim']} != {BOARD_DIM}")
        return True

    moves = rec["moves"]

    py_game = Game(BOARD_DIM)
    py_game.addPlayer(User(), isBlack=True)
    py_game.addPlayer(User(), isBlack=False)

    cpp_game  = pyspiel.load_game("blitzgo")
    cpp_state = cpp_game.new_initial_state()

    ok = True
    for i, action in enumerate(moves):
        x, y = action % BOARD_DIM, action // BOARD_DIM
        py_result = py_game.placeStone((x, y))
        if py_result != 0:
            print(f"WARN [{name}] move {i} action={action} rejected by Python engine (result={py_result}), stopping replay")
            break
        cpp_state.apply_action(action)

        if cpp_state.is_terminal():
            break

        py  = py_extract(py_game)
        cpp = cpp_extract(cpp_state)
        if not compare(i, action, py, cpp, verbose=verbose):
            ok = False
            break

    status = "OK  " if ok else "FAIL"
    print(f"{status} [{name}] ({len(moves)} moves)")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve(path):
    if os.path.exists(path):
        return path
    name = path if path.endswith(".json") else f"{path}.json"
    candidate = os.path.join(RECORDINGS_DIR, name)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Recording not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs C++ BlitzGo engines on recordings.")
    parser.add_argument("recordings", nargs="*", help="Recording files or names (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full board at each failure")
    args = parser.parse_args()

    if args.recordings:
        paths = [resolve(r) for r in args.recordings]
    else:
        paths = sorted(
            os.path.join(RECORDINGS_DIR, f)
            for f in os.listdir(RECORDINGS_DIR)
            if f.endswith(".json")
        )

    results = [test_recording(p, verbose=args.verbose) for p in paths]
    passed = sum(results)
    total  = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
