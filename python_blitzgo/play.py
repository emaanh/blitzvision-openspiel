import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from game import Game
from player import User

BOARD_DIM = 13
RECORDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "open_spiel", "games", "blitzgo", "recordings")
)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def action_to_position(action):
    """Flat action index → board.py (x, y) where y=0 is top row."""
    return action % BOARD_DIM, action // BOARD_DIM


def position_to_action(x, y):
    return y * BOARD_DIM + x


def rc_to_position(row, col):
    """Display (row, col) 1-indexed, row 13=top → board.py (x, y)."""
    return col - 1, BOARD_DIM - row


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def grid_to_string(grid, size):
    header = "   " + "".join(f"{c:2}" for c in range(1, size + 1))
    lines = [header]
    for y in range(size):
        label = size - y
        row = f"{label:2}  "
        for x in range(size):
            cell = grid[y][x]
            if cell is None:
                row += "· "
            elif cell.isBlack:
                row += "○ "
            else:
                row += "● "
        row += str(label)
        lines.append(row)
    lines.append(header)
    return "\n".join(lines)


def game_to_string(game):
    board = game.board
    black = next(p for p in game.players if p.isBlack)
    white = next(p for p in game.players if not p.isBlack)
    current = game.currPlayer()
    stones_str    = grid_to_string(board.stones,    board.size)
    territory_str = grid_to_string(board.territory, board.size)
    turn = "○ Black" if current.isBlack else "● White"
    counts = (f"{turn} to play  |  ○ {board.territory_counts[black]}"
              f"  ● {board.territory_counts[white]}")
    return stones_str + "\n" + territory_str + "\n" + counts


def print_result(game):
    black = next(p for p in game.players if p.isBlack)
    white = next(p for p in game.players if not p.isBlack)
    bc = game.board.territory_counts[black]
    wc = game.board.territory_counts[white]
    if bc > wc:
        print("Black wins.")
    elif wc > bc:
        print("White wins.")
    else:
        print("Draw.")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def new_recording(name):
    return {
        "version": 1,
        "board_dim": BOARD_DIM,
        "recorded_at": datetime.now().isoformat(),
        "name": name,
        "moves": [],
    }


def save_recording(recording, name):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = os.path.join(RECORDINGS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(recording, f, indent=2)
    print(f"Game saved to {path}")


def load_recording(path):
    with open(path) as f:
        return json.load(f)


def resolve_replay_path(path):
    if os.path.exists(path):
        return path
    name = path if path.endswith(".json") else f"{path}.json"
    candidate = os.path.join(RECORDINGS_DIR, name)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Recording not found: {path}")


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------

def make_game():
    game = Game(BOARD_DIM)
    game.addPlayer(User(), isBlack=True)
    game.addPlayer(User(), isBlack=False)
    return game


def play_human(record_name):
    recording = new_recording(record_name) if record_name else None
    game = make_game()

    try:
        while not game.checkGameOver():
            print(game_to_string(game))
            while True:
                try:
                    raw = input("Enter move (row col): ").strip().split()
                    row, col = int(raw[0]), int(raw[1])
                    if not (1 <= row <= BOARD_DIM and 1 <= col <= BOARD_DIM):
                        print("Out of bounds, try again.")
                        continue
                    x, y = rc_to_position(row, col)
                    result = game.placeStone((x, y))
                    if result == 0:
                        if recording is not None:
                            recording["moves"].append(position_to_action(x, y))
                        break
                    elif result == 2:
                        print("Repeated position, try again.")
                    else:
                        print("Illegal move, try again.")
                except (ValueError, IndexError):
                    print("Invalid input, enter two numbers e.g. '13 7'")

        print(game_to_string(game))
        print_result(game)
    except KeyboardInterrupt:
        print()

    if recording is not None and recording["moves"]:
        save_recording(recording, record_name)


def play_replay(path, sleep_secs):
    recording = load_recording(resolve_replay_path(path))
    stored_dim = recording.get("board_dim")
    if stored_dim != BOARD_DIM:
        sys.exit(
            f"Error: engine is {BOARD_DIM}x{BOARD_DIM} "
            f"but recording is {stored_dim}x{stored_dim}."
        )

    game = make_game()
    for action in recording["moves"]:
        print(game_to_string(game))
        time.sleep(sleep_secs)
        game.placeStone(action_to_position(action))

    print(game_to_string(game))
    print_result(game)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Play BlitzGo (Python engine)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", metavar="NAME", nargs="?", const="",
                      help="Record the game. Optionally provide a name.")
    mode.add_argument("--replay", metavar="FILE",
                      help="Replay a recorded game from FILE.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between moves during replay (default: 1.0).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replay:
        play_replay(args.replay, args.sleep)
    else:
        record_name = None
        if args.record is not None:
            record_name = args.record if args.record else datetime.now().strftime("%Y%m%d_%H%M%S")
        play_human(record_name)


if __name__ == "__main__":
    main()
