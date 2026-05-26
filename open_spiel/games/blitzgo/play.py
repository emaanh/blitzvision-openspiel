import argparse
import json
import os
import sys
import time
from datetime import datetime

import pyspiel

GAME = pyspiel.load_game("blitzgo")
BOARD_DIM = GAME.observation_tensor_shape()[1]
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")


def rc_to_action(row, col):
    r = BOARD_DIM - row
    c = col - 1
    return r * BOARD_DIM + c


def action_to_rc(action):
    row = BOARD_DIM - action // BOARD_DIM
    col = action % BOARD_DIM + 1
    return row, col


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
    filename = os.path.join(RECORDINGS_DIR, f"{name}.json")
    with open(filename, "w") as f:
        json.dump(recording, f, indent=2)
    print(f"Game saved to {filename}")


def load_recording(path):
    with open(path) as f:
        return json.load(f)


def play_human(record_name):
    recording = new_recording(record_name) if record_name else None
    state = GAME.new_initial_state()

    try:
        while not state.is_terminal():
            os.system("clear")
            print(state)
            legal = set(state.legal_actions())
            try:
                raw = input("Enter move (row col): ").strip().split()
                row, col = int(raw[0]), int(raw[1])
                action = rc_to_action(row, col)
                if action not in legal:
                    print("Illegal move, try again.")
                    continue
                state.apply_action(action)
                if recording is not None:
                    recording["moves"].append(action)
            except (ValueError, IndexError):
                print("Invalid input, enter two numbers e.g. '4 3'")

        os.system("clear")
        print(state)
        print("Game over. Returns:", state.returns())
    except KeyboardInterrupt:
        print()

    if recording is not None and recording["moves"]:
        save_recording(recording, record_name)


def resolve_replay_path(path):
    if os.path.exists(path):
        return path
    name = path if path.endswith(".json") else f"{path}.json"
    candidate = os.path.join(RECORDINGS_DIR, name)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Recording not found: {path}")


def play_replay(path, sleep_secs):
    recording = load_recording(resolve_replay_path(path))
    stored_dim = recording.get("board_dim")
    if stored_dim != BOARD_DIM:
        sys.exit(
            f"Error: C++ is compiled for {BOARD_DIM}x{BOARD_DIM} "
            f"but this recording is {stored_dim}x{stored_dim}. "
            f"Recompile with BOARD_DIM={stored_dim} or use a matching recording."
        )

    state = GAME.new_initial_state()

    for action in recording["moves"]:
        os.system("clear")
        row, col = action_to_rc(action)
        print(f"Replaying move: ({row}, {col})")
        print(state)
        time.sleep(sleep_secs)
        state.apply_action(action)

    os.system("clear")
    print(state)
    print("Game over. Returns:", state.returns())


def parse_args():
    parser = argparse.ArgumentParser(description="Play BlitzGo")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", metavar="NAME", nargs="?", const="",
                      help="Record the game. Optionally provide a name.")
    mode.add_argument("--replay", metavar="FILE",
                      help="Replay a recorded game from FILE.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between moves during replay (default: 1.0).")
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
