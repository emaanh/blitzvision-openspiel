#!/usr/bin/env python3
"""Play BlitzGo with alpha-beta search using territory as the leaf eval."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "build_rel", "python"))

import numpy as np
import pyspiel
from open_spiel.python.algorithms.minimax import alpha_beta_search

BOARD_DIM = 7
BOARD_CELLS = BOARD_DIM * BOARD_DIM
DEPTH = 6


def territory_eval(state):
    tensor = np.array(state.observation_tensor(0)).reshape(5, BOARD_DIM, BOARD_DIM)
    black = tensor[0].sum() + tensor[2].sum()
    white = tensor[1].sum() + tensor[3].sum()
    return black - white


def action_to_rc(action):
    return BOARD_DIM - action // BOARD_DIM, action % BOARD_DIM + 1


def rc_to_action(row, col):
    return (BOARD_DIM - row) * BOARD_DIM + (col - 1)


def main():
    game = pyspiel.load_game("blitzgo")
    state = game.new_initial_state()

    while not state.is_terminal():
        print(state)
        current = state.current_player()

        if current == 0:
            print(f"Black thinking (depth {DEPTH})...")
            t0 = time.time()
            _, action = alpha_beta_search(
                game,
                state=state,
                value_function=territory_eval,
                maximum_depth=DEPTH,
                maximizing_player_id=0,
            )
            elapsed = time.time() - t0
            row, col = action_to_rc(action)
            print(f"Black plays ({row}, {col})  time={elapsed:.1f}s")
        else:
            while True:
                try:
                    raw = input("White move (row col): ").strip().split()
                    row, col = int(raw[0]), int(raw[1])
                    action = rc_to_action(row, col)
                    if action in state.legal_actions():
                        break
                    print("Illegal move, try again.")
                except (ValueError, IndexError):
                    print("Enter two numbers e.g. '7 7'")

        state.apply_action(action)

    print(state)
    print("Returns:", state.returns())


if __name__ == "__main__":
    main()
