# BlitzGo

A custom two-player territory game, built on top of OpenSpiel, with the goal of training an AlphaZero agent to play it.

BlitzGo is not Go. It has its own rules around territory, enclosures, and captures — designed from scratch.

---

## The Game

Two players (Black and White) take turns placing stones on a grid. The goal is to control territory.

- **Stones** count as territory for the player who placed them.
- **Enclosures** — surround a region with your stones and you claim everything inside. Enemy stones inside get captured and removed.
- If you enclose a region that was already claimed by your opponent, you take it.
- **Terminal condition** — the game ends when all claimed territory is stable (no cell has two or more diagonal enemy stones threatening it).

Territory score = your stones + your enclosed cells. Highest score wins.

---

## Setup

First time:

```bash
./install.sh
```

Build:

```bash
./open_spiel/scripts/build_and_run_tests.sh --build_only=true
```

Set your Python path so the bindings are found:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/blitzvision-openspiel
export PYTHONPATH=$PYTHONPATH:/path/to/blitzvision-openspiel/build/python
```

---

## Playing

```bash
python3 open_spiel/games/blitzgo/play.py
```

Enter moves as `row col` (1-indexed, matching what's printed on the board). Example: `4 3`.

### Record a game

```bash
# give it a name
python3 open_spiel/games/blitzgo/play.py --record mygame

# or let it generate one from the timestamp
python3 open_spiel/games/blitzgo/play.py --record
```

Saves to `open_spiel/games/blitzgo/recordings/<name>.json`.

### Replay a game

```bash
python3 open_spiel/games/blitzgo/play.py --replay recordings/mygame.json

# speed it up or slow it down
python3 open_spiel/games/blitzgo/play.py --replay recordings/mygame.json --sleep 0.3
```

Default sleep between moves is 1 second.

---

## Recording format

```json
{
  "version": 1,
  "board_dim": 5,
  "recorded_at": "2026-05-24T13:00:00",
  "name": "mygame",
  "moves": [12, 7, 24]
}
```

Moves are flat cell indices. To convert back to `(row, col)`:
- `row = BOARD_DIM - action // BOARD_DIM`
- `col = action % BOARD_DIM + 1`

---

## Board size

Set `BOARD_DIM` at the top of `open_spiel/games/blitzgo/blitzgo.h`. Rebuild after changing it.

---

## Project structure

```
open_spiel/games/blitzgo/
  blitzgo.h       — game constants, class declarations
  blitzgo.cc      — full game logic (placement, enclosures, territory, DFS)
  play.py         — interactive play, recording, replay
  recordings/     — saved games (created on first record)

python_blitzgo/
  cleaner_board.py — original Python prototype (reference)
```

---

## Goal

Train an AlphaZero agent (MCTS + ResNet) to play BlitzGo. The C++ OpenSpiel implementation is the environment. The observation tensor shape is `[5, BOARD_DIM, BOARD_DIM]` — 5 feature planes for the network input.

---

## Built with

- [OpenSpiel](https://github.com/google-deepmind/open_spiel) — game framework
- C++20, Python 3.11+, Clang
