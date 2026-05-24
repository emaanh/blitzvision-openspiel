// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/blitzgo/blitzgo.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace blitzgo {

constexpr uint8_t kWallNorth = 1 << 0;
constexpr uint8_t kWallSouth = 1 << 1;
constexpr uint8_t kWallWest  = 1 << 2;
constexpr uint8_t kWallEast  = 1 << 3;

inline int CountBits(uint8_t x) { return __builtin_popcount(x); }
inline Player Enemy(Player p) { return 1 - p; }
inline uint8_t PlayerToStone(Player p) { return static_cast<uint8_t>(p + 1); }
inline Player StoneToPlayer(uint8_t stone) {
  SPIEL_DCHECK_NE(stone, 0);
  return static_cast<Player>(stone - 1);
}

struct Side { int delta; uint8_t wall; bool on_edge; };
struct Direction { int seed; bool valid; uint8_t marker; };

namespace {

// ---- Game registration (do not modify) ------------------------------------

const GameType kGameType{
    /*short_name=*/"blitzgo",
    /*long_name=*/"BlitzGo",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {},
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<BlitzGoGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

// ---------------------------------------------------------------------------
// BlitzGoGame
// ---------------------------------------------------------------------------

BlitzGoGame::BlitzGoGame(const GameParameters& params)
    : Game(kGameType, params) {}

int BlitzGoGame::NumDistinctActions() const {
  return BOARD_CELLS;
}

std::unique_ptr<State> BlitzGoGame::NewInitialState() const {
  return std::make_unique<BlitzGoState>(shared_from_this());
}

std::vector<int> BlitzGoGame::ObservationTensorShape() const {
  // Shape expected by AlphaZero's convolutional network: [planes, rows, cols]
  return {kNumObservationPlanes, BOARD_DIM, BOARD_DIM};
}

int BlitzGoGame::MaxGameLength() const {
  return BOARD_CELLS * 2; //upper bound
}

// ---------------------------------------------------------------------------
// BlitzGoState
// ---------------------------------------------------------------------------

BlitzGoState::BlitzGoState(std::shared_ptr<const Game> game)
    : State(game), stones_{}, enclosures_{}, current_player_(0),
      outcome_(kInvalidPlayer), territory_{} {}

Player BlitzGoState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : current_player_;
}

std::vector<Action> BlitzGoState::LegalActions() const {
  //   Superko filtering goes here too.
  if(IsTerminal()) return {};
  std::vector<Action> legal_actions;
  legal_actions.reserve(BOARD_CELLS);
  for(int cell = 0; cell < BOARD_CELLS; cell++) {
    if(stones_[cell] == 0) legal_actions.push_back(cell);
  }
  return legal_actions;
}

void BlitzGoState::PlaceStone(int cell, Player player) {
  SPIEL_DCHECK_EQ(stones_[cell], 0);
  stones_[cell] = PlayerToStone(player);
  territory_[player]++;
}

void BlitzGoState::RemoveStone(int cell) {
  SPIEL_DCHECK_NE(stones_[cell], 0);
  Player owner = StoneToPlayer(stones_[cell]);
  stones_[cell] = 0;
  territory_[owner]--;
}

void BlitzGoState::ClaimEnclosure(int cell, Player player) {
  SPIEL_DCHECK_EQ(enclosures_[cell], 0);
  enclosures_[cell] = PlayerToStone(player);
  territory_[player]++;
}

void BlitzGoState::ReleaseEnclosure(int cell) {
  SPIEL_DCHECK_NE(enclosures_[cell], 0);
  territory_[StoneToPlayer(enclosures_[cell])]--;
  enclosures_[cell] = 0;
}

void BlitzGoState::SwitchPlayer() {
  current_player_ = Enemy(current_player_);
}

bool BlitzGoState::HasEnclosurePotential(int cell) const {
  return true;
}

bool BlitzGoState::IsRegionEnclosed(int start, uint8_t direction,
                                     uint8_t (&visited)[BOARD_CELLS]) {
  std::array<int, BOARD_CELLS> stack;
  int top = 0;
  uint8_t walls_touched = 0;
  const uint8_t my_stone = PlayerToStone(current_player_);
  const uint8_t enemy_stone = PlayerToStone(Enemy(current_player_));

  visited[start] = direction;
  stack[top++] = start;

  auto visit = [&](int n) -> bool {
    if (stones_[n] == my_stone) return true;  // own stone acts as wall
    if (visited[n] != 0 && visited[n] != direction) return false;
    if (visited[n] == 0) { visited[n] = direction; stack[top++] = n; }
    return true;
  };

  while (top > 0) {
    int c = stack[--top];
    int row = c / BOARD_DIM;
    int col = c % BOARD_DIM;

    const Side sides[4] = {
      { -BOARD_DIM, kWallNorth, row == 0           },
      { +BOARD_DIM, kWallSouth, row == BOARD_DIM-1 },
      { -1,         kWallWest,  col == 0            },
      { +1,         kWallEast,  col == BOARD_DIM-1  },
    };
    for (const Side& s : sides) {
      if (s.on_edge) walls_touched |= s.wall;
      else if (!visit(c + s.delta)) return false;
    }

    if (CountBits(walls_touched) >= 3) return false;
  }

  return true;
}

bool BlitzGoState::TryEnclose(int cell) {
  const uint8_t my_stone = PlayerToStone(current_player_);
  const uint8_t enemy_stone = PlayerToStone(Enemy(current_player_));

  if (enclosures_[cell] == my_stone) {
    ReleaseEnclosure(cell);
    return false;
  }

  uint8_t visited[BOARD_CELLS] = {};
  bool captured = false;
  int row = cell / BOARD_DIM;
  int col = cell % BOARD_DIM;

  const Direction dirs[4] = {
    { cell - BOARD_DIM, row > 0,           1 },
    { cell + 1,         col < BOARD_DIM-1, 2 },
    { cell + BOARD_DIM, row < BOARD_DIM-1, 3 },
    { cell - 1,         col > 0,           4 },
  };

  for (const Direction& dir : dirs) {
    if (!dir.valid) continue;
    if (stones_[dir.seed] == my_stone) continue;  // no region to enclose on own side
    if (!IsRegionEnclosed(dir.seed, dir.marker, visited)) continue;

    for (int c = 0; c < BOARD_CELLS; c++) {
      if (visited[c] != dir.marker) continue;
      if (enclosures_[c] == enemy_stone) ReleaseEnclosure(c);
      if (enclosures_[c] == 0) ClaimEnclosure(c, current_player_);
      if (stones_[c] == enemy_stone) {
        RemoveStone(c);
        captured = true;
      }
    }
  }

  return captured;
}

void BlitzGoState::DoApplyAction(Action action) {
  PlaceStone(action, current_player_);
  if (HasEnclosurePotential(action)) {
    bool captured = TryEnclose(action);
    bool penetrated = captured && enclosures_[action] == Enemy(current_player_);
    if (penetrated) {
      // emptyEnemyEnclosure(action); //update enclosures for the enemy with DFS
      std::cout << "penetrated" << std::endl;
    }
  }
  SwitchPlayer();
}

bool BlitzGoState::IsTerminal() const {
  return outcome_ != kInvalidPlayer;
}

std::vector<double> BlitzGoState::Returns() const {
  // TODO: return {black_result, white_result} using +1.0 / -1.0 / 0.0.
  return {0.0, 0.0};
}

void BlitzGoState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  // TODO: fill `values` with kNumObservationPlanes feature planes.
  //   values has size ObservationTensorSize() = planes * rows * cols.
  //   AlphaZero's ResNet reads this as [planes, board_size, board_size].
  std::fill(values.begin(), values.end(), 0.f);
}

//Generated by Claude.
std::string BlitzGoState::GridToString(
    const std::array<uint8_t, BOARD_CELLS>& grid) const {
  constexpr int kReserve = (BOARD_DIM + 2) * (BOARD_DIM * 4 + 8) + 64;
  std::string result;
  result.reserve(kReserve);

  absl::StrAppend(&result, "   ");
  for (int col = 1; col <= BOARD_DIM; ++col)
    absl::StrAppend(&result, col < 10 ? " " : "", std::to_string(col));
  absl::StrAppend(&result, "\n");

  for (int row = 0; row < BOARD_DIM; ++row) {
    int label = BOARD_DIM - row;
    absl::StrAppend(&result, label < 10 ? " " : "", std::to_string(label), "  ");
    for (int col = 0; col < BOARD_DIM; ++col) {
      uint8_t cell = grid[row * BOARD_DIM + col];
      if (cell == 0)      absl::StrAppend(&result, "\xC2\xB7 ");
      else if (cell == 1) absl::StrAppend(&result, "\xE2\x97\x8B ");
      else                absl::StrAppend(&result, "\xE2\x97\x8F ");
    }
    absl::StrAppend(&result, std::to_string(label), "\n");
  }

  absl::StrAppend(&result, "   ");
  for (int col = 1; col <= BOARD_DIM; ++col)
    absl::StrAppend(&result, col < 10 ? " " : "", std::to_string(col));
  absl::StrAppend(&result, "\n");

  return result;
}

std::string BlitzGoState::ToString() const {
  std::string result;
  absl::StrAppend(&result, GridToString(stones_));
  absl::StrAppend(&result, GridToString(enclosures_));
  absl::StrAppend(&result,
      current_player_ == 0 ? "\xE2\x97\x8B Black" : "\xE2\x97\x8F White",
      " to play  |  \xE2\x97\x8B ", std::to_string(territory_[0]),
      "  \xE2\x97\x8F ", std::to_string(territory_[1]), "\n");
  return result;
}

std::string BlitzGoState::ActionToString(Player player, Action action) const {
  int row = BOARD_DIM - action / BOARD_DIM;
  int col = action % BOARD_DIM + 1;
  return absl::StrCat("(", row, ",", col, ")");
}

std::string BlitzGoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return HistoryString();
}

std::string BlitzGoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToString();
}

std::unique_ptr<State> BlitzGoState::Clone() const {
  return std::make_unique<BlitzGoState>(*this);
}

}  // namespace blitzgo
}  // namespace open_spiel
