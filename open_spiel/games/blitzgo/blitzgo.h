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

#ifndef OPEN_SPIEL_GAMES_BLITZGO_H_
#define OPEN_SPIEL_GAMES_BLITZGO_H_

constexpr int BOARD_DIM = 13;
constexpr int BOARD_CELLS = BOARD_DIM * BOARD_DIM;

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace blitzgo {

inline constexpr int kNumPlayers = 2;
inline constexpr Player kBlack = 0;
inline constexpr Player kWhite = 1;

// Number of observation planes for the AlphaZero network input tensor.
// Adjust if you add more feature planes.
inline constexpr int kNumObservationPlanes = 5;


class BlitzGoState : public State {
 public:
  explicit BlitzGoState(std::shared_ptr<const Game> game);
  BlitzGoState(const BlitzGoState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::string GridToString(const std::array<uint8_t, BOARD_CELLS>& grid) const;
  void PlaceStone(int cell, Player player);
  void RemoveStone(int cell);
  void ClaimEnclosure(int cell, Player player);
  void ReleaseEnclosure(int cell);
  void SwitchPlayer();
  void ResolveSuicide();
  bool HasEnclosurePotential(int cell) const;
  bool TryEnclose(int cell);
  bool IsRegionEnclosed(int start, uint8_t marker, uint8_t (&visited)[BOARD_CELLS]) const;
  void ReleaseEnemyEnclosures(int cell);
  bool CheckInvariants() const;
  bool IsCellStable(int cell) const;
  Player ComputeOutcome() const;

  std::array<uint8_t, BOARD_CELLS> stones_;
  std::array<uint8_t, BOARD_CELLS> enclosures_;
  std::unordered_set<uint64_t> seen_positions_;
  Player current_player_;
  Player outcome_;
  int territory_[2];
  int suicide_cell_;
};

class BlitzGoGame : public Game {
 public:
  explicit BlitzGoGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;

};

}  // namespace blitzgo
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BLITZGO_H_
