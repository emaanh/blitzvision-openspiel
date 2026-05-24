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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace blitzgo {
namespace {

namespace testing = open_spiel::testing;

void BasicBlitzGoTests() {
  testing::LoadGameTest("blitzgo");
  testing::RandomSimTest(*LoadGame("blitzgo"), /*num_sims=*/3);
  testing::RandomSimTest(*LoadGame("blitzgo(board_size=5)"), /*num_sims=*/3);
}

void ObservationTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("blitzgo(board_size=5)");
  std::unique_ptr<State> state = game->NewInitialState();

  // Verify tensor shape matches game declaration.
  std::vector<int> shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(shape.size(), 3);
  SPIEL_CHECK_EQ(shape[0], kNumObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], 5);
  SPIEL_CHECK_EQ(shape[2], 5);

  int tensor_size = game->ObservationTensorSize();
  std::vector<float> obs(tensor_size);
  state->ObservationTensor(0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(static_cast<int>(obs.size()), tensor_size);
}

void SuperkoTest() {
  // Verify that placing stones which recreate a prior board state is illegal.
  // TODO(you): add a concrete sequence that triggers superko once your
  //   capture rules are implemented.
  std::shared_ptr<const Game> game = LoadGame("blitzgo(board_size=5)");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

}  // namespace
}  // namespace blitzgo
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::blitzgo::BasicBlitzGoTests();
  open_spiel::blitzgo::ObservationTensorTest();
  open_spiel::blitzgo::SuperkoTest();
  return 0;
}
