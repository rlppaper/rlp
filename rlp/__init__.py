from gymnasium.envs.registration import register

import constants
import envs.puzzle_env
import puzzle

register(
    id="rlp/Puzzle-v0",
    entry_point="rlp.envs:PuzzleEnv",
)
