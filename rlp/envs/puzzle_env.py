import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import numpy as np
import pygame

from rlp import puzzle as rp

FPS = 60

class PuzzleEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self,
                 puzzle: str,
                 render_mode: str = "human",
                 window_width: int = 128,
                 window_height: int = 128,
                 params: str | None = None,
                 n_envs: int = 1
    ):
        EzPickle.__init__(
            self,
            puzzle,
            render_mode,
            window_width,
            window_height,
            params,
            n_envs
        )

        super().__init__()
        self.n_envs = n_envs

        # Puzzle parameters
        self.puzzle_name = puzzle
        self.params = params
        # Pygame window dimensions
        self.window_width = window_width
        self.window_height = window_height

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(0, 255, (3, self.window_width, self.window_height), dtype=np.uint8),
            }
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the keyboard key to press
        I.e. 0 corresponds to "arrow_up", 1 to "arrow_down" etc.
        """
        action_keys = rp.api.specific.get_action_keys(puzzle)
        self._action_to_key = {i: key for i, key in zip(
            np.arange(len(action_keys)), action_keys)}

        # For the CTRL / SHIFT modifiers
        self.modifiers_down = np.zeros((2), dtype=bool)
        self.modifiers_value = 0

        self.action_space = spaces.Discrete(len(self._action_to_key))

        # In the future, the "undo" moves may be 
        # extended by an optional flag that removes the "undo" moves and 
        # forces the agent to never make any errors
        self.reward_from_game_status = {
            "-1": -100,
            "0": 0,
            "1": +100,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.puzzle = None
        self.clock = None
        self.ticks = 0

        self.puzzle = rp.Puzzle(self.puzzle_name, self.window_width,
                                self.window_height, self.params, False if self.render_mode == "human" else True)

    def _get_obs(self):
        return {
            "pixels": np.transpose(
                np.array(pygame.surfarray.pixels3d(self.puzzle.surf)), axes=(2, 1, 0)
            )
        }

    def _get_info(self):
        return {
            "puzzle_state": self.puzzle.get_puzzle_state()
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize a new puzzle
        self.puzzle.new_game()
        self.ticks = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action to the keyboard input
        key = self._action_to_key[action]

        # CTRL/SHIFT: Consider an action to be changing the status only,
        # but for the other keys, consider it as a press AND release

        if key == pygame.K_LCTRL or key == pygame.K_LSHIFT:
            modifier = 0 if key == pygame.K_LCTRL else 1
            self.modifiers_down[modifier] ^= 1
            self.modifiers_value = (pygame.KMOD_CTRL if self.modifiers_down[0] else 0) + (
                pygame.KMOD_SHIFT if self.modifiers_down[1] else 0)
        else:
            self.puzzle.process_key(pygame.KEYDOWN, key, self.modifiers_value)

        # Status: {+1: completed, 0: ongoing, -1: failed/died/stuck}
        status = self.puzzle.game_status()
        observation = self._get_obs()
        info = self._get_info()

        reward = self.reward_from_game_status[str(status)]
        
        # An episode is done iff the puzzle backend deems it
        # to be in a "solved" or "failed" state
        terminated = True if abs(status) == 1 else False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        if self.render_mode == "human":
            pygame.event.pump()
            # Delays the next frame to keep it relatively accurate to the desired framerate
            self.clock.tick(self.metadata["render_fps"])
        self.puzzle.proceed_animation()

        if not self.render_mode == "human":  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.puzzle.window)), axes=(2, 1, 0)
            )

    def close(self):
        if self.puzzle is not None:
            self.puzzle.destroy()
