#!/usr/bin/env python
import os
import fnmatch

import numpy as np
import gymnasium as gym
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env

import rlp

class SaveOnBestEpisodeLengthCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, puzzle_name:str, verbose: int = 1):
        super(SaveOnBestEpisodeLengthCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, f"best_model_{puzzle_name}")
        self.best_mean_length = np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, _ = ts2xy(load_results(self.log_dir), "timesteps")
            # compute exact episode lengths
            x[1:] = x[1:]-x[0:-1]
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_length = np.mean(x[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean length: {self.best_mean_length:.2f} - Last mean length per episode: {mean_length:.2f}")

                # New best model, you could save the agent here
                if mean_length < self.best_mean_length:
                    self.best_mean_length = mean_length
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

if __name__ == "__main__":

    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument('-t', '--timesteps', type=int, default=2000000,
                        help='Number of timesteps during training. Default: 2000000')
    parser.add_argument('-g', '--gui', action='store_true',
                        help="Set this to enable GUI mode.")
    args = parser.parse_args()

    render_mode = "human" if args.gui else "rgb_array"
    
    log_dir = f"data/PPO_{args.timesteps}/{args.puzzle}_{args.arg}/"
    os.makedirs(log_dir, exist_ok=True)


    env = make_vec_env('rlp/Puzzle-v0', 
                       env_kwargs=dict(puzzle=args.puzzle,
                                       render_mode=render_mode, 
                                       params=args.arg), 
                       n_envs=1, 
                       monitor_dir=log_dir)
    
    callback = SaveOnBestEpisodeLengthCallback(check_freq=1000, 
                                               log_dir=log_dir, 
                                               puzzle_name=args.puzzle) 

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    pygame.quit()

