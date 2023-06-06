#!/usr/bin/env python
import os

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


import rlp


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument('-g', '--gui', action='store_true',
                        help="Set this to enable GUI mode.")
    args = parser.parse_args()

    render_mode = "human" if args.gui else "rgb_array"

    log_dir = f"data/random/{args.puzzle}_{args.arg}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"log_dir = {log_dir}")
    
    env = gym.make('rlp/Puzzle-v0', puzzle=args.puzzle,
                   render_mode=render_mode, params=args.arg)
    env = Monitor(env, log_dir, override_existing=False)

    max_timesteps = 10000
    episodes = 1000
    env = TimeLimit(env, max_timesteps) # truncate if we hit an episode's time limit
    
    observation, info = env.reset(seed=42)
    while episodes > 0:
        action = env.action_space.sample()  # a random policy
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            episodes -= 1
            if truncated:
                print("Episode truncated")
            if terminated:
                print("Episode terminated")
    env.close()

