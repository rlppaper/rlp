#!/usr/bin/env python
import os

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import rlp


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument('-t', '--timesteps', type=int, default=2000000,
                        help='Number of timesteps during training')
    parser.add_argument('-g', '--gui', action='store_true',
                        help="Set this to enable GUI mode.")
    args = parser.parse_args()

    render_mode = "human" if args.gui else "rgb_array"

    log_dir = f"data/trained/{args.puzzle}_{args.arg}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"log_dir = {log_dir}")
    
    env = gym.make('rlp/Puzzle-v0', puzzle=args.puzzle,
                   render_mode=render_mode, params=args.arg)
    env = Monitor(env, log_dir, override_existing=True)
    max_timesteps = 10000
    env = TimeLimit(env, max_timesteps)
    
    # load the model we trained earlier
    model = PPO.load(f"data/PPO_{args.timesteps}/{args.puzzle}_{args.arg}/best_model_{args.puzzle}", env=env)

    episodes = 1000
    
    obs, info = env.reset()
    while episodes > 0:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            episodes -= 1
            obs, info = env.reset()
            if terminated:
                print("terminated")
            if truncated:
                print("truncated")
    env.close()

