# rlp - Reinforcement Learning for Simon Tatham's Portable Puzzle Collection

## Description

This code supports the NeurIPS 2023 Submission _Reinforcement Learning Benchmark for Logic Puzzles_.

We provide `rlp`, a Reinforcement Learning (RL) environment based on [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/), designed for use with [Farama's Gymnasium](https://gymnasium.farama.org/) RL tools.

Along with `rlp`, you may find scripts that enable reproduction of the results presented in the paper. To this end, we give instructions on how to use them below.

## Installation Guide

### Requirements

[CMake](https://cmake.org/) 3.26, Python 3.10, a reasonably recent C compiler such as GCC or Clang

### Step-by-step Guide

First, clone the git repository to your local machine.

    git clone <repo>

Step into the directory.

    cd rlp

Install all required packages and build the C libraries.

    ./install.sh

Activate `rlp`'s virtual Python environment.

    source rlpvenv/bin/activate

## Usage Guide

After successfully following the Installation Guide, you can now run the `rlp` environment!

When initializing a puzzle, you must supply the desired puzzle's name. The names are expected to be all-lower case, one word. You may refer to `experiment_commands.txt` for the puzzle names, as well as the exact commands to reproduce the results presented in the paper.

### Train an Agent

In order to train an agent in a specific puzzle, run the following command in the repository's top level.

    ./run_training.py --puzzle <name of puzzle> --args <args>

Run `./run_training.py --help` for the full range of customizable options.

### Run a previously trained Agent

In order to run an agent previously trained on a specific puzzle, run the following command in the repository's top level.

    ./run_trained_agent.py --puzzle <name of puzzle> --arg <parameters>

Run `./run_trained_agent.py --help` for the full range of customizable options.

### Random Agent

To have an agent perform random actions in one of the puzzles, run the following command in the repository's top level:

    ./run_random.py --puzzle <name of puzzle> --arg <parameters>

Run `./run_random.py --help` for the full range of customizable options.

### Manual Play

To manually play one of the puzzles, run the following command in the repository's top level:

    ./run_puzzle.py --puzzle <name of puzzle>

Run `./run_puzzle.py --help` for the full range of customizable options.

## Developer Notes

### Custom Reward Structure

One can use a Gymnasium environment wrapper ([Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/)) to give out custom rewards in order to improve the agent's learning process.
The puzzle's internal game state is provided in the `info` dict created by the environment after each `step()`. Its attributes can be accessed using

    info['puzzle_state']['<attribute name>']

An example can be found in `custom_rewards_example.py`.

## License

The `rlp` code is released under the CC-BY-NC 4.0 license. For more information, see [LICENSE](LICENSE).

Simon Tatham's Portable Puzzle Collection is licensed under the MIT License, see [puzzles/LICENCE](puzzles/LICENCE).