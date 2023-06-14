# RLP - Reinforcement Learning Benchmark for Logic Puzzles

## Description

This code supports the NeurIPS 2023 Submission _Reinforcement Learning Benchmark for Logic Puzzles_.

We provide `RLP`, a Reinforcement Learning (RL) environment based on [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/), designed for use with [Farama's Gymnasium](https://gymnasium.farama.org/) RL tools.

Along with `RLP`, you may find scripts that enable reproduction of the results presented in the paper. To this end, we give instructions on how to use them below.

<img src='media/bridges_demo.gif' width='19.6%' />
<img src='media/flood_demo.gif' width='19.6%' />
<img src='media/filling_demo.gif' width='19.6%' />
<img src='media/rect_demo.gif' width='19.6%' />
<img src='media/dominosa_demo.gif' width='19.6%' />

## Installation Guide

### Requirements

[CMake](https://cmake.org/) 3.26, Python 3.10, a reasonably recent C compiler such as GCC or Clang.

We only tested the code on Linux.

### Step-by-step Guide

First, clone the git repository to your local machine.

```shell
git clone https://github.com/rlppaper/rlp.git
```

Step into the directory.

```shell
cd rlp
```

Install all required packages and build the C libraries.

```shell
./install.sh
```

Activate `RLP`'s virtual Python environment.

```shell
source rlpvenv/bin/activate
```

## Usage Guide

After successfully following the Installation Guide, you can now run the `RLP` environment!

When initializing a puzzle, you must supply the desired puzzle's name. Refer to the [list of puzzle names](#list-of-puzzles).

You may find the exact commands to reproduce the paper's experiments in `experiment_commands.txt`.

### Train an Agent

In order to train an agent in a specific puzzle, run the following command in the repository's top level.

```shell
./run_training.py --puzzle <name of puzzle> --arg <parameters>
```

Run `./run_training.py --help` for the full range of customizable options.

Check the [list of puzzle names](#list-of-puzzles).

### Run a previously trained Agent

In order to run an agent previously trained on a specific puzzle, run the following command in the repository's top level.

```shell
./run_trained_agent.py --puzzle <name of puzzle> --arg <parameters>
```

Run `./run_trained_agent.py --help` for the full range of customizable options.

Check the [list of puzzle names](#list-of-puzzles).

### Random Agent

To have an agent perform random actions in one of the puzzles, run the following command in the repository's top level:

```shell
./run_random.py --puzzle <name of puzzle> --arg <parameters>
```

Run `./run_random.py --help` for the full range of customizable options.

Check the [list of puzzle names](#list-of-puzzles).

### Manual Play

To manually play one of the puzzles, run the following command in the repository's top level:

```shell
./run_puzzle.py --puzzle <name of puzzle>
```

Run `./run_puzzle.py --help` for the full range of customizable options.

Check the [list of puzzle names](#list-of-puzzles).

### List of Puzzles

<table>
  <tbody>
    <tr>
      <td width='20%' align='center'>
        blackbox
      </td>
      <td width='20%' align='center'>
        bridges
      </td>
      <td width='20%' align='center'>
        cube
      </td>
      <td width='20%' align='center'>
        dominosa
      </td>
      <td width='20%' align='center'>
        fifteen
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        filling
      </td>
      <td width='20%' align='center'>
        flip
      </td>
      <td width='20%' align='center'>
        flood
      </td>
      <td width='20%' align='center'>
        galaxies
      </td>
      <td width='20%' align='center'>
        guess
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        inertia
      </td>
      <td width='20%' align='center'>
        keen
      </td>
      <td width='20%' align='center'>
        lightup
      </td>
      <td width='20%' align='center'>
        magnets
      </td>
      <td width='20%' align='center'>
        map
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        mines
      </td>
      <td width='20%' align='center'>
        mosaic
      </td>
      <td width='20%' align='center'>
        net
      </td>
      <td width='20%' align='center'>
        netslide
      </td>
      <td width='20%' align='center'>
        palisade
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        pattern
      </td>
      <td width='20%' align='center'>
        pearl
      </td>
      <td width='20%' align='center'>
        pegs
      </td>
      <td width='20%' align='center'>
        range
      </td>
      <td width='20%' align='center'>
        rect
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        samegame
      </td>
      <td width='20%' align='center'>
        signpost
      </td>
      <td width='20%' align='center'>
        singles
      </td>
      <td width='20%' align='center'>
        sixteen
      </td>
      <td width='20%' align='center'>
        slant
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        solo
      </td>
      <td width='20%' align='center'>
        tents
      </td>
      <td width='20%' align='center'>
        towers
      </td>
      <td width='20%' align='center'>
        tracks
      </td>
      <td width='20%' align='center'>
        twiddle
      </td>
    </tr>
    <tr>
      <td width='20%' align='center'>
        undead
      </td>
      <td width='20%' align='center'>
        unequal
      </td>
      <td width='20%' align='center'>
        unruly
      </td>
      <td width='20%' align='center'>
      </td>
      <td width='20%' align='center'>
      </td>
    </tr>
  </tbody>
</table>

## Developer Notes

### Custom Reward Structure

One can use a Gymnasium environment wrapper ([Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/)) to give out custom rewards in order to improve the agent's learning process.
The puzzle's internal game state is provided in the `info` dict created by the environment after each `step()`. Its attributes can be accessed using

```python
info['puzzle_state']['<attribute name>']
```

An example can be found in `custom_rewards_example.py`.

## License

The `RLP` code is released under the CC BY-NC 4.0 license. For more information, see [LICENSE](LICENSE).

Simon Tatham's Portable Puzzle Collection is licensed under the MIT License, see [puzzles/LICENCE](puzzles/LICENCE).
