# KillFire: Forest Fire Suppression Simulator

A multi-agent RL environment using PettingZoo and RLlib, simulating forest fire suppression by helicopter, drone, and groundcrew.

## Features

- 20x20 grid world: tree, fire, suppressed states
- 3 agents: helicopter, drone, groundcrew, each with unique suppression ability
- Fire spreads probabilistically each step
- Shared team reward for fire suppression and penalty for burning
- Designed for multi-agent RL (single shared policy)
- Matplotlib grid visualization

## Setup

```bash
pip install -r requirements.txt