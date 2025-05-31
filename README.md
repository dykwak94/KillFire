# KillFire: Forest Fire Suppression Simulator

A multi-agent RL environment using PettingZoo and RLlib, simulating forest fire suppression by helicopter, drone, and groundcrew.
📄 [View KillFire Report (PDF)](docs/KillFire_Report.pdf)

<img src="image_dump/KillFire_screenshot.png" alt="KillFire Screenshot" width="200"/>

## Features

- 20x20 grid world: tree, fire, suppressed states
- 3 agents: helicopter, drone, groundcrew, each with unique suppression ability
- Fire spreads probabilistically each step
- Shared team reward for fire suppression and penalty for burning
- Designed for multi-agent RL (single shared policy)
- Matplotlib grid visualization

## Setup

```bash
git clone https://github.com/dykwak94/KillFire.git
cd ~/KillFire
pip install -r requirements.txt
```
## Run default simulation
This shows the forest fire spreading model and three agents on the environment.\
It has no RL property used, just random simulation.
```bash
cd ~/KillFIre
PYTHONPATH=.. python main.py
```
## Train 
```bash
cd ~/KillFire/train
PYTHONPATH=.. python train_rllib.py
```
## Training Result (Batch size = 400, Iteration = 1000)
### Parameters
`FIRE_SPREAD_PROB = 0.01`\
`initial_fires=3`\
`SUPPRESSED_COEFF = 988 `\
`ONFIRE_COEFF = 2.83`\


<p align="center">
  <img src="Compare%20with%20Random_results/MAPPO%20vs%20Random%20%28m%3D988%2C%20n%3D2.83%29.png" alt="Comparison" height="300"/>
  &nbsp;&nbsp;&nbsp;
  <img src="image_dump/KillFire%20Demo.gif" alt="KillFire Demo" height="300"/>
</p>

