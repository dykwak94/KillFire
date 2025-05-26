# environment/forest_env.py

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

# Agent settings
AGENT_NAMES = ['helicopter', 'drone', 'groundcrew']
AGENT_COLORS = {'helicopter': 'blue', 'drone': 'purple', 'groundcrew': 'black'}
AGENT_SUPPRESS_RANGE = {
    'helicopter': [(0,0), (-1,0), (1,0), (0,-1), (0,1)],  # current + 4-adjacent
    'drone': [(0,0), (1,0)],                             # current + right
    'groundcrew': [(0,0)]                                # current only
}

GRID_SIZE = 20
FIRE_SPREAD_PROB = 0.01
TREE, FIRE, SUPPRESSED = 0, 1, 2
CELL_STATE_NAMES = ['tree', 'fire', 'suppressed']

MOVE_MAP = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}

ACTION_MEANINGS = [
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "suppress",
    "stay"
]

class ForestFireEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "KillFire-v0"}

    def __init__(self, grid_size=GRID_SIZE, max_steps=50, initial_fires=3):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.initial_fires = initial_fires

        self.agents = AGENT_NAMES.copy()
        self.possible_agents = AGENT_NAMES.copy()
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=2, shape=(grid_size, grid_size, 3), dtype=np.float32
            ) for agent in self.agents
        }
        self.agent_positions = {}
        self.grid = None
        self.steps = 0
        self._rewards = None
        self._cumulative_rewards = None
        self._terminations = None
        self._truncations = None
        self._infos = None

    def reset(self, seed=None, options=None):
        self.agents = AGENT_NAMES.copy()
        self.agent_positions = {a: (self.grid_size // 2, self.grid_size // 2) for a in self.agents}
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # all trees

        # Randomly ignite initial fires
        fire_cells = set()
        rng = np.random.default_rng(seed)
        while len(fire_cells) < self.initial_fires:
            x, y = rng.integers(0, self.grid_size, 2)
            if (x, y) != (self.grid_size // 2, self.grid_size // 2):
                fire_cells.add((x, y))
        for x, y in fire_cells:
            self.grid[x, y] = FIRE

        self.steps = 0

        self._rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        self._infos = {agent: {} for agent in self.agents}
        obs = self._get_observations()
        return obs, self._infos

    def step(self, actions):
        assert set(actions.keys()) == set(self.agents)
        # Move phase
        new_positions = {}
        occupied = set(self.agent_positions.values())
        for agent in self.agents:
            act = actions[agent]
            pos = self.agent_positions[agent]
            if act in [0, 1, 2, 3]:  # Move
                dx, dy = MOVE_MAP[act]
                nx, ny = pos[0] + dx, pos[1] + dy
                # Stay within bounds and not move to occupied
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in occupied:
                        occupied.discard(pos)  # Does nothing if pos not in set
                        occupied.add((nx, ny))
                        new_positions[agent] = (nx, ny)
                    else:
                        new_positions[agent] = pos  # Can't move, blocked

                else:
                    new_positions[agent] = pos  # Out of bounds
            else:
                new_positions[agent] = pos  # Not a move action

        self.agent_positions = new_positions

        # Suppression phase
        fires_suppressed = 0
        for agent in self.agents:
            act = actions[agent]
            if act == 4:  # Suppress
                x, y = self.agent_positions[agent]
                for dx, dy in AGENT_SUPPRESS_RANGE[agent]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.grid[nx, ny] == FIRE:
                            self.grid[nx, ny] = SUPPRESSED
                            fires_suppressed += 1
        # Fire spread phase
        fire_cells = np.argwhere(self.grid == FIRE)
        new_fires = []
        for x, y in fire_cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx, ny] == TREE and np.random.rand() < FIRE_SPREAD_PROB:
                        new_fires.append((nx, ny))
        for x, y in new_fires:
            self.grid[x, y] = FIRE

        # Reward: +1 per fire suppressed, -1 * number of current fire cells
        reward = fires_suppressed - 0.3* np.count_nonzero(self.grid == FIRE)
        for agent in self.agents:
            self._rewards[agent] = reward
            self._cumulative_rewards[agent] += reward
        #print("step: ", self.steps, "reward: ", rewards)

        self.steps += 1
        trunc = self.steps >= self.max_steps
        for agent in self.agents:
            self._terminations[agent] = False
            self._truncations[agent] = trunc

        obs = self._get_observations()
        infos = self._infos
        terminations = self._terminations.copy()
        truncations = self._truncations.copy()
        rewards = self._rewards.copy()

        # Remove agents if the episode ends (but not in this env: they persist)
        if trunc:
            self.agents = []
        
        return obs, rewards, terminations, truncations, infos

    def _get_observations(self):
        obs = {}
        # Each agent sees the same grid, but PettingZoo expects per-agent obs
        obs_grid = self.grid[..., None].astype(np.float32).copy()
        for agent in self.agents:
            obs[agent] = obs_grid
        stacked_obs = np.concatenate([obs["helicopter"], obs["drone"], obs["groundcrew"]], axis=-1).astype(np.float32)
        return {agent: stacked_obs for agent in self.agents}

    
    def render(self, mode="human", save_dir=None):
        from .render import render_grid
        agent_pos = {a: self.agent_positions[a] for a in self.agents}
        render_grid(self.grid, agent_pos,step=self.steps, save_dir=save_dir)

    def close(self):
        pass

