# main.py
'''
from environment.forest_env import ForestFireEnv, AGENT_NAMES
import time
import numpy as np

def random_policy(obs):
    # Suppress if standing on fire, else move randomly
    grid = obs[..., 0]
    moves = [0, 1, 2, 3, 5]  # Up, Down, Left, Right, Stay
    suppress = 4
    for agent in AGENT_NAMES:
        # center of grid (where agent starts)
        pos = np.argwhere(grid != 0)
        if pos.size > 0:
            # Prefer suppress if fire at current pos
            if grid[pos[0][0], pos[0][1]] == 1:
                return suppress
    return np.random.choice(moves)

if __name__ == "__main__":
    env = ForestFireEnv(max_steps=100, initial_fires=5)
    obs, infos = env.reset()
    done = False
    step = 0
    while not done:
        actions = {}
        for agent in env.agents:
            # Always suppress if fire at current pos, else random
            x, y = env.agent_positions[agent]
            if env.grid[x, y] == 1:
                actions[agent] = 4
            else:
                actions[agent] = np.random.randint(0, 6)
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        step += 1
        done = all(truncs.values()) or all(terms.values())
        time.sleep(0.3)
    print("Episode finished.")
'''
from environment.forest_env import ForestFireEnv, AGENT_NAMES
import time
import numpy as np
import csv
import os


EPISODES = 10
returns_log = []
for ep in range(EPISODES):
    # Make a subfolder for each episode
    save_dir = f"sim_results/ep_{ep+1:02d}"
    os.makedirs(save_dir, exist_ok=True)
    env = ForestFireEnv(max_steps=100, initial_fires=5)
    obs, infos = env.reset()
    done = False
    step = 0
    while not done:
        actions = {}
        for agent in env.agents:
            x, y = env.agent_positions[agent]
            if env.grid[x, y] == 1:
                actions[agent] = 4
            else:
                actions[agent] = np.random.randint(0, 6)
        obs, rewards, terms, truncs, infos = env.step(actions)
        # Pass the save_dir and step so each frame is saved
        env.render(mode="human", save_dir=save_dir)
        step += 1
        done = all(truncs.values()) or all(terms.values())
        # time.sleep(0.1)  # Optional: speed up or slow down rendering
    returns_log.append({'episode': ep+1, **env._cumulative_rewards})
    print(f"Episode {ep+1} finished.")

# Save results to CSV
with open('sim_results/returns_log.csv', 'w', newline='') as csvfile:
    fieldnames = ['episode'] + list(env._cumulative_rewards.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in returns_log:
        writer.writerow(row)

import imageio
import os

def make_gif_from_folder(folder, gif_name):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    images = [imageio.v2.imread(os.path.join(folder, f)) for f in files]
    imageio.mimsave(gif_name, images, duration=0.2)

# For each episode folder:
base_dir = "sim_results"
for ep in range(1, EPISODES+1):
    folder = os.path.join(base_dir, f"ep_{ep:02d}")
    gif_name = os.path.join(base_dir, f"ep_{ep:02d}.gif")
    make_gif_from_folder(folder, gif_name)