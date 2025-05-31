# Random Policy Training code (Is it fine to say 'training'??)
# No algorithms are used and every actions of each step is determined randomly
# Last Update : 2025.05.31
import numpy as np
import csv
import multiprocessing as mp
from environment.forest_env import ForestFireEnv, AGENT_NAMES

BATCH_SIZE = 400
MAX_STEPS_PER_EPISODE = 50
NUM_ITERATIONS = 10000
ITERATION_CSV = "./random_train_result/random_train_iteration.csv" # customize it
NUM_WORKERS = 8  # Or set manually

def random_actions(action_space):
    return {agent: action_space[agent].sample() for agent in AGENT_NAMES}

def run_batch(_):
    env = ForestFireEnv(max_steps=MAX_STEPS_PER_EPISODE)
    episode_reward_means = []
    steps_this_batch = 0

    while steps_this_batch < BATCH_SIZE:
        obs, info = env.reset()
        episode_rewards = {agent: 0.0 for agent in AGENT_NAMES}
        done = False
        step = 0
        while not done and step < MAX_STEPS_PER_EPISODE:
            actions = random_actions(env.action_spaces)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent in AGENT_NAMES:
                episode_rewards[agent] += rewards[agent]
            step += 1
            done = all(truncations.values())
        steps_this_batch += 1
        mean_reward = np.mean(list(episode_rewards.values()))
        episode_reward_means.append(mean_reward)
    # Return episode means for this batch (iteration)
    return episode_reward_means

def main():
    print(f"Running with {NUM_WORKERS} worker processes...")
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(run_batch, range(NUM_ITERATIONS))

    # Compute per-iteration means (average over each batch)
    iteration_means = [np.mean(batch) for batch in results]

    # Save iteration-level means
    with open(ITERATION_CSV, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['iteration', 'mean_reward'])
        for idx, reward in enumerate(iteration_means, 1):
            writer.writerow([idx, reward])

    print(f"\nSaved {len(iteration_means)} iteration mean rewards to {ITERATION_CSV}")

if __name__ == "__main__":
    main()
