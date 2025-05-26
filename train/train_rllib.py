# train/train_rllib.py

import os
import sys
import ray
import imageio
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.utils.conversions import parallel_to_aec
from environment.forest_env import ForestFireEnv
from ray.tune.registry import register_env

from environment.forest_env import ForestFireEnv, AGENT_NAMES
from ray.rllib.algorithms.callbacks import DefaultCallbacks

def env_creator(config):
    base_env = ForestFireEnv(max_steps=config.get("max_steps", 50))
    aec_env = parallel_to_aec(base_env)
    return PettingZooEnv(aec_env)

register_env("KillFire-v0", env_creator)
'''
from environment.render import render_grid


class ImageLogger(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Set up directory for images for this episode
        self.save_dir = f"./tmp/episode_{episode.episode_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.images = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        # WARNING: Accessing env internals. Works for single-env vector only!
        env = base_env.get_sub_environments()[0]
        grid = getattr(env, 'grid', None)
        agent_positions = getattr(env, 'agent_positions', None)
        #print(grid, agent_positions)
        if grid is not None and agent_positions is not None:
            step_num = episode.length
            render_grid(grid, agent_positions, step=step_num, save_dir=self.save_dir)
            self.images.append(f"{self.save_dir}/step_{step_num:03d}.png")
    
    def on_episode_end(self, *, worker, base_env, episode, env_index, **kwargs):
        # Make gif at the end of the episode
        if hasattr(self, 'images') and self.images:
            images = [imageio.imread(img) for img in self.images]
            gif_path = f"{self.save_dir}/episode_{episode.episode_id}.gif"
            imageio.mimsave(gif_path, images, duration=0.2)
            print(f"Saved episode GIF: {gif_path}")
            self.images = []
'''

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    '''
    config = (
        PPOConfig()
        .environment("KillFire-v0")
        .env_runners(num_env_runners=0)
        .framework("torch")
        .multi_agent(
            policies={
                "shared_policy": (None, (20, 20, 1), 6, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .rl_module(
            model_config={
                "encoder_config": {
                    "conv_filters":[
                        [16, [3, 3], 1],
                        [32, [3, 3], 1],
                        [64, [3, 3], 1],
                    ],
                    "conv_activation": "relu",
                    "post_fcnet_hiddens": [256, 128],
                    "post_fcnet_activation": "relu"
                }    
            }
        )
        .training(train_batch_size=400)
    )
    '''
    config = (
        PPOConfig()
        .environment("KillFire-v0")
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .training(
            model={
                "conv_filters": [
                    [16, [3, 3], 2],  # (18, 18, 16)
                    [32, [3, 3], 3],  # (8, 8, 32)
                    [64, [3, 3], 3],  # (3, 3, 64)
                ],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [256, 256],
                "post_fcnet_activation": "relu",
            },
            train_batch_size=400,
        )
        #.callbacks(ImageLogger)
    )
    stop = {
        "training_iteration": 100
    }

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        storage_path="/home/davisk/KillFire/results",
        checkpoint_at_end=True,
        checkpoint_freq=10,
        verbose=1
    )
    ray.shutdown()
