# train/train_rllib.py
# Basic training code can be used for hyperparameter optimization too
# Last Update : 2025.05.31

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

ITERATION = int(os.environ.get("ITERATION", 10000))
def env_creator(config):
    base_env = ForestFireEnv(max_steps=config.get("max_steps", 50))
    aec_env = parallel_to_aec(base_env)
    return PettingZooEnv(aec_env)

register_env("KillFire-v0", env_creator)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    config = (
        PPOConfig()
        .environment("KillFire-v0")
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=7)
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
        "training_iteration": ITERATION
    }

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        storage_path="../results"
        #storage_path="/Users/davisk/KillFire/results", # for MacOS
        #storage_path="/home/davisk/KillFire/results", # for Ubuntu
        checkpoint_at_end=True,
        checkpoint_freq=10,
        verbose=1
    )
    ray.shutdown()
'''
    Config for other ray version
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
