import os
os.environ['MUJOCO_GL'] = 'osmesa'
from stable_baselines3 import ExploreGoSAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from cdmc.eval import evaluate
from cdmc.env.wrappers import make_env
import pandas as pd
import numpy as np
import json
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int)
parser.add_argument('--explore_steps',type=int)
parser.add_argument('--domain_name',type=str)
parser.add_argument('--task_name',type=str)
parser.add_argument('--num_episodes',type=int)
parser.add_argument('--ep_length',type=int)
parser.add_argument('--path_prefix',type=str)
args = parser.parse_args()

domain_name = args.domain_name#'reacher'
task_name = args.task_name#'easy'
seed = args.seed
explore_steps = args.explore_steps
num_train_contexts=5
episode_length = args.ep_length

base_path = f'{args.path_prefix}/{domain_name}_{task_name}/ep_length_{episode_length}_{num_train_contexts}_contexts/{explore_steps}_expl_steps/'

num_episodes = args.num_episodes
action_repeat = 4

results, types, steps, seeds, explore_steps_list, contexts = [], [], [], [], [], []

train_context_file = f"{domain_name}_{task_name}_contexts/{num_train_contexts}_contexts_startseed_{seed*num_train_contexts}.json"
test_context_file = f"{domain_name}_{task_name}_contexts/empty.json"
with open(train_context_file, 'r') as file:
    train_contexts = json.load(file)
with open(test_context_file, 'r') as file:
    test_contexts = json.load(file)
train_env = make_env(
    domain_name=domain_name,
    task_name=task_name,
    seed=seed,
    episode_length=episode_length,
    action_repeat=action_repeat,
    from_pixels=False,
    states=train_contexts['states'],
)
train_env = DummyVecEnv([lambda: train_env])
test_env = make_env(
    domain_name=domain_name,
    task_name=task_name,
    seed=seed+42,
    episode_length=episode_length,
    action_repeat=action_repeat,
    from_pixels=False,
    states=test_contexts['states'],
)
test_env = DummyVecEnv([lambda: test_env])

config = {
    "seed":seed,
    "use_sde":False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "ent_coef": "auto",
    "learning_starts": 10_000,
    "buffer_size":100_000,
    "batch_size":128,
    "tau":0.005,
    "gamma":.99,
    "gradient_steps":1,
    "target_update_interval":1,
    "train_freq":1,
    "learning_rate":0.001,
    "n_envs":1,
    "max_grad_norm":1,
    "device":"cuda",
    "max_pure_expl_steps": explore_steps,
    "include_pure_experience": False,
}
base_model_train = ExploreGoSAC(
    "MlpPolicy", 
    train_env, 
    train_freq=config["train_freq"],
    buffer_size=config["buffer_size"], 
    learning_starts=config["learning_starts"], 
    learning_rate=config["learning_rate"], 
    tau=config["tau"],
    use_sde=config["use_sde"],
    sde_sample_freq=config["sde_sample_freq"],
    use_sde_at_warmup=config["use_sde_at_warmup"],
    ent_coef=config["ent_coef"],
    verbose=1, 
    device=config["device"],
    tensorboard_log="logs/tensorboard",
    max_pure_expl_steps=explore_steps,
)
base_model_test = ExploreGoSAC(
    "MlpPolicy", 
    test_env, 
    train_freq=config["train_freq"],
    buffer_size=config["buffer_size"], 
    learning_starts=config["learning_starts"], 
    learning_rate=config["learning_rate"], 
    tau=config["tau"],
    use_sde=config["use_sde"],
    sde_sample_freq=config["sde_sample_freq"],
    use_sde_at_warmup=config["use_sde_at_warmup"],
    ent_coef=config["ent_coef"],
    verbose=1, 
    device=config["device"],
    tensorboard_log="logs/tensorboard",
    max_pure_expl_steps=explore_steps,
)

for checkpoint in range(1,500_000//25_000+1):
    print("checkpoint: ", checkpoint, flush=True)
    train_agent = ExploreGoSAC.load(path=base_path+f'seed_{seed}/rl_model_{checkpoint*25_000}_steps',
                                env=train_env)
    test_agent = ExploreGoSAC.load(path=base_path+f'seed_{seed}/rl_model_{checkpoint*25_000}_steps',
                                env=test_env)
    step = 25_000 * checkpoint
    train_mean_reward = evaluate_policy(train_agent, train_env, num_episodes)
    steps.append(step);seeds.append(seed);explore_steps_list.append(explore_steps)
    types.append('train');results.append(train_mean_reward[0])
    print('mean train episode rewards: ', train_mean_reward, flush=True)

    test_mean_reward = evaluate_policy(test_agent, test_env, num_episodes)
    steps.append(step);seeds.append(seed);explore_steps_list.append(explore_steps)
    types.append('test');results.append(test_mean_reward[0])
    print('mean test episode rewards: ', test_mean_reward, flush=True)
        
df = pd.DataFrame({'results': results, 'type': types, 'steps': steps, 
                    'seeds':seeds, 'explore_steps':explore_steps_list})

out_file = f'checkpoint_data/len_{episode_length}/{domain_name}_{task_name}_{num_train_contexts}-contexts_{explore_steps}_expsteps_seed_{seed}_checkpoint_data.csv'
df.to_csv(out_file)