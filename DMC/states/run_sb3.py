import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

import torch
#os.environ['MUJOCO_GL'] = 'glfw'

from stable_baselines3 import ExploreGoSAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from cdmc.env.wrappers import make_env

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
import shimmy

import cProfile, pstats
import json

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int)
parser.add_argument('--domain',type=str)
parser.add_argument('--task',type=str)
parser.add_argument('--exp_steps',type=int)
parser.add_argument('--wandb_name',type=str)
parser.add_argument('--episode_length',type=int)
parser.add_argument('--num_train_contexts',type=int)
parser.add_argument('--replay_buffer_size',type=int)
args = parser.parse_args()

domain_name = args.domain#"finger"
task_name = args.task#"turn_easy"
train_context_file = f"{domain_name}_{task_name}_contexts/{args.num_train_contexts}_contexts_startseed_{args.seed*args.num_train_contexts}.json"
test_context_file = f"{domain_name}_{task_name}_contexts/empty.json"
seed = args.seed
episode_length = args.episode_length
action_repeat = 4
n_eval_episodes = 100
eval_freq = 10_000
save_freq = 25_000


with open(train_context_file, 'r') as file:
		train_contexts = json.load(file)
with open(test_context_file, 'r') as file:
    test_contexts = json.load(file)
env = make_env(
    domain_name=domain_name,
    task_name=task_name,
    seed=seed,
    episode_length=episode_length,
    action_repeat=action_repeat,
    from_pixels=False,
    states=train_contexts['states'],
)
env = DummyVecEnv([lambda: env])
env = VecMonitor(env)

test_env = make_env(
    domain_name=domain_name,
    task_name=task_name,
    seed=seed+42,
    episode_length=episode_length,
    action_repeat=action_repeat,
    from_pixels=False,
    states=test_contexts['states'],
)

config = {
    "seed":seed,
    "use_sde":False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "ent_coef": "auto",
    "learning_starts": 10_000,
    "buffer_size":args.replay_buffer_size,
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
    "max_pure_expl_steps": args.exp_steps,
    "include_pure_experience": False,
}

policy_kwargs = {'net_arch': [256, 256], 'share_features_extractor': False, 'normalize_images':False}
config['policy_kwargs'] = policy_kwargs

with wandb.init(
        project=f"ContextualDMC",
        name=args.wandb_name,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    ):
    wandb_callback = WandbCallback()
    callback_list = [wandb_callback]

    model = ExploreGoSAC(
        "MlpPolicy", 
        env, 
        train_freq=config["train_freq"],
        policy_kwargs=config["policy_kwargs"],
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
        max_pure_expl_steps=args.exp_steps,
        )

    wandb.config['task']=args.domain+'_'+args.task
    wandb.config['num_train_contexts']=args.num_train_contexts
    wandb.config['replay_buffer_size']=args.replay_buffer_size
    wandb.config['episode_length']=args.episode_length

    eval_callback = EvalCallback(test_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, verbose=0)
    callback_list.append(eval_callback)

    # look up CheckpointCallback if you want to store network checkpoints or replay buffers during training 
    # https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
    checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=f"logs/{domain_name}_{task_name}/ep_length_{args.episode_length}_{args.num_train_contexts}_contexts/{config['max_pure_expl_steps']}_expl_steps/seed_{config['seed']}/",
    name_prefix="rl_model",
    save_replay_buffer=False,
    save_vecnormalize=True,
    )
    callback_list.append(checkpoint_callback)

    model.learn(total_timesteps=500_000, log_interval=5, callback=callback_list)
    env.close()
    test_env.close()
