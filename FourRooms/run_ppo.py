from stable_baselines3 import ExploreGoPPO
from stable_baselines3.common.buffers import AsyncRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.uncertainties import CountSAUncertainty, EpisodicCountSAUncertainty

import torch as th
import argparse
import uuid

import dill
from four_room.wrappers import gym_wrapper
from arch import *

import gymnasium as gym
from four_room.env import FourRoomsEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs='+', type=int, default=0, help="Provide the seeds for the agents to be trained")
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--arch_size", type=str, default='large')
parser.add_argument('--num_training_levels', type=int, default=200)
parser.add_argument("--max_pure_expl_steps", type=int, default=0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--pure_exploration', action='store_true')
parser.add_argument("--init", type=str, default='orthogonal')

args = parser.parse_args()

CONFIGS_DIR = 'config/'
LOGS_DIR = 'logs/'


class ExplorationCoverageCallback(BaseCallback):
    def __init__(self, log_freq, total_states, num_actions, verbose=0):
        super(ExplorationCoverageCallback, self).__init__(verbose)
        self.state_action_coverage_set = set()
        self.log_freq = log_freq
        self.total_state_actions = total_states*num_actions

    def _on_step(self) -> bool:
        for i, obs in enumerate(self.locals['env'].buf_obs[None]):
            action = self.locals['actions'][i]
            self.state_action_coverage_set.add(hash((hash(obs.data.tobytes()), hash(action.data.tobytes()))))

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record('train/state_action_coverage_exploration', len(self.state_action_coverage_set) / self.total_state_actions)

        return True

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open(CONFIGS_DIR + f'train.pl', 'rb') as file:
	train_config = dill.load(file)

with open(CONFIGS_DIR + f'validation.pl', 'rb') as file:
    val_config = dill.load(file)

num_train_configs = len(train_config['topologies'])

config = {
    "batch_size":256,
    "clip_range":args.clip_range,
    "ent_coef":args.entropy_coef,
    "gamma":.99,
    "gae_lambda":0.95,
    "n_epochs":args.num_epochs,
    "n_steps": 256,
    "learning_rate":args.learning_rate,
    "n_envs":50,
    "device":"cuda" if th.cuda.is_available() else "cpu",
    "max_pure_expl_steps": args.max_pure_expl_steps,
    "beta": args.beta,
    "pure_beta": 0.1,
    "pure_exploration": args.pure_exploration,
    "initialisation": args.init,
    "pure_initialisation": args.init,
}

if args.arch_size == 'small':
    net_arch = []
elif args.arch_size == 'large':
    net_arch = [256]
policy_kwargs = dict(
    features_extractor_class = CNN, 
    features_extractor_kwargs = {'features_dim': 512, 'init_function': config['initialisation']}, 
    normalize_images=False, 
    net_arch=net_arch
    )
pure_policy_kwargs = dict(
    features_extractor_class = CNN, 
    features_extractor_kwargs = {'features_dim': 512, 'init_function': config['pure_initialisation']}, 
    normalize_images=False, 
    net_arch=net_arch
    )
config['policy_kwargs'] = policy_kwargs

for seed in args.seeds:
    config['seed'] = seed
    mpes = config['max_pure_expl_steps']
    eval_env = make_vec_env('MiniGrid-FourRooms-v1', 
                            n_envs=1, 
                            seed=config['seed'], 
                            vec_env_cls=DummyVecEnv, 
                            wrapper_class=gym_wrapper, 
                            env_kwargs={'agent_pos': val_config['agent positions'],
                                        'goal_pos': val_config['goal positions'],
                                        'doors_pos': val_config['topologies'],
                                        'agent_dir': val_config['agent directions'],
                                        'size':19,
                                        'max_steps':100},
                            wrapper_kwargs={'original_obs': True})

    train_env = make_vec_env('MiniGrid-FourRooms-v1', 
                            n_envs=config["n_envs"], 
                            seed=config['seed'], 
                            vec_env_cls=DummyVecEnv, 
                            wrapper_class=gym_wrapper, 
                            env_kwargs={'agent_pos': train_config['agent positions'],
                                        'goal_pos': train_config['goal positions'],
                                        'doors_pos': train_config['topologies'],
                                        'agent_dir': train_config['agent directions'],
                                        'size':19,
                                        'max_steps':100},
                            wrapper_kwargs={'original_obs': True})
    


    global_uncertainty = CountSAUncertainty((8*8*4+3)*4*args.num_training_levels*3, train_env.observation_space.shape, device=config['device'])
    uncertainty = EpisodicCountSAUncertainty(config['n_envs'], 100, train_env.observation_space.shape, device=config['device'], global_uncertainty=global_uncertainty)

    pure_global_uncertainty = CountSAUncertainty((8*8*4+3)*4*args.num_training_levels*3, train_env.observation_space.shape, device=config['device'])
    pure_uncertainty = EpisodicCountSAUncertainty(config['n_envs'], 100, train_env.observation_space.shape, device=config['device'], global_uncertainty=pure_global_uncertainty)

    callback_list = []
    eval_callback = EvalCallback(eval_env, n_eval_episodes=len(val_config['topologies']), eval_freq=max(100_000 // config["n_envs"], 1), verbose=0)
    callback_list.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(4_000_000 // config["n_envs"], 1),
        save_path=LOGS_DIR + f"PPO/{args.max_pure_expl_steps}_{config['seed']}/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callback_list.append(checkpoint_callback)

    exp_callback = ExplorationCoverageCallback(log_freq=100_000, total_states=(8*8*4+3)*4*args.num_training_levels, num_actions=3)
    callback_list.append(exp_callback)

    # Delete the following lines if you don't want to use wandb for logging results
    import wandb
    from wandb.integration.sb3 import WandbCallback
    with wandb.init(
            project="ExploreGo",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            tags=["FourRooms", "PPO"],
            config=config,
            ):
        wandb_callback = WandbCallback()

        model = ExploreGoPPO(
            'MlpPolicy',
            train_env, 
            beta=config["beta"],
            pure_beta=config["pure_beta"],
            uncertainty=uncertainty,
            pure_uncertainty=pure_uncertainty,
            tensorboard_log=LOGS_DIR + "logging/", 
            policy_kwargs=policy_kwargs, 
            pure_policy_kwargs=pure_policy_kwargs,
            rollout_buffer_class=AsyncRolloutBuffer,
            learning_rate=config["learning_rate"], 
            batch_size=config["batch_size"], 
            gamma=config["gamma"], 
            n_epochs=config["n_epochs"], 
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            gae_lambda=config["gae_lambda"],
            n_steps=config["n_steps"],
            seed=config['seed'],
            device=config["device"],
            max_pure_expl_steps=config["max_pure_expl_steps"],
            )

        run_id = f"run_{uuid.uuid4()}_{config['seed']}"
        model.learn(total_timesteps=8_000_000, callback=callback_list, tb_log_name=run_id)
        train_env.close()
        eval_env.close()
