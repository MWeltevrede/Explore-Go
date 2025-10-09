from stable_baselines3 import UncertaintyDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.dqn.upolicies import UncertaintyMlpPolicy
from stable_baselines3.common.uncertainties import CountSAUncertainty, EpisodicCountSAUncertainty
from stable_baselines3.common.ubuffers import ExploreGoUncertaintyReplayBuffer

import torch as th
import torch.nn.functional as F
import numpy as np
import argparse
import uuid

from arch import *

import dill
from four_room.wrappers import gym_wrapper

import gymnasium as gym
from four_room.env import FourRoomsEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs='+', type=int, default=0, help="Provide the seeds for the agents to be trained")
parser.add_argument("--exp_frac", type=float, default=0.125)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--tau", type=float, default=0.05)
parser.add_argument("--u_tau", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--u_lr", type=float, default=0.001)
parser.add_argument("--arch_size", type=str, default='large')
parser.add_argument("--max_pure_expl_steps", type=int, default=0)
parser.add_argument('--include_pure_experience', action='store_true')
parser.add_argument('--e_greedy', action='store_true')
parser.add_argument('--num_training_levels', type=int, default=200)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--lam', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--small_rnd', action='store_true')

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

class BufferCoverageCallback(BaseCallback):
    """
    Custom callback for calculating the policy optimality and plotting it in tensorboard.
    """
    def __init__(self, freq, total_states, num_actions, verbose=0):
        super(BufferCoverageCallback, self).__init__(verbose)
        self.freq = freq
        self.num_actions = num_actions
        self.total_states = total_states
        self.total_state_actions = total_states*num_actions

        with open(CONFIGS_DIR + f'train_reachable_space.pl', 'rb') as file:
            self.data = dill.load(file)



    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            state_action_count = dict()
            for obs in self.data:
                state_action_count[hash(obs.data.tobytes())] = dict()

            if self.model.replay_buffer.full:
                for i, obs_stack in enumerate(self.model.replay_buffer.observations):
                    for j, obs in enumerate(obs_stack):
                        obs_hash = hash(obs.data.tobytes())
                        action_hash = self.model.replay_buffer.actions[i,j].data.tobytes()
                        if obs_hash in state_action_count:
                            # if this is not the case this is because it is a terminal state which is just in the buffer for bootstrapping reasons
                            if action_hash in state_action_count[obs_hash]:
                                state_action_count[obs_hash][action_hash] += 1
                            else:
                                state_action_count[obs_hash][action_hash] = 1
                        else:
                            print(obs[0] == obs[-1])
            else:
                for i, obs_stack in enumerate(self.model.replay_buffer.observations[:self.model.replay_buffer.pos]):
                    for j, obs in enumerate(obs_stack):
                        obs_hash = hash(obs.data.tobytes())
                        action_hash = self.model.replay_buffer.actions[i,j].data.tobytes()
                        if obs_hash in state_action_count:
                            # if this is not the case this is because it is a terminal state which is just in the buffer for bootstrapping reasons
                            if action_hash in state_action_count[obs_hash]:
                                state_action_count[obs_hash][action_hash] += 1
                            else:
                                state_action_count[obs_hash][action_hash] = 1
                        else:
                            print(obs[0] == obs[-1])

            zero_count = 0
            state_actions_missing = 0

            for k,v in state_action_count.items():
                if len(v.keys()) == 0:
                    zero_count += 1
                if len(v.keys()) < self.num_actions:
                    state_actions_missing += (self.num_actions - len(v.keys()))

            self.logger.record('train/state_coverage', (self.total_states - zero_count) / self.total_states)
            self.logger.record('train/state_action_coverage', (self.total_state_actions - state_actions_missing) / self.total_state_actions)



        return True

class PolicyOptimalityCallback(BaseCallback):
    """
    Custom callback for calculating the policy optimality and plotting it in tensorboard.
    """
    def __init__(self, freq, num_training_levels, verbose=0):
        super(PolicyOptimalityCallback, self).__init__(verbose)
        self.freq = freq
        self.num_training_levels = num_training_levels

        with open(CONFIGS_DIR + f'train_reachable_space.pl', 'rb') as file:
            data = dill.load(file)
        self.reachable_batch = th.as_tensor(data, device=th.device('cuda'))

        with open(CONFIGS_DIR + f'train_reachable_space_opt_actions.pl', 'rb') as file:
            self.optimal_actions = dill.load(file)

        with open(CONFIGS_DIR + f'obs_to_q_values_map.pl', 'rb') as file:
            self.obs_to_optimal_values = dill.load(file)


    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            max_action = []
            max_values = []
            num_batches = int(self.num_training_levels * 0.2)
            batch_size = self.reachable_batch.shape[0] // num_batches
            for start_ind in range(0, self.reachable_batch.shape[0], batch_size):
                with th.no_grad():
                    values = self.model.q_net(self.reachable_batch[start_ind:start_ind+batch_size])
                    max_action.append(values.max(dim=-1)[1].cpu().numpy())
                    max_values.append(values.max(dim=-1)[0].cpu().numpy())

            max_action = np.concatenate(max_action, axis=0)
            max_values = np.concatenate(max_values, axis=0)
            
            same_sum = 0
            value_diff = 0
            for i in range(self.reachable_batch.shape[0]):
                max_v = max_values[i].item()
                max_a = max_action[i].item()
                if max_a in self.optimal_actions[i]:
                    same_sum += 1

                opt_v = max(self.obs_to_optimal_values[self.reachable_batch[i].cpu().numpy().data.tobytes()])
                value_diff += abs(max_v - opt_v)

            self.logger.record('eval/policy_optimality', same_sum / self.reachable_batch.shape[0])
            self.logger.record('eval/policy_optimality_values', value_diff)

            if self.model.replay_buffer.full:
                states_in_buffer = np.unique(self.model.replay_buffer.observations.reshape(-1, *self.reachable_batch[0].shape), axis=0)
                max_action = []
                max_values = []
                batch_size = 2048
                start_ind = 0
                while start_ind < states_in_buffer.shape[0]:
                # for start_ind in range(0, states_in_buffer.shape[0], batch_size):
                    with th.no_grad():
                        values = self.model.q_net(th.as_tensor(states_in_buffer[start_ind:start_ind+batch_size], device=th.device('cuda')))
                        max_action.append(values.max(dim=-1)[1].cpu().numpy())
                        max_values.append(values.max(dim=-1)[0].cpu().numpy())
                    start_ind += batch_size

                max_action = np.concatenate(max_action, axis=0)
                max_values = np.concatenate(max_values, axis=0)
                
                same_sum = 0
                value_diff = 0
                for i in range(states_in_buffer.shape[0]):
                    max_v = max_values[i].item()
                    max_a = max_action[i].item()

                    opt_v = max(self.obs_to_optimal_values[states_in_buffer[i].data.tobytes()])
                    opt_a = np.argwhere(self.obs_to_optimal_values[states_in_buffer[i].data.tobytes()] == np.amax(self.obs_to_optimal_values[states_in_buffer[i].data.tobytes()])).flatten().tolist()
                    value_diff += abs(max_v - opt_v)
                    if max_a in opt_a:
                        same_sum += 1

                self.logger.record('eval/policy_optimality_buffer', same_sum / states_in_buffer.shape[0])
                self.logger.record('eval/policy_optimality_values_buffer', value_diff / states_in_buffer.shape[0])




        return True

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open(CONFIGS_DIR + f'train.pl', 'rb') as file:
	train_config = dill.load(file)

with open(CONFIGS_DIR + f'validation.pl', 'rb') as file:
    val_config = dill.load(file)

num_train_configs = len(train_config['topologies'])

config = {
    "exp_frac":args.exp_frac,
    "exploration_final_eps": 0.1,
    "learning_starts": 256,
    "buffer_size":500_000,
    "batch_size":256,
    "tau":args.tau,
    "u_tau": args.u_tau,
    "gamma":.99,
    "gradient_steps":args.gradient_steps,
    "target_update_interval":50,
    "train_freq":50,
    "learning_rate":args.lr,
    "u_learning_rate":args.u_lr,
    "n_envs":50,
    "max_grad_norm":1,
    "device":"cuda" if th.cuda.is_available() else "cpu",
    "max_pure_expl_steps": args.max_pure_expl_steps,
    "include_pure_experience": args.include_pure_experience,
    "double_q": False,
    "num_training_levels": args.num_training_levels,
    "split_uncertainty": True,
    "beta": args.beta,
    "lambda": args.lam,
    "alpha": args.alpha,
    "initialisation": 'orthogonal',
    "dont_bootstrap_terminal": True,
}

for seed in args.seeds:
    config['seed'] = seed
    ipe = config['include_pure_experience']
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

    if not args.e_greedy:
        global_uncertainty = CountSAUncertainty((8*8*4+3)*4*args.num_training_levels*3, train_env.observation_space.shape, device=config['device'])
        uncertainty = EpisodicCountSAUncertainty(config['n_envs'], 100, train_env.observation_space.shape, device=config['device'], global_uncertainty=global_uncertainty)
        replay_buffer_kwargs = dict(
                    uncertainty=uncertainty,
                    state_action_bonus=True,
                    uncertainty_of_sampling=False,
                    episodic_discount=True,
                    split_uncertainty=True,
                    include_pure_experience=args.include_pure_experience,
        )
        config['replay_buffer_kwargs'] = replay_buffer_kwargs
    else:
        uncertainty = "egreedy"
        replay_buffer_kwargs = dict(
                        uncertainty=uncertainty,
                        state_action_bonus=True,
                        uncertainty_of_sampling=False,
                        episodic_discount=False,
                        split_uncertainty=False,
        )
        config['replay_buffer_kwargs'] = replay_buffer_kwargs

    if args.arch_size == 'small':
        net_arch = []
    elif args.arch_size == 'large':
        net_arch = [256]
    policy_kwargs = dict(
                    features_extractor_class = CNN, 
                    features_extractor_kwargs = {'features_dim': 512, 'init_function': config['initialisation']}, 
                    normalize_images=False, 
                    net_arch=net_arch,
                    beta=config['beta'],
                    lam=config['lambda'],
                    alpha=config['alpha'],
                    u_lr=config["u_learning_rate"],
                    n_envs=config['n_envs'],
                    )
    config['policy_kwargs'] = policy_kwargs



    callback = EvalCallback(eval_env, n_eval_episodes=len(val_config['topologies']), eval_freq=max(100_000 // config["n_envs"], 1), verbose=0)
    callback_list = [callback]

    if args.include_pure_experience:
        save_path = LOGS_DIR + f"DQN/{args.include_pure_experience}_{args.max_pure_expl_steps}_{args.lam}_{args.alpha}_{config['seed']}/"
    else:
        save_path = LOGS_DIR + f"DQN/{args.beta}_{args.max_pure_expl_steps}_{args.lam}_{args.alpha}_{config['seed']}/"
    checkpoint_callback = CheckpointCallback(
        save_freq=max(4_000_000 // config["n_envs"], 1),
        save_path=save_path,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callback_list.append(checkpoint_callback)

    policy_callback = PolicyOptimalityCallback(100_000, config['num_training_levels'])
    callback_list.append(policy_callback)

    exp_callback = ExplorationCoverageCallback(log_freq=100_000, total_states=(8*8*4+3)*4*args.num_training_levels, num_actions=3)
    callback_list.append(exp_callback)

    buffer_callback = BufferCoverageCallback(freq=100_000, total_states=(8*8*4+3)*4*args.num_training_levels, num_actions=3)
    callback_list.append(buffer_callback)



    # Delete the following lines if you don't want to use wandb for logging results
    import wandb
    from wandb.integration.sb3 import WandbCallback
    with wandb.init(
            project="ExploreGo",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            tags=["FourRooms", "DQN"],
            config=config,
            ):
        wandb_callback = WandbCallback()

        model = UncertaintyDQN(
            UncertaintyMlpPolicy,
            train_env, 
            config['beta'],
            lam=config['lambda'],
            alpha=config['alpha'],
            uncertainty=uncertainty,
            learning_starts=config["learning_starts"],
            tensorboard_log=LOGS_DIR + "logging/", 
            policy_kwargs=policy_kwargs, 
            learning_rate=config["learning_rate"], 
            buffer_size=config["buffer_size"], 
            batch_size=config["batch_size"], 
            tau=config["tau"], u_tau=config["u_tau"], gamma=config["gamma"], 
            train_freq=(config["train_freq"] // config["n_envs"], "step"), 
            gradient_steps=config["gradient_steps"], 
            target_update_interval=config["target_update_interval"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exp_frac"],
            max_grad_norm=config["max_grad_norm"],
            seed=config["seed"],
            device=config["device"],
            max_pure_expl_steps=config["max_pure_expl_steps"],
            replay_buffer_class=ExploreGoUncertaintyReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            double_q=config["double_q"],
            )

        run_id = f"run_{uuid.uuid4()}_{config['seed']}"
        model.learn(total_timesteps=8_000_000, callback=callback_list, tb_log_name=run_id)
        train_env.close()
        eval_env.close()
