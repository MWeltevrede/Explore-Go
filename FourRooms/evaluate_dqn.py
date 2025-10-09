import gymnasium as gym
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

import torch as th
import dill
import numpy as np
import time

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

from stable_baselines3 import UncertaintyDQN as DQN

t0 = time.time()

env_size = 19
original_obs = True

methods_to_evaluate = ["0.01_0_1.0_0.0", "0.01_50_1.0_0.0"]

logs_dir = "logs/"

seeds = [0]
at_steps = list(range(500,8_001,500))

method = []
step = []
winrate = []
version_data = []

with open(f'config/test.pl', 'rb') as file:
	test_config = dill.load(file)
num_test_configurations = len(test_config['topologies'])

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
								render_mode='rgb_array', 
								agent_pos=test_config['agent positions'],
								goal_pos=test_config['goal positions'],
								doors_pos=test_config['topologies'],
								agent_dir=test_config['agent directions'], size=env_size), original_obs=original_obs)

for meth in methods_to_evaluate:
	for v in seeds:
		for at_step in at_steps:
			file_name = f"{logs_dir}DQN/{meth}_{v}/rl_model_{at_step}000_steps"

			model = DQN.load(file_name, env=env)

			device = th.device('cuda')
			wins = 0

			for _ in range(num_test_configurations):
				obs, _ = env.reset()
				terminated = False
				truncated = False
				while not (terminated or truncated):
					state = th.as_tensor(obs, device=device).unsqueeze(0)
					with th.no_grad():
						action = model.q_net(state).max(dim=-1)[1].cpu().numpy()
					obs, reward, terminated, truncated, info = env.step(action)

					if terminated:
						wins += 1

			method.append(meth)
			step.append(at_step)
			winrate.append(wins / num_test_configurations)
			version_data.append(v)


data = {'method':method, 'step':step, 'winrate':winrate, 'version':version_data}

with open(f'data/test_data_dqn.pl', 'wb') as file:
	dill.dump(data, file)



method = []
step = []
winrate = []
version_data = []

with open(f'config/train.pl', 'rb') as file:
	test_config = dill.load(file)
num_test_configurations = len(test_config['topologies'])

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
								render_mode='rgb_array', 
								agent_pos=test_config['agent positions'],
								goal_pos=test_config['goal positions'],
								doors_pos=test_config['topologies'],
								agent_dir=test_config['agent directions'], size=env_size), original_obs=original_obs)

for meth in methods_to_evaluate:
	for v in seeds:
		for at_step in at_steps:
			file_name = f"{logs_dir}DQN/{meth}_{v}/rl_model_{at_step}000_steps"

			model = DQN.load(file_name, env=env)

			device = th.device('cuda')
			wins = 0

			for _ in range(num_test_configurations):
				obs, _ = env.reset()
				terminated = False
				truncated = False
				while not (terminated or truncated):
					state = th.as_tensor(obs, device=device).unsqueeze(0)
					with th.no_grad():
						action = model.q_net(state).max(dim=-1)[1].cpu().numpy()

					obs, reward, terminated, truncated, info = env.step(action)

					if terminated:
						wins += 1

			method.append(meth)
			step.append(at_step)
			winrate.append(wins / num_test_configurations)
			version_data.append(v)


data = {'method':method, 'step':step, 'winrate':winrate, 'version':version_data}

with open(f'data/train_data_dqn.pl', 'wb') as file:
	dill.dump(data, file)






with open(f'config/train.pl', 'rb') as file:
	training_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
								render_mode='rgb_array', 
								agent_pos=training_config['agent positions'],
								goal_pos=training_config['goal positions'],
								doors_pos=training_config['topologies'],
								agent_dir=training_config['agent directions'], size=env_size), original_obs=original_obs)

with open(f'config/train_reachable_space.pl', 'rb') as file:
	data = dill.load(file)
reachable_batch = th.as_tensor(data, device=th.device('cuda'))
reachable_state_space_size = reachable_batch.shape[0]

with open(f'config/train_reachable_space_opt_actions.pl', 'rb') as file:
	optimal_actions = dill.load(file)

with open(f'config/obs_to_q_values_map.pl', 'rb') as file:
	obs_to_optimal_values = dill.load(file)

non_optimal_states = dict()
method = []
step = []
version_data = []
difference_to_opt = []
difference_to_opt_value = []

for meth in methods_to_evaluate:
	for v in seeds:
		non_optimal_states[v] = list()
		for at_step in at_steps:
			file_name = f"{logs_dir}DQN/{meth}_{v}/rl_model_{at_step}000_steps"

			model = DQN.load(file_name, env=env)

			device = th.device('cuda')

			with th.no_grad():
				max_action = []
				max_values = []
				num_batches = int(200 * 0.2)
				batch_size = reachable_batch.shape[0] // num_batches
				for start_ind in range(0, reachable_batch.shape[0], batch_size):
					with th.no_grad():
						values = model.q_net(reachable_batch[start_ind:start_ind+batch_size])
						max_action.append(values.max(dim=-1)[1].cpu().numpy())
						max_values.append(values.max(dim=-1)[0].cpu().numpy())

				max_action = np.concatenate(max_action, axis=0)
				max_values = np.concatenate(max_values, axis=0)

			same_sum = 0
			value_diff = 0
			for i in range(reachable_state_space_size):
				max_a = max_action[i].item()
				if max_a in optimal_actions[i]:
					same_sum += 1
				else:
					non_optimal_states[v].append(i)

				opt_v = max(obs_to_optimal_values[reachable_batch[i].cpu().numpy().data.tobytes()])
				value_diff += abs(max_values[i].item() - opt_v)

			method.append(meth)
			step.append(at_step)
			difference_to_opt.append(same_sum / reachable_state_space_size)
			difference_to_opt_value.append(value_diff)
			version_data.append(v)


data = {'method':method, 'step':step, 'difference to optimal':difference_to_opt, 'version':version_data}

with open(f'data/train_diff_to_opt_data_dqn.pl', 'wb') as file:
	dill.dump(data, file)

data = {'method':method, 'step':step, 'difference to optimal value':difference_to_opt_value, 'version':version_data}

with open(f'data/train_diff_to_opt_value_data_dqn.pl', 'wb') as file:
	dill.dump(data, file)
