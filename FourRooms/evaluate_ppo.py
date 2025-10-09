import gymnasium as gym
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper

import torch as th
import dill
import time

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

from stable_baselines3 import ExploreGoPPO

t0 = time.time()

env_size = 19
original_obs = True

methods_to_evaluate = ["0", "50"]

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
			file_name = f"{logs_dir}PPO/{meth}_{v}/rl_model_{at_step}000_steps"

			model = ExploreGoPPO.load(file_name, env=env)

			device = th.device('cuda')
			wins = 0

			for _ in range(num_test_configurations):
				obs, _ = env.reset()
				terminated = False
				truncated = False
				while not (terminated or truncated):
					state = th.as_tensor(obs, device=device).unsqueeze(0)
					with th.no_grad():
						action, _, _ = model.policy(state, deterministic=True)
						action = action.cpu().numpy()
					obs, reward, terminated, truncated, info = env.step(action)

					if terminated:
						wins += 1

			method.append(meth)
			step.append(at_step)
			winrate.append(wins / num_test_configurations)
			version_data.append(v)


data = {'method':method, 'step':step, 'winrate':winrate, 'version':version_data}

with open(f'data/test_data_ppo.pl', 'wb') as file:
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
			file_name = f"{logs_dir}PPO/{meth}_{v}/rl_model_{at_step}000_steps"

			model = ExploreGoPPO.load(file_name, env=env)

			device = th.device('cuda')
			wins = 0

			for _ in range(num_test_configurations):
				obs, _ = env.reset()
				terminated = False
				truncated = False
				while not (terminated or truncated):
					state = th.as_tensor(obs, device=device).unsqueeze(0)
					with th.no_grad():
						action, _, _ = model.policy(state, deterministic=True)
						action = action.cpu().numpy()

					obs, reward, terminated, truncated, info = env.step(action)

					if terminated:
						wins += 1

			method.append(meth)
			step.append(at_step)
			winrate.append(wins / num_test_configurations)
			version_data.append(v)


data = {'method':method, 'step':step, 'winrate':winrate, 'version':version_data}

with open(f'data/train_data_ppo.pl', 'wb') as file:
	dill.dump(data, file)

print(f"Evaluation done in {time.time() - t0}s")
