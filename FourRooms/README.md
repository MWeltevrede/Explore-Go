# Installation
Add your WandB API key to `recipe.def` and build the container:
```
	apptainer build explore_go.sif recipe.def
```
Generate additional data for analysis:
```
	apptainer run explore_go.sif python generate_reachable_data.py
```


# Usage
PPO (baseline)
```
	apptainer run --nv explore_go.sif python run_ppo.py --seeds 0
```

DQN (baseline)
```
	apptainer run --nv explore_go.sif python run_dqn.py --seeds 0
```

DQN+TEE (baseline)
```
	apptainer run --nv explore_go.sif python run_dqn.py --seeds 0 --beta 0.5 --lam 0.9 --alpha 25
```

PPO+Explore-Go (ours)
```
	apptainer run --nv explore_go.sif python run_ppo.py --seeds 0 --max_pure_expl_steps 50
```

DQN+Explore-Go (ours)
```
	apptainer run --nv explore_go.sif python run_dqn.py --seeds 0 --max_pure_expl_steps 50
```