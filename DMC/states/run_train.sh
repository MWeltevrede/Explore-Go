#!/bin/bash

for i in {0..29}
do
	apptainer run --nv cdmc.sif \
	/bin/bash -c "\
		python run_sb3.py \
		--exp_steps 200 \
		--wandb_name ExploreGo \
		--replay_buffer_size 500_000 \
		--num_train_contexts 5 \
		--seed $i \
		--episode_length 500\
		--domain finger \
		--task turn_easy" 
done