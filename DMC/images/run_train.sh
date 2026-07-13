#!/bin/bash


apptainer exec --nv cdmc.sif /bin/bash -c "python -m cdmc.train --max_pure_expl_steps=200 --episode_length=500 --save_freq=25k --buffer_size=100000 --batch_size=512 --num_shared_layers=4 --projection_dim=50 --train_steps=500k --algorithm=rad --log_dir=logs --seed=0 --domain_name=finger --task_name=turn_easy --train_context_file=finger_turn_easy_contexts/30_contexts_startseed_0.json --test_context_file=finger_turn_easy_contexts/empty.json"
