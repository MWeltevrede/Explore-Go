#!/bin/bash

apptainer exec --nv --bind train_dir:/results/train_dir vizdoom.sif /bin/bash -c "python evaluate_milestones.py --env=my_way_home_train --env_frameskip=4 --num_workers=1 --num_envs_per_worker=1"
