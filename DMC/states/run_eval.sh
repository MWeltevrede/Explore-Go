DOMAIN_NAME=finger
TASK_NAME=turn_easy
EP_LENGTH=500
PATH_PREFIX=logs/
for STEPS in 200
do
    for SEED in {0..29}
    do  
        apptainer run --nv cdmc.sif \
            /bin/bash -c "python get_data_from_checkpoint.py \
                            --path_prefix $PATH_PREFIX \
                            --domain_name $DOMAIN_NAME \
                            --task_name $TASK_NAME \
                            --explore_steps $STEPS \
                            --seed $SEED \
                            --ep_length $EP_LENGTH \
                            --num_episodes 100 \
                            " > out
    done
done