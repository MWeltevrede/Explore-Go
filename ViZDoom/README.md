# Introduction
This repo is an adaptation of the [Sample Factory](https://github.com/alex-petrenko/sample-factory) library that includes an [implementation of E3B](https://github.com/facebookresearch/e3b/tree/main/minihack/sample-factory) exploration from [Exploration via Elliptical Episodic Bonuses, Henaff et al. 2022](https://arxiv.org/pdf/2210.05805) and has an implementation for the Explore-Go algorithm and the ViZDoom My Way Home scenario train-test split as used in the paper [Training on Irrelevant States Implies Data Augmentation: Generalization in Contextual MDPs, Weltevrede et al. 2026](https://arxiv.org/abs/2406.08069).


# Installation
Add your WandB API key to the recipe_vizdoom.def and build the container:
```
apptainer build slurm_scripts/vizdoom.sif recipe_vizdoom.def
```


# Usage
For an example of how to train locally, see script `run_train.sh`.

To run using slurm, adapt the `run_slurm_explorego.py` file to suit your system, copy the `slurm_scripts` directory to your cluster, and run the bash script `run_slurm.sh` for training.  


For evaluation, run `run_eval.sh` locally.