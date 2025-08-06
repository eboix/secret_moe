#!/bin/bash
#SBATCH --job-name=sae_batch
#SBATCH --output=sae_batch_in_%j.out
#SBATCH --error=sae_batch_in_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --account=bbjr-delta-gpu

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env2

# Change to the notebooks directory
cd /u/eboix/moe_distillation/sae_demo

srun python train_mlp_sae.py --save_dir ./saes --model_name EleutherAI/pythia-160m-deduped --io in --layers 6 --architectures top_k --use_wandb --num_tokens 50000000