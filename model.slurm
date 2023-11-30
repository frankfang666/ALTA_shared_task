#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --partition=deeplearn
#SBATCH -A punim1846
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn
#SBATCH --job-name=MODEL
#SBATCH -o slurm.model.out
#SBATCH -e slurm.model.err
data_dir='/data/gpfs/projects/punim1846/yunhfang/MCS_Project_Local/alta2023_public_data/'
model=roberta_base
python Training.py -m ${model} -p ${data_dir}
