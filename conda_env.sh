#!/bin/bash
#SBATCH --job-name=esanli
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
##SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=72:0:0
#SBATCH --output=%J.log

##python infer.py
##python npy.py
python calculate_psnr.py