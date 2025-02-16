#!/bin/bash
#SBATCH --job-name=mkizil19
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
##SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
##SBATCH --constraint=nvidia_a100
#SBATCH --constraint=tesla_v100
##SBATCH --constraint=tesla_t4
##SBATCH --constraint=nvidia_a40|tesla_v100
#SBATCH --mem=20G
#SBATCH --time=24:0:0
#SBATCH --output=logs/vidode/%J.log



echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo
n="$5"
##module load anaconda/3.6
##source
eval "$(micromamba shell hook --shell bash)"

micromamba activate OpenSTL
wandb login --relogin cc89c1f3f40cc95516b0af2df32cb6b924177898 ##Enes
echo 'number of processors:'$(nproc)
nvidia-smi
##3module load cuda/11.8.0   
###ule load cudnn/8.6.0/cuda-11.x
####python ttest.py


module load cuda/11.8.0

cd /kuacc/users/hpc-esanli/Vid-ODE
CUDA_VISIBLE_DEVICES=0 python main.py --phase train --diff_weight 0.1 --dataset penn --extrap --name "$n"
