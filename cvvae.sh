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

eval "$(micromamba shell hook --shell bash)"

micromamba activate OpenSTL

echo 'number of processors:'$(nproc)
nvidia-smi


module load cuda/11.8.0

cd /kuacc/users/hpc-esanli/Vid-ODE
##python3 cvvae_inference_video.py \
##  --vae_path /kuacc/users/hpc-esanli/CV-VAE \
##  --root_dir  /datasets/tai-chi/taichi_png \
##  --save_dir  /kuacc/users/hpc-esanli/CV-VAE/output \
##  --height 128 \
##  --width 128 

python3 cvvae_inference_video.py \
  --vae_path /kuacc/users/hpc-esanli/CV-VAE \
  --root_dir  /kuacc/users/hpc-esanli/CV-VAE/Penn_Action \
  --save_dir  /kuacc/users/hpc-esanli/CV-VAE/output \
  --height 256 \
  --width 256 
