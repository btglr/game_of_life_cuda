#!/bin/bash
#SBATCH --error=jobs/life_cuda_SM.%J.err
#SBATCH --output=jobs/life_cuda_SM.%J.out
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:4
#SBATCH -p instant

#SBATCH --time=00:10:00

cd ~/CHPS0911/Projet\ -\ Marathon2016/2016/life
make

echo '== Shared Memory Kernel =='
time ./life_cuda_SM.bin < judge.in