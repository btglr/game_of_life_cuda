#!/bin/bash
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:4
#SBATCH -p instant

#SBATCH --time=00:10:00

cd ~/CHPS0911/Projet\ -\ Marathon2016/2016/life
make

echo '== Global Memory Kernel =='
./life_cuda_GM judge.in judge_cuda_GM.txt

echo '== Shared Memory Kernel =='
./life_cuda_SM judge.in judge_cuda_SM.txt

echo '== Shared Memory Kernel + Pitch =='
./life_cuda_pitch judge.in judge_cuda_pitch.txt