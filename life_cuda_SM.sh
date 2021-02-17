#!/bin/bash
#SBATCH --error=jobs/life_cuda_SM.%J.err
#SBATCH --output=jobs/life_cuda_SM.%J.out
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:4
#SBATCH -p instant

#SBATCH --time=00:10:00

export OMP_NUM_THREADS=14

cd ~/CHPS0911/Projet\ -\ Marathon2016/2016/life || exit

printf "=== COMPILATION ===\n"

make

printf "\n=== Shared Memory Kernel ===\n"

(time ./life_cuda_SM.bin < judge.in > judge_cuda.out) 2>&1

RESULT=$(diff judge_cuda.out judge_serial.out)

if [ "$RESULT" == '' ]
  then
    printf "\nFILES ARE EQUAL"
  else
    printf "\nFILES ARE NOT EQUAL\n\n"
    echo "$RESULT"
fi