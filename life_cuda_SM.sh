#!/bin/bash
#SBATCH --error=jobs/cuda_SM/life_cuda_SM.%J.err
#SBATCH --output=jobs/cuda_SM/life_cuda_SM.%J.out
#SBATCH -N 1
#SBATCH -c 14
#SBATCH --gres=gpu:2
#SBATCH -p instant

#SBATCH --time=00:10:00

if [ -z "$FULL_EXEC" ]; then
	FULL_EXEC=0
else
  FULL_EXEC=1
fi

if [ "$FULL_EXEC" == 1 ]; then
  printf "=== COMPILATION ===\n"

  module load cuda/11.0
  make life_cuda_SM
fi

printf "\n=== Shared Memory Kernel ===\n"

export OMP_NUM_THREADS=28
(time ./life_cuda_SM.bin < inputs/judge.in > outputs/judge_cuda.out) 2>&1

RESULT=$(diff outputs/judge_cuda.out outputs/judge_serial.out)

if [ "$RESULT" == '' ]
  then
    printf "\nFILES ARE EQUAL"
  else
    printf "\nFILES ARE NOT EQUAL\n\n"
    echo "$RESULT"
fi

if [ "$FULL_EXEC" == 1 ]; then
  printf "\n\n=== nvprof ===\n\n"
  (nvprof ./life_cuda_SM.bin < inputs/judge.in > /dev/null) 2>&1
fi

# Delete empty job files
find jobs/ -size 0 -delete