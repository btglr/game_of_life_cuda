#!/bin/bash
#SBATCH --error=jobs/cuda_GM/life_cuda_GM.%J.err
#SBATCH --output=jobs/cuda_GM/life_cuda_GM.%J.out
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
  make life_cuda_GM
fi

printf "\n=== Global Memory Kernel ===\n"

(time ./life_cuda_GM.bin < inputs/judge.in > outputs/judge_cuda_GM.out) 2>&1

RESULT=$(diff outputs/judge_cuda_GM.out outputs/judge_serial.out)

if [ "$RESULT" == '' ]
  then
    printf "\nFILES ARE EQUAL"
  else
    printf "\nFILES ARE NOT EQUAL\n\n"
    echo "$RESULT"
fi

if [ "$FULL_EXEC" == 1 ]; then
  printf "\n\n=== nvprof ===\n\n"
  (nvprof ./life_cuda_GM.bin < inputs/judge.in > /dev/null) 2>&1
fi

# Delete empty job files
find jobs/ -size 0 -delete