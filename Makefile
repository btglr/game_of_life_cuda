CC=cc
NVCC=nvcc

FLAGS=-O3
FLAGS_CUDA=-O3 -arch=sm_60 \
               -gencode=arch=compute_60,code=sm_60 \
               -Xcompiler -fopenmp \

all: life life_cuda_SM life_cuda_GM life_cuda_pitch

life: life.c
	$(CC) $(FLAGS) life.c -o life.bin

life_cuda_GM: life_cuda_GM.cu
	$(NVCC) $(FLAGS_CUDA) life_cuda_GM.cu -o life_cuda_GM.bin

life_cuda_SM: life_cuda_SM.cu
	$(NVCC) $(FLAGS_CUDA) life_cuda_SM.cu -o life_cuda_SM.bin

life_cuda_pitch: life_cuda_pitch.cu
	$(NVCC) $(FLAGS_CUDA) life_cuda_pitch.cu -o life_cuda_pitch.bin

clean:
	rm -f *.bin *.out
