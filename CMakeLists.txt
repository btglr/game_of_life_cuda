cmake_minimum_required(VERSION 3.17)
project(life CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(life_cuda_SM life_cuda_SM.cu)
add_executable(life_cuda_GM life_cuda_GM.cu)
add_executable(life_cuda_pitch life_cuda_pitch.cu)
add_executable(life_serial life_serial.c)
add_executable(life life.c)

set_target_properties(
        life_cuda_SM
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
        life_cuda_GM
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
        life_cuda_pitch
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
