cmake_minimum_required(VERSION 3.29)
project(cuda_mpi_openmp CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(cuda_mpi_openmp STATIC library.cu
		lab1/src/main.cu)

set_target_properties(cuda_mpi_openmp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
