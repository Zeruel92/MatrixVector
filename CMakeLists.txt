cmake_minimum_required(VERSION 3.20)
project(MatrixVector C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-O3 -DMAX_ITERATIONS=100 -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-D_DEBUG -DMAX_ITERATIONS=1 -Wall -Wextra")
set(CMAKE_C_COMPILER mpiicc)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(matrix_vector matrix_vector.c)
target_link_libraries(matrix_vector ${MPI_LIBRARIES})
