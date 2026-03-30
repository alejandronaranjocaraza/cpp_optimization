# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Name project
project(basic_optimization)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Find libs
find_package(Eigen3 REQUIRED)

# Include directories to look for header files
include_directories(include)

# Create executable "optimize" by compiling source files
add_executable(optimize
  main.cpp
  src/funcs.cpp
  src/grads.cpp
  src/optimization.cpp
)

# Include libs
target_include_directories(optimize PRIVATE ${EIGEN3_INCLUDE_DIR})


