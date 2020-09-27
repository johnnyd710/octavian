cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(main VERSION 0.1.0)

set(Torch_DIR "/home/johnny/git/machine-learning-algorithms/deps/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(CTest)
enable_testing()

add_executable(main main.cpp lib)

target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
