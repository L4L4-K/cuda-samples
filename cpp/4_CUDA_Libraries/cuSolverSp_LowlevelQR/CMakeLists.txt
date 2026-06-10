# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

# JP: `cuSolverSp_LowlevelQR`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。 CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
project(cuSolverSp_LowlevelQR LANGUAGES C CXX)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for cuSolverSp_LowlevelQR
add_executable(cuSolverSp_LowlevelQR cuSolverSp_LowlevelQR.cpp mmio.c mmio_wrapper.cpp)

target_compile_options(cuSolverSp_LowlevelQR PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cuSolverSp_LowlevelQR PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuSolverSp_LowlevelQR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverSp_LowlevelQR PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverSp_LowlevelQR PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LowlevelQR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap2D_5pt_n32.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_command(TARGET cuSolverSp_LowlevelQR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap2D_5pt_n100.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_command(TARGET cuSolverSp_LowlevelQR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap3D_7pt_n20.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
