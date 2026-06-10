# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAdd_nvrtc LANGUAGES C CXX CUDA)

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
# Add sample target executable
add_executable(vectorAdd_nvrtc vectorAdd.cpp)

target_compile_options(vectorAdd_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(vectorAdd_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(vectorAdd_nvrtc PRIVATE
    # JP: nvrtc: NVRTC/JIT は実行時に device code を compile/link します。生成した module と kernel 名が launch と対応します。
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy vectorAdd_kernel.cu to the output directory
add_custom_command(TARGET vectorAdd_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
