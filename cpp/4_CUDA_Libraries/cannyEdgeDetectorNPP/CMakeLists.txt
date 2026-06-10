# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(cannyEdgeDetectorNPP LANGUAGES CXX)

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
include_directories(
    ../../../Common
    ../../../Common/UtilNPP
)

# Source file
find_package(FreeImage)

if(${FreeImage_FOUND})
    # Add target for cannyEdgeDetectorNPP
    add_executable(cannyEdgeDetectorNPP cannyEdgeDetectorNPP.cpp)

target_compile_options(cannyEdgeDetectorNPP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cannyEdgeDetectorNPP PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(cannyEdgeDetectorNPP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(cannyEdgeDetectorNPP PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FreeImage_INCLUDE_DIRS}
    )

    target_link_libraries(cannyEdgeDetectorNPP PRIVATE
        # JP: `nppc`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
        CUDA::nppc
        CUDA::nppisu
        CUDA::nppif
        CUDA::cudart
        ${FreeImage_LIBRARIES}
    )

    # Copy data files to output directory
    add_custom_command(TARGET cannyEdgeDetectorNPP POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/teapot512.pgm
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(WIN32)
        add_custom_command(TARGET cannyEdgeDetectorNPP
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${FreeImage_LIBRARY}/../FreeImage.dll
        ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
        )
    endif()
else()
    message(STATUS "FreeImage not found - will not build sample 'cannyEdgeDetectorNPP'")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
