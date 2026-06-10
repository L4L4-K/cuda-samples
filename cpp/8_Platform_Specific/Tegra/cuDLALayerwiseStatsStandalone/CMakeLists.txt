# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

# JP: `cuDLALayerwiseStatsStandalone`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
project(cuDLALayerwiseStatsStandalone LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)

find_package(NVSCI)
find_library(CUDLA_LIB cudla PATHS ${CUDAToolkit_LIBRARY_DIR} ${CMAKE_LIBRARY_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CUDLA_LIB)
        if(NVSCI_FOUND)
            # Source file
            # Add target for cuDLALayerwiseStatsStandalone
            add_executable(cuDLALayerwiseStatsStandalone main.cpp)

            target_compile_options(cuDLALayerwiseStatsStandalone PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(cuDLALayerwiseStatsStandalone PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(cuDLALayerwiseStatsStandalone PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(cuDLALayerwiseStatsStandalone PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                ${NVSCI_INCLUDE_DIRS}
            )

            target_link_libraries(cuDLALayerwiseStatsStandalone
                ${CUDLA_LIB}
                ${NVSCI_LIBRARIES}
            )
        else()
            message(STATUS "NvSCI not found - will not build sample 'cuDLALayerwiseStatsStandalone'")
        endif()
    else()
        message(STATUS "CUDLA not found - will not build sample 'cuDLALayerwiseStatsStandalone'")
    endif()
else()
    message(STATUS "Will not build sample cuDLALayerwiseStatsStandalone - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/InstallSamples.cmake)
setup_samples_install()
