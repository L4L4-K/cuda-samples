# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(cudaNvSciNvMedia)
add_subdirectory(cudaNvSciBufMultiplanar)
add_subdirectory(cuDLAErrorReporting)
add_subdirectory(cuDLAHybridMode)
add_subdirectory(cuDLALayerwiseStatsHybrid)
add_subdirectory(cuDLALayerwiseStatsStandalone)
add_subdirectory(cuDLAStandaloneMode)
add_subdirectory(EGLSync_CUDAEvent_Interop)
add_subdirectory(fluidsGLES)
add_subdirectory(nbody_opengles)
add_subdirectory(simpleGLES)
add_subdirectory(simpleGLES_EGLOutput)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
