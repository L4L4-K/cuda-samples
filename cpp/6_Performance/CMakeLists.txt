# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(LargeKernelParameter)
add_subdirectory(UnifiedMemoryPerf)
add_subdirectory(alignedTypes)
add_subdirectory(cudaGraphsPerfScaling)
add_subdirectory(transpose)
