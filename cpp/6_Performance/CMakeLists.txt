# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

add_subdirectory(LargeKernelParameter)
add_subdirectory(UnifiedMemoryPerf)
add_subdirectory(alignedTypes)
# JP: `cudaGraphsPerfScaling`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
add_subdirectory(cudaGraphsPerfScaling)
add_subdirectory(transpose)
