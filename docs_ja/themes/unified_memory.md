# CUDA Unified Memory

English anchor: Unified Memory lets CPU and GPU use one managed pointer while the driver migrates pages.

> **日本語**
> Unified Memory は CPU と GPU が同じ pointer を使える memory model です。ただし、migration、prefetch、advice、同期を読まなければ performance と visibility を誤解します。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

Unified Memory は CPU と GPU が同じ pointer を使える memory model です。ただし、migration、prefetch、advice、同期を読まなければ performance と visibility を誤解します。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Unified Memory は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- Unified Memory では、どの thread/process/API が resource を所有するかを先に分けます。
- Unified Memory の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cudaMallocManaged / cudaFree` | managed allocation と解放です。 | CPU/GPU の両方で使う pointer lifetime を確認します。 |
| `cudaMemPrefetchAsync` | 指定 device へ pages を移動する hint です。 | stream ordering と access 直前の配置を確認します。 |
| `cudaMemAdvise` | access pattern hint です。 | read-mostly や preferred location の意図を読みます。 |
| `ManagedMemoryResource` | Python CUDA の managed allocator です。 | Buffer close と stream sync を対応させます。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [UnifiedMemoryStreams](../../cpp/0_Introduction/UnifiedMemoryStreams/README.ja.md): UM と stream ordering を読みます。
- [conjugateGradientUM](../../cpp/4_CUDA_Libraries/conjugateGradientUM/README.ja.md): solver flow と UM を読みます。
- [UnifiedMemoryPerf](../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md): UM performance modes を読みます。
- [blurImageUnifiedMemory](../../python/1_GettingStarted/blurImageUnifiedMemory/README.ja.md): Python UM workflow を読みます。
- [memoryResources](../../python/2_CoreConcepts/memoryResources/README.ja.md): ManagedMemoryResource を読みます。

## Representative Code

### `cpp/0_Introduction/UnifiedMemoryStreams/CMakeLists.txt`

Source: cpp/0_Introduction/UnifiedMemoryStreams/CMakeLists.txt:1-57
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(UnifiedMemoryStreams LANGUAGES C CXX CUDA)

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

# This sample is not supported on QNX
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(STATUS "Will not build sample ${PROJECT_NAME} - not supported on QNX")
    return()
endif()

# FindOpenMP: request COMPONENTS CXX only. From CMake 3.31 on, a bare find_package(OpenMP) also
# probes OpenMP for CUDA when the project enables CUDA; that check can fail even when C++ OpenMP
# works, and this sample only needs the C++ OpenMP package. Link OpenMP::OpenMP_CXX for libs/headers.
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
    # Add target for UnifiedMemoryStreams
    add_executable(UnifiedMemoryStreams UnifiedMemoryStreams.cu)

target_compile_options(UnifiedMemoryStreams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(UnifiedMemoryStreams PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(UnifiedMemoryStreams PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(UnifiedMemoryStreams PUBLIC
        # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
        CUDA::cublas
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "OpenMP not found - will not build sample 'UnifiedMemoryStreams'")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/UnifiedMemoryStreams/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/UnifiedMemoryStreams/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/4_CUDA_Libraries/conjugateGradientUM/CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/conjugateGradientUM/CMakeLists.txt:1-48
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(conjugateGradientUM LANGUAGES CXX)

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
# Add target for conjugateGradientUM
add_executable(conjugateGradientUM main.cpp)

target_compile_options(conjugateGradientUM PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(conjugateGradientUM PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(conjugateGradientUM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(conjugateGradientUM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(conjugateGradientUM PRIVATE
    CUDA::cudart
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::cublas
    CUDA::cusparse
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/conjugateGradientUM/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/4_CUDA_Libraries/conjugateGradientUM/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/6_Performance/UnifiedMemoryPerf/CMakeLists.txt`

Source: cpp/6_Performance/UnifiedMemoryPerf/CMakeLists.txt:1-41
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(UnifiedMemoryPerf LANGUAGES C CXX CUDA)

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
# Add target for UnifiedMemoryPerf
add_executable(UnifiedMemoryPerf helperFunctions.cpp matrixMultiplyPerf.cu commonKernels.cu)

target_compile_options(UnifiedMemoryPerf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(UnifiedMemoryPerf PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(UnifiedMemoryPerf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(UnifiedMemoryPerf PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/6_Performance/UnifiedMemoryPerf/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/6_Performance/UnifiedMemoryPerf/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. Unified Memory に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- Unified Memory の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- Unified Memory は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- Unified Memory に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Memory](memory.md): memory ownership を読みます。
- [Streams And Events](streams_events.md): prefetch ordering を読みます。
- [Performance](performance.md): migration cost を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
