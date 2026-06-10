# CUDA Memory

English anchor: CUDA samples move data among host memory, device memory, pinned memory, managed memory, arrays, mapped memory, and external resources.

> **日本語**
> CUDA memory を読む目的は、pointer がどこを指し、誰が所有し、どの processor からいつ見えるかを明確にすることです。`malloc`、`cudaMalloc`、`cudaMallocHost`、`cudaMallocManaged`、external memory は似た pointer に見えても lifetime と access rule が違います。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

CUDA memory を読む目的は、pointer がどこを指し、誰が所有し、どの processor からいつ見えるかを明確にすることです。`malloc`、`cudaMalloc`、`cudaMallocHost`、`cudaMallocManaged`、external memory は似た pointer に見えても lifetime と access rule が違います。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Memory は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- Host memory は CPU から自然に読めますが、GPU から直接読めるとは限りません。
- Device memory は GPU work の主な storage で、host は CUDA API を通して扱います。
- Pinned memory は DMA と async copy の前提になり、managed memory は CPU/GPU が同じ pointer を共有します。

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cudaMalloc / cudaFree` | device memory の確保と解放です。 | byte size と cleanup path を対応させます。 |
| `cudaMallocHost / cudaFreeHost` | page-locked host memory を確保します。 | async copy と overlap の前提を確認します。 |
| `cudaMemcpy / cudaMemcpyAsync` | host/device 間の visibility を作る転送です。 | direction enum と stream dependency を確認します。 |
| `cudaMallocManaged / cudaMemPrefetchAsync` | Unified Memory allocation と migration hint です。 | 同期と access order を読みます。 |
| `cudaExternalMemory / cudaGraphics*` | 外部 API が所有する resource を CUDA に見せます。 | map/unmap と owner lifetime を区別します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [vectorAdd](../../cpp/0_Introduction/vectorAdd/README.ja.md): H2D、kernel、D2H、free の基本形です。
- [UnifiedMemoryStreams](../../cpp/0_Introduction/UnifiedMemoryStreams/README.ja.md): Unified Memory と stream を読みます。
- [simpleZeroCopy](../../cpp/0_Introduction/simpleZeroCopy/README.ja.md): mapped host memory を読みます。
- [UnifiedMemoryPerf](../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md): managed memory performance を読みます。
- [memoryResources](../../python/2_CoreConcepts/memoryResources/README.ja.md): Python Buffer lifetime を読みます。

## Representative Code

### `cpp/0_Introduction/vectorAdd/CMakeLists.txt`

Source: cpp/0_Introduction/vectorAdd/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAdd LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Source file
# Add target for vectorAdd
add_executable(vectorAdd vectorAdd.cu)

target_compile_options(vectorAdd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(vectorAdd PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(vectorAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/vectorAdd/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/vectorAdd/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

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

### `cpp/0_Introduction/simpleZeroCopy/CMakeLists.txt`

Source: cpp/0_Introduction/simpleZeroCopy/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleZeroCopy LANGUAGES C CXX CUDA)

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
# Add target for simpleZeroCopy
add_executable(simpleZeroCopy simpleZeroCopy.cu)

target_compile_options(simpleZeroCopy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleZeroCopy PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleZeroCopy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/simpleZeroCopy/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/simpleZeroCopy/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. allocation API と解放 API を対応表にします。
2. copy/mapping/migration が input visibility を作る場所を探します。
3. kernel/library call が読む pointer と書く pointer を分けます。
4. validation 前に結果が host から見えるか確認します。
5. cleanup 前に非同期 work が完了しているか確認します。

## Common Mistakes

- element count を byte count として API に渡す。
- D2H と H2D の direction enum を取り違える。
- managed memory なら同期や prefetch が不要だと思う。
- external resource の owner を CUDA 側だと誤解する。

## Performance Notes

- pageable host memory では async copy の overlap が制限されることがあります。
- coalesced global memory access は thread index と data layout の両方で決まります。
- Unified Memory の page fault は timing に混ざるため warmup と prefetch を確認します。

## Exercises

- vectorAdd の host/device pointer を表にする。
- UnifiedMemoryPerf の data movement path を図にする。
- external memory sample で CUDA と外部 API の owner を分ける。

## Cross-Theme Links

- [Unified Memory](unified_memory.md): managed memory を詳しく読みます。
- [Streams And Events](streams_events.md): Async copy と stream ordering を読みます。
- [Performance](performance.md): memory bandwidth と measurement を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
