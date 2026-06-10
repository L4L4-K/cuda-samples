# Multi-GPU, P2P, And IPC

English anchor: Multi-GPU samples coordinate memory, contexts, peer access, processes, and interconnect topology across devices.

> **日本語**
> multi-GPU/P2P/IPC sample は、複数 device、process、context の間で memory と work をどう分担するかを示します。device owner、peer capability、IPC handle、topology を先に分けます。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

multi-GPU/P2P/IPC sample は、複数 device、process、context の間で memory と work をどう分担するかを示します。device owner、peer capability、IPC handle、topology を先に分けます。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- Multi-GPU, P2P, And IPC は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- multi-GPU/P2P/IPC では、どの thread/process/API が resource を所有するかを先に分けます。
- multi-GPU/P2P/IPC の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cudaGetDeviceCount / cudaSetDevice` | device enumeration と selection です。 | current device を確認します。 |
| `cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess` | P2P capability と有効化です。 | device pair と error path を確認します。 |
| `cudaMemcpyPeerAsync` | device 間転送です。 | source/destination device と stream を確認します。 |
| `cudaIpcGetMemHandle / cudaIpcOpenMemHandle` | process 間 memory handle です。 | owner process と close/free を確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [simpleP2P](../../cpp/0_Introduction/simpleP2P/README.ja.md): peer access を読みます。
- [simpleIPC](../../cpp/0_Introduction/simpleIPC/README.ja.md): IPC handle を読みます。
- [simpleMultiGPU](../../cpp/0_Introduction/simpleMultiGPU/README.ja.md): work split を読みます。
- [MonteCarloMultiGPU](../../cpp/5_Domain_Specific/MonteCarloMultiGPU/README.ja.md): domain workload 分割を読みます。
- [multiGPUGradientAverage](../../python/4_DistributedComputing/multiGPUGradientAverage/README.ja.md): Python distributed GPU を読みます。

## Representative Code

### `cpp/0_Introduction/simpleP2P/CMakeLists.txt`

Source: cpp/0_Introduction/simpleP2P/CMakeLists.txt:1-41
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleP2P LANGUAGES C CXX CUDA)

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
# Add target for simpleP2P
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample simpleP2P - not supported on aarch64")
else()
    add_executable(simpleP2P simpleP2P.cu)

    target_compile_options(simpleP2P PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleP2P PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleP2P PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/simpleP2P/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/simpleP2P/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/0_Introduction/simpleIPC/CMakeLists.txt`

Source: cpp/0_Introduction/simpleIPC/CMakeLists.txt:1-41
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleIPC LANGUAGES C CXX CUDA)

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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Will not build sample simpleIPC - not supported on aarch64")
else()
    # Source file
    # Add target for simpleIPC
    add_executable(simpleIPC simpleIPC.cu ../../../Common/helper_multiprocess.cpp)

    target_compile_options(simpleIPC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleIPC PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleIPC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/simpleIPC/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/simpleIPC/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/0_Introduction/simpleMultiGPU/CMakeLists.txt`

Source: cpp/0_Introduction/simpleMultiGPU/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleMultiGPU LANGUAGES C CXX CUDA)

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
# Add target for simpleMultiGPU
add_executable(simpleMultiGPU simpleMultiGPU.cu)

target_compile_options(simpleMultiGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleMultiGPU PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleMultiGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/simpleMultiGPU/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/simpleMultiGPU/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. multi-GPU/P2P/IPC に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- multi-GPU/P2P/IPC の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- multi-GPU/P2P/IPC は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- multi-GPU/P2P/IPC に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Memory](memory.md): device ごとの owner を読みます。
- [Streams And Events](streams_events.md): device 間 copy ordering を読みます。
- [Performance](performance.md): P2P bandwidth/latency を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
