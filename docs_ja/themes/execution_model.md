# CUDA Execution Model

English anchor: CUDA samples launch work as grids of thread blocks, where each thread runs the same kernel code on different data.

> **日本語**
> CUDA の execution model は、host が kernel を投入し、device が grid、block、thread の階層で device code を実行するという考え方です。sample を読むときは、host から device work に渡る境界、各 thread が担当する data、host が完了を待つ位置を分けます。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

CUDA の execution model は、host が kernel を投入し、device が grid、block、thread の階層で device code を実行するという考え方です。sample を読むときは、host から device work に渡る境界、各 thread が担当する data、host が完了を待つ位置を分けます。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Execution Model は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- grid は全体の仕事、block は協調できる thread の単位、thread は個々の data を処理する単位です。
- block 間は通常同期できません。block 内だけが shared memory と barrier を共有できます。
- kernel launch は非同期なので、launch 行だけでは host から結果を読めるとは限りません。

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `kernel<<<grid, block, sharedMem, stream>>>()` | Runtime API の kernel launch 構文です。 | grid/block/shared memory/stream と後続同期を確認します。 |
| `cuLaunchKernel` | Driver API の launch です。 | CUfunction、argument 配列、shared memory、stream を対応させます。 |
| `cudaGetLastError` | 直近の launch 設定 error を確認します。 | 完了待ちではないため同期 API と分けます。 |
| `cudaDeviceSynchronize / cudaStreamSynchronize` | host が device work を待つ境界です。 | correctness 待ちか timing 待ちかを区別します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [vectorAdd](../../cpp/0_Introduction/vectorAdd/README.ja.md): one-thread-per-element pattern を読みます。
- [matrixMul](../../cpp/0_Introduction/matrixMul/README.ja.md): 2D grid と block 内協調を読みます。
- [simpleOccupancy](../../cpp/0_Introduction/simpleOccupancy/README.ja.md): launch shape と occupancy を読みます。
- [LargeKernelParameter](../../cpp/6_Performance/LargeKernelParameter/README.ja.md): kernel parameter と launch 境界を読みます。
- [vectorAdd](../../python/1_GettingStarted/vectorAdd/README.ja.md): Python launch wrapper と比較します。

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

### `cpp/0_Introduction/matrixMul/CMakeLists.txt`

Source: cpp/0_Introduction/matrixMul/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMul LANGUAGES C CXX CUDA)

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
# Add target for asyncAPI
add_executable(matrixMul matrixMul.cu)

target_compile_options(matrixMul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMul PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/matrixMul/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/matrixMul/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/0_Introduction/simpleOccupancy/CMakeLists.txt`

Source: cpp/0_Introduction/simpleOccupancy/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleOccupancy LANGUAGES C CXX CUDA)

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
# Add target for simpleOccupancy
add_executable(simpleOccupancy simpleOccupancy.cu)

target_compile_options(simpleOccupancy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleOccupancy PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleOccupancy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/simpleOccupancy/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/simpleOccupancy/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. kernel launch 行で grid、block、shared memory、stream をメモします。
2. kernel 内の `blockIdx`、`threadIdx`、`blockDim` が data index に変わる式を探します。
3. rounded-up grid の場合、out-of-range を防ぐ条件分岐を確認します。
4. launch 後の error check と同期 API を分けて読みます。
5. validation が同期後に置かれているか確認します。

## Common Mistakes

- launch が非同期なのに直後に host で結果を読む。
- thread count と element count を同じものとして扱う。
- block 間で `__syncthreads()` 相当の同期ができると思う。
- occupancy が高ければ常に速いと判断する。

## Performance Notes

- block size は memory coalescing、register pressure、occupancy、shared memory 使用量に影響します。
- small kernel を何度も launch する sample では launch overhead が支配的になることがあります。
- 測定前には warmup と同期範囲を確認します。

## Exercises

- vectorAdd の block size を変えたときの grid 計算式を説明する。
- matrixMul の 2D thread/block mapping を図にする。
- launch error と実行時 error が見える API を書き分ける。

## Cross-Theme Links

- [Kernel Launch And Indexing](kernel_indexing.md): index 計算を詳しく読みます。
- [Streams And Events](streams_events.md): launch をどの queue に入れるかを読みます。
- [Performance](performance.md): launch overhead と occupancy を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
