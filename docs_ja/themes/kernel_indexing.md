# Kernel Launch And Indexing

English anchor: CUDA kernels map grid, block, and thread coordinates to data indexes.

> **日本語**
> kernel indexing は、CUDA の並列実行を data layout に結び付ける場所です。1D vector、2D image、matrix tile、3D volume では index 式が変わりますが、目的は「この thread がどの要素を読む/書くか」を明確にすることです。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

kernel indexing は、CUDA の並列実行を data layout に結び付ける場所です。1D vector、2D image、matrix tile、3D volume では index 式が変わりますが、目的は「この thread がどの要素を読む/書くか」を明確にすることです。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- Kernel Launch And Indexing は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- `threadIdx` は block 内 coordinate、`blockIdx` は grid 内 block coordinate です。
- `blockDim` と `gridDim` は shape で、data size そのものではありません。
- global index は coordinate 変換であり、memory address とは stride や pitch を通して対応します。

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `blockIdx / threadIdx / blockDim / gridDim` | thread coordinate から global data index を作ります。 | 1D/2D/3D の式と boundary check を一緒に読みます。 |
| `dim3` | grid/block を multi-dimensional shape として表します。 | x/y/z の意味が data layout と一致しているか確認します。 |
| `cudaMallocPitch / cudaMemcpy2D` | pitch を持つ 2D memory を扱います。 | width と pitch の単位が byte か element かを分けます。 |
| `__launch_bounds__` | compiler に launch 上限や occupancy hint を与えます。 | performance hint であり correctness とは分けます。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [vectorAdd](../../cpp/0_Introduction/vectorAdd/README.ja.md): 1D indexing の基本です。
- [matrixMul](../../cpp/0_Introduction/matrixMul/README.ja.md): 2D matrix coordinate を読みます。
- [transpose](../../cpp/6_Performance/transpose/README.ja.md): coalescing と tile indexing を読みます。
- [reduction](../../cpp/2_Concepts_and_Techniques/reduction/README.ja.md): multiple elements per thread を読みます。
- [volumeFiltering](../../cpp/5_Domain_Specific/volumeFiltering/README.ja.md): 3D volume coordinate を読みます。

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

### `cpp/6_Performance/transpose/CMakeLists.txt`

Source: cpp/6_Performance/transpose/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(transpose LANGUAGES C CXX CUDA)

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
# Add target for transpose
add_executable(transpose transpose.cu)

target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(transpose PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/6_Performance/transpose/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/6_Performance/transpose/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. kernel の最初にある index 計算式を見つけます。
2. data shape、stride、pitch、leading dimension と index 式を対応させます。
3. write address が thread ごとに一意か、atomic が必要かを確認します。
4. boundary check が read と write の両方を守るか確認します。
5. CPU reference が同じ layout 前提か確認します。

## Common Mistakes

- 1D 式を 2D data にそのまま使う。
- pitch が byte 単位なのに element 単位で足す。
- read は guard しているが write の guard を忘れる。
- grid size を切り下げて末尾 data を処理しない。

## Performance Notes

- adjacent threads が adjacent addresses を読むと coalescing しやすくなります。
- 2D tile では x/y の割り当てが bank conflict に影響します。
- boundary branch は必要ですが、全 thread が複雑に分岐する設計は見直します。

## Exercises

- vectorAdd の global index を別 block size で手計算する。
- transpose の read/write address を thread 4 個分追う。
- matrixMul の row/col と A/B/C address 式を図にする。

## Cross-Theme Links

- [Execution Model](execution_model.md): grid/block/thread の実行単位を確認します。
- [Shared Memory](shared_memory.md): tile indexing と barrier を確認します。
- [Performance](performance.md): coalescing と occupancy を確認します。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
