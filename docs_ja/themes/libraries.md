# CUDA Libraries

English anchor: CUDA libraries submit optimized device work through handles, descriptors, plans, and workspaces.

> **日本語**
> CUDA library sample では、kernel を自分で書かなくても library call が device work を投入します。handle、descriptor、plan、workspace、data layout、stream association、cleanup を対応させます。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

CUDA library sample では、kernel を自分で書かなくても library call が device work を投入します。handle、descriptor、plan、workspace、data layout、stream association、cleanup を対応させます。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Libraries は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- CUDA Libraries では、どの thread/process/API が resource を所有するかを先に分けます。
- CUDA Libraries の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cublasCreate / cublasDestroy` | cuBLAS handle の lifetime です。 | stream、math mode、pointer mode を確認します。 |
| `cufftPlan* / cufftExec*` | FFT plan と実行です。 | batch、layout、in-place/out-of-place を確認します。 |
| `cusparse descriptor APIs` | sparse descriptor を作ります。 | index base、format、workspace を確認します。 |
| `cusolverDn / cusolverSp` | solver routines です。 | workspace、info、pivot、validation を確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [simpleCUBLAS](../../cpp/4_CUDA_Libraries/simpleCUBLAS/README.ja.md): handle と operation を読みます。
- [matrixMulCUBLAS](../../cpp/4_CUDA_Libraries/matrixMulCUBLAS/README.ja.md): manual GEMM と cuBLAS を比較します。
- [simpleCUFFT](../../cpp/4_CUDA_Libraries/simpleCUFFT/README.ja.md): FFT plan を読みます。
- [cuSolverDn_LinearSolver](../../cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/README.ja.md): solver workspace を読みます。
- [nvJPEG](../../cpp/4_CUDA_Libraries/nvJPEG/README.ja.md): decoder state を読みます。

## Representative Code

### `cpp/4_CUDA_Libraries/simpleCUBLAS/CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/simpleCUBLAS/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleCUBLAS LANGUAGES CXX)

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
# Add target for simpleCUBLAS
add_executable(simpleCUBLAS simpleCUBLAS.cpp)

target_compile_options(simpleCUBLAS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCUBLAS PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(simpleCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simpleCUBLAS PRIVATE
    CUDA::cudart
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::cublas
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/simpleCUBLAS/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/4_CUDA_Libraries/simpleCUBLAS/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/4_CUDA_Libraries/matrixMulCUBLAS/CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/matrixMulCUBLAS/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulCUBLAS LANGUAGES CXX)

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
# Add target for matrixMulCUBLAS
add_executable(matrixMulCUBLAS matrixMulCUBLAS.cpp)

target_compile_options(matrixMulCUBLAS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(matrixMulCUBLAS PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(matrixMulCUBLAS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(matrixMulCUBLAS PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(matrixMulCUBLAS PRIVATE
    CUDA::cudart
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::cublas
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/matrixMulCUBLAS/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/4_CUDA_Libraries/matrixMulCUBLAS/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/4_CUDA_Libraries/simpleCUFFT/CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/simpleCUFFT/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleCUFFT LANGUAGES CUDA)

# Disable response file for libraries on QNX as qcc does not support lib paths with double quotes
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
endif()

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
# Add target for simpleCUFFT
add_executable(simpleCUFFT simpleCUFFT.cu)

target_compile_options(simpleCUFFT PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(simpleCUFFT PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(simpleCUFFT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(simpleCUFFT PRIVATE
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::cufft
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/simpleCUFFT/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/4_CUDA_Libraries/simpleCUFFT/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. CUDA Libraries に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- CUDA Libraries の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- CUDA Libraries は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- CUDA Libraries に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Memory](memory.md): input/output buffer を読みます。
- [Streams And Events](streams_events.md): library stream association を読みます。
- [Performance](performance.md): workspace と timing を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
