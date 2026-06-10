# CUDA Runtime, Driver, And NVRTC

English anchor: Samples use Runtime API, Driver API, NVRTC, JIT linking, PTX, and NVVM at different abstraction levels.

> **日本語**
> Runtime API は便利な host API、Driver API は context/module/function handle を明示する API、NVRTC/JIT/NVVM は device code を compile/link する仕組みです。compile、load、lookup、launch を分けて読みます。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

Runtime API は便利な host API、Driver API は context/module/function handle を明示する API、NVRTC/JIT/NVVM は device code を compile/link する仕組みです。compile、load、lookup、launch を分けて読みます。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Runtime, Driver, And NVRTC は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- Runtime/Driver/NVRTC では、どの thread/process/API が resource を所有するかを先に分けます。
- Runtime/Driver/NVRTC の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cudaMalloc / kernel<<<>>>` | Runtime API の基本形です。 | 暗黙 context と error check を確認します。 |
| `cuInit / cuCtxCreate` | Driver API 初期化と context 作成です。 | CU* handle lifetime を追います。 |
| `cuModuleLoad / cuModuleGetFunction` | module と function handle を得ます。 | PTX/cubin と symbol name を確認します。 |
| `nvrtcCreateProgram / nvrtcCompileProgram` | runtime compilation です。 | compile option、log、generated code を確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [vectorAddDrv](../../cpp/0_Introduction/vectorAddDrv/README.ja.md): Driver API vectorAdd を読みます。
- [matrixMulDrv](../../cpp/0_Introduction/matrixMulDrv/README.ja.md): Driver API matrixMul を読みます。
- [vectorAdd_nvrtc](../../cpp/0_Introduction/vectorAdd_nvrtc/README.ja.md): NVRTC compile から launch までを読みます。
- [ptxjit](../../cpp/3_CUDA_Features/ptxjit/README.ja.md): PTX JIT を読みます。
- [simple](../../cpp/7_libNVVM/simple/README.ja.md): libNVVM flow を読みます。

## Representative Code

### `cpp/0_Introduction/vectorAddDrv/CMakeLists.txt`

Source: cpp/0_Introduction/vectorAddDrv/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAddDrv LANGUAGES C CXX CUDA)

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
```

> JP: この抜粋は `cpp/0_Introduction/vectorAddDrv/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/vectorAddDrv/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/0_Introduction/matrixMulDrv/CMakeLists.txt`

Source: cpp/0_Introduction/matrixMulDrv/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(matrixMulDrv LANGUAGES C CXX CUDA)

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
```

> JP: この抜粋は `cpp/0_Introduction/matrixMulDrv/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/matrixMulDrv/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。

### `cpp/0_Introduction/vectorAdd_nvrtc/CMakeLists.txt`

Source: cpp/0_Introduction/vectorAdd_nvrtc/CMakeLists.txt:1-47
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(vectorAdd_nvrtc LANGUAGES C CXX CUDA)

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
# Add sample target executable
add_executable(vectorAdd_nvrtc vectorAdd.cpp)

target_compile_options(vectorAdd_nvrtc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(vectorAdd_nvrtc PRIVATE cxx_std_17 cuda_std_17)

target_link_libraries(vectorAdd_nvrtc PRIVATE
    # JP: nvrtc: NVRTC/JIT は実行時に device code を compile/link します。生成した module と kernel 名が launch と対応します。
    CUDA::nvrtc
    CUDA::cuda_driver
)

# Copy vectorAdd_kernel.cu to the output directory
add_custom_command(TARGET vectorAdd_nvrtc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd_kernel.cu ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/0_Introduction/vectorAdd_nvrtc/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

> JP: `cpp/0_Introduction/vectorAdd_nvrtc/CMakeLists.txt` はこのテーマを読むための代表例です。API 名だけでなく、所有権、同期位置、検証位置を抜粋内で対応付けます。


## Reading Steps

1. Runtime/Driver/NVRTC に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- Runtime/Driver/NVRTC の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- Runtime/Driver/NVRTC は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- Runtime/Driver/NVRTC に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Execution Model](execution_model.md): launch shape を読みます。
- [Memory](memory.md): allocation lifetime を読みます。
- [Debugging, Profiling, And Testing](debugging_profiling_testing.md): compile log と error code を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
