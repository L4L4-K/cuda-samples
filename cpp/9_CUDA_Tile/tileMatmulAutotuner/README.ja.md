# tileMatmulAutotuner - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA Tile C++ sample demonstrating how to autotune tile sizes and optimization hints for matrix multiplication with FP16 inputs and FP32 accumulation when compiling with nvrtc or nvcc.

The sample explores combinations of tile sizes and optimization hints, compiles the Tile C++ matrix multiplication kernel, executes it, and reports the best measured configuration. The launch grid is derived from the selected tile size and matrix dimensions. The search space is read from `autotuner_search_space.conf` so one can edit tile sizes and hint values without rebuilding the sample.

Original README headings: `tileMatmulAutotuner`, `Description`, `Running`, `Run autotuner. Validation is disabled by default.`, `Select a backend`, `Enable CPU validation`, `To run faster, skip warmups and set iteration to 1`, `Show all options`, `Command-Line Options`, `Search Space Configuration`, `Prerequisites`

> **日本語**
> `cpp/9_CUDA_Tile/tileMatmulAutotuner` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `tileMatmulAutotuner` as a focused example of the CUDA concepts used in `cpp/9_CUDA_Tile/tileMatmulAutotuner`.
> **日本語**
> この sample の目的は、`tileMatmulAutotuner` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Tensor Cores And WMMA, Shared Memory, Performance を具体的に追うことです。
>
> **学習メモ**
> 最初に `backend_common.h, backend_nvcc.h, backend_nvrtc.h, matmul.cu, matmul_autotuner.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `autotuner_search_space.conf`: Supporting file used by `autotuner_search_space.conf`.
- `backend_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `backend_nvcc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `backend_nvrtc.h`: Host/device declarations, helper types, constants, or library wrappers.
- `matmul.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `matmul_autotuner.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `backend_common.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
- Inside the kernel, map thread/block indexes to tile elements and check the barrier around shared memory reuse.
- Move, map, or expose input data so GPU work can read the intended values.
- Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.
- Synchronize only at the required correctness or timing boundary.
- Validate results against the CPU/reference path, generated artifact, or expected status message.
- Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.

> **日本語**
> 実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。
>
> **学習メモ**
> CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。

## Concrete Reading Path

- `backend_common.h`: focus on `nvrtc`, `launch`.
- `backend_nvcc.h`: focus on `CUDA`, `CUDA_INCLUDE_PATH`.
- `backend_nvrtc.h`: focus on `CUDA`, `nvrtc`, `nvrtcResult`, `nvrtcCreateProgram`, `nvrtcCompileProgram`.
- `matmul.cu`: focus on `CUDA`.
- `matmul_autotuner.cpp`: focus on `launch`, `CUDA`, `nvrtc`, `CUdeviceptr`, `cuMemAlloc`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/CMakeLists.txt:1-71
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(tileMatmulAutotuner LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

get_filename_component(CUDA_TOOLKIT_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program(TILEIRAS_EXECUTABLE tileiras
    HINTS "${CUDA_TOOLKIT_BIN_DIR}"
    REQUIRED
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --enable-tile")

if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common ../Benchmark_Common)

# Source files
add_executable(tileMatmulAutotuner
    matmul_autotuner.cpp
    autotuner_search_space.conf
)

target_compile_features(tileMatmulAutotuner PRIVATE cxx_std_20 cuda_std_20)

target_include_directories(tileMatmulAutotuner PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR)
file(TO_CMAKE_PATH "${CUDA_INCLUDE_DIR}" CUDA_INCLUDE_DIR_FOR_DEFINE)
file(TO_CMAKE_PATH "${TILEIRAS_EXECUTABLE}" TILEIRAS_EXECUTABLE_FOR_DEFINE)
file(TO_CMAKE_PATH "${CMAKE_CUDA_COMPILER}" NVCC_EXECUTABLE_FOR_DEFINE)

target_compile_definitions(tileMatmulAutotuner PRIVATE
    CUDA_INCLUDE_PATH="${CUDA_INCLUDE_DIR_FOR_DEFINE}"
    NVCC_PATH="${NVCC_EXECUTABLE_FOR_DEFINE}"
    TILEIRAS_PATH="${TILEIRAS_EXECUTABLE_FOR_DEFINE}"
)

target_link_libraries(tileMatmulAutotuner PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    # JP: nvrtc: NVRTC/JIT は実行時に device code を compile/link します。生成した module と kernel 名が launch と対応します。
    CUDA::nvrtc
)

add_custom_command(TARGET tileMatmulAutotuner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu
    ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/autotuner_search_space.conf
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `backend_common.h`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_common.h:29-47
```cpp
#pragma once

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <helper_string.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <process.h>
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_common.h:195-214
```cpp
    fprintf(stderr, "Error: %s:%d: %s\n", filename, line_number, message.c_str());
    exit(EXIT_FAILURE);
}

inline char *copyFilePath(const std::string& path) {
    char *file_path = reinterpret_cast<char *>(malloc(path.length() + 1));
    if (file_path == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for file path\n");
        exit(EXIT_FAILURE);
    }
    std::memcpy(file_path, path.c_str(), path.length() + 1);
    return file_path;
}

inline char *findSampleFile(const char *filename, const char *executable_path) {
    if (executable_path != NULL) {
        std::filesystem::path executable_dir =
            std::filesystem::path(executable_path).parent_path();
        if (!executable_dir.empty()) {
            std::filesystem::path candidate = executable_dir / filename;
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `backend_nvcc.h`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvcc.h:29-47
```cpp
#pragma once

#include "backend_common.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

inline CompiledKernel compileFileWithNVCC(const char *filename,
                                          int sm_value,
                                          int block_m, int block_n, int block_k,
                                          const std::vector<std::string>& extra_flags) {
    // Check CUDA include path for cuda_fp16.h
    const char *include_path = CUDA_INCLUDE_PATH;
    if (include_path[0] == '\0') {
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvcc.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `backend_nvrtc.h`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h:29-47
```cpp
#pragma once

#include "backend_common.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nvrtc.h>

#define NVRTC_SAFE_CALL(Name, x)                                             \
    do {                                                                     \
        nvrtcResult result = x;                                              \
        if (result != NVRTC_SUCCESS) {                                       \
            std::cerr << "\nerror: " << Name << " failed with error " <<     \
                      nvrtcGetErrorString(result);                           \
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h:152-165
```cpp
    }
    std::streampos pos = inputFile.tellg();
    size_t inputSize = pos;
    char * memBlock = new char [inputSize + 1];
    inputFile.seekg (0, std::ios::beg);
    inputFile.read (memBlock, inputSize);
    inputFile.close();
    memBlock[inputSize] = '\x0';

    // Compile the source string to PTX and Tile IR.
    // JP: この連続する anchor 群では NVRTC/JIT compile/link output です。compile option、log、生成 code と後続 module/kernel の対応 を確認します。
    nvrtcProgram prog;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, memBlock,
    "testprog", 0, NULL, NULL));
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h:172-191
```cpp
    log[logSize] = '\x0';
    std::cerr << "\n compilation log ---\n";
    std::cerr << log;
    std::cerr << "\n end log ---\n\n";
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(log);
    NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

    // Fetch Tile IR and compile it to cubin before benchmarking.
    size_t tileIRSize;
    // JP: この連続する anchor 群では NVRTC/JIT compile/link output です。compile option、log、生成 code と後続 module/kernel の対応 を確認します。
    NVRTC_SAFE_CALL("nvrtcGetTileIRSize", nvrtcGetTileIRSize(prog, &tileIRSize));
    std::vector<char> tileIR(tileIRSize);
    NVRTC_SAFE_CALL("nvrtcGetTileIR", nvrtcGetTileIR(prog, tileIR.data()));
    CompiledKernel kernel;
    kernel.image = compileTileIRToCubin(tileIR.data(), tileIR.size(), sm_value);
    NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));
    delete[] memBlock;
    return kernel;
}
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matmul.cu`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul.cu:43-57
```cuda
#include "cuda_tile.h"
#include <cuda_fp16.h>

namespace ct = cuda::tiles;

extern "C" __tile_global__ void matmul_tile(float* __restrict__ _C,
                                             const __half* __restrict__ _A,
                                             const __half* __restrict__ _B,
                                             int _M, int _N, int _K) {
    float* C = ct::assume_aligned<16>(_C);
    const __half* A = ct::assume_aligned<16>(_A);
    const __half* B = ct::assume_aligned<16>(_B);
    auto M = ct::assume_divisible<16>(_M);
    auto N = ct::assume_divisible<16>(_N);
    auto K = ct::assume_divisible<16>(_K);
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `matmul_autotuner.cpp`

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp:48-66
```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "backend_common.h"
#include "backend_nvcc.h"
#include "backend_nvrtc.h"
#include "matmul_benchmark.h"
#include <helper_cuda_drvapi.h>

// global SM value (compute capability)
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp:117-136
```cpp
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUdevice device;
    int major = 0, minor = 0;

    // initialize the CUDA Driver API
    checkCudaErrors(cuInit(0));

    // get the first device (device 0)
    checkCudaErrors(cuDeviceGet(&device, 0));
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    printf("GPU Compute Capability: %d.%d\n", major, minor);
    smValue = major * 10 + minor;
}

CompiledKernel compileFile(const char *filename,
                           int block_m, int block_n, int block_k,
                           CompilerBackend compiler_backend,
                           const std::vector<std::string>& extra_flags = {}) {
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp:158-177
```cpp
    };

    checkCudaErrors(cuModuleLoadData(&module, compiled_kernel.image.data()));

    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, kMatmulKernelName));
    // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                gridDimX, gridDimY, 1,  // grid dim
                1, 1, 1,                // block dim
                sMem, 0,                // shared mem, stream
                args,                   // arguments
                NULL));
    checkCudaErrors(cuCtxSynchronize());

    // cleanup
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuModuleUnload(module));
}

void autotuner(int M, int N, int K,
```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp:290-309
```cpp
    printf("  LOAD_LATENCY=%d, STORE_LATENCY=%d, grid_x=%d, grid_y=%d\n",
           best->load_latency, best->store_latency, best->grid_x, best->grid_y);
    printf("  Performance: %.1f GFLOPS, %.3f ms, %.1f GB/s\n",
           best->result.gflops, best->result.time_ms, best->result.bandwidth_gb_s);

    // JP: `cuMemFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
}

int main(int argc, char** argv) {
    std::vector<char*> benchmark_argv;
    CompilerBackend compiler_backend = parseCompilerBackendArgs(argc, argv, benchmark_argv);
    parse_benchmark_args(static_cast<int>(benchmark_argv.size()), benchmark_argv.data());
    print_device_info();

    // initialize CUDA and get compute capability
    setSMValue();

```

> JP: この抜粋は `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `nvrtc` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemAlloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuMemcpyHtoD` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuMemFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_INCLUDE_PATH` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `nvrtcResult` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `CUmodule` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `nvrtcCreateProgram` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `nvrtcCompileProgram` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `nvrtcGetProgramLogSize` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |
| `nvrtcGetProgramLog` | 実行時 compile/link の境界です。log、module、kernel name の対応を確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- Tensor Core sample では tile size、alignment、precision、accumulator の型が正しさと性能を決めます。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- matrixMul 系では global memory の値を shared memory tile に移し、barrier 後に再利用します。性能の主役は arithmetic だけでなく memory reuse です。
- cuBLAS/Tensor Core 版がある場合は、同じ数学でも API、data layout、precision、workspace の責任分担が変わります。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target tileMatmulAutotuner
ctest --test-dir build -R tileMatmulAutotuner
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `nvrtc` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Tensor Cores And WMMA](../../../docs_ja/themes/tensor_cores_wmma.md): Tensor Core、tile、precision、fragment の制約を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
