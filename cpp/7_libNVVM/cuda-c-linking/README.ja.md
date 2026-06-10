# cuda-c-linking - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Introduction ============

This sample demonstrates linking a libnvvm-generated module with an existing CUDA C library. The LLVM C++ API is used to generate an LLVM IR module that conforms to the NVVM IR specification and contains a call to an externally- defined function, and this module is compiled to PTX with libnvvm. The JIT linker (part of the CUDA Driver API) is then used to assemble the PTX and link it with the math library, creating a linked CUBIN image. This image is then

> **日本語**
> `cpp/7_libNVVM/cuda-c-linking` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cuda-c-linking` as a focused example of the CUDA concepts used in `cpp/7_libNVVM/cuda-c-linking`.
> **日本語**
> この sample の目的は、`cuda-c-linking` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Memory, Kernel Launch And Indexing, Execution Model を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda-c-linking.cpp, math-funcs.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `cuda-c-linking.cpp`: Host-side setup, API calls, validation, and cleanup.
- `math-funcs.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuda-c-linking.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.
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

- `cuda-c-linking.cpp`: focus on `CUDA`, `Device`, `cudaModule`, `CUresult`, `CUDA_SUCCESS`.
- `math-funcs.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `gridDim`, `launch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/7_libNVVM/cuda-c-linking/CMakeLists.txt:2-20
```cmake
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/cuda-c-linking/CMakeLists.txt:33-52
```cmake
                 "LLVM development libraries v7 to v14, opaque pointers are "
                 "not supported in libNVVM for pre-Blackwell architectures.")
  return()
endif ()

# JP: `add_executable`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
add_executable(cuda-c-linking cuda-c-linking.cpp)

add_test(NAME cuda-c-linking
   COMMAND cuda-c-linking
   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(cuda-c-linking ${NVVM_LIB} ${CUDA_LIB})

# See https://llvm.org/docs/CMake.html#developing-llvm-passes-out-of-source
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(cuda-c-linking ${llvm_libs})

```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda-c-linking.cpp`

Source: cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp:29-47
```cpp
#include <cassert>
#include <cuda.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <nvvm.h>
#include <string>

#include "DDSWriter.h"

static_assert(sizeof(void *) == 8, "Only 64bit targets are supported.");
using namespace llvm;
```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp:58-77
```cpp
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
static void __checkCudaErrors(CUresult err, const char *filename, int line)
{
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    assert(filename);
    if (CUDA_SUCCESS != err) {
        const char    *ename = NULL;
        const CUresult res   = cuGetErrorName(err, &ename);
        fprintf(stderr,
                "CUDA API Error %04d: \"%s\" from file <%s>, "
                "line %i.\n",
                err,
                ((CUDA_SUCCESS == res) ? ename : "Unknown"),
                filename,
                line);
        exit(err);
    }
}

```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp:162-200
```cpp
    nvvmResult  res       = nvvmCompileProgram(compileUnit, 1, options);
    if (res != NVVM_SUCCESS) {
        errs() << "nvvmCompileProgram failed!\n";
        size_t logSize;
        nvvmGetProgramLogSize(compileUnit, &logSize);
        char *msg = new char[logSize];
        nvvmGetProgramLog(compileUnit, msg);
        errs() << msg << "\n";
        delete[] msg;
        exit(EXIT_FAILURE);
    }

    // Get the result PTX size and source.
    size_t ptxSize = 0;
    checkNVVMCall(nvvmGetCompiledResultSize(compileUnit, &ptxSize));
    char *ptx = new char[ptxSize];
    checkNVVMCall(nvvmGetCompiledResult(compileUnit, ptx));

    // Clean-up libNVVM.
    checkNVVMCall(nvvmDestroyProgram(&compileUnit));

    return std::string(ptx);
}

int main(int argc, char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "cuda-c-linking");

    // Locate the pre-built library.
    std::string      libpath0 = sys::fs::getMainExecutable(argv[0], (void *)main);
    SmallString<256> libpath(libpath0);
    const char      *mathlibFile = "libmathfuncs64.a";
    sys::path::remove_filename(libpath);
    sys::path::append(libpath, mathlibFile);

    if (!sys::fs::exists(libpath.c_str())) {
        errs() << "Unable to locate math library, expected at " << libpath << '\n';
        return EXIT_FAILURE;
    }
```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp:313-337
```cpp
    const unsigned gridSizeZ  = 1;

    // Execute the kernel.
    outs() << "Launching kernel\n";
    void *params[] = {&devBuffer};
    // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    checkCudaErrors(cuLaunchKernel(
        function, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, 0, NULL, params, NULL));

    // Retrieve the result data from the device.
    // JP: `cuMemcpyDtoH`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cuMemcpyDtoH(&data[0], devBuffer, sizeof(float) * width * height * 4));

    writeDDS("mandelbrot.dds", data, width, height);
    outs() << "Output saved to mandelbrot.dds\n";

    // Cleanup.
    delete[] data;
    // JP: `cuMemFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cuMemFree(devBuffer));
    checkCudaErrors(cuModuleUnload(cudaModule));
    checkCudaErrors(cuCtxDestroy(context));

    return 0;
}
```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `math-funcs.cu`

Source: cpp/7_libNVVM/cuda-c-linking/math-funcs.cu:34-87
```cuda
extern "C" __device__ void mandelbrot(float *Data)
{

    // Which pixel am I?
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned DataX  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned DataY  = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned Width  = gridDim.x * blockDim.x;
    unsigned Height = gridDim.y * blockDim.y;

    float R, G, B, A;

    // Scale coordinates to (-2.5, 1) and (-1, 1)

    float NormX = (float)DataX / (float)Width;
    NormX *= 3.5f;
    NormX -= 2.5f;

    float NormY = (float)DataY / (float)Height;
    NormY *= 2.0f;
    NormY -= 1.0f;

    float X0 = NormX;
    float Y0 = NormY;

    float X = 0.0f;
    float Y = 0.0f;

    unsigned Iter    = 0;
    unsigned MaxIter = 1000;

    // Iterate
    while (X * X + Y * Y < 4.0f && Iter < MaxIter) {
        float XTemp = X * X - Y * Y + X0;
        Y           = 2.0f * X * Y + Y0;

        X = XTemp;

        Iter++;
    }

    unsigned ColorG = Iter % 50;
    unsigned ColorB = Iter % 25;

    R = 0.0f;
    G = (float)ColorG / 50.0f;
    B = (float)ColorB / 25.0f;
    A = 1.0f;

    Data[DataY * Width * 4 + DataX * 4 + 0] = R;
    Data[DataY * Width * 4 + DataX * 4 + 1] = G;
    Data[DataY * Width * 4 + DataX * 4 + 2] = B;
    Data[DataY * Width * 4 + DataX * 4 + 3] = A;
}
```

> JP: この抜粋は `cpp/7_libNVVM/cuda-c-linking/math-funcs.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `CUBIN` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaModule` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuMemAlloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuLaunchKernel` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cuMemcpyDtoH` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cuMemFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- NVRTC/Driver/JIT 系では、compile/load/link と kernel launch が別の段階です。compile log、module、function handle、launch parameter の対応が重要です。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target cuda-c-linking
ctest --test-dir build -R cuda-c-linking
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

Run the sample as documented and compare its output with the original README, validation message, generated file, or reference result.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `CUBIN` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
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
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
