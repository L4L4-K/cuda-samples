# uvmlite - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Unified Virtual Memory Lite (UVM-lite) From NVVM IR ===================================================

This document is for the programming language and compiler implementers who target NVVM IR and plan to support Unified Virtual Memory Lite (UVM-lite) in their language.  It provides the low-level details related to supporting kernel launches at the NVVM IR level.

> **日本語**
> `cpp/7_libNVVM/uvmlite` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `uvmlite` as a focused example of the CUDA concepts used in `cpp/7_libNVVM/uvmlite`.
> **日本語**
> この sample の目的は、`uvmlite` の小さな実装を通して Runtime, Driver, And NVRTC, Multi-GPU, P2P, And IPC, Streams And Events, Unified Memory, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `uvmlite.c` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `uvmlite.c`: Host-side setup, API calls, validation, and cleanup.
- `uvmlite64.ll`: Supporting file used by `uvmlite64.ll`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `uvmlite.c` first and locate the host-side setup or Python entry point.
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

- `uvmlite.c`: focus on `cuDevice`, `cuDeviceGetAttribute`, `cuPointerGetAttribute`, `CUDA`, `CUresult`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/7_libNVVM/uvmlite/CMakeLists.txt:2-20
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

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/uvmlite/CMakeLists.txt:29-48
```cmake
set(CMAKE_INSTALL_RPATH ${LIBNVVM_HOME})
set(CMAKE_INCLUDE_CURRENT_DIR YES)
set_property(SOURCE uvmlite.c
             PROPERTY COMPILE_DEFINITIONS LIBCUDADEVRT="${CUDADEVRT_LIB}")

# JP: `add_executable`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
add_executable(uvmlite uvmlite.c)

add_test(NAME uvmlite COMMAND uvmlite WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(uvmlite ${NVVM_LIB} ${CUDA_LIB})

if (WIN32)
  set (LIB_ARCH_SUFFIX "/x64")
  set_target_properties(uvmlite PROPERTIES COMPILE_FLAGS "/wd4996")
else (WIN32)
  set (LIB_ARCH_SUFFIX "64")
  set_target_properties(uvmlite PROPERTIES
                        LINK_FLAGS "-Wl,-rpath,${LIBNVVM_RPATH}")
endif ()
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `uvmlite.c`

Source: cpp/7_libNVVM/uvmlite/uvmlite.c:24-47
```c
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <builtin_types.h>
#include <cuda.h>
#include <math.h>
#include <nvvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define ERROR_IF(expr)                                                  \
    if (expr) {                                                         \
        fprintf(stderr, "Failed check at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                             \
    }

// If 'err' is non-zero, emit an error message and exit.
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/uvmlite.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/uvmlite/uvmlite.c:93-116
```c
    if (result != NVVM_SUCCESS) {
        char  *Msg = NULL;
        size_t LogSize;
        fprintf(stderr, "nvvmCompileProgram: Failed\n");
        nvvmGetProgramLogSize(program, &LogSize);
        Msg = (char *)malloc(LogSize);
        nvvmGetProgramLog(program, Msg);
        fprintf(stderr, "%s\n", Msg);
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(Msg);
        exit(EXIT_FAILURE);
    }

    size_t ptxSize = 0;
    result         = nvvmGetCompiledResultSize(program, &ptxSize);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "nvvmGetCompiledResultSize: Failed\n");
        exit(EXIT_FAILURE);
    }

    char *ptx = malloc(ptxSize);
    assert(ptx);
    result = nvvmGetCompiledResult(program, ptx);
    if (result != NVVM_SUCCESS) {
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/uvmlite.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/uvmlite/uvmlite.c:156-175
```c
// JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
static CUdevice cudaDeviceInit(int *major, int *minor)
{
    assert(major && minor);
    // Count the number of CUDA compute capable devices..
    CUresult err         = cuInit(0);
    int      deviceCount = 0;
    if (CUDA_SUCCESS == err)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    // Get the first device discovered (device 0) and print its name.
    // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    CUdevice cuDevice = 0;
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));
    char name[128] = {0};
    checkCudaErrors(cuDeviceGetName(name, sizeof(name), cuDevice));
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/uvmlite.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/7_libNVVM/uvmlite/uvmlite.c:310-329
```c
    printf("The host added 1 and 11 to xxx and yyy.\n");

    // Launch the kernel with the following parameters.
    {
        void *params[] = {(void *)&devp_xxx};
        // JP: `cuLaunchKernel`: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        checkCudaErrors(cuLaunchKernel(hKernel, nBlocks, 1, 1, nThreads, 1, 1, 0, NULL, params, NULL));
    }
    checkCudaErrors(cuCtxSynchronize());

    printf("kernel added 20 and 30 to xxx and yyy, respectively.\n");
    printf("The final value checked in the host: xxx = %d, yyy = %d\n", *p_xxx, *p_yyy);

    if (hModule) {
        // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }
    if (hContext) {
        checkCudaErrors(cuCtxDestroy(hContext));
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/uvmlite.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `uvmlite64.ll`

Source: cpp/7_libNVVM/uvmlite/uvmlite64.ll:2-61
```llvm
;
; Redistribution and use in source and binary forms, with or without
; modification, are permitted provided that the following conditions
; are met:
;  * Redistributions of source code must retain the above copyright
;    notice, this list of conditions and the following disclaimer.
;  * Redistributions in binary form must reproduce the above copyright
;    notice, this list of conditions and the following disclaimer in the
;    documentation and/or other materials provided with the distribution.
;  * Neither the name of NVIDIA CORPORATION nor the names of its
;    contributors may be used to endorse or promote products derived
;    from this software without specific prior written permission.
;
; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
; EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
; PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
; CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
; EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
; PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
; OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
; (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
; OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; the initial value of xxx is 10
@xxx = internal addrspace(1) global i32 10, align 4

; the initial value of yyy is 100
@yyy = internal addrspace(1) global i32 100, align 4

@llvm.used = appending global [3 x i8*] [i8* bitcast (i8* addrspacecast (i32 addrspace(1)* @xxx to i8*) to i8*), i8* bitcast (i8* addrspacecast (i32 addrspace(1)* @yyy to i8*) to i8*), i8* bitcast (void (i32*)* @test_kernel to i8*)], section "llvm.metadata"

; %ptr can be in the managed space, and its address can be directly used in the host and device.
; See the uvmlite.c, which passes the device pointer of xxx as the kernel parameter.
; This kernel also directly accesses @yyy, which is also managed.
define void @test_kernel(i32* nocapture %ptr) nounwind alwaysinline {
  ; *%ptr = *%ptr + 20
  %gen2other = addrspacecast i32* %ptr to i32 addrspace(1)*
  %tmp1 = load i32, i32 addrspace(1)* %gen2other, align 4
  %add = add nsw i32 %tmp1, 20
  store i32 %add, i32 addrspace(1)* %gen2other, align 4

  ; @yyy = @yyy + 30
  %tmp2 = load i32, i32 addrspace(1)* @yyy, align 4
  %add3 = add nsw i32 %tmp2, 30
  store i32 %add3, i32 addrspace(1)* @yyy, align 4
  ret void
}

!nvvm.annotations = !{!7, !8, !9}
!nvvmir.version = !{!6}

!6 = !{i32 2, i32 0}
!7 = !{i32 addrspace(1)* @xxx, !"managed", i32 1}
!8 = !{i32 addrspace(1)* @yyy, !"managed", i32 1}
!9 = !{void (i32*)* @test_kernel, !"kernel", i32 1}
```

> JP: この抜粋は `cpp/7_libNVVM/uvmlite/uvmlite64.ll` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuPointerGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CU_POINTER_ATTRIBUTE_IS_MANAGED` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuModuleGetGlobal` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUdevice` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CU_POINTER_ATTRIBUTE_HOST_POINTER` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaDeviceInit` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `CUcontext` | Driver API の handle 境界です。context/module/function と error code を追います。 |

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
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。
- Unified Memory は pointer を共有しますが、migration、prefetch、同期の理解は必要です。

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
cmake --build build --target uvmlite
ctest --test-dir build -R uvmlite
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cuPointerGetAttribute` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
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
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Unified Memory](../../../docs_ja/themes/unified_memory.md): managed memory、migration、prefetch の意味を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
