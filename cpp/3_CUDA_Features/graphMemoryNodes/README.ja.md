# graphMemoryNodes - Graph Memory Nodes - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A demonstration of memory allocations and frees within CUDA graphs using Graph APIs and Stream Capture APIs.

CUDA Graphs, Stream Capture

Original README headings: `graphMemoryNodes - Graph Memory Nodes`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/3_CUDA_Features/graphMemoryNodes` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `graphMemoryNodes` as a focused example of the CUDA concepts used in `cpp/3_CUDA_Features/graphMemoryNodes`.
> **日本語**
> この sample の目的は、`graphMemoryNodes` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Graphs, Streams And Events, Unified Memory, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `graphMemoryNodes.cu` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `graphMemoryNodes.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `graphMemoryNodes.cu` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `graphMemoryNodes.cu`: focus on `launch`, `CUDA`, `cudaGraphNode_t`, `cudaStreamSynchronize`, `cudaGraphLaunch`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/3_CUDA_Features/graphMemoryNodes/CMakeLists.txt:1-37
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(graphMemoryNodes LANGUAGES C CXX CUDA)

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
# Add target for graphMemoryNodes
add_executable(graphMemoryNodes graphMemoryNodes.cu)

target_compile_options(graphMemoryNodes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(graphMemoryNodes PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(graphMemoryNodes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/3_CUDA_Features/graphMemoryNodes/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `graphMemoryNodes.cu`

Source: cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu:25-63
```cuda
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では memory ownership と host/device transfer、kernel launch と thread indexing、stream/event による非同期実行と同期 を確認します。英語の識別子/API/出力文字列は保持します。

// System includes
#include <assert.h>
#include <climits>
#include <stdio.h>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define THREADS_PER_BLOCK  512
#define ALLOWABLE_VARIANCE 1.e-6f
#define NUM_ELEMENTS       8000000

// Stores the square of each input element in output array
__global__ void squareArray(const float *input, float *output, int numElements)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        output[idx] = input[idx] * input[idx];
    }
}

// Stores the negative of each input element in output array
__global__ void negateArray(const float *input, float *output, int numElements)
{
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
```

> JP: この抜粋は `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu:96-115
```cuda
    size_t numBlocks = hostArrays->numElements / (size_t)THREADS_PER_BLOCK;
    if ((numBlocks % (size_t)THREADS_PER_BLOCK) != 0) {
        numBlocks++;
    }

    hostArrays->input     = (float *)malloc(bytes);
    hostArrays->square    = (float *)malloc(bytes);
    hostArrays->negSquare = (float *)malloc(bytes);
    hostArrays->bytes     = bytes;
    hostArrays->numBlocks = numBlocks;

    fillRandomly(hostArrays->input, hostArrays->numElements);
    fillRandomly(hostArrays->square, hostArrays->numElements);
    fillRandomly(hostArrays->negSquare, hostArrays->numElements);
}

// JP: `cudaGraphExec_t`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
void createFreeGraph(cudaGraphExec_t *graphExec, float *dPtr)
{
    cudaGraph_t     graph;
```

> JP: この抜粋は `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu:204-223
```cuda
                                             &allocNodeSquare,
                                             1,
                                             d_input,
                                             hostArrays->input,
                                             hostArrays->bytes,
                                             // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                                             cudaMemcpyHostToDevice));

    void *squareKernelArgs[3]     = {(void *)&d_input, (void *)&d_square, (void *)&(hostArrays->numElements)};
    kernelNodeParams.func         = (void *)squareArray;
    kernelNodeParams.kernelParams = (void **)squareKernelArgs;

    // Square kernel depends on copyNodeInput to ensure all data is on the device
    // before kernel launch.
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphAddKernelNode(&squareKernelNode, graph, &copyNodeInput, 1, &kernelNodeParams));

    checkCudaErrors(cudaGraphAddMemcpyNode1D(&copyNodeSquare,
                                             graph,
                                             &squareKernelNode,
```

> JP: この抜粋は `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu:337-356
```cuda
    checkCudaErrors(cudaMallocAsync(&d_square, hostArrays->bytes, stream1));

    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(d_input, hostArrays->input, hostArrays->bytes, cudaMemcpyHostToDevice, stream1));
    squareArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(d_input, d_square, hostArrays->numElements);
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(squareKernelCompleteEvent, stream1));

    checkCudaErrors(cudaStreamWaitEvent(stream2, squareKernelCompleteEvent, 0));
    checkCudaErrors(cudaMemcpyAsync(hostArrays->square, d_square, hostArrays->bytes, cudaMemcpyDeviceToHost, stream2));

    // JP: `cudaFreeAsync`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeAsync(d_input, stream1));
    checkCudaErrors(cudaMallocAsync(&d_negSquare, hostArrays->bytes, stream1));
    negateArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
        d_square, d_negSquare, hostArrays->numElements);
    checkCudaErrors(cudaEventRecord(negateKernelCompleteEvent, stream1));
    checkCudaErrors(
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        cudaMemcpyAsync(hostArrays->negSquare, d_negSquare, hostArrays->bytes, cudaMemcpyDeviceToHost, stream1));
```

> JP: この抜粋は `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaGraphLaunch` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphNode_t` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaFreeAsync` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGraphAddMemFreeNode` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMallocAsync` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaGraphExec_t` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphInstantiate` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaGraphAddMemAllocNode` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
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

- graph sample では、最初の capture/instantiate と繰り返し launch を分けて読みます。launch overhead を減らす代わりに依存関係と buffer lifetime が固定されます。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target graphMemoryNodes
ctest --test-dir build -R graphMemoryNodes
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
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `launch` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
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
