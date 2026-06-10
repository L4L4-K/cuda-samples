# nvJPEG_encoder - NVJPEG Encoder - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA Sample that demonstrates single encoding of jpeg images using NVJPEG Library.

Image Encoding, NVJPEG Library

Original README headings: `nvJPEG_encoder - NVJPEG Encoder`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/4_CUDA_Libraries/nvJPEG_encoder` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `nvJPEG_encoder` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries/nvJPEG_encoder`.
> **日本語**
> この sample の目的は、`nvJPEG_encoder` の小さな実装を通して CUDA Libraries, Streams And Events, Synchronization And Atomics, Performance, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `nvJPEG_encoder.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG

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
- `encode_output/img1.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img2.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img3.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img4.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img5.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img6.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img7.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `encode_output/img8.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img1.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img2.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img3.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img4.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img5.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img6.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img7.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `images/img8.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nvJPEG_encoder.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `nvJPEG_encoder.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `nvJPEG_encoder.cpp`: focus on `nvjpeg_handle`, `CUDA`, `nvjpegEncoderParamsSetSamplingFactors`, `cudaMalloc`, `cudaFree`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/nvJPEG_encoder/CMakeLists.txt:1-60
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

# JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
project(nvJPEG_encoder LANGUAGES CXX)

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
# Add target for nvJPEG_encoder
add_executable(nvJPEG_encoder nvJPEG_encoder.cpp)

target_compile_options(nvJPEG_encoder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(nvJPEG_encoder PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(nvJPEG_encoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(nvJPEG_encoder PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvJPEG_encoder PRIVATE
    CUDA::cudart
    CUDA::nvjpeg
)

# Copy data to the output directory
add_custom_command(TARGET nvJPEG_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/images
    ${CMAKE_CURRENT_BINARY_DIR}/images
)

add_custom_command(TARGET nvJPEG_encoder POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/encode_output
    ${CMAKE_CURRENT_BINARY_DIR}/encode_output
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/nvJPEG_encoder/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvJPEG_encoder.cpp`

Source: cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp:32-53
```cpp
#include <cuda_runtime_api.h>

#include "helper_nvJPEG.hxx"


// JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
// JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
int dev_free(void *p) { return (int)cudaFree(p); }

// JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

struct encode_params_t
{
    std::string input_dir;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp:115-134
```cpp
        // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
        nvjpegChromaSubsampling_t subsampling;
        int                       widths[NVJPEG_MAX_COMPONENT];
        int                       heights[NVJPEG_MAX_COMPONENT];
        if (NVJPEG_STATUS_SUCCESS
            != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights)) {
            std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
            return 1;
        }

        // image information
        std::cout << "Image is " << nComponent << " channels." << std::endl;
        for (int i = 0; i < nComponent; i++) {
            std::cout << "Channel #" << i << " size: " << widths[i] << " x " << heights[i] << std::endl;
        }

        switch (subsampling) {
        case NVJPEG_CSS_444:
            std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
            break;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp:174-193
```cpp
                                      (unsigned int)widths[0],
                                      (unsigned int)widths[0]}};

            int nReturnCode = 0;

            // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
            cudaDeviceSynchronize();

            nReturnCode = nvjpegDecode(nvjpeg_handle, jpeg_state, dpImage, nSize, output_format, &imgdesc, NULL);

            // alternatively decode by stages
            /*int nReturnCode = nvjpegDecodeCPU(nvjpeg_handle, dpImage, nSize, output_format, &imgdesc, NULL);
            // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
            nReturnCode = nvjpegDecodeMixed(nvjpeg_handle, NULL);
            nReturnCode = nvjpegDecodeGPU(nvjpeg_handle, NULL);*/
            // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
            cudaDeviceSynchronize();

            if (nReturnCode != 0) {
                std::cerr << "Error in nvjpegDecode." << std::endl;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp:379-398
```cpp

    return -1;
}


int main(int argc, const char *argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 || (pidx = findParamIndex(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0]
                  << " -i images_dir  [-o output_dir] [-device=device_id]"
                     "[-q quality][-s 420/444] [-fmt output_format] [-huf 0]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
        std::cout << "\toutput_dir\t:\tWrite encoded images as jpeg to this directory" << std::endl;
        std::cout << "\tdevice_id\t:\tWhich device to use for encoding" << std::endl;
        std::cout << "\tQuality\t:\tUse image quality [default 70]" << std::endl;
        std::cout << "\tsubsampling\t:\tUse Subsampling [420, 444]" << std::endl;
        std::cout << "\toutput_format\t:\tnvJPEG output format for encoding. One "
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `nvjpeg_handle` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `nvjpegEncoderParamsSetSamplingFactors` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `nvjpegOutputFormat_t` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaGetErrorString` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEventSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventElapsedTime` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `nvjpegInputFormat_t` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaEvent_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- library sample では、CUDA kernel を直接書かなくても library call が device work を投入します。handle/descriptor/workspace の lifetime を kernel launch と同じ厳しさで追います。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target nvJPEG_encoder
ctest --test-dir build -R nvJPEG_encoder
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `nvjpeg_handle` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
