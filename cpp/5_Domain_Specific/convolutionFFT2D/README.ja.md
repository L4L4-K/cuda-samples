# convolutionFFT2D - FFT-Based 2D Convolution - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates how 2D convolutions with very large kernel sizes can be efficiently implemented using FFT transformations.

Image Processing, CUFFT Library

Original README headings: `convolutionFFT2D - FFT-Based 2D Convolution`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/convolutionFFT2D` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `convolutionFFT2D` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/convolutionFFT2D`.
> **日本語**
> この sample の目的は、`convolutionFFT2D` の小さな実装を通して CUDA Libraries, Memory, Kernel Launch And Indexing, Execution Model, Debugging, Profiling, And Testing を具体的に追うことです。
>
> **学習メモ**
> 最初に `convolutionFFT2D.cu, convolutionFFT2D.cuh, convolutionFFT2D_common.h, convolutionFFT2D_gold.cpp, main.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `convolutionFFT2D.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `convolutionFFT2D.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `convolutionFFT2D_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `convolutionFFT2D_gold.cpp`: Host-side setup, API calls, validation, and cleanup.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `convolutionFFT2D.cu` first and locate the host-side setup or Python entry point.
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

- `convolutionFFT2D.cu`: focus on `cudaResourceDesc`, `cudaTextureDesc`, `launch`, `cudaDestroyTextureObject`, `cudaResourceTypeLinear`.
- `convolutionFFT2D.cuh`: focus on `blockDim`, `blockIdx`, `threadIdx`, `cudaTextureObject_t`, `launch`.
- `convolutionFFT2D_common.h`: focus on control flow and helper functions.
- `convolutionFFT2D_gold.cpp`: focus on control flow and helper functions.
- `main.cpp`: focus on `cudaMalloc`, `cudaFree`, `CUDA`, `cufftComplex`, `cudaMemcpy`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/convolutionFFT2D/CMakeLists.txt:1-46
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(convolutionFFT2D LANGUAGES C CXX CUDA)

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
# Add target for convolutionFFT2D
add_executable(convolutionFFT2D convolutionFFT2D.cu convolutionFFT2D_gold.cpp main.cpp)

target_compile_options(convolutionFFT2D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(convolutionFFT2D PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(convolutionFFT2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(convolutionFFT2D PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(convolutionFFT2D PUBLIC
    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    CUDA::cufft
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `convolutionFFT2D.cu`

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu:24-47
```cuda
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では kernel launch と thread indexing、stream/event による非同期実行と同期 を確認します。英語の識別子/API/出力文字列は保持します。

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Other helpers
#include <helper_cuda.h>

// Project includes
#include "convolutionFFT2D.cuh"
#include "convolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void
padKernel(float *d_Dst, float *d_Src, int fftH, int fftW, int kernelH, int kernelW, int kernelY, int kernelX)
{
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu:70-105
```cuda

    checkCudaErrors(cudaCreateTextureObject(&texFloat, &texRes, &texDescr, NULL));
#endif

    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    padKernel_kernel<<<grid, threads>>>(d_Dst,
                                        d_Src,
                                        fftH,
                                        fftW,
                                        kernelH,
                                        kernelW,
                                        kernelY,
                                        kernelX
#if (USE_TEXTURE)
                                        ,
                                        texFloat
#endif
    );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");

#if (USE_TEXTURE)
    // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaDestroyTextureObject(texFloat));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(float *d_Dst,
                                     float *d_Src,
                                     int    fftH,
                                     int    fftW,
                                     int    dataH,
                                     int    dataW,
                                     int    kernelW,
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `convolutionFFT2D.cuh`

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh:29-47
```cuda
#define USE_TEXTURE  1
#define POWER_OF_TWO 1

#if (USE_TEXTURE)
#define LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define SET_FLOAT_BASE
#else
#define LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

#include "convolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(float *d_Dst,
                                 float *d_Src,
                                 int    fftH,
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh:54-73
```cuda
                                 ,
                                 cudaTextureObject_t texFloat
#endif
)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y < kernelH && x < kernelW) {
        int ky = y - kernelY;

        if (ky < 0) {
            ky += fftH;
        }

        int kx = x - kernelX;

        if (kx < 0) {
            kx += fftW;
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `convolutionFFT2D_common.h`

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_common.h:29-91
```cpp
#ifndef CONVOLUTIONFFT2D_COMMON_H
#define CONVOLUTIONFFT2D_COMMON_H

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

extern "C" void convolutionClampToBorderCPU(float *h_Result,
                                            float *h_Data,
                                            float *h_Kernel,
                                            int    dataH,
                                            int    dataW,
                                            int    kernelH,
                                            int    kernelW,
                                            int    kernelY,
                                            int    kernelX);

extern "C" void padKernel(float *d_PaddedKernel,
                          float *d_Kernel,
                          int    fftH,
                          int    fftW,
                          int    kernelH,
                          int    kernelW,
                          int    kernelY,
                          int    kernelX);

extern "C" void padDataClampToBorder(float *d_PaddedData,
                                     float *d_Data,
                                     int    fftH,
                                     int    fftW,
                                     int    dataH,
                                     int    dataW,
                                     int    kernelH,
                                     int    kernelW,
                                     int    kernelY,
                                     int    kernelX);

extern "C" void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH, int fftW, int padding);

extern "C" void spPostprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX, uint padding, int dir);

extern "C" void spPreprocess2D(void *d_Dst, void *d_Src, uint DY, uint DX, uint padding, int dir);

extern "C" void spProcess2D(void *d_Data, void *d_Data0, void *d_Kernel0, uint DY, uint DX, int dir);

#endif // CONVOLUTIONFFT2D_COMMON_H
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `convolutionFFT2D_gold.cpp`

Source: cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_gold.cpp:29-72
```cpp
#include <assert.h>

#include "convolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
// Reference straightforward CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionClampToBorderCPU(float *h_Result,
                                            float *h_Data,
                                            float *h_Kernel,
                                            int    dataH,
                                            int    dataW,
                                            int    kernelH,
                                            int    kernelW,
                                            int    kernelY,
                                            int    kernelX)
{
    for (int y = 0; y < dataH; y++)
        for (int x = 0; x < dataW; x++) {
            double sum = 0;

            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++)
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++) {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0)
                        dy = 0;

                    if (dx < 0)
                        dx = 0;

                    if (dy >= dataH)
                        dy = dataH - 1;

                    if (dx >= dataW)
                        dx = dataW - 1;

                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }

            h_Result[y * dataW + x] = (float)sum;
        }
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_gold.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/5_Domain_Specific/convolutionFFT2D/main.cpp:31-54
```cpp
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

// Helper functions for CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "convolutionFFT2D_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
int snapTransformSize(int dataSize)
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/convolutionFFT2D/main.cpp:87-106
```cpp

    float *d_Data, *d_PaddedData, *d_Kernel, *d_PaddedKernel;

    fComplex *d_DataSpectrum, *d_KernelSpectrum;

    // JP: `cufftHandle`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    cufftHandle fftPlanFwd, fftPlanInv;

    bool                bRetVal;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    printf("Testing built-in R2C / C2R FFT-based convolution\n");
    const int kernelH = 7;
    const int kernelW = 6;
    const int kernelY = 3;
    const int kernelX = 4;
    const int dataH   = 2000;
    const int dataW   = 2000;
    const int fftH    = snapTransformSize(dataH + kernelH - 1);
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/convolutionFFT2D/main.cpp:119-138
```cpp
    checkCudaErrors(cudaMalloc((void **)&d_PaddedData, fftH * fftW * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_PaddedKernel, fftH * fftW * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_DataSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    checkCudaErrors(cudaMalloc((void **)&d_KernelSpectrum, fftH * (fftW / 2 + 1) * sizeof(fComplex)));
    // JP: `cudaMemset`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(d_KernelSpectrum, 0, fftH * (fftW / 2 + 1) * sizeof(fComplex)));

    printf("...generating random input data\n");
    srand(2010);

    for (int i = 0; i < dataH * dataW; i++) {
        h_Data[i] = getRand();
    }

    for (int i = 0; i < kernelH * kernelW; i++) {
        h_Kernel[i] = getRand();
    }

    printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/convolutionFFT2D/main.cpp:153-172
```cpp

    // Not including kernel transformation into time measurement,
    // since convolution kernel is not changed very frequently
    printf("...transforming convolution kernel\n");
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedKernel, (cufftComplex *)d_KernelSpectrum));

    printf("...running GPU FFT convolution: ");
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)d_PaddedData, (cufftComplex *)d_DataSpectrum));
    modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_PaddedData));

    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
```

> JP: この抜粋は `cpp/5_Domain_Specific/convolutionFFT2D/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cufftComplex` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaResourceDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaTextureDesc` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemset` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaDestroyTextureObject` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。

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
cmake --build build --target convolutionFFT2D
ctest --test-dir build -R convolutionFFT2D
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample validates the library result against a CPU/reference path or reports the documented success status.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaMalloc` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
