# EGLStream_CUDA_Interop - EGLStream CUDA Interop - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Demonstrates data exchange between CUDA and EGL Streams.

EGLStreams Interop

Original README headings: `EGLStream_CUDA_Interop - EGLStream CUDA Interop`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `EGLStream_CUDA_Interop` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop`.
> **日本語**
> この sample の目的は、`EGLStream_CUDA_Interop` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda_consumer.cpp, cuda_consumer.h, cuda_producer.cpp, cuda_producer.h, eglstrm_common.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The graphics, display, or platform stack named by the English README

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
- `cuda_consumer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuda_consumer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_f_1.yuv`: Supporting file used by `cuda_f_1.yuv`.
- `cuda_f_2.yuv`: Supporting file used by `cuda_f_2.yuv`.
- `cuda_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuda_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `cuda_yuv_f_1.yuv`: Supporting file used by `cuda_yuv_f_1.yuv`.
- `cuda_yuv_f_2.yuv`: Supporting file used by `cuda_yuv_f_2.yuv`.
- `eglstrm_common.cpp`: Host-side setup, API calls, validation, and cleanup.
- `eglstrm_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `main.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuda_consumer.cpp` first and locate the host-side setup or Python entry point.
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

- `cuda_consumer.cpp`: focus on `cuStatus`, `cudaEgl`, `CUDA_SUCCESS`, `cudaConsumer`, `CUDA`.
- `cuda_consumer.h`: focus on `cudaConsumer`, `CUresult`, `CUDA`, `cudaEGL`, `CUcontext`.
- `cuda_producer.cpp`: focus on `cudaProducer`, `cuStatus`, `CUDA_SUCCESS`, `cudaEgl`, `CUresult`.
- `cuda_producer.h`: focus on `cudaProducer`, `CUresult`, `CUdeviceptr`, `CUarray`, `cudaEGL`.
- `eglstrm_common.cpp`: focus on `cudaIndex`, `CUDA`, `Device`.
- `eglstrm_common.h`: focus on `cudaEGL`.
- `main.cpp`: focus on `cudaProducer`, `cudaConsumer`, `CUDA_SUCCESS`, `cuCtxPushCurrent`, `cuCtxPopCurrent`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/CMakeLists.txt:1-58
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(EGLStream_CUDA_Interop LANGUAGES C CXX CUDA)

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

find_package(EGL)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Source file
    if(${EGL_FOUND})
        # Add target for EGLStream_CUDA_Interop
        add_executable(EGLStream_CUDA_Interop cuda_consumer.cpp cuda_producer.cpp eglstrm_common.cpp main.cpp)

        target_compile_options(EGLStream_CUDA_Interop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

        target_compile_features(EGLStream_CUDA_Interop PRIVATE cxx_std_17 cuda_std_17)

        set_target_properties(EGLStream_CUDA_Interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_include_directories(EGLStream_CUDA_Interop PUBLIC
            ${EGL_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(EGLStream_CUDA_Interop
            ${EGL_LIBRARY}
            CUDA::cuda_driver
        )

    else()
        message(STATUS "EGL not found - will not build sample 'EGLStream_CUDA_Interop'")
    endif()
else()
    message(STATUS "Will not build sample EGLStream_CUDA_Interop - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp:33-51
```cpp
#include "cuda_consumer.h"

#include <helper_cuda_drvapi.h>

#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

int checkbuf(FILE *fp1, FILE *fp2);

// JP: `cudaConsumerTest`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
CUresult cudaConsumerTest(test_cuda_consumer_s *data, const char *fileName)
{
    CUresult           cuStatus = CUDA_SUCCESS;
    CUarray            cudaArr  = NULL;
    CUeglFrame         cudaEgl;
    CUgraphicsResource cudaResource;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp:144-163
```cpp
                            copyHeight >>= 1;
                        }
                    }
                }
                if (i == 0) {
                    pCudaCopyMem = (unsigned char *)malloc(bufferSize);
                    if (pCudaCopyMem == NULL) {
                        printf("pCudaCopyMem malloc failed\n");
                        goto done;
                    }
                }
                memset(pCudaCopyMem, 0, bufferSize);
                if (data->pitchLinearOutput) {
                    // JP: `cuStatus`, `cuMemcpyDtoH`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                    cuStatus = cuMemcpyDtoH(pCudaCopyMem, pDevPtr, bufferSize);
                    if (cuStatus != CUDA_SUCCESS) {
                        printf("cuda_consumer: pitch linear Memcpy failed, bufferSize =%d\n", bufferSize);
                        goto done;
                    }
                    cuStatus = cuCtxSynchronize();
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp:225-244
```cpp
            else {
                printf("Frame check Passed\n");
            }
            if (pCudaCopyMem) {
                // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
                free(pCudaCopyMem);
                pCudaCopyMem = NULL;
            }
            cuStatus = cuEGLStreamConsumerReleaseFrame(&data->cudaConn, cudaResource, NULL);
            if (cuStatus != CUDA_SUCCESS) {
                printf("cuEGLStreamConsumerReleaseFrame failed with cuStatus = %d\n", cuStatus);
                goto done;
            }
        }
        else {
            printf("cuda AcquireFrame FAILED with  cuStatus=%d\n", cuStatus);
            goto done;
        }
    }

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp:289-308
```cpp

// JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer, CUdevice device)
{
    CUresult status = CUDA_SUCCESS;
    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    int  major = 0, minor = 0;
    char deviceName[256];
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuDeviceGetName(deviceName, 256, device));
    printf("CUDA Consumer on GPU Device %d: \"%s\" with compute capability "
           "%d.%d\n\n",
           device,
           deviceName,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.h:33-66
```cpp
#ifndef _CUDA_CONSUMER_H_
#define _CUDA_CONSUMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay   g_display;

typedef struct _test_cuda_consumer_s
{
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUcontext             context;
    CUeglStreamConnection cudaConn;
    bool                  pitchLinearOutput;
    unsigned int          width;
    unsigned int          height;
    const char           *fileName1;
    const char           *fileName2;
    const char           *outFile1;
    const char           *outFile2;
    unsigned int          frameCount;
} test_cuda_consumer_s;

void     cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult cuda_consumer_deinit(test_cuda_consumer_s *cudaConsumer);
CUresult cudaConsumerTest(test_cuda_consumer_s *data, const char *outFile);
// JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer, CUdevice device);
#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_producer.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp:33-51
```cpp
#include "cuda_producer.h"

#include <helper_cuda_drvapi.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

// JP: `cudaProducerReadYUVFrame`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
static CUresult cudaProducerReadYUVFrame(FILE          *file,
                                         unsigned int   frameNum,
                                         unsigned int   width,
                                         unsigned int   height,
                                         unsigned char *pBuff)
{
    int            bOrderUV = 0;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp:297-316
```cpp

// JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer, CUdevice device)
{
    CUresult status = CUDA_SUCCESS;
    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    int  major = 0, minor = 0;
    char deviceName[256];
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    checkCudaErrors(cuDeviceGetName(deviceName, 256, device));
    printf("CUDA Producer on GPU Device %d: \"%s\" with compute capability "
           "%d.%d\n\n",
           device,
           deviceName,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp:411-428
```cpp

CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer)
{
    if (cudaProducer->pBuff)
        // JP: `cudaProducer`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(cudaProducer->pBuff);

    checkCudaErrors(cuMemFree(cudaProducer->cudaPtrARGB[0]));
    checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[0]));
    checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[1]));
    checkCudaErrors(cuMemFree(cudaProducer->cudaPtrYUV[2]));
    checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrARGB[0]));
    checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[0]));
    checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[1]));
    checkCudaErrors(cuArrayDestroy(cudaProducer->cudaArrYUV[2]));

    return cuEGLStreamProducerDisconnect(&cudaProducer->cudaConn);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_producer.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.h:33-74
```cpp
#ifndef _CUDA_PRODUCER_H_
#define _CUDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay   g_display;

typedef struct _test_cuda_producer_s
{
    //  Stream params
    char                 *fileName1;
    char                 *fileName2;
    unsigned char        *pBuff;
    int                   frameCount;
    bool                  isARGB;
    bool                  pitchLinearOutput;
    unsigned int          width;
    unsigned int          height;
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUcontext             context;
    CUeglStreamConnection cudaConn;
    CUdeviceptr           cudaPtrARGB[1];
    CUdeviceptr           cudaPtrYUV[3];
    CUarray               cudaArrARGB[1];
    CUarray               cudaArrYUV[3];
    EGLStreamKHR          eglStream;
    EGLDisplay            eglDisplay;
} test_cuda_producer_s;

void     cudaProducerInit(test_cuda_producer_s *cudaProducer,
                          EGLDisplay            eglDisplay,
                          EGLStreamKHR          eglStream,
                          TestArgs             *args);
CUresult cudaProducerTest(test_cuda_producer_s *parserArg, char *file);
CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
// JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer, CUdevice device);
#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `eglstrm_common.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.cpp:33-51
```cpp
#include "eglstrm_common.h"

EGLStreamKHR eglStream;
EGLDisplay   g_display;
EGLAttrib    cudaIndex;

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_DECL)
typedef void (*extlst_fnptr_t)(void);
static struct
{
    extlst_fnptr_t *fnptr;
    char const     *name;
} extensionList[] = {EXTENSION_LIST(EXTLST_ENTRY)};

int eglSetupExtensions(void)
{
    unsigned int i;

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `eglstrm_common.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.h:33-101
```cpp
#ifndef _EGLSTRM_COMMON_H_
#define _EGLSTRM_COMMON_H_

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "cuda.h"
#include "cudaEGL.h"
#include "helper_cuda_drvapi.h"

#define EXTENSION_LIST(T)                                                                  \
    T(PFNEGLCREATESTREAMKHRPROC, eglCreateStreamKHR)                                       \
    T(PFNEGLDESTROYSTREAMKHRPROC, eglDestroyStreamKHR)                                     \
    T(PFNEGLQUERYSTREAMKHRPROC, eglQueryStreamKHR)                                         \
    T(PFNEGLQUERYSTREAMU64KHRPROC, eglQueryStreamu64KHR)                                   \
    T(PFNEGLQUERYSTREAMTIMEKHRPROC, eglQueryStreamTimeKHR)                                 \
    T(PFNEGLSTREAMATTRIBKHRPROC, eglStreamAttribKHR)                                       \
    T(PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR)                     \
    T(PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR)                     \
    T(PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, eglStreamConsumerGLTextureExternalKHR) \
    T(PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC, eglGetStreamFileDescriptorKHR)                 \
    T(PFNEGLQUERYDEVICESEXTPROC, eglQueryDevicesEXT)                                       \
    T(PFNEGLGETPLATFORMDISPLAYEXTPROC, eglGetPlatformDisplayEXT)                           \
    T(PFNEGLQUERYDEVICEATTRIBEXTPROC, eglQueryDeviceAttribEXT)                             \
    T(PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC, eglCreateStreamFromFileDescriptorKHR)

#define eglCreateStreamKHR                    my_eglCreateStreamKHR
#define eglDestroyStreamKHR                   my_eglDestroyStreamKHR
#define eglQueryStreamKHR                     my_eglQueryStreamKHR
#define eglQueryStreamu64KHR                  my_eglQueryStreamu64KHR
#define eglQueryStreamTimeKHR                 my_eglQueryStreamTimeKHR
#define eglStreamAttribKHR                    my_eglStreamAttribKHR
#define eglStreamConsumerAcquireKHR           my_eglStreamConsumerAcquireKHR
#define eglStreamConsumerReleaseKHR           my_eglStreamConsumerReleaseKHR
#define eglStreamConsumerGLTextureExternalKHR my_eglStreamConsumerGLTextureExternalKHR
#define eglGetStreamFileDescriptorKHR         my_eglGetStreamFileDescriptorKHR
#define eglCreateStreamFromFileDescriptorKHR  my_eglCreateStreamFromFileDescriptorKHR
#define eglQueryDevicesEXT                    my_eglQueryDevicesEXT
#define eglGetPlatformDisplayEXT              my_eglGetPlatformDisplayEXT
#define eglQueryDeviceAttribEXT               my_eglQueryDeviceAttribEXT

#define EXTLST_DECL(tx, x)   tx my_##x = NULL;
#define EXTLST_EXTERN(tx, x) extern tx my_##x;
#define EXTLST_ENTRY(tx, x)  {(extlst_fnptr_t *)&my_##x, #x},

#define MAX_STRING_SIZE 256
#define WIDTH           720
#define HEIGHT          480

typedef struct _TestArgs
{
    char        *infile1;
    char        *infile2;
    bool         isARGB;
    unsigned int inputWidth;
    unsigned int inputHeight;
    bool         pitchLinearOutput;
} TestArgs;

int  eglSetupExtensions(void);
int  EGLStreamInit(int *dev);
void EGLStreamFini(void);
#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/main.cpp:36-71
```cpp
#include "cudaEGL.h"
#include "cuda_consumer.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"

/* ------  globals ---------*/

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

#define NUM_TRAILS 4

bool signal_stop = 0;

static void sig_handler(int sig)
{
    signal_stop = 1;
    printf("Signal: %d\n", sig);
}

int main(int argc, char **argv)
{
    TestArgs     args;
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUresult     curesult = CUDA_SUCCESS;
    unsigned int i, j;
    EGLint       streamState = 0;

    test_cuda_consumer_s cudaConsumer;
    test_cuda_producer_s cudaProducer;

    memset(&cudaProducer, 0, sizeof(test_cuda_producer_s));
    memset(&cudaConsumer, 0, sizeof(test_cuda_consumer_s));

    // Hook up Ctrl-C handler
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaProducer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuStatus` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaConsumer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEgl` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuCtxPopCurrent` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaConn` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuCtxPushCurrent` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaArr` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuCtxSynchronize` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaPtrYUV` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaConsumerTest` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDA_ERROR_UNKNOWN` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

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

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Sample-Specific Notes

- stream sample では、copy と kernel が本当に重なるには pinned memory、non-default stream、依存 event の条件がそろう必要があります。

> **日本語**
> この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。
>
> **学習メモ**
> 似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target EGLStream_CUDA_Interop
ctest --test-dir build -R EGLStream_CUDA_Interop
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample may display a window or produce/validate image-like output; exact visuals depend on platform support.
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

- `cudaProducer` の直前と直後で、どの memory/resource が有効になったかをメモする。
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
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
