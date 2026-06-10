# EGLStream_CUDA_CrossGPU - EGLStream_CUDA_CrossGPU - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Demonstrates CUDA and EGL Streams interop, where consumer's EGL Stream is on one GPU and producer's on other and both consumer-producer are different processes.

EGLStreams Interop

Original README headings: `EGLStream_CUDA_CrossGPU - EGLStream_CUDA_CrossGPU`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `EGLStream_CUDA_CrossGPU` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU`.
> **日本語**
> この sample の目的は、`EGLStream_CUDA_CrossGPU` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Graphs, Multi-GPU, P2P, And IPC, Streams And Events, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuda_consumer.cpp, cuda_consumer.h, cuda_producer.cpp, cuda_producer.h, eglstrm_common.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support
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
- `cuda_producer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `cuda_producer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `eglstrm_common.cpp`: Host-side setup, API calls, validation, and cleanup.
- `eglstrm_common.h`: Host/device declarations, helper types, constants, or library wrappers.
- `helper.h`: Host/device declarations, helper types, constants, or library wrappers.
- `kernel.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
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

- `cuda_consumer.cpp`: focus on `cudaConsumer`, `cuStatus`, `CUDA_SUCCESS`, `CUresult`, `CUDA`.
- `cuda_consumer.h`: focus on `CUresult`, `cudaConsumer`, `CUstream`, `cudaError_t`, `CUDA`.
- `cuda_producer.cpp`: focus on `cudaProducer`, `cudaEgl`, `CUDA_SUCCESS`, `CUresult`, `cudaPtr`.
- `cuda_producer.h`: focus on `CUresult`, `CUdeviceptr`, `cudaProducer`, `CUeglFrame`, `cudaEgl`.
- `eglstrm_common.cpp`: focus on `CUDA`, `cudaDevIndexProd`, `Device`, `cudaDevIndexCons`.
- `eglstrm_common.h`: focus on `cudaEGL`, `cudaDevIndexCons`, `cudaDevIndexProd`.
- `helper.h`: focus on `cuInit`, `CUDA_SUCCESS`, `CUresult`, `cuDeviceGetCount`, `cuDeviceGetAttribute`.
- `kernel.cu`: focus on `blockDim`, `blockIdx`, `threadIdx`, `cudaSuccess`, `cudaError_t`.
- `main.cpp`: focus on `cudaConsumer`, `cudaProducer`, `CUDA_SUCCESS`, `CUDA_ERROR_UNKNOWN`, `cudaEgl1`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/CMakeLists.txt:1-57
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(EGLStream_CUDA_CrossGPU LANGUAGES C CXX CUDA)

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
            # Add target for EGLStream_CUDA_CrossGPU
            add_executable(EGLStream_CUDA_CrossGPU cuda_consumer.cpp cuda_producer.cpp eglstrm_common.cpp kernel.cu main.cpp)

            target_compile_options(EGLStream_CUDA_CrossGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(EGLStream_CUDA_CrossGPU PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(EGLStream_CUDA_CrossGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(EGLStream_CUDA_CrossGPU PUBLIC
                ${EGL_INCLUDE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
            )

            target_link_libraries(EGLStream_CUDA_CrossGPU
                ${EGL_LIBRARY}
                CUDA::cuda_driver
            )
    else()
        message(STATUS "EGL not found - will not build sample 'EGLStream_CUDA_CrossGPU'")
    endif()
else()
    message(STATUS "Will not build sample EGLStream_CUDA_CrossGPU - requires Linux OS")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp:33-51
```cpp
#include "cuda_consumer.h"

#include <cuda_runtime.h>
#include <math.h>
#include <unistd.h>

#include "eglstrm_common.h"

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
CUgraphicsResource cudaResource;

static int    count_acq           = 0;
static double acquire_time[25000] = {0}, total_time_acq = 0;

static int    count_rel       = 0;
static double rel_time[25000] = {0}, total_time_rel = 0;

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp:195-214
```cpp
    // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    CUdevice          device;
    CUresult          status          = CUDA_SUCCESS;
    CUctxCreateParams ctxCreateParams = {};

    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuDeviceGet(&device, cudaConsumer->cudaDevId))) {
        printf("failed to get CUDA device\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuCtxCreate(&cudaConsumer->context, &ctxCreateParams, 0, device))) {
        printf("failed to create CUDA context\n");
        return status;
    }

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp:241-270
```cpp
    int      bufferSize;

    cudaConsumer->charCnt = args->charCnt;
    bufferSize            = args->charCnt;

    cudaConsumer->pCudaCopyMem = (unsigned char *)malloc(bufferSize);
    if (cudaConsumer->pCudaCopyMem == NULL) {
        printf("Cuda Consumer: malloc failed\n");
        goto done;
    }

    status = cuStreamCreate(&cudaConsumer->consCudaStream, 0);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Consumer: cuStreamCreate failed, status:%d\n", status);
        goto done;
    }

    atexit(acquireApiStat);
done:
    return status;
}

CUresult cuda_consumer_Deinit(test_cuda_consumer_s *cudaConsumer)
{
    if (cudaConsumer->pCudaCopyMem) {
        // JP: `cudaConsumer`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(cudaConsumer->pCudaCopyMem);
    }
    return cuEGLStreamConsumerDisconnect(&cudaConsumer->cudaConn);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_consumer.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.h:33-75
```cpp
#ifndef _CUDA_CONSUMER_H_
#define _CUDA_CONSUMER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"

typedef struct _test_cuda_consumer_s
{
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUcontext             context;
    CUeglStreamConnection cudaConn;
    int                   cudaDevId;
    EGLDisplay            eglDisplay;
    EGLStreamKHR          eglStream;
    unsigned int          charCnt;
    char                 *cudaBuf;
    bool                  profileAPI;
    unsigned char        *pCudaCopyMem;
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    CUstream              consCudaStream;
} test_cuda_consumer_s;

CUresult    cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult    cuda_consumer_Deinit(test_cuda_consumer_s *cudaConsumer);
CUresult    cudaConsumerAcquireFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult    cudaConsumerReleaseFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult    cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer);
cudaError_t cudaConsumer_filter(CUstream cStream,
                                char    *pSrc,
                                int      width,
                                int      height,
                                char     expectedVal,
                                char     newVal,
                                int      frameNumber);
cudaError_t cudaGetValueMismatch(void);

#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_producer.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp:33-51
```cpp
#include "cuda_producer.h"

#include <cuda_runtime.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "math.h"

int         cudaPresentReturnData = INIT_DATA;
int         fakePresent           = 0;
CUeglFrame  fakeFrame;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp:187-206
```cpp
    // JP: この連続する anchor 群では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
    CUdevice          device;
    CUresult          status          = CUDA_SUCCESS;
    CUctxCreateParams ctxCreateParams = {};

    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuDeviceGet(&device, cudaProducer->cudaDevId))) {
        printf("failed to get CUDA device\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuCtxCreate(&cudaProducer->context, &ctxCreateParams, 0, device))) {
        printf("failed to create CUDA context\n");
        return status;
    }

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp:234-253
```cpp
    int      bufferSize;

    cudaProducer->charCnt = args->charCnt;
    bufferSize            = cudaProducer->charCnt;

    cudaProducer->tempBuff = (char *)malloc(bufferSize);
    if (!cudaProducer->tempBuff) {
        printf("Cuda Producer: Failed to allocate image buffer\n");
        status = CUDA_ERROR_UNKNOWN;
        goto done;
    }
    memset((void *)cudaProducer->tempBuff, INIT_DATA, cudaProducer->charCnt);

    // Fill this init data
    // JP: `cuMemAlloc`, `cudaProducer`, `cudaPtr`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    status = cuMemAlloc(&cudaProducer->cudaPtr, bufferSize);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
        goto done;
    }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp:293-304
```cpp

CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer)
{
    if (cudaProducer->tempBuff) {
        // JP: `cudaProducer`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(cudaProducer->tempBuff);
    }
    if (cudaProducer->cudaPtr) {
        cuMemFree(cudaProducer->cudaPtr);
    }
    return cuEGLStreamProducerDisconnect(&cudaProducer->cudaConn);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuda_producer.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.h:33-74
```cpp
#ifndef _CUDA_PRODUCER_H_
#define _CUDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaEGL.h"
#include "eglstrm_common.h"

typedef struct _test_cuda_producer_s
{
    //  Stream params
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUcontext             context;
    CUeglStreamConnection cudaConn;
    int                   cudaDevId;
    EGLStreamKHR          eglStream;
    EGLDisplay            eglDisplay;
    unsigned int          charCnt;
    bool                  profileAPI;
    char                 *tempBuff;
    CUdeviceptr           cudaPtr;
    CUdeviceptr           cudaPtr1;
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    CUstream              prodCudaStream;
} test_cuda_producer_s;

CUresult    cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args);
CUresult    cudaProducerPresentFrame(test_cuda_producer_s *parserArg, CUeglFrame cudaEgl, int t);
CUresult    cudaProducerReturnFrame(test_cuda_producer_s *parserArg, CUeglFrame cudaEgl, int t);
CUresult    cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
CUresult    cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer);
cudaError_t cudaProducer_filter(CUstream cStream,
                                char    *pSrc,
                                int      width,
                                int      height,
                                char     expectedVal,
                                char     newVal,
                                int      frameNumber);
void        cudaProducerPrepareFrame(CUeglFrame *cudaEgl, CUdeviceptr cudaPtr, int bufferSize);
#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `eglstrm_common.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.cpp:33-51
```cpp
#include "eglstrm_common.h"

EGLStreamKHR g_producerEglStream  = EGL_NO_STREAM_KHR;
EGLStreamKHR g_consumerEglStream  = EGL_NO_STREAM_KHR;
EGLDisplay   g_producerEglDisplay = EGL_NO_DISPLAY;
EGLDisplay   g_consumerEglDisplay = EGL_NO_DISPLAY;
int          cudaDevIndexProd     = -1;
int          cudaDevIndexCons     = -1;

#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_DECL)
typedef void (*extlst_fnptr_t)(void);
static struct
{
    extlst_fnptr_t *fnptr;
    char const     *name;
    bool            is_dgpu; // This function is need only for dgpu case
} extensionList[] = {EXTENSION_LIST(EXTLST_ENTRY)};

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.cpp:154-173
```cpp
    }

    if (!isConsumer) { // Producer

        if (fileDesc == EGL_NO_FILE_DESCRIPTOR_KHR) {
            // JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
            printf("Cuda Producer received bad file descriptor\n");
            eglStatus = EGL_FALSE;
            goto Done;
        }

        int egl_device_id    = 0;
        int egl_cuda_devices = 0;
        for (egl_device_id = 0; egl_device_id < numDevices; egl_device_id++) {
            EGLAttrib cuda_device = -1;
            eglStatus             = eglQueryDeviceAttribEXT(devices[egl_device_id], EGL_CUDA_DEVICE_NV, &cuda_device);
            if (eglStatus == EGL_TRUE) {
                egl_cuda_devices++;
                if (isCrossDevice && (egl_cuda_devices > 1)) {
                    // We select second EGL-CUDA Capable device for producer.
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `eglstrm_common.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.h:33-51
```cpp
#ifndef _EGLSTRM_COMMON_H_
#define _EGLSTRM_COMMON_H_

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "cuda.h"
#include "cudaEGL.h"
#define TIME_DIFF(end, start) (getMicrosecond(end) - getMicrosecond(start))
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `helper.h`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h:29-47
```cpp
#include "eglstrm_common.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <cuda.h>

int  parseCmdLine(int argc, char *argv[], TestArgs *args);
void printUsage(void);
int  NUMTRIALS   = 10;
int  profileAPIs = 0;

bool verbose       = 0;
bool isCrossDevice = 0;

// Parse the command line options. Returns FAILURE on a parse error, SUCCESS
// otherwise.
int parseCmdLine(int argc, char *argv[], TestArgs *args)
{
    int i;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h:101-125
```cpp
    }

    if (isCrossDevice) {
        int deviceCount = 0;

        // JP: `cuInit`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
        CUresult error_id = cuInit(0);
        if (error_id != CUDA_SUCCESS) {
            printf("cuInit(0) returned %d\n", error_id);
            // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        error_id = cuDeviceGetCount(&deviceCount);
        if (error_id != CUDA_SUCCESS) {
            printf("cuDeviceGetCount returned %d\n", (int)error_id);
            printf("Result = FAIL\n");
            exit(EXIT_FAILURE);
        }

        int      iGPUexists = 0;
        // JP: この anchor では Driver API の CU* handle と cu* call です。context/module/function/device memory の所有と error boundary を確認します。
        CUdevice dev;
        for (dev = 0; dev < deviceCount; ++dev) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `kernel.cu`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu:33-61
```cuda
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "eglstrm_common.h"

extern bool isCrossDevice;

__device__ static unsigned int numErrors = 0, errorFound = 0;
__device__ void                checkProducerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    // JP: `blockDim`, `blockIdx`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Producer FOUND:%d expected: %d at %d for trial %d %d\n",
               data[blockDim.x * blockIdx.x + threadIdx.x],
               expectedVal,
               (blockDim.x * blockIdx.x + threadIdx.x),
               frameNumber,
               numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__device__ void checkConsumerDataGPU(char *data, int size, char expectedVal, int frameNumber)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu:107-159
```cuda
    }
    writeDataToBuffer<<<(width * height) / 1024, 1024, 1, pStream>>>(pSrc, newVal);
    return cudaSuccess;
};

// JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
cudaError_t cudaConsumer_filter(cudaStream_t cStream,
                                char        *pSrc,
                                int          width,
                                int          height,
                                char         expectedVal,
                                char         newVal,
                                int          frameNumber)
{
    // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
    testKernelConsumer<<<(width * height) / 1024, 1024, 1, cStream>>>(
        pSrc, width * height, expectedVal, newVal, frameNumber);
    writeDataToBuffer<<<(width * height) / 1024, 1024, 1, cStream>>>(pSrc, newVal);
    return cudaSuccess;
};

cudaError_t cudaGetValueMismatch()
{
    int         numErr_h;
    int        *numErr_d = NULL;
    cudaError_t err      = cudaSuccess;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    err                  = cudaMalloc(&numErr_d, sizeof(int));
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMalloc failed with %s\n", cudaGetErrorString(err));
        return err;
    }
    getNumErrors<<<1, 1>>>(numErr_d);
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaDeviceSynchronize failed with %s\n", cudaGetErrorString(err));
    }
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    err = cudaMemcpy(&numErr_h, numErr_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMemcpy failed with %s\n", cudaGetErrorString(err));
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFree(numErr_d);
        return err;
    }
    err = cudaFree(numErr_d);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaFree failed with %s\n", cudaGetErrorString(err));
        return err;
    }
    if (numErr_h > 0) {
        return cudaErrorUnknown;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `main.cpp`

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp:29-47
```cpp
#include "cudaEGL.h"
#include "cuda_consumer.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"
#include "helper.h"
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif

bool        signal_stop = 0;
extern bool verbose;

static void sig_handler(int sig)
{
    signal_stop = 1;
    printf("Signal: %d\n", sig);
}

void DoneCons(int consumerStatus, int send_fd)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp:79-98
```cpp
        exit(EXIT_FAILURE);
    }
}

int WIDTH = 8192, HEIGHT = 8192;
int main(int argc, char **argv)
{
    TestArgs                   args           = {0, false};
    // JP: driver_api: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
    CUresult                   curesult       = CUDA_SUCCESS;
    unsigned int               j              = 0;
    cudaError_t                err            = cudaSuccess;
    EGLNativeFileDescriptorKHR fileDescriptor = EGL_NO_FILE_DESCRIPTOR_KHR;
    struct timespec            start, end;
    CUeglFrame                 cudaEgl1, cudaEgl2;
    int                        consumerStatus = 0;
    int                        send_fd        = -1;

    if (parseCmdLine(argc, argv, &args) < 0) {
        printUsage();
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp:154-173
```cpp
        cudaConsumer.eglDisplay = g_consumerEglDisplay;

        // Send the EGL stream FD to producer
        fileDescriptor = eglGetStreamFileDescriptorKHR(cudaConsumer.eglDisplay, cudaConsumer.eglStream);
        if (EGL_NO_FILE_DESCRIPTOR_KHR == fileDescriptor) {
            // JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
            printf("%s: Cuda Consumer could not get EGL file descriptor.\n", __func__);
            eglDestroyStreamKHR(cudaConsumer.eglDisplay, cudaConsumer.eglStream);
            consumerStatus = -1;
            DoneCons(consumerStatus, send_fd);
        }

        if (verbose)
            printf("%s: Cuda Consumer EGL stream FD obtained : %d.\n", __func__, fileDescriptor);

        int res = -1;
        res     = EGLStreamSendfd(send_fd, fileDescriptor);
        if (-1 == res) {
            printf("%s: Cuda Consumer could not send EGL file descriptor.\n", __func__);
            consumerStatus = -1;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaProducer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaConsumer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDA_SUCCESS` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUresult` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaEgl` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuStatus` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUdeviceptr` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaPtr` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUDA_ERROR_UNKNOWN` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaConn` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `CUeglFrame` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDeviceGetAttribute` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `Device` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。
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
cmake --build build --target EGLStream_CUDA_CrossGPU
ctest --test-dir build -R EGLStream_CUDA_CrossGPU
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
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

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
- device ごとの ownership と、peer/IPC で共有される resource を分けて書く。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
