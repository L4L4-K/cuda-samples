# particles - Particles - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample uses CUDA to simulate and visualize a large set of particles and their physical interaction.  Adding "-particles=<N>" to the command line will allow users to set # of particles for simulation.  This example implements a uniform grid data structure using either atomic operations or a fast radix sort from the Thrust library

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation, Performance Strategies

Original README headings: `particles - Particles`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/2_Concepts_and_Techniques/particles` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `particles` as a focused example of the CUDA concepts used in `cpp/2_Concepts_and_Techniques/particles`.
> **日本語**
> この sample の目的は、`particles` の小さな実装を通して CUDA Libraries, CUDA Graphs, Shared Memory, Streams And Events, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `particleSystem.cpp, particleSystem.cuh, particleSystem.h, particleSystem_cuda.cu, particles.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `data/ref_particles.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/particles.doc`: Supporting file used by `doc/particles.doc`.
- `doc/particles.pdf`: Supporting file used by `doc/particles.pdf`.
- `doc/screenshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `particleSystem.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particleSystem.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particleSystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `particleSystem_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particles_kernel_impl.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `render_particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `render_particles.h`: Host/device declarations, helper types, constants, or library wrappers.
- `shaders.cpp`: Host-side setup, API calls, validation, and cleanup.
- `shaders.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `particleSystem.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `particleSystem.cpp`: focus on `CUDA`, `cudaMalloc`, `cudaFree`, `CUDART_PI_F`, `cudaGraphicsResource`.
- `particleSystem.cuh`: focus on `cudaGraphicsResource`, `CUDA`, `cudaInit`.
- `particleSystem.h`: focus on `CUDA`, `cudaGraphicsResource`.
- `particleSystem_cuda.cu`: focus on `cudaGraphicsResource`, `thrust::device_ptr`, `launch`, `CUDA`, `cudaMemcpy`.
- `particles.cpp`: focus on `CUDA`, `gridDim`, `cudaInit`, `cudaDeviceSynchronize`, `atomic`.
- `particles_kernel.cuh`: focus on control flow and helper functions.
- `particles_kernel_impl.cuh`: focus on `cudaParams`, `threadIdx`, `blockIdx`, `blockDim`, `__shared__`.
- `render_particles.cpp`: focus on `CUDA`.
- `render_particles.h`: focus on control flow and helper functions.
- `shaders.cpp`: focus on control flow and helper functions.
- Additional source files: 1 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/2_Concepts_and_Techniques/particles/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(particles LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particleSystem.cpp`

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp:13-32
```cpp
// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include "particleSystem.h"

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_gl.h>
#include <math.h>
#include <memory.h>

#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp:149-168
```cpp
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles * 4];
    m_hVel = new float[m_numParticles * 4];
    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL) {
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp:244-263
```cpp
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

// step the simulation
void ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL) {
        dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particleSystem.cuh`

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.cuh:29-73
```cuda
extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    // JP: `cudaGraphicsResource`, `cuda_vbo_resource`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void  unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticles);

    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
}
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particleSystem.h`

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.h:29-47
```cpp
#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING  0

#include <helper_functions.h>

#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
    ~ParticleSystem();

    enum ParticleConfig { CONFIG_RANDOM, CONFIG_GRID, _NUM_CONFIGS };
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem.h:129-148
```cpp
    uint m_colorVBO; // vertex buffer object for colors

    float *m_cudaPosVBO;   // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO; // these are the CUDA deviceMem Color

    // JP: `cudaGraphicsResource`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
    struct cudaGraphicsResource *m_cuda_posvbo_resource;   // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3     m_gridSize;
    uint      m_numGridCells;

    StopWatchInterface *m_timer;

    uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particleSystem_cuda.cu`

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu:32-50
```cuda
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string.h>

#include "particles_kernel_impl.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu:65-95
```cuda
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    void allocateArray(void **devPtr, size_t size) { checkCudaErrors(cudaMalloc(devPtr, size)); }

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    void freeArray(void *devPtr) { checkCudaErrors(cudaFree(devPtr)); }

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    void threadSync() { checkCudaErrors(cudaDeviceSynchronize()); }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
    }

    // JP: `cudaGraphicsResource`, `cuda_vbo_resource`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu:152-171
```cuda
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
        calcHashD<<<numBlocks, numThreads>>>(gridParticleHash, gridParticleIndex, (float4 *)pos, numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particles.cpp`

Source: cpp/2_Concepts_and_Techniques/particles/particles.cpp:30-48
```cpp
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particles.cpp:143-162
```cpp
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL);
    psystem->reset(ParticleSystem::CONFIG_GRID);

    if (bUseOpenGL) {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particles.cpp:197-216
```cpp
}

void runBenchmark(int iterations, char *exec_path)
{
    printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    for (int i = 0; i < iterations; ++i) {
        psystem->update(timestep);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer) / (float)iterations);

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u "
           "particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles) / fAvgSeconds,
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particles.cpp:661-680
```cpp
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKsample);

    printf("NOTE: The CUDA Samples are not meant for performance measurements. "
           "Results may vary when GPU Boost is enabled.\n\n");

    numParticles  = NUM_PARTICLES;
    // JP: `gridDim`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    uint gridDim  = GRID_SIZE;
    numIterations = 0;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particles_kernel.cuh`

Source: cpp/2_Concepts_and_Techniques/particles/particles_kernel.cuh:29-60
```cuda
#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float  globalDamping;
    float  particleRadius;

    uint3  gridSize;
    uint   numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
};

#endif
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particles_kernel_impl.cuh`

Source: cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh:33-51
```cuda
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <cooperative_groups.h>
#include <math.h>
#include <stdio.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

// for cuda::std::get
#include <cuda/std/utility>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh:70-89
```cuda
        float3          vel     = make_float3(velData.x, velData.y, velData.z);

        vel += cudaParams.gravity * deltaTime;
        vel *= cudaParams.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

// set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - cudaParams.particleRadius) {
            pos.x = 1.0f - cudaParams.particleRadius;
            vel.x *= cudaParams.boundaryDamping;
        }

        if (pos.x < -1.0f + cudaParams.particleRadius) {
            pos.x = -1.0f + cudaParams.particleRadius;
            vel.x *= cudaParams.boundaryDamping;
        }
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh:140-159
```cuda
__global__ void calcHashD(uint   *gridParticleHash,  // output
                          uint   *gridParticleIndex, // output
                          float4 *pos,               // input: positions
                          uint    numParticles)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles)
        return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash    = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index]  = hash;
    gridParticleIndex[index] = index;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `render_particles.cpp`

Source: cpp/2_Concepts_and_Techniques/particles/render_particles.cpp:24-47
```cpp
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では stream/event による非同期実行と同期、CUDA Graph の依存関係と replay、Runtime/Driver/NVRTC の境界 を確認します。英語の識別子/API/出力文字列は保持します。

#include <assert.h>
#include <math.h>
#include <stdio.h>

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
    : m_pos(0)
    , m_numParticles(0)
    , m_pointSize(1.0f)
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/render_particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `render_particles.h`

Source: cpp/2_Concepts_and_Techniques/particles/render_particles.h:29-76
```cpp
#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setVertexBuffer(unsigned int vbo, int numParticles);
    void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }

    enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPHERES, PARTICLE_NUM_MODES };

    void display(DisplayMode mode = PARTICLE_POINTS);
    void displayGrid();

    void setPointSize(float size) { m_pointSize = size; }
    void setParticleRadius(float r) { m_particleRadius = r; }
    void setFOV(float fov) { m_fov = fov; }
    void setWindowSize(int w, int h)
    {
        m_window_w = w;
        m_window_h = h;
    }

protected: // methods
    void   _initGL();
    void   _drawPoints();
    GLuint _compileProgram(const char *vsource, const char *fsource);

protected: // data
    float *m_pos;
    int    m_numParticles;

    float m_pointSize;
    float m_particleRadius;
    float m_fov;
    int   m_window_w, m_window_h;

    GLuint m_program;

    GLuint m_vbo;
    GLuint m_colorVBO;
};

#endif //__ RENDER_PARTICLES__
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/render_particles.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `shaders.cpp`

Source: cpp/2_Concepts_and_Techniques/particles/shaders.cpp:29-66
```cpp
#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(uniform float pointRadius; // point size in world space
                                     uniform float pointScale;  // scale to calculate size in pixels
                                     uniform float densityScale;
                                     uniform float densityOffset;
                                     void          main() {
                                         // calculate window-space point size
                                         vec3  posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
                                         float dist   = length(posEye);
                                         gl_PointSize = pointRadius * (pointScale / dist);

                                         gl_TexCoord[0] = gl_MultiTexCoord0;
                                         gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

                                         gl_FrontColor = gl_Color;
                                     });

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(void main() {
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy      = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0)
        discard; // kill pixels outside circle

    N.z = sqrt(1.0 - mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
});
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/shaders.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `shaders.h`

Source: cpp/2_Concepts_and_Techniques/particles/shaders.h:29-30
```cpp
extern const char *vertexShader;
extern const char *spherePixelShader;
```

> JP: この抜粋は `cpp/2_Concepts_and_Techniques/particles/shaders.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaParams` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaInit` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `thrust::device_ptr` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `atomic` | 複数 thread が同じ address を更新する箇所です。競合と順序の意味を確認します。 |
| `gridDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
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
cmake --build build --target particles
ctest --test-dir build -R particles
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaParams` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Performance](../../../docs_ja/themes/performance.md): memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
