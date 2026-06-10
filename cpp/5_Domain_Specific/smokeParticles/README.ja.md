# smokeParticles - Smoke Particles - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

Smoke simulation with volumetric shadows using half-angle slicing technique. Uses CUDA for procedural simulation, Thrust Library for sorting algorithms, and OpenGL for graphics rendering.

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation

Original README headings: `smokeParticles - Smoke Particles`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/smokeParticles` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `smokeParticles` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/smokeParticles`.
> **日本語**
> この sample の目的は、`smokeParticles` の小さな実装を通して CUDA Libraries, CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `GLSLProgram.cpp, GLSLProgram.h, GpuArray.h, ParticleSystem.cpp, ParticleSystem.cuh` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `GLSLProgram.cpp`: Host-side setup, API calls, validation, and cleanup.
- `GLSLProgram.h`: Host/device declarations, helper types, constants, or library wrappers.
- `GpuArray.h`: Host/device declarations, helper types, constants, or library wrappers.
- `ParticleSystem.cpp`: Host-side setup, API calls, validation, and cleanup.
- `ParticleSystem.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `ParticleSystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `ParticleSystem_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `SmokeRenderer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SmokeRenderer.h`: Host/device declarations, helper types, constants, or library wrappers.
- `SmokeShaders.cpp`: Host-side setup, API calls, validation, and cleanup.
- `SmokeShaders.h`: Host/device declarations, helper types, constants, or library wrappers.
- `data/floortile.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_smokePart_pos.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `data/ref_smokePart_vel.bin`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_lg.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_md.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_sm.png`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/smokeParticles.doc`: Supporting file used by `doc/smokeParticles.doc`.
- `doc/smokeParticles.pdf`: Supporting file used by `doc/smokeParticles.pdf`.
- `framebufferObject.cpp`: Host-side setup, API calls, validation, and cleanup.
- `framebufferObject.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvMath.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvMatrix.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvQuaternion.h`: Host/device declarations, helper types, constants, or library wrappers.
- `nvVector.h`: Host/device declarations, helper types, constants, or library wrappers.
- `particleDemo.cpp`: Host-side setup, API calls, validation, and cleanup.
- `particles_kernel.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `particles_kernel_device.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `renderbuffer.cpp`: Host-side setup, API calls, validation, and cleanup.
- `renderbuffer.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `GLSLProgram.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
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

- `GLSLProgram.cpp`: focus on control flow and helper functions.
- `GLSLProgram.h`: focus on control flow and helper functions.
- `GpuArray.h`: focus on `CUDA`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaGraphicsResource`.
- `ParticleSystem.cpp`: focus on `CUDART_PI_F`, `CUDA`.
- `ParticleSystem.cuh`: focus on control flow and helper functions.
- `ParticleSystem.h`: focus on `CUDA`.
- `ParticleSystem_cuda.cu`: focus on `thrust::device_ptr`, `thrust::make_zip_iterator`, `cudaAddressModeWrap`, `CUDA`, `cudaMemcpyHostToDevice`.
- `SmokeRenderer.cpp`: focus on control flow and helper functions.
- `SmokeRenderer.h`: focus on control flow and helper functions.
- `SmokeShaders.cpp`: focus on control flow and helper functions.
- Additional source files: 12 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/smokeParticles/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(smokeParticles LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `GLSLProgram.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/GLSLProgram.cpp:29-47
```cpp
#include <stdlib.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "GLSLProgram.h"

GLSLProgram::GLSLProgram(const char *vsource, const char *fsource) { mProg = compileProgram(vsource, 0, fsource); }

GLSLProgram::GLSLProgram(const char *vsource, const char *gsource, const char *fsource, GLenum gsInput, GLenum gsOutput)
{
    mProg = compileProgram(vsource, gsource, fsource, gsInput, gsOutput);
}

GLSLProgram::~GLSLProgram()
{
    if (mProg) {
        glDeleteProgram(mProg);
    }
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GLSLProgram.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `GLSLProgram.h`

Source: cpp/5_Domain_Specific/smokeParticles/GLSLProgram.h:31-72
```cpp
#ifndef GLSL_PROGRAM_H
#define GLSL_PROGRAM_H

#include <stdio.h>

class GLSLProgram
{
public:
    // construct program from strings
    GLSLProgram(const char *vsource, const char *fsource);
    GLSLProgram(const char *vsource,
                const char *gsource,
                const char *fsource,
                GLenum      gsInput  = GL_POINTS,
                GLenum      gsOutput = GL_TRIANGLE_STRIP);
    ~GLSLProgram();

    void enable();
    void disable();

    void setUniform1f(const GLchar *name, GLfloat x);
    void setUniform2f(const GLchar *name, GLfloat x, GLfloat y);
    void setUniform3f(const char *name, float x, float y, float z);
    void setUniform4f(const char *name, float x, float y, float z, float w);
    void setUniformfv(const GLchar *name, GLfloat *v, int elementSize, int count = 1);
    void setUniformMatrix4fv(const GLchar *name, GLfloat *m, bool transpose);

    void bindTexture(const char *name, GLuint tex, GLenum target, GLint unit);

    inline GLuint getProgId() { return mProg; }

private:
    GLuint checkCompileStatus(GLuint shader, GLint *status);
    GLuint compileProgram(const char *vsource,
                          const char *gsource,
                          const char *fsource,
                          GLenum      gsInput  = GL_POINTS,
                          GLenum      gsOutput = GL_TRIANGLE_STRIP);
    GLuint mProg;
};

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GLSLProgram.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `GpuArray.h`

Source: cpp/5_Domain_Specific/smokeParticles/GpuArray.h:30-48
```cpp
   Class to represent an array in GPU and CPU memory
*/

#include <stdio.h>
#include <stdlib.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_gl.h>

template <class T> class GpuArray
{
public:
    GpuArray();
    ~GpuArray();

    enum Direction {
        HOST_TO_DEVICE,
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GpuArray.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/GpuArray.h:148-167
```cpp
    if (m_dptr) {
        freeDevice();
    }
}

template <class T> void GpuArray<T>::allocHost() { m_hptr = (T *)new T[m_size]; }

template <class T> void GpuArray<T>::freeHost()
{
    if (m_hptr) {
        delete[] m_hptr;
        m_hptr = 0;
    }
}

template <class T> void GpuArray<T>::allocDevice()
{
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&m_dptr[0], m_size * sizeof(T)));

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GpuArray.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/GpuArray.h:171-190
```cpp
}

template <class T> void GpuArray<T>::freeDevice()
{
    if (m_dptr[0]) {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFree(m_dptr[0]));
        m_dptr[0] = 0;
    }

    if (m_dptr[1]) {
        checkCudaErrors(cudaFree(m_dptr[1]));
        m_dptr[1] = 0;
    }
}

template <class T> GLuint GpuArray<T>::createVbo(size_t size, bool useElementArray)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GpuArray.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/GpuArray.h:235-254
```cpp

template <class T> void GpuArray<T>::map()
{
    if (m_vbo[0]) {
        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_vbo_resource[0], 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dptr[0], &num_bytes, m_cuda_vbo_resource[0]));
    }

    if (m_doubleBuffer && m_vbo[1]) {
        checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_vbo_resource[1], 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dptr[1], &num_bytes, m_cuda_vbo_resource[1]));
    }
}

template <class T> void GpuArray<T>::unmap()
{
    if (m_vbo[0]) {
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/GpuArray.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ParticleSystem.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cpp:25-47
```cpp
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では stream/event による非同期実行と同期、Runtime/Driver/NVRTC の境界、performance measurement と memory access pattern を確認します。英語の識別子/API/出力文字列は保持します。

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <memory.h>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

// includes for OpenGL
#include <helper_gl.h>

// includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ParticleSystem.cuh"
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ParticleSystem.cuh`

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cuh:29-47
```cuda
#include "particles_kernel.cuh"

extern "C"
{
    void initCuda(bool bUseGL);
    void setParameters(SimParams *hostParams);
    void createNoiseTexture(int w, int h, int d);

    void
    integrateSystem(float4 *oldPos, float4 *newPos, float4 *oldVel, float4 *newVel, float deltaTime, int numParticles);

    void calcDepth(float4 *pos,
                   float  *keys,    // output
                   uint   *indices, // output
                   float3  sortVector,
                   int     numParticles);

    void sortParticles(float *sortKeys, uint *indices, uint numParticles);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ParticleSystem.h`

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem.h:29-47
```cpp
#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>

#include "GpuArray.h"
#include "nvMath.h"
#include "particles_kernel.cuh"
#include "vector_functions.h"

using namespace nv;

// CUDA BodySystem: runs on the GPU
class ParticleSystem
{
public:
    ParticleSystem(uint numParticles, bool bUseVBO = true, bool bUseGL = true);
    ~ParticleSystem();

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ParticleSystem_cuda.cu`

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu:30-48
```cuda
This file contains simple wrapper functions that call the CUDA kernels
*/
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

// includes for OpenGL
#include <helper_gl.h>

// includes
#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <string.h>

#include "ParticleSystem.cuh"
#include "particles_kernel_device.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu:54-94
```cuda
    cudaArray *noiseArray;

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(cudaParams, hostParams, sizeof(SimParams)));
    }

    // Round a / b to nearest higher integer value
    int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks  = iDivUp(n, numThreads);
    }

    inline float frand() { return rand() / (float)RAND_MAX; }

    // create 3D texture containing random values
    void createNoiseTexture(int w, int h, int d)
    {
        cudaExtent size     = make_cudaExtent(w, h, d);
        size_t     elements = size.width * size.height * size.depth;

        float *volumeData = (float *)malloc(elements * 4 * sizeof(float));
        float *ptr        = volumeData;

        for (size_t i = 0; i < elements; i++) {
            *ptr++ = frand() * 2.0f - 1.0f;
            *ptr++ = frand() * 2.0f - 1.0f;
            *ptr++ = frand() * 2.0f - 1.0f;
            *ptr++ = frand() * 2.0f - 1.0f;
        }

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        checkCudaErrors(cudaMalloc3DArray(&noiseArray, &channelDesc, size));

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr =
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu:98-117
```cuda
        // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(volumeData);

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType         = cudaResourceTypeArray;
        texRes.res.array.array = noiseArray;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.addressMode[0]   = cudaAddressModeWrap;
        texDescr.addressMode[1]   = cudaAddressModeWrap;
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SmokeRenderer.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.cpp:30-48
```cpp
    This class renders particles using OpenGL and GLSL shaders
*/

#include <math.h>
#include <stdlib.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "SmokeRenderer.h"
#include "SmokeShaders.h"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#define USE_MBLUR         1
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.cpp:80-99
```cpp
    , m_imageTex(0)
    , m_depthTex(0)
    , m_imageFbo(0)
{
    // load shader programs
    m_simpleProg         = new GLSLProgram(particleVS, simplePS);
    m_particleProg       = new GLSLProgram(mblurVS, mblurGS, particlePS);
    m_particleShadowProg = new GLSLProgram(mblurVS, mblurGS, particleShadowPS);

    m_blurProg       = new GLSLProgram(passThruVS, blurPS);
    m_displayTexProg = new GLSLProgram(passThruVS, texture2DPS);

    // create buffer for light shadows
    createLightBuffer();

    glutReportErrors();
}

SmokeRenderer::~SmokeRenderer()
{
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SmokeRenderer.h`

Source: cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.h:31-49
```cpp
#ifndef SMOKE_RENDERER_H
#define SMOKE_RENDERER_H

#include "GLSLProgram.h"
#include "framebufferObject.h"
#include "nvMath.h"

using namespace nv;

class SmokeRenderer
{
public:
    SmokeRenderer(int maxParticles);
    ~SmokeRenderer();

    enum DisplayMode { POINTS, SPRITES, VOLUMETRIC, NUM_MODES };

    enum Target { LIGHT_BUFFER, SCENE_BUFFER };

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SmokeShaders.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/SmokeShaders.cpp:30-48
```cpp
#define STRINGIFY(A) #A

// particle vertex shader
const char *particleVS = STRINGIFY(
  uniform float pointRadius;  // point size in world space         \n
  uniform float pointScale;   // scale to calculate size in pixels \n
  uniform vec4 eyePos;                                             \n
  void main()                                                      \n
  {
    \n
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                          \n
    gl_Position = gl_ModelViewProjectionMatrix *wpos;              \n

    // calculate window-space point size                           \n
    vec4 eyeSpacePos = gl_ModelViewMatrix *wpos;                   \n
    float dist = length(eyeSpacePos.xyz);                          \n
    gl_PointSize = pointRadius * (pointScale / dist);              \n

    gl_TexCoord[0] = gl_MultiTexCoord0; // sprite texcoord         \n
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/SmokeShaders.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `SmokeShaders.h`

Source: cpp/5_Domain_Specific/smokeParticles/SmokeShaders.h:29-33
```cpp
extern const char *particleVS;
extern const char *particleSpherePS, *simplePS, *particlePS, *particleShadowPS;
extern const char *mblurVS, *mblurGS;
extern const char *passThruVS, *texture2DPS, *blurPS;
extern const char *floorVS, *floorPS;
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/SmokeShaders.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `framebufferObject.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/framebufferObject.cpp:3-21
```cpp
     Aaron Lefohn    (lefohn@cs.ucdavis.edu)
     Robert Strzodka (strzodka@stanford.edu)
     Adam Moerschell (atmoerschell@ucdavis.edu)
 All rights reserved.

 This software is licensed under the BSD open-source license. See
 http://www.opensource.org/licenses/bsd-license.php for more detail.

 *************************************************************
 Redistribution and use in source and binary forms, with or
 without modification, are permitted provided that the following
 conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/framebufferObject.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `framebufferObject.h`

Source: cpp/5_Domain_Specific/smokeParticles/framebufferObject.h:3-21
```cpp
  Aaron Lefohn   (lefohn@cs.ucdavis.edu)
  Robert Strzodka (strzodka@stanford.edu)
  Adam Moerschell (atmoerschell@ucdavis.edu)
 All rights reserved.

 This software is licensed under the BSD open-source license. See
 http://www.opensource.org/licenses/bsd-license.php for more detail.

 *************************************************************
 Redistribution and use in source and binary forms, with or
 without modification, are permitted provided that the following
 conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/framebufferObject.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvMath.h`

Source: cpp/5_Domain_Specific/smokeParticles/nvMath.h:25-91
```cpp
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
       // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
       FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
       REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
       INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
       BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
       LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
       CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
       ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
       POSSIBILITY OF SUCH DAMAGE.


    Cass Everitt - cass@r3.nu
*/

#ifndef NV_MATH_H
#define NV_MATH_H

#include <math.h>
#include <nvMatrix.h>
#include <nvQuaternion.h>
#include <nvVector.h>

#define NV_PI float(3.1415926535897932384626433832795)

namespace nv {

typedef vec2<float>        vec2f;
typedef vec3<float>        vec3f;
typedef vec3<int>          vec3i;
typedef vec3<unsigned int> vec3ui;
typedef vec4<float>        vec4f;
typedef matrix4<float>     matrix4f;
typedef quaternion<float>  quaternionf;

inline void applyRotation(const quaternionf &r)
{
    float angle;
    vec3f axis;
    r.get_value(axis, angle);
    glRotatef(angle / 3.1415926f * 180.0f, axis[0], axis[1], axis[2]);
}
}; // namespace nv

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/nvMath.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvMatrix.h`

Source: cpp/5_Domain_Specific/smokeParticles/nvMatrix.h:27-45
```cpp
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/nvMatrix.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvQuaternion.h`

Source: cpp/5_Domain_Specific/smokeParticles/nvQuaternion.h:27-45
```cpp
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/nvQuaternion.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nvVector.h`

Source: cpp/5_Domain_Specific/smokeParticles/nvVector.h:27-45
```cpp
    All rights reserved.

    Redistribution and use in source and binary forms, with or
    without modification, are permitted provided that the following
    conditions are met:

     * Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution.

     * The names of contributors to this software may not be used
       to endorse or promote products derived from this software
       without specific prior written permission.

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/nvVector.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particleDemo.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp:30-48
```cpp
    CUDA particle system with volumetric shadows
    sgreen 11/2008

    This sample demonstrates a technique for rendering realistic volumetric
    shadows through a cloud of particles. It uses CUDA for the simulation and
    depth sorting of the particles, and OpenGL for rendering.

    See the accompanying documentation for more details on the algorithm.

    This file handles OpenGL initialization and the user interface.
*/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <helper_gl.h>
#include <math.h>
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp:155-174
```cpp
void runEmitter();

// initialize particle system
void initParticles(int numParticles, bool bUseVBO, bool bUseGL)
{
    psystem = new ParticleSystem(numParticles, bUseVBO, bUseGL);
    psystem->reset(ParticleSystem::CONFIG_RANDOM);

    if (bUseVBO) {
        renderer = new SmokeRenderer(numParticles);
        renderer->setLightTarget(vec3f(0.0, 1.0, 0.0));

        sdkCreateTimer(&timer);
    }
}

void cleanup()
{
    if (psystem) {
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp:852-871
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

    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particles_kernel.cuh`

Source: cpp/5_Domain_Specific/smokeParticles/particles_kernel.cuh:29-52
```cuda
#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

struct SimParams
{
    float3 gravity;
    float  globalDamping;
    float  noiseFreq;
    float  noiseAmp;
    float3 cursorPos;

    float  time;
    float3 noiseSpeed;
};

struct float4x4
{
    float m[16];
};

#endif
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particles_kernel.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `particles_kernel_device.cuh`

Source: cpp/5_Domain_Specific/smokeParticles/particles_kernel_device.cuh:33-51
```cuda
#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

// for cuda::std::get
#include <cuda/std/utility>

cudaTextureObject_t noiseTex;
// simulation parameters
__constant__ SimParams cudaParams;

// look up in 3D noise texture
```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particles_kernel_device.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/smokeParticles/particles_kernel_device.cuh:91-110
```cuda

        // apply procedural noise
        float3 noise = noise3D(pos * cudaParams.noiseFreq + cudaParams.time * cudaParams.noiseSpeed, noiseTex);
        vel += noise * cudaParams.noiseAmp;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        vel *= cudaParams.globalDamping;

        // store new position and velocity
        cuda::std::get<0>(t) = make_float4(pos, age);
        cuda::std::get<1>(t) = make_float4(vel, velData.w);
    }
};

struct calcDepth_functor
{
    float3 sortVector;

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/particles_kernel_device.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `renderbuffer.cpp`

Source: cpp/5_Domain_Specific/smokeParticles/renderbuffer.cpp:15-33
```cpp
  Aaron Lefohn (lefohn@cs.ucdavis.edu)
  Adam Moerschell (atmoerschell@ucdavis.edu)
 All rights reserved.

 This software is licensed under the BSD open-source license. See
 http://www.opensource.org/licenses/bsd-license.php for more detail.

 *************************************************************
 Redistribution and use in source and binary forms, with or
 without modification, are permitted provided that the following
 conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/renderbuffer.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `renderbuffer.h`

Source: cpp/5_Domain_Specific/smokeParticles/renderbuffer.h:15-33
```cpp
  Aaron Lefohn (lefohn@cs.ucdavis.edu)
  Adam Moerschell (atmoerschell@ucdavis.edu)
 All rights reserved.

 This software is licensed under the BSD open-source license. See
 http://www.opensource.org/licenses/bsd-license.php for more detail.

 *************************************************************
 Redistribution and use in source and binary forms, with or
 without modification, are permitted provided that the following
 conditions are met:

 Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

```

> JP: この抜粋は `cpp/5_Domain_Specific/smokeParticles/renderbuffer.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `thrust::device_ptr` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaParams` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `thrust::make_zip_iterator` | CUDA library call です。handle/descriptor/workspace と data layout を確認します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyToSymbol` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaExtent` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaAddressModeWrap` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaCreateTextureObject` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGraphicsResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |

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
cmake --build build --target smokeParticles
ctest --test-dir build -R smokeParticles
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
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `thrust::device_ptr` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
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
