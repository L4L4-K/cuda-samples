# nbody - CUDA N-Body Simulation - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA.  This sample accompanies the GPU Gems 3 chapter "Fast N-Body Simulation with CUDA".  With CUDA 5.5, performance on Tesla K20c has increased to over 1.8TFLOP/s single precision.  Double Performance has also improved on all Kepler and Fermi GPU architectures as well.  Starting in CUDA 4.0, the nBody sample has been updated to take advantage of new features to easily scale the n-body simulation across multiple GPUs in a single PC.  Adding "-numbodies=<bodies>" to the command line will allow users to set # of bodies for simulation.  Adding “-numdevices=<N>” to the command line option will cause the sample to use N devices (if available) for simulation.  In this mode, the position and velocity data for all bodies are read from system memory using “zero copy” rather than from device memory.  For a small number of devices (4 or fewer) and a large enough number of bodies, bandwidth is not a bottleneck so we can achieve strong scaling across these devices.

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation

Original README headings: `nbody - CUDA N-Body Simulation`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/nbody` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `nbody` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/nbody`.
> **日本語**
> この sample の目的は、`nbody` の小さな実装を通して CUDA Graphs, Multi-GPU, P2P, And IPC, Shared Memory, Streams And Events, Synchronization And Atomics を具体的に追うことです。
>
> **学習メモ**
> 最初に `bodysystem.h, bodysystemcpu.h, bodysystemcpu_impl.h, bodysystemcuda.cu, bodysystemcuda.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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

- `.vscode/c_cpp_properties.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `.vscode/extensions.json`: Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior.
- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `bodysystem.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bodysystemcpu.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bodysystemcpu_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bodysystemcuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `bodysystemcuda.h`: Host/device declarations, helper types, constants, or library wrappers.
- `bodysystemcuda_impl.h`: Host/device declarations, helper types, constants, or library wrappers.
- `doc/nbody_gems3_ch31.pdf`: Supporting file used by `doc/nbody_gems3_ch31.pdf`.
- `doc/screenshot_lg.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_md.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `doc/screenshot_sm.jpg`: Input, reference, generated-data description, or documentation used by the sample.
- `nbody.cpp`: Host-side setup, API calls, validation, and cleanup.
- `render_particles.cpp`: Host-side setup, API calls, validation, and cleanup.
- `render_particles.h`: Host/device declarations, helper types, constants, or library wrappers.
- `tipsy.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `bodysystem.h` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
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

- `bodysystem.h`: focus on control flow and helper functions.
- `bodysystemcpu.h`: focus on control flow and helper functions.
- `bodysystemcpu_impl.h`: focus on control flow and helper functions.
- `bodysystemcuda.cu`: focus on `blockDim`, `launch`, `CUDA`, `threadIdx`, `cudaGraphicsResource`.
- `bodysystemcuda.h`: focus on `CUDA`, `cudaEvent_t`, `cudaGraphicsResource`.
- `bodysystemcuda_impl.h`: focus on `CUDA`, `cudaHostAlloc`, `cudaHostAllocMapped`, `cudaHostAllocPortable`, `cudaSetDevice`.
- `nbody.cpp`: focus on `CUDA`, `cudaEventRecord`, `cudaDeviceProp`, `cudaEventSynchronize`, `cudaGetDeviceProperties`.
- `render_particles.cpp`: focus on control flow and helper functions.
- `render_particles.h`: focus on control flow and helper functions.
- `tipsy.h`: focus on control flow and helper functions.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/nbody/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(nbody LANGUAGES C CXX CUDA)

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

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystem.h`

Source: cpp/5_Domain_Specific/nbody/bodysystem.h:29-47
```cpp
#ifndef __BODYSYSTEM_H__
#define __BODYSYSTEM_H__

#include <algorithm>

enum NBodyConfig { NBODY_CONFIG_RANDOM, NBODY_CONFIG_SHELL, NBODY_CONFIG_EXPAND, NBODY_NUM_CONFIGS };

enum BodyArray {
    BODYSYSTEM_POSITION,
    BODYSYSTEM_VELOCITY,
};

template <typename T> struct vec3
{
    typedef float Type;
}; // dummy
template <> struct vec3<float>
{
    typedef float3 Type;
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystem.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystemcpu.h`

Source: cpp/5_Domain_Specific/nbody/bodysystemcpu.h:29-78
```cpp
#ifndef __BODYSYSTEMCPU_H__
#define __BODYSYSTEMCPU_H__

#include "bodysystem.h"

// CPU Body System
template <typename T> class BodySystemCPU : public BodySystem<T>
{
public:
    BodySystemCPU(int numBodies);
    virtual ~BodySystemCPU();

    virtual void loadTipsyFile(const std::string &filename);

    virtual void update(T deltaTime);

    virtual void setSoftening(T softening) { m_softeningSquared = softening * softening; }
    virtual void setDamping(T damping) { m_damping = damping; }

    virtual T   *getArray(BodyArray array);
    virtual void setArray(BodyArray array, const T *data);

    virtual unsigned int getCurrentReadBuffer() const { return 0; }

    virtual unsigned int getNumBodies() const { return m_numBodies; }

protected:             // methods
    BodySystemCPU() {} // default constructor

    virtual void _initialize(int numBodies);
    virtual void _finalize();

    void _computeNBodyGravitation();
    void _integrateNBodySystem(T deltaTime);

protected: // data
    int  m_numBodies;
    bool m_bInitialized;

    T *m_pos;
    T *m_vel;
    T *m_force;

    T m_softeningSquared;
    T m_damping;
};

#include "bodysystemcpu_impl.h"

#endif // __BODYSYSTEMCPU_H__
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcpu.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystemcpu_impl.h`

Source: cpp/5_Domain_Specific/nbody/bodysystemcpu_impl.h:25-47
```cpp
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では stream/event による非同期実行と同期、performance measurement と memory access pattern を確認します。英語の識別子/API/出力文字列は保持します。

#include <algorithm>
#include <assert.h>
#include <helper_cuda.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "bodysystemcpu.h"
#include "tipsy.h"

#ifdef OPENMP
#include <omp.h>
#endif

template <typename T>
BodySystemCPU<T>::BodySystemCPU(int numBodies)
    : m_numBodies(numBodies)
    , m_bInitialized(false)
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcpu_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcpu_impl.h:66-85
```cpp
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    assert(!m_bInitialized);

    m_numBodies = numBodies;

    m_pos   = new T[m_numBodies * 4];
    m_vel   = new T[m_numBodies * 4];
    m_force = new T[m_numBodies * 3];

    memset(m_pos, 0, m_numBodies * 4 * sizeof(T));
    memset(m_vel, 0, m_numBodies * 4 * sizeof(T));
    memset(m_force, 0, m_numBodies * 3 * sizeof(T));

    m_bInitialized = true;
}

template <typename T> void BodySystemCPU<T>::_finalize()
{
    assert(m_bInitialized);

```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcpu_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystemcuda.cu`

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda.cu:29-81
```cuda
#include <helper_cuda.h>
#include <math.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cooperative_groups.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#include "bodysystem.h"

__constant__ float  softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    // JP: `cudaMemcpyToSymbol`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    return cudaMemcpyToSymbol(softeningSquared, &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64, &softeningSq, sizeof(double), 0, cudaMemcpyHostToDevice);
}

// JP: shared_memory: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
template <class T> struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template <typename T> __device__ T rsqrt_T(T x) { return rsqrt(x); }

template <> __device__ float rsqrt_T<float>(float x) { return rsqrtf(x); }

```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda.cu:182-201
```cuda
    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position, oldPos, numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in
    // bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda.cu:232-283
```cuda
            (void **)&(deviceData[0].dPos[1 - currentRead]), &bytes, pgres[1 - currentRead]));
    }

    for (unsigned int dev = 0; dev != numDevices; dev++) {
        if (numDevices > 1) {
            cudaSetDevice(dev);
        }

        int numBlocks     = (deviceData[dev].numBodies + blockSize - 1) / blockSize;
        int numTiles      = (numBodies + blockSize - 1) / blockSize;
        int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos

        integrateBodies<T>
            // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
            <<<numBlocks, blockSize, sharedMemSize>>>((typename vec4<T>::Type *)deviceData[dev].dPos[1 - currentRead],
                                                      (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
                                                      (typename vec4<T>::Type *)deviceData[dev].dVel,
                                                      deviceData[dev].offset,
                                                      deviceData[dev].numBodies,
                                                      deltaTime,
                                                      damping,
                                                      numTiles);

        if (numDevices > 1) {
            // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
            checkCudaErrors(cudaEventRecord(deviceData[dev].event));
            // MJH: Hack on older driver versions to force kernel launches to flush!
            cudaStreamQuery(0);
        }

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1) {
        for (unsigned int dev = 0; dev < numDevices; dev++) {
            // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
            checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
        }
    }

    if (bUsePBO) {
        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        checkCudaErrors(cudaGraphicsUnmapResources(2, pgres, 0));
    }
}

// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float>     *deviceData,
                                          cudaGraphicsResource **pgres,
                                          unsigned int           currentRead,
                                          float                  deltaTime,
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystemcuda.h`

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda.h:29-52
```cpp
#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__

#include "bodysystem.h"

template <typename T> struct DeviceData
{
    T           *dPos[2]; // mapped host pointers
    T           *dVel;
    // JP: `cudaEvent_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <typename T> class BodySystemCUDA : public BodySystem<T>
{
public:
    BodySystemCUDA(unsigned int numBodies,
                   unsigned int numDevices,
                   unsigned int blockSize,
                   bool         usePBO,
                   bool         useSysMem = false,
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `bodysystemcuda_impl.h`

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h:25-47
```cpp
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では memory ownership と host/device transfer、stream/event による非同期実行と同期、CUDA Graph の依存関係と replay を確認します。英語の識別子/API/出力文字列は保持します。

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <math.h>
#include <memory.h>
#include <vector>

template <typename T>
void integrateNbodySystem(DeviceData<T>         *deviceData,
                          // JP: `cudaGraphicsResource`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
                          cudaGraphicsResource **pgres,
                          unsigned int           currentRead,
                          float                  deltaTime,
                          float                  damping,
                          unsigned int           numBodies,
                          unsigned int           numDevices,
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h:93-112
```cpp

    m_numBodies = numBodies;

    unsigned int memSize = sizeof(T) * 4 * numBodies;

    m_deviceData = new DeviceData<T>[m_numDevices];

    // divide up the workload amongst Devices
    float *weights = new float[m_numDevices];
    int   *numSms  = new int[m_numDevices];
    float  total   = 0;

    for (unsigned int i = 0; i < m_numDevices; i++) {
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));

        // Choose the weight based on the Compute Capability
        // We estimate that a CC2.0 SM is about 4.0x faster than a CC 1.x SM for
        // this application (since a 15-SM GF100 is about 2X faster than a 30-SM
        // GT200).
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h:155-192
```cpp
        memset(m_hPos[1], 0, memSize);
        memset(m_hVel, 0, memSize);

        for (unsigned int i = 0; i < m_numDevices; i++) {
            if (m_numDevices > 1) {
                checkCudaErrors(cudaSetDevice(i));
            }

            // JP: `cudaEventCreate`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
            checkCudaErrors(cudaEventCreate(&m_deviceData[i].event));
            checkCudaErrors(cudaHostGetDevicePointer((void **)&m_deviceData[i].dPos[0], (void *)m_hPos[0], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void **)&m_deviceData[i].dPos[1], (void *)m_hPos[1], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void **)&m_deviceData[i].dVel, (void *)m_hVel, 0));
        }
    }
    else {
        m_hPos[0] = new T[m_numBodies * 4];
        m_hVel    = new T[m_numBodies * 4];

        memset(m_hPos[0], 0, memSize);
        memset(m_hVel, 0, memSize);

        checkCudaErrors(cudaSetDevice(m_devID));
        // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaEventCreate(&m_deviceData[0].event));

        if (m_bUsePBO) {
            // create the position pixel buffer objects for rendering
            // we will actually compute directly from this memory in CUDA too
            glGenBuffers(2, (GLuint *)m_pbo);

            for (int i = 0; i < 2; ++i) {
                glBindBuffer(GL_ARRAY_BUFFER, m_pbo[i]);
                glBufferData(GL_ARRAY_BUFFER, memSize, m_hPos[0], GL_DYNAMIC_DRAW);

                int size = 0;
                glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint *)&size);

```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h:369-388
```cpp

    if (!m_bUseSysMem) {
        if (pgres) {
            // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
            checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
            checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
            size_t bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ddata, &bytes, pgres));
        }

        // JP: `cudaMemcpy`, `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        checkCudaErrors(cudaMemcpy(hdata, ddata, m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        if (pgres) {
            checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
        }
    }

    return hdata;
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `nbody.cpp`

Source: cpp/5_Domain_Specific/nbody/nbody.cpp:29-56
```cpp
#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>
#include <paramgl.h>

#include "bodysystemcpu.h"
#include "bodysystemcuda.h"
#include "cuda_runtime.h"
#include "render_particles.h"

```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/nbody.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/nbody.cpp:166-187
```cpp
cudaEvent_t hostMemSyncEvent;

template <typename T> class NBodyDemo
{
public:
    static void Create() { m_singleton = new NBodyDemo; }
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    static void Destroy() { delete m_singleton; }

    static void init(int  numBodies,
                     int  numDevices,
                     int  blockSize,
                     bool usePBO,
                     bool useHostMem,
                     bool useP2P,
                     bool useCpu,
                     int  devID)
    {
        m_singleton->_init(numBodies, numDevices, blockSize, usePBO, useHostMem, useP2P, useCpu, devID);
    }

    static void reset(int numBodies, NBodyConfig config) { m_singleton->_reset(numBodies, config); }
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/nbody.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/nbody.cpp:989-1008
```cpp
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    bool bTestResults = true;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("\n> Command line options\n");
        showHelp();
        return 0;
    }

    printf("Run \"nbody -benchmark [-numbodies=<numBodies>]\" to measure "
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/nbody.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `render_particles.cpp`

Source: cpp/5_Domain_Specific/nbody/render_particles.cpp:29-51
```cpp
#include "render_particles.h"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

// includes for OpenGL
#include <helper_gl.h>

// includes
#include <assert.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>

#define GL_POINT_SPRITE_ARB             0x8861
#define GL_COORD_REPLACE_ARB            0x8862
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV 0x8642

ParticleRenderer::ParticleRenderer()
    : m_pos(0)
    , m_numParticles(0)
    , m_pointSize(1.0f)
    , m_spriteSize(2.0f)
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/render_particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/nbody/render_particles.cpp:335-354
```cpp
    return (B0 * pA + B1 * pB + B2 * vA + B3 * vB);
}

unsigned char *createGaussianMap(int N)
{
    float         *M = new float[2 * N * N];
    unsigned char *B = new unsigned char[4 * N * N];
    float          X, Y, Y2, Dist;
    float          Incr = 2.0f / N;
    int            i    = 0;
    int            j    = 0;
    Y                   = -1.0f;

    // float mmax = 0;
    for (int y = 0; y < N; y++, Y += Incr) {
        Y2 = Y * Y;
        X  = -1.0f;

        for (int x = 0; x < N; x++, X += Incr, i += 2, j += 4) {
            Dist = (float)sqrtf(X * X + Y2);
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/render_particles.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `render_particles.h`

Source: cpp/5_Domain_Specific/nbody/render_particles.h:29-80
```cpp
#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setPositions(double *pos, int numParticles);
    void setBaseColor(float color[4]);
    void setColors(float *color, int numParticles);
    void setPBO(unsigned int pbo, int numParticles, bool fp64);

    enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPRITES, PARTICLE_SPRITES_COLOR, PARTICLE_NUM_MODES };

    void display(DisplayMode mode = PARTICLE_POINTS);

    void setPointSize(float size) { m_pointSize = size; }
    void setSpriteSize(float size) { m_spriteSize = size; }

    void resetPBO();

protected: // methods
    void _initGL();
    void _createTexture(int resolution);
    void _drawPoints(bool color = false);

protected: // data
    float  *m_pos;
    double *m_pos_fp64;
    int     m_numParticles;

    float m_pointSize;
    float m_spriteSize;

    unsigned int m_vertexShader;
    unsigned int m_vertexShaderPoints;
    unsigned int m_pixelShader;
    unsigned int m_programPoints;
    unsigned int m_programSprites;
    unsigned int m_texture;
    unsigned int m_pbo;
    unsigned int m_vboColor;

    float m_baseColor[4];

    bool m_bFp64Positions;
};

#endif //__ RENDER_PARTICLES__
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/render_particles.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `tipsy.h`

Source: cpp/5_Domain_Specific/nbody/tipsy.h:3-21
```cpp
#ifndef __TIPSY_H__
#define __TIPSY_H__

#include <string>

using namespace std;

#define MAXDIM 3

typedef float Real;

struct gas_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real rho;
    Real temp;
    Real hsmooth;
```

> JP: この抜粋は `cpp/5_Domain_Specific/nbody/tipsy.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaEventRecord` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `cudaGraphicsResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaEventCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventDestroy` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaEvent_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaEventSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cudaDeviceProp` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaError_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaMemcpyToSymbol` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |

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
- performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target nbody
ctest --test-dir build -R nbody
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
- peer access が有効な device pair と、単に複数 GPU が存在することを混同する。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaEventRecord` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
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

- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Multi-GPU, P2P, And IPC](../../../docs_ja/themes/multi_gpu_p2p_ipc.md): device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。
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
