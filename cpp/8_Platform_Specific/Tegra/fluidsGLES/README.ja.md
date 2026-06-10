# fluidsGLES - Fluids (OpenGLES Version) - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

An example of fluid simulation using CUDA and CUFFT, with OpenGLES rendering.

Graphics Interop, CUFFT Library, Physically-Based Simulation

Original README headings: `fluidsGLES - Fluids (OpenGLES Version)`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/8_Platform_Specific/Tegra/fluidsGLES` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `fluidsGLES` as a focused example of the CUDA concepts used in `cpp/8_Platform_Specific/Tegra/fluidsGLES`.
> **日本語**
> この sample の目的は、`fluidsGLES` の小さな実装を通して CUDA Libraries, CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `defines.h, fluidsGLES.cpp, fluidsGLES_kernels.cu, fluidsGLES_kernels.cuh, fluidsGLES_kernels.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG
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
- `data/ref_fluidsGLES.ppm`: Input, reference, generated-data description, or documentation used by the sample.
- `defines.h`: Host/device declarations, helper types, constants, or library wrappers.
- `fluidsGLES.cpp`: Host-side setup, API calls, validation, and cleanup.
- `fluidsGLES_kernels.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `fluidsGLES_kernels.cuh`: CUDA header with device functions, kernels, templates, or shared constants.
- `fluidsGLES_kernels.h`: Host/device declarations, helper types, constants, or library wrappers.
- `graphics_interface.h`: Host/device declarations, helper types, constants, or library wrappers.
- `mesh.frag.glsl`: Supporting file used by `mesh.frag.glsl`.
- `mesh.vert.glsl`: Supporting file used by `mesh.vert.glsl`.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `defines.h` first and locate the host-side setup or Python entry point.
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

- `defines.h`: focus on control flow and helper functions.
- `fluidsGLES.cpp`: focus on `CUDA`, `cudaFree`, `CUFFT`, `cufftHandle`, `launch`.
- `fluidsGLES_kernels.cu`: focus on `threadIdx`, `CUDA`, `launch`, `blockIdx`, `blockDim`.
- `fluidsGLES_kernels.cuh`: focus on `cudaTextureObject_t`.
- `fluidsGLES_kernels.h`: focus on `cudaTextureObject_t`.
- `graphics_interface.h`: focus on `CUDA`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/CMakeLists.txt:1-23
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/Modules")

project(fluidsGLES LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 87 110)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../../Common)
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/CMakeLists.txt:46-65
```cmake
                    ${CUDAToolkit_INCLUDE_DIRS}
                )

                target_link_libraries(fluidsGLES
                    CUDA::cuda_driver
                    # JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
                    CUDA::cufft
                    ${EGL_LIBRARY}
                    ${X11_LIBRARIES}
                    ${OPENGL_LIBRARIES}
                )

                # Copy the .glsl files to the output directory
                add_custom_command(TARGET fluidsGLES POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.frag.glsl
                    ${CMAKE_CURRENT_SOURCE_DIR}/mesh.vert.glsl
                    ${CMAKE_CURRENT_BINARY_DIR}
                )
            else()
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `defines.h`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/defines.h:29-48
```cpp
#ifndef DEFINES_H
#define DEFINES_H

#define DIM   512                 // Square size of solver domain
#define DS    (DIM * DIM)         // Total domain size
#define CPADW (DIM / 2 + 1)       // Padded width for real->complex in-place FFT
#define RPADW (2 * (DIM / 2 + 1)) // Padded width for real->complex in-place FFT
#define PDS   (DIM * CPADW)       // Padded total domain size

#define DT    0.09f        // Delta T for interative solver
#define VIS   0.0025f      // Viscosity constant
#define FORCE (5.8f * DIM) // Force scale factor
#define FR    4            // Force update radius

#define TILEX 64 // Tile width
#define TILEY 64 // Tile height
#define TIDSX 64 // Tids in X
#define TIDSY 4  // Tids in Y

#endif
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/defines.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `fluidsGLES.cpp`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp:14-32
```cpp
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void error_exit(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    exit(1);
}

// GLES related includes and Xlib and EGL stuff
#include "graphics_interface.h"
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp:62-81
```cpp

void cleanup(void);
void reshape(int x, int y);

// CUFFT plan handle
// JP: `cufftHandle`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
cufftHandle   planr2c;
cufftHandle   planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData     *hvfield = NULL;
cData     *dvfield = NULL;
static int wWidth  = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static int          clicked  = 0;
static int          fpsCount = 0;
static int          fpsLimit = 1;
StopWatchInterface *timer    = NULL;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp:229-248
```cpp
    }

    glAttachShader(new_shaderprogram, shader);

    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(data);
}

GLuint ShaderCreate(const char *vshader_filename, const char *fshader_filename)
{
    printf("Loading GLSL shaders %s %s\n", vshader_filename, fshader_filename);

    GLuint new_shaderprogram = glCreateProgram();

    GET_GLERROR(0);
    if (vshader_filename)
        readAndCompileShaderFromGLSLFile(new_shaderprogram, vshader_filename, GL_VERTEX_SHADER);

    GET_GLERROR(0);
    if (fshader_filename)
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp:451-470
```cpp

    displayFrame();

    fbo->unbindRenderPath();

    // compare to offical reference image, printing PASS or FAIL.
    printf("> (Frame %d) Readback BackBuffer\n", 100);
    g_CheckRender->readback(wWidth, wHeight);
    g_CheckRender->savePPM("fluidsGLES.ppm", true, NULL);

    if (!g_CheckRender->PPMvsPPM("fluidsGLES.ppm", ref_file, MAX_EPSILON_ERROR, 0.25f)) {
        g_TotalErrors++;
    }
}

// Run fluids Simulation
bool runFluidsSimulation(int argc, char **argv, char *ref_file)
{


```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `fluidsGLES_kernels.cu`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cu:29-66
```cuda
#include <cuda_runtime.h>
#include <cufft.h>       // CUDA FFT Libraries
#include <helper_cuda.h> // Helper functions for CUDA Error handling
#include <stdio.h>
#include <stdlib.h>

// OpenGL Graphics includes
#include <GLES3/gl31.h>

// FluidsGLES CUDA kernel definitions
#include "fluidsGLES_kernels.cuh"

// Texture object for reading velocity field
cudaTextureObject_t texObj;
static cudaArray   *array = NULL;

// Particle data
extern GLuint                       vbo;               // OpenGL vertex buffer object
// JP: `cudaGraphicsResource`, `cuda_vbo_resource`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
extern size_t      tPitch;
// JP: `cufftHandle`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
extern cufftHandle planr2c;
extern cufftHandle planc2r;
cData             *vxfield = NULL;
cData             *vyfield = NULL;

void setupTexture(int x, int y)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    // JP: `cudaMallocArray`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    cudaMallocArray(&array, &desc, y, x);
    getLastCudaError("cudaMalloc failed");

    cudaResourceDesc texRes;
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cu:80-122
```cuda
    checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}

void updateTexture(cData *data, size_t wib, size_t h, size_t pitch)
{
    checkCudaErrors(cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice));
}

void deleteTexture(void)
{
    // JP: `cudaDestroyTextureObject`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaDestroyTextureObject(texObj));
    checkCudaErrors(cudaFreeArray(array));
}

// Note that these kernels are designed to work with arbitrary
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch)
{
    // JP: `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int    tx = threadIdx.x;
    int    ty = threadIdx.y;
    cData *fj = (cData *)((char *)v + (ty + spy) * pitch) + tx + spx;

    cData vterm = *fj;
    tx -= r;
    ty -= r;
    float s = 1.f / (1.f + tx * tx * tx * tx + ty * ty * ty * ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

// This method performs the velocity advection step, where we
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `fluidsGLES_kernels.cuh`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cuh:29-72
```cuda
#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

#include "defines.h"

// Vector data type used to velocity and force fields
typedef float2 cData;

void setupTexture(int x, int y);
void updateTexture(cData *data, size_t w, size_t h, size_t pitch);
void deleteTexture(void);

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void
advectVelocity_k(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt, int lb, cudaTextureObject_t tex);

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the wave wave vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt, float visc, int lb);

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void updateVelocity_k(cData *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch);

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void advectParticles_k(cData *part, cData *v, int dx, int dy, float dt, int lb, size_t pitch);

#endif
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cuh` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `fluidsGLES_kernels.h`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.h:13-54
```cpp
#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

// Vector data type used to velocity and force fields
typedef float2 cData;

void setupTexture(int x, int y);
void updateTexture(cData *data, size_t w, size_t h, size_t pitch);
void deleteTexture(void);

// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void addForces_k(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void
advectVelocity_k(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt, int lb, cudaTextureObject_t tex);

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the wave wave vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void diffuseProject_k(cData *vx, cData *vy, int dx, int dy, float dt, float visc, int lb);

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void updateVelocity_k(cData *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch);

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).
__global__ void advectParticles_k(cData *part, cData *v, int dx, int dy, float dt, int lb, size_t pitch);

#endif
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `graphics_interface.h`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/graphics_interface.h:29-47
```cpp
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>

Display *display;
int      screen;
Window   win = 0;

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#define GET_GLERROR(ret)                                                                   \
                                                                                           \
    {                                                                                      \
        GLenum err = glGetError();                                                         \
        if (err != GL_NO_ERROR) {                                                          \
            fprintf(stderr, "[%s line %d] OpenGL Error: 0x%x\n", __FILE__, __LINE__, err); \
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/graphics_interface.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/graphics_interface.h:112-131
```cpp
        error_exit("EGL failed to initialize\n");

    if (!eglChooseConfig(eglDisplay, configAttrs, NULL, 0, &configCount) || !configCount)
        error_exit("EGL failed to return any matching configurations\n");

    configList = (EGLConfig *)malloc(configCount * sizeof(EGLConfig));

    if (!eglChooseConfig(eglDisplay, configAttrs, configList, configCount, &configCount) || !configCount)
        error_exit("EGL failed to populate configuration list\n");

    Window               xRootWindow = DefaultRootWindow(display);
    XSetWindowAttributes xCreateWindowAttributes;
    xCreateWindowAttributes.event_mask = ExposureMask;
    win                                = XCreateWindow(display,
                        xRootWindow,
                        0,
                        0,
                        width,
                        height,
                        0,
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/graphics_interface.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `mesh.frag.glsl`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.frag.glsl:28-29
```glsl
void main()
{
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.frag.glsl` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `mesh.vert.glsl`

Source: cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.vert.glsl:28-37
```glsl
attribute vec4 a_position;

uniform mat4 projection;
uniform mat4 modelview;

void main()
{
  gl_PointSize = 1.0;
  gl_Position = projection * modelview * a_position;
}
```

> JP: この抜粋は `cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.vert.glsl` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `launch` | Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。 |
| `threadIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `CUFFT` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `blockIdx` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `blockDim` | thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。 |
| `cufftHandle` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpy` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cufft` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaGraphicsResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsGLRegisterBuffer` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaGraphicsUnregisterResource` | CUDA Graph の node、capture、instantiate、launch、update の境界を表します。 |
| `cudaTextureObject_t` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |

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
cmake --build build --target fluidsGLES
ctest --test-dir build -R fluidsGLES
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
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。

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
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Libraries](../../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [CUDA Graphs](../../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Streams And Events](../../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
