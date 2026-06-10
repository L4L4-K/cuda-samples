# cuSolverSp_LinearSolver - cuSolverSp Linear Solver - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A CUDA Sample that demonstrates cuSolverSP's LU, QR and Cholesky factorization.

Linear Algebra, CUSOLVER Library

Original README headings: `cuSolverSp_LinearSolver - cuSolverSp Linear Solver`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `cuSolverSp_LinearSolver` as a focused example of the CUDA concepts used in `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver`.
> **日本語**
> この sample の目的は、`cuSolverSp_LinearSolver` の小さな実装を通して Runtime, Driver, And NVRTC, CUDA Libraries, Shared Memory, Streams And Events, Memory を具体的に追うことです。
>
> **学習メモ**
> 最初に `cuSolverSp_LinearSolver.cpp, mmio.c, mmio.h, mmio_wrapper.cpp` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

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
- `cuSolverSp_LinearSolver.cpp`: Host-side setup, API calls, validation, and cleanup.
- `lap2D_5pt_n100.mtx`: Supporting file used by `lap2D_5pt_n100.mtx`.
- `lap3D_7pt_n20.mtx`: Supporting file used by `lap3D_7pt_n20.mtx`.
- `mmio.c`: Host-side setup, API calls, validation, and cleanup.
- `mmio.h`: Host/device declarations, helper types, constants, or library wrappers.
- `mmio_wrapper.cpp`: Host-side setup, API calls, validation, and cleanup.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `cuSolverSp_LinearSolver.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Create library handles, descriptors, plans, or workspaces before the library call.
- Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.
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

- `cuSolverSp_LinearSolver.cpp`: focus on `CUDA`, `cudaMemcpyAsync`, `cudaMalloc`, `cudaFree`, `cudaMemcpyHostToDevice`.
- `mmio.c`: focus on control flow and helper functions.
- `mmio.h`: focus on control flow and helper functions.
- `mmio_wrapper.cpp`: focus on `cuGet`, `cuComplex`, `cuDoubleComplex`, `CUDA`, `cusolverDn`.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/CMakeLists.txt:1-62
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

# JP: `cuSolverSp_LinearSolver`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。 CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
project(cuSolverSp_LinearSolver LANGUAGES C CXX)

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
# Add target for cuSolverSp_LinearSolver
add_executable(cuSolverSp_LinearSolver cuSolverSp_LinearSolver.cpp mmio.c mmio_wrapper.cpp)

target_compile_options(cuSolverSp_LinearSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(cuSolverSp_LinearSolver PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuSolverSp_LinearSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSolverSp_LinearSolver PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuSolverSp_LinearSolver PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LinearSolver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap2D_5pt_n100.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy data files to output directory
add_custom_command(TARGET cuSolverSp_LinearSolver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lap3D_7pt_n20.mtx
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `cuSolverSp_LinearSolver.cpp`

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp:32-73
```cpp
 extension .mtx).
 *  For example, the user can download matrices in Florida Sparse Matrix
 Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by the switch -R<solver> and
 *  to provide the path of the matrix by the switch -F<file>, then
 *  the program solves
 *          A*x = b
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|+|b|)
 *
 *  How does it work?
 *     The example solves A*x = b by the following steps
 *  step 1: B = A(Q,Q)
 *     Q is the ordering to minimize zero fill-in.
 *     The user can choose symrcm or symamd.
 *  step 2: solve B*z = Q*b
 *  step 3: x = inv(Q)*z
 *
 *  Above three steps can be combined by the formula
 *        (Q*A*Q')*(Q*x) = (Q*b)
 *
 *  The elapsed time is also reported so the user can compare efficiency of
 different solvers.
 *
 *  How to use
        // JP: `cuSolverSp_LinearSolver`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。 CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
        /cuSolverSp_LinearSolver            // Default: Cholesky, symrcm &
 file=lap2D_5pt_n100.mtx
 *     ./cuSolverSp_LinearSolver -R=chol  -file=<file>   // cholesky
 factorization
 *     ./cuSolverSp_LinearSolver -R=lu -P=symrcm -file=<file>     // symrcm + LU
 with partial pivoting
 *     ./cuSolverSp_LinearSolver -R=qr -P=symamd -file=<file>     // symamd + QR
 factorization
 *
 *
 *  Remark: the absolute error on solution x is meaningless without knowing
 condition number of A.
 *     The relative error on residual should be close to machine zero,
 i.e. 1.e-15.
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp:79-98
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cusolverSp.h"
#include "cusparse.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char    *filename,
                       char     elem_type,
                       bool     csrFormat,
                       int     *m,
                       int     *n,
                       int     *nnz,
                       T_ELEM **aVal,
                       int    **aRowInd,
                       int    **aColInd,
                       int      extendSymMatrix);
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp:288-310
```cpp
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverSpCreate(&handle));
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamCreate(&stream));
    /* bind stream to cusparse and cusolver*/
    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverSpSetStream(handle, stream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

    /* configure matrix descriptor*/
    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (baseA) {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    }
    else {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp:436-455
```cpp
        h_Qb[row] = h_b[h_Q[row]];
    }

    printf("step 4: prepare data on device\n");
    checkCudaErrors(
        // JP: `cudaMemcpyAsync`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
        cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
        cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrColIndB, h_csrColIndB, sizeof(int) * nnzA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA, cudaMemcpyHostToDevice, stream));

    printf("step 5: solve A*x = b on CPU \n");
    start = second();

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `mmio.c`

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c:11-29
```c
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "mmio.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_, double **val_, int **I_, int **J_)
{
    FILE       *f;
    MM_typecode matcode;
    int         M, N, nz;
    int         i;
    double     *val;
    int        *I, *J;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c:56-75
```c
    *N_  = N;
    *nz_ = nz;

    /* reserve memory for matrices */

    I   = (int *)malloc(nz * sizeof(int));
    J   = (int *)malloc(nz * sizeof(int));
    val = (double *)malloc(nz * sizeof(double));

    *val_ = val;
    *I_   = I;
    *J_   = J;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nz; i++) {
        if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) {
            return -1;
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c:365-384
```c
    char *str = mm_typecode_to_str(matcode);
    int   ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(str);
    if (ret_code != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[], double val[], MM_typecode matcode)
{
    FILE *f;
    int   i;

    if (strcmp(fname, "stdout") == 0)
        f = stdout;
    else if ((f = fopen(fname, "w")) == NULL)
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `mmio.h`

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.h:10-28
```cpp
#ifndef MM_IO_H
#define MM_IO_H

#include <stdio.h>

#if defined(__cplusplus)
extern "C"
{
#endif /* __cplusplus */

#define MM_MAX_LINE_LENGTH  1025
#define MatrixMarketBanner  "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

    typedef char MM_typecode[4];

    char *mm_typecode_to_str(MM_typecode matcode);

    int mm_read_banner(FILE *f, MM_typecode *matcode);
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `mmio_wrapper.cpp`

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp:4-22
```cpp
#include <cusolverDn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mmio.h"

/* avoid Windows warnings (for example: strcpy, fscanf, etc.) */
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

/* various __inline__ __device__  function to initialize a T_ELEM */
// JP: `cuGet`: Driver API は CU* handle を明示的に扱います。context/module/function の所有と error check を追います。
template <typename T_ELEM> __inline__ T_ELEM cuGet(int);
template <> __inline__ float                 cuGet<float>(int x) { return float(x); }

template <> __inline__ double cuGet<double>(int x) { return double(x); }

```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp:266-285
```cpp
        tempColInd = (int *)malloc((*nnz + count) * sizeof(int));
        if (mm_is_real(matcode) || mm_is_integer(matcode)) {
            tempVal = (double *)malloc((*nnz + count) * sizeof(double));
        }
        else {
            tempVal = (double *)malloc(2 * (*nnz + count) * sizeof(double));
        }
        // copy the elements regular and transposed locations
        for (j = 0, i = 0; i < (*nnz); i++) {
            tempRowInd[j] = trow[i];
            tempColInd[j] = tcol[i];
            if (mm_is_real(matcode) || mm_is_integer(matcode)) {
                tempVal[j] = tval[i];
            }
            else {
                tempVal[2 * j]     = tval[2 * i];
                tempVal[2 * j + 1] = tval[2 * i + 1];
            }
            j++;
            if (trow[i] != tcol[i]) {
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp:307-326
```cpp
            }
        }
        (*nnz) += count;
        // free temporary storage
        // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        free(trow);
        free(tcol);
        free(tval);
    }
    else {
        tempRowInd = trow;
        tempColInd = tcol;
        tempVal    = tval;
    }
    // life time of (trow, tcol, tval) is over.
    // please use COO format (tempRowInd, tempColInd, tempVal)

    // use qsort to sort COO format
    work = (struct cooFormat *)malloc(sizeof(struct cooFormat) * (*nnz));
    if (NULL == work) {
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp:393-412
```cpp

        *aColInd = cscColPtr;
        *aRowInd = (int *)malloc((*nnz) * sizeof(int));
    }

    /* transfrom the matrix values of type double into one of the cusparse library types */
    *aVal = (T_ELEM *)malloc((*nnz) * sizeof(T_ELEM));

    for (i = 0; i < (*nnz); i++) {
        if (csrFormat) {
            (*aColInd)[i] = tempColInd[i];
        }
        else {
            (*aRowInd)[i] = tempRowInd[i];
        }
        if (mm_is_real(matcode) || mm_is_integer(matcode)) {
            (*aVal)[i] = cuGet<T_ELEM>(tempVal[work[i].p]);
        }
        else {
            (*aVal)[i] = cuGet<T_ELEM>(tempVal[2 * work[i].p], tempVal[2 * work[i].p + 1]);
```

> JP: この抜粋は `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cuGet` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaMemcpyAsync` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cudaMalloc` | device 側 storage を確保する API です。対応する cleanup と byte size を確認します。 |
| `cuComplex` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cuDoubleComplex` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaMemcpyHostToDevice` | host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。 |
| `cusparseHandle` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUDA_R_64F` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cuSolverSp_LinearSolver` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `cudaDeviceSynchronize` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cusparse` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUSPARSE_INDEX_32I` | Driver API の handle 境界です。context/module/function と error code を追います。 |
| `CUSOLVER` | Driver API の handle 境界です。context/module/function と error code を追います。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。
- shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。

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
cmake --build build --target cuSolverSp_LinearSolver
ctest --test-dir build -R cuSolverSp_LinearSolver
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
- shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。
- different stream 間に依存があるのに event や explicit sync を置かない。
- leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。
- JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cuGet` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- shared memory tile の producer、consumer、barrier を図にする。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- handle/descriptor/workspace の作成、利用、破棄を対応表にする。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [Runtime, Driver, And NVRTC](../../../docs_ja/themes/runtime_driver_nvrtc.md): Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。
- [CUDA Libraries](../../../docs_ja/themes/libraries.md): handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。
- [Shared Memory](../../../docs_ja/themes/shared_memory.md): block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
