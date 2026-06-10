/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では memory ownership と host/device transfer、stream/event による非同期実行と同期、Unified Memory の migration と同期 を確認します。英語の識別子/API/出力文字列は保持します。

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>      // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradientUM";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        }
        else {
            I[1] = 2;
        }

        start        = (i - 1) * 3 + 2;
        J[start]     = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start]     = val[start - 1];
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

int main(int argc, char **argv)
{
    int         N = 0, nz = 0, *I = NULL, *J = NULL;
    float      *val      = NULL;
    const float tol      = 1e-5f;
    const int   max_iter = 10000;
    float      *x;
    float      *rhs;
    float       a, b, na, r0, r1;
    float       dot;
    float      *r, *p, *Ax;
    int         k;
    float       alpha, beta, alpham1;

    printf("Starting [%s]...\n", sSDKname);

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int            devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (!deviceProp.managedMemory) {
        // This samples requires being run on a device that supports Unified Memory
        // JP: managed_memory: Unified Memory は CPU/GPU で同じ pointer を使います。prefetch や同期で移動タイミングを意識します。
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount,
           deviceProp.major,
           deviceProp.minor);

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    N  = 1048576;
    nz = (N - 2) * 3 + 4;

    // JP: この連続する anchor 群では Unified Memory allocation/prefetch/advice です。migration、host/device visibility、同期位置 を確認します。
    cudaMallocManaged((void **)&I, sizeof(int) * (N + 1));
    cudaMallocManaged((void **)&J, sizeof(int) * nz);
    cudaMallocManaged((void **)&val, sizeof(float) * nz);

    genTridiag(I, J, val, N, nz);

    cudaMallocManaged((void **)&x, sizeof(float) * N);
    cudaMallocManaged((void **)&rhs, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        rhs[i] = 1.0;
        x[i]   = 0.0;
    }

    /* Get handle to the CUBLAS context */
    // JP: `cublasHandle_t`, `cublasHandle`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus           = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // temp memory for CG
    // JP: この連続する anchor 群では Unified Memory allocation/prefetch/advice です。migration、host/device visibility、同期位置 を確認します。
    checkCudaErrors(cudaMallocManaged((void **)&r, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&p, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&Ax, N * sizeof(float)));

    /* Wrap raw data into cuSPARSE generic API objects */
    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    cusparseSpMatDescr_t matA = NULL;
    checkCudaErrors(cusparseCreateCsr(
        &matA, N, N, nz, I, J, val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecx, N, x, CUDA_R_32F));
    cusparseDnVecDescr_t vecp = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, p, CUDA_R_32F));
    cusparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecAx, N, Ax, CUDA_R_32F));

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        r[i] = rhs[i];
    }

    alpha   = 1.0;
    alpham1 = -1.0;
    beta    = 0.0;
    r0      = 0.;

    /* Allocate workspace for cuSPARSE */
    size_t bufferSize = 0;
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            matA,
                                            vecx,
                                            &beta,
                                            vecAx,
                                            CUDA_R_32F,
                                            CUSPARSE_SPMV_ALG_DEFAULT,
                                            &bufferSize));
    void *buffer = NULL;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    checkCudaErrors(cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 matA,
                                 vecx,
                                 &beta,
                                 vecAx,
                                 CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                 buffer));

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);

    k = 1;

    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            b            = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
        }
        else {
            cublasStatus = cublasScopy(cublasHandle, N, r, 1, p, 1);
        }

        checkCudaErrors(cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha,
                                     matA,
                                     vecp,
                                     &beta,
                                     vecAx,
                                     CUDA_R_32F,
                                     CUSPARSE_SPMV_ALG_DEFAULT,
                                     buffer));
        // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
        cublasStatus = cublasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        a            = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
        na           = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

        r0           = r1;
        cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);
        // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    printf("Final residual: %e\n", sqrt(r1));

    fprintf(stdout, "&&&& conjugateGradientUM %s\n", (sqrt(r1) < tol) ? "PASSED" : "FAILED");

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++) {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++) {
            rsum += val[j] * x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err) {
            err = diff;
        }
    }

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    if (matA) {
        checkCudaErrors(cusparseDestroySpMat(matA));
    }
    if (vecx) {
        checkCudaErrors(cusparseDestroyDnVec(vecx));
    }
    if (vecAx) {
        checkCudaErrors(cusparseDestroyDnVec(vecAx));
    }
    if (vecp) {
        checkCudaErrors(cusparseDestroyDnVec(vecp));
    }

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    cudaFree(I);
    cudaFree(J);
    cudaFree(val);
    cudaFree(x);
    cudaFree(rhs);
    cudaFree(r);
    cudaFree(p);
    cudaFree(Ax);

    printf("Test Summary:  Error amount = %f, result = %s\n", err, (k <= max_iter) ? "SUCCESS" : "FAILURE");
    exit((k <= max_iter) ? EXIT_SUCCESS : EXIT_FAILURE);
}
