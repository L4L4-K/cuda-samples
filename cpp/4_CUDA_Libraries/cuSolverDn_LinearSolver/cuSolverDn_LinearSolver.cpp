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
// JP: この file では memory ownership と host/device transfer、stream/event による非同期実行と同期、CUDA library の handle/descriptor/workspace を確認します。英語の識別子/API/出力文字列は保持します。

/*
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The user has to prepare a sparse matrix of "matrix market format" (with
 * extension .mtx). For example, the user can download matrices in Florida
 * Sparse Matrix Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by switch -R<solver> and
 *  to provide the path of the matrix by switch -F<file>, then
 *  the program solves
 *          A*x = b  where b = ones(m,1)
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|)
 *
 *  The elapsed time is also reported so the user can compare efficiency of
 * different solvers.
 *
 *  How to use
 *      ./cuSolverDn_LinearSolver                     // Default: cholesky
 *     ./cuSolverDn_LinearSolver -R=chol -filefile>   // cholesky factorization
 *     ./cuSolverDn_LinearSolver -R=lu -file<file>     // LU with partial
 * pivoting
 *     ./cuSolverDn_LinearSolver -R=qr -file<file>     // QR factorization
 *
 *  Remark: the absolute error on solution x is meaningless without knowing
 * condition number of A. The relative error on residual should be close to
 * machine zero, i.e. 1.e-15.
 */

#include <assert.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
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

void UsageDN(void)
{
    printf("<options>\n");
    printf("-h          : display this help\n");
    printf("-R=<name>    : choose a linear solver\n");
    printf("              chol (cholesky factorization), this is default\n");
    printf("              qr   (QR factorization)\n");
    printf("              lu   (LU factorization)\n");
    printf("-lda=<int> : leading dimension of A , m by default\n");
    printf("-file=<filename>: filename containing a matrix in MM format\n");
    printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit(0);
}

/*
 *  solve A*x = b by Cholesky factorization
 *
 */
// JP: `cusolverDnHandle_t`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double *Acopy, int lda, const double *b, double *x)
{
    int              bufferSize = 0;
    int             *info       = NULL;
    double          *buffer     = NULL;
    double          *A          = NULL;
    int              h_info     = 0;
    double           start, stop;
    double           time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double *)Acopy, lda, &bufferSize));

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));

    // prepare a copy of A because potrf will overwrite A with L
    // JP: `cudaMemcpy`, `cudaMemcpyDeviceToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info));

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf(stdout, "timing: cholesky = %10.6f sec\n", time_solve);

    if (info) {
        // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }

    return 0;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
// JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
int linearSolverLU(cusolverDnHandle_t handle, int n, const double *Acopy, int lda, const double *b, double *x)
{
    int     bufferSize = 0;
    int    *info       = NULL;
    double *buffer     = NULL;
    double *A          = NULL;
    int    *ipiv       = NULL; // pivoting sequence
    int     h_info     = 0;
    double  start, stop;
    double  time_solve;

    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double *)Acopy, lda, &bufferSize));

    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int) * n));

    // prepare a copy of A because getrf will overwrite A with L
    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf(stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info) {
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }
    if (ipiv) {
        checkCudaErrors(cudaFree(ipiv));
    }

    return 0;
}

/*
 *  solve A*x = b by QR
 *
 */
// JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
int linearSolverQR(cusolverDnHandle_t handle, int n, const double *Acopy, int lda, const double *b, double *x)
{
    cublasHandle_t cublasHandle     = NULL; // used in residual evaluation
    int            bufferSize       = 0;
    int            bufferSize_geqrf = 0;
    int            bufferSize_ormqr = 0;
    int           *info             = NULL;
    double        *buffer           = NULL;
    double        *A                = NULL;
    double        *tau              = NULL;
    int            h_info           = 0;
    double         start, stop;
    double         time_solve;
    const double   one = 1.0;

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double *)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(
        handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, NULL, x, n, &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);

    bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf : bufferSize_ormqr;

    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));
    checkCudaErrors(cudaMalloc((void **)&tau, sizeof(double) * n));

    // prepare a copy of A because getrf will overwrite A with L
    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    // compute QR factorization
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: QR factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(
        // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
        cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, A, lda, tau, x, n, buffer, bufferSize, info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(cublasHandle,
                                CUBLAS_SIDE_LEFT,
                                CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT,
                                n,
                                1,
                                &one,
                                A,
                                lda,
                                x,
                                n));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf(stdout, "timing: QR = %10.6f sec\n", time_solve);

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    if (cublasHandle) {
        checkCudaErrors(cublasDestroy(cublasHandle));
    }
    if (info) {
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }
    if (tau) {
        checkCudaErrors(cudaFree(tau));
    }

    return 0;
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
        UsageDN();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "R")) {
        char *solverType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

        if (solverType) {
            if ((STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0)
                && (STRCASECMP(solverType, "qr") != 0)) {
                printf("\nIncorrect argument passed to -R option\n");
                UsageDN();
            }
            else {
                opts.testFunc = solverType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName) {
            opts.sparse_mat_filename = fileName;
        }
        else {
            printf("\nIncorrect filename passed to -file \n ");
            UsageDN();
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "lda")) {
        opts.lda = getCmdLineArgumentInt(argc, (const char **)argv, "lda");
    }
}

int main(int argc, char *argv[])
{
    struct testOpts    opts;
    cusolverDnHandle_t handle       = NULL;
    cublasHandle_t     cublasHandle = NULL; // used in residual evaluation
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t       stream       = NULL;

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
    int lda   = 0; // leading dimension in dense matrix

    // CSR(A) from I/O
    int    *h_csrRowPtrA = NULL;
    int    *h_csrColIndA = NULL;
    double *h_csrValA    = NULL;

    double *h_A = NULL; // dense matrix from CSR(A)
    double *h_x = NULL; // a copy of d_x
    double *h_b = NULL; // b = ones(m,1)
    double *h_r = NULL; // r = b - A*x, a copy of d_r

    double *d_A = NULL; // a copy of h_A
    double *d_x = NULL; // x = A \ b
    double *d_b = NULL; // a copy of h_b
    double *d_r = NULL; // r = b - A*x

    // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one       = 1.0;

    double x_inf  = 0.0;
    double r_inf  = 0.0;
    double A_inf  = 0.0;
    int    errors = 0;

    parseCommandLineArguments(argc, argv, opts);

    if (NULL == opts.testFunc) {
        opts.testFunc = "chol"; // By default running Cholesky as NO solver
                                // selected with -R option.
    }

    findCudaDevice(argc, (const char **)argv);

    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename == NULL) {
        opts.sparse_mat_filename = sdkFindFilePath("gr_900_900_crg.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find gr_900_900_crg.mtx\n");
    }
    else {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }

    if (opts.sparse_mat_filename == NULL) {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return EXIT_FAILURE;
    }

    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename,
                                   'd',
                                   true,
                                   &rowsA,
                                   &colsA,
                                   &nnzA,
                                   &h_csrValA,
                                   &h_csrRowPtrA,
                                   &h_csrColIndA,
                                   true)) {
        exit(EXIT_FAILURE);
    }
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}

    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    if (rowsA != colsA) {
        fprintf(stderr, "Error: only support square matrix\n");
        exit(EXIT_FAILURE);
    }

    printf("step 2: convert CSR(A) to dense matrix\n");

    lda = opts.lda ? opts.lda : rowsA;
    if (lda < rowsA) {
        fprintf(stderr, "Error: lda must be greater or equal to dimension of A\n");
        exit(EXIT_FAILURE);
    }

    h_A = (double *)malloc(sizeof(double) * lda * colsA);
    h_x = (double *)malloc(sizeof(double) * colsA);
    h_b = (double *)malloc(sizeof(double) * rowsA);
    h_r = (double *)malloc(sizeof(double) * rowsA);
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    assert(NULL != h_A);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    memset(h_A, 0, sizeof(double) * lda * colsA);

    for (int row = 0; row < rowsA; row++) {
        const int start = h_csrRowPtrA[row] - baseA;
        const int end   = h_csrRowPtrA[row + 1] - baseA;
        for (int colidx = start; colidx < end; colidx++) {
            const int    col     = h_csrColIndA[colidx] - baseA;
            const double Areg    = h_csrValA[colidx];
            h_A[row + col * lda] = Areg;
        }
    }

    printf("step 3: set right hand side vector (b) to 1\n");
    for (int row = 0; row < rowsA; row++) {
        h_b[row] = 1.0;
    }

    // verify if A is symmetric or not.
    if (0 == strcmp(opts.testFunc, "chol")) {
        int issym = 1;
        for (int j = 0; j < colsA; j++) {
            for (int i = j; i < rowsA; i++) {
                double Aij = h_A[i + j * lda];
                double Aji = h_A[j + i * lda];
                if (Aij != Aji) {
                    issym = 0;
                    break;
                }
            }
        }
        if (!issym) {
            printf("Error: A has no symmetric pattern, please use LU or QR \n");
            exit(EXIT_FAILURE);
        }
    }

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamCreate(&stream));

    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double) * lda * colsA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

    printf("step 4: prepare data on device\n");
    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double) * lda * colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

    printf("step 5: solve A*x = b \n");
    // d_A and d_b are read-only
    if (0 == strcmp(opts.testFunc, "chol")) {
        linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x);
    }
    else if (0 == strcmp(opts.testFunc, "lu")) {
        linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x);
    }
    else if (0 == strcmp(opts.testFunc, "qr")) {
        linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x);
    }
    else {
        fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
        exit(EXIT_FAILURE);
    }
    printf("step 6: evaluate residual\n");
    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice));

    // r = b - A*x
    // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    checkCudaErrors(cublasDgemm_v2(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, 1, colsA, &minus_one, d_A, lda, d_x, rowsA, &one, d_r, rowsA));

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = mat_norminf(rowsA, colsA, h_A, lda);

    printf("|b - A*x| = %E \n", r_inf);
    printf("|A| = %E \n", A_inf);
    printf("|x| = %E \n", x_inf);
    printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

    if (handle) {
        // JP: この連続する anchor 群では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
        checkCudaErrors(cusolverDnDestroy(handle));
    }
    if (cublasHandle) {
        checkCudaErrors(cublasDestroy(cublasHandle));
    }
    if (stream) {
        // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaStreamDestroy(stream));
    }

    if (h_csrValA) {
        free(h_csrValA);
    }
    if (h_csrRowPtrA) {
        free(h_csrRowPtrA);
    }
    if (h_csrColIndA) {
        free(h_csrColIndA);
    }

    if (h_A) {
        free(h_A);
    }
    if (h_x) {
        free(h_x);
    }
    if (h_b) {
        free(h_b);
    }
    if (h_r) {
        free(h_r);
    }

    if (d_A) {
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaFree(d_A));
    }
    if (d_x) {
        checkCudaErrors(cudaFree(d_x));
    }
    if (d_b) {
        checkCudaErrors(cudaFree(d_b));
    }
    if (d_r) {
        checkCudaErrors(cudaFree(d_r));
    }

    return 0;
}
