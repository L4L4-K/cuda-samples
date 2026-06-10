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
// JP: この file では memory ownership と host/device transfer、kernel launch と thread indexing、stream/event による非同期実行と同期 を確認します。英語の識別子/API/出力文字列は保持します。

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>

#include "jacobi.h"

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int  old            = *address_as_ull, assumed;

    do {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
        // NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

static __global__ void
JacobiMethod(const float *A, const double *b, const float conv_threshold, double *x, double *x_new, double *sum)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block  cta = cg::this_thread_block();
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ double x_shared[N_ROWS]; // N_ROWS == n
    __shared__ double b_shared[ROWS_PER_CTA + 1];

    for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
        x_shared[i] = x[i];
    }

    if (threadIdx.x < ROWS_PER_CTA) {
        int k = threadIdx.x;
        for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS);
             k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
            b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
        }
    }

    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (int k = 0, i = blockIdx.x * ROWS_PER_CTA; (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
        double rowThreadSum = 0.0;
        for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x) {
            rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
        }

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
        }

        if (tile32.thread_rank() == 0) {
            atomicAdd(&b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
        }
    }

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(cta);

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (threadIdx.x < ROWS_PER_CTA) {
        cg::thread_block_tile<ROWS_PER_CTA> tile8    = cg::tiled_partition<ROWS_PER_CTA>(cta);
        double                              temp_sum = 0.0;

        int k = threadIdx.x;

        for (int i = k + (blockIdx.x * ROWS_PER_CTA); (k < ROWS_PER_CTA) && (i < N_ROWS);
             k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
            double dx = b_shared[i % (ROWS_PER_CTA + 1)];
            dx /= A[i * N_ROWS + i];

            x_new[i] = (x_shared[i] + dx);
            temp_sum += fabs(dx);
        }

        for (int offset = tile8.size() / 2; offset > 0; offset /= 2) {
            temp_sum += tile8.shfl_down(temp_sum, offset);
        }

        if (tile8.thread_rank() == 0) {
            atomicAdd(sum, temp_sum);
        }
    }
}

// Thread block size for finalError kernel should be multiple of 32
static __global__ void finalError(double *x, double *g_sum)
{
    // Handle to thread block group
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    cg::thread_block         cta = cg::this_thread_block();
    // JP: この anchor では shared memory の block-local scratchpad です。producer/consumer の順序と必要な barrier を確認します。
    extern __shared__ double warpSum[];
    double                   sum = 0.0;

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = globalThreadId; i < N_ROWS; i += blockDim.x * gridDim.x) {
        double d = x[i] - 1.0;
        sum += fabs(d);
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        sum += tile32.shfl_down(sum, offset);
    }

    if (tile32.thread_rank() == 0) {
        // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
        warpSum[threadIdx.x / warpSize] = sum;
    }

    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(cta);

    double blockSum = 0.0;
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    if (threadIdx.x < (blockDim.x / warpSize)) {
        blockSum = warpSum[threadIdx.x];
    }

    if (threadIdx.x < 32) {
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            blockSum += tile32.shfl_down(blockSum, offset);
        }
        if (tile32.thread_rank() == 0) {
            atomicAdd(g_sum, blockSum);
        }
    }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(const float  *A,
                                                   const double *b,
                                                   const float   conv_threshold,
                                                   const int     max_iter,
                                                   double       *x,
                                                   double       *x_new,
                                                   // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                                                   cudaStream_t  stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3            nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
    // JP: `cudaGraph_t`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = NULL;

    double  sum   = 0.0;
    double *d_sum = NULL;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));

    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t              memcpyNode, jacobiKernelNode, memsetNode;
    cudaMemcpy3DParms            memcpyParams = {0};
    cudaMemsetParams             memsetParams = {0};

    memsetParams.dst   = (void *)d_sum;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
    memsetParams.elementSize = sizeof(float);
    memsetParams.width       = 2;
    memsetParams.height      = 1;

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    checkCudaErrors(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
    nodeDependencies.push_back(memsetNode);

    cudaKernelNodeParams NodeParams0, NodeParams1;
    NodeParams0.func           = (void *)JacobiMethod;
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    NodeParams0.gridDim        = nblocks;
    NodeParams0.blockDim       = nthreads;
    NodeParams0.sharedMemBytes = 0;
    void *kernelArgs0[6]       = {
        (void *)&A, (void *)&b, (void *)&conv_threshold, (void *)&x, (void *)&x_new, (void *)&d_sum};
    NodeParams0.kernelParams = kernelArgs0;
    NodeParams0.extra        = NULL;

    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphAddKernelNode(
        &jacobiKernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &NodeParams0));

    nodeDependencies.clear();
    nodeDependencies.push_back(jacobiKernelNode);

    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr   = make_cudaPitchedPtr(d_sum, sizeof(double), 1, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr   = make_cudaPitchedPtr(&sum, sizeof(double), 1, 1);
    memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
    // JP: `cudaMemcpyDeviceToHost`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    memcpyParams.kind     = cudaMemcpyDeviceToHost;

    checkCudaErrors(
        cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));

    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    NodeParams1.func           = (void *)JacobiMethod;
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    NodeParams1.gridDim        = nblocks;
    NodeParams1.blockDim       = nthreads;
    NodeParams1.sharedMemBytes = 0;
    void *kernelArgs1[6]       = {
        (void *)&A, (void *)&b, (void *)&conv_threshold, (void *)&x_new, (void *)&x, (void *)&d_sum};
    NodeParams1.kernelParams = kernelArgs1;
    NodeParams1.extra        = NULL;

    int k = 0;
    for (k = 0; k < max_iter; k++) {
        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        checkCudaErrors(cudaGraphExecKernelNodeSetParams(
            graphExec, jacobiKernelNode, ((k & 1) == 0) ? &NodeParams0 : &NodeParams1));
        checkCudaErrors(cudaGraphLaunch(graphExec, stream));
        // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (sum <= conv_threshold) {
            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x            = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
            if ((k & 1) == 0) {
                // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
            }
            else {
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
            }

            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
            // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
            checkCudaErrors(cudaStreamSynchronize(stream));
            printf("GPU iterations : %d\n", k + 1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(d_sum));
    return sum;
}

double JacobiMethodGpuCudaGraphExecUpdate(const float  *A,
                                          const double *b,
                                          const float   conv_threshold,
                                          const int     max_iter,
                                          double       *x,
                                          double       *x_new,
                                          // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                                          cudaStream_t  stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3            nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = NULL;

    double  sum = 0.0;
    double *d_sum;
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));

    int k = 0;
    for (k = 0; k < max_iter; k++) {
        // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
        if ((k & 1) == 0) {
            // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
            JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x, x_new, d_sum);
        }
        else {
            JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x_new, x, d_sum);
        }
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
        // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
        checkCudaErrors(cudaStreamEndCapture(stream, &graph));

        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        if (graphExec == NULL) {
            checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        }
        else {
            cudaGraphExecUpdateResult updateResult_out;
            checkCudaErrors(cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult_out));
            if (updateResult_out != cudaGraphExecUpdateSuccess) {
                if (graphExec != NULL) {
                    checkCudaErrors(cudaGraphExecDestroy(graphExec));
                }
                printf("k = %d graph update failed with error - %d\n", k, updateResult_out);
                checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            }
        }
        checkCudaErrors(cudaGraphLaunch(graphExec, stream));
        // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (sum <= conv_threshold) {
            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x            = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
            if ((k & 1) == 0) {
                // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
            }
            else {
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
            }

            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
            // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
            checkCudaErrors(cudaStreamSynchronize(stream));
            printf("GPU iterations : %d\n", k + 1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaFree(d_sum));
    return sum;
}

double JacobiMethodGpu(const float  *A,
                       const double *b,
                       const float   conv_threshold,
                       const int     max_iter,
                       double       *x,
                       double       *x_new,
                       // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
                       cudaStream_t  stream)
{
    // CTA size
    dim3 nthreads(256, 1, 1);
    // grid size
    dim3 nblocks((N_ROWS / ROWS_PER_CTA) + 2, 1, 1);

    double  sum = 0.0;
    double *d_sum;
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaMalloc(&d_sum, sizeof(double)));
    int k = 0;

    for (k = 0; k < max_iter; k++) {
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
        if ((k & 1) == 0) {
            // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
            JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x, x_new, d_sum);
        }
        else {
            JacobiMethod<<<nblocks, nthreads, 0, stream>>>(A, b, conv_threshold, x_new, x, d_sum);
        }
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
        // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
        checkCudaErrors(cudaStreamSynchronize(stream));

        if (sum <= conv_threshold) {
            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
            nblocks.x            = (N_ROWS / nthreads.x) + 1;
            size_t sharedMemSize = ((nthreads.x / 32) + 1) * sizeof(double);
            if ((k & 1) == 0) {
                // JP: この連続する anchor 群では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x_new, d_sum);
            }
            else {
                finalError<<<nblocks, nthreads, sharedMemSize, stream>>>(x, d_sum);
            }

            // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream));
            // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
            checkCudaErrors(cudaStreamSynchronize(stream));
            printf("GPU iterations : %d\n", k + 1);
            printf("GPU error : %.3e\n", sum);
            break;
        }
    }

    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaFree(d_sum));
    return sum;
}
