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

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;
#include <curand_kernel.h>
#include <helper_cuda.h>

#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024 * 1024)

// Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT)
{
    float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
    return (callValue > 0.0F) ? callValue : 0.0F;
}

__device__ inline double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
    double callValue = S * exp(MuByT + VBySqrtT * r) - X;
    return (callValue > 0.0) ? callValue : 0.0;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
// JP: `curandState`: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
static __global__ void MonteCarloOneBlockPerOption(curandState *__restrict rngStates,
                                                   const __TOptionData *__restrict d_OptionData,
                                                   __TOptionValue *__restrict d_CallValue,
                                                   int pathN,
                                                   int optionN)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block          cta    = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int       SUM_N = THREAD_N;
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ real s_SumCall[SUM_N];
    __shared__ real s_Sum2Call[SUM_N];

    // determine global thread id
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Copy random number state to local memory for efficiency
    // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
    curandState localState = rngStates[tid];
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x) {
        const real S        = d_OptionData[optionIndex].S;
        const real X        = d_OptionData[optionIndex].X;
        const real MuByT    = d_OptionData[optionIndex].MuByT;
        const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

        // Cycle through the entire samples array:
        // derive end stock price for each path
        // accumulate partial integrals into intermediate shared memory buffer
        // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
        for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x) {
            __TOptionValue sumCall = {0, 0};

#pragma unroll 8
            for (int i = iSum; i < pathN; i += SUM_N) {
                // JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
                real r         = curand_normal(&localState);
                real callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
                sumCall.Expected += callValue;
                sumCall.Confidence += callValue * callValue;
            }

            s_SumCall[iSum]  = sumCall.Expected;
            s_Sum2Call[iSum] = sumCall.Confidence;
        }

        // Reduce shared memory accumulators
        // and write final result to global memory
        // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
        cg::sync(cta);
        sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call, cta, tile32, &d_CallValue[optionIndex]);
    }
}

// JP: この anchor では CUDA library/NPP resource call です。handle/descriptor/workspace/allocation の作成、利用、破棄 を確認します。
static __global__ void rngSetupStates(curandState *rngState, int device_id)
{
    // determine global thread id
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each threadblock gets different seed,
    // Threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState[tid]);
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////

extern "C" void initMonteCarloGPU(TOptionPlan *plan)
{
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc(&plan->d_OptionData, sizeof(__TOptionData) * (plan->optionCount)));
    checkCudaErrors(cudaMalloc(&plan->d_CallValue, sizeof(__TOptionValue) * (plan->optionCount)));
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost(&plan->h_OptionData, sizeof(__TOptionData) * (plan->optionCount)));
    // Allocate internal device memory
    checkCudaErrors(cudaMallocHost(&plan->h_CallValue, sizeof(__TOptionValue) * (plan->optionCount)));
    // Allocate states for pseudo random number generators
    checkCudaErrors(cudaMalloc((void **)&plan->rngStates, plan->gridSize * THREAD_N * sizeof(curandState)));
    // JP: `cudaMemset`, `curandState`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(plan->rngStates, 0, plan->gridSize * THREAD_N * sizeof(curandState)));

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
    getLastCudaError("rngSetupStates kernel failed.\n");
}

// Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
    for (int i = 0; i < plan->optionCount; i++) {
        const double RT    = plan->optionData[i].R * plan->optionData[i].T;
        const double sum   = plan->h_CallValue[i].Expected;
        const double sum2  = plan->h_CallValue[i].Confidence;
        const double pathN = plan->pathN;
        // Derive average from the total sum and discount by riskfree rate
        plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
        // Standard deviation
        double stdDev = sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
        // Confidence width; in 95% of all cases theoretical value lies within these
        // borders
        plan->callValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
    }

    // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFree(plan->rngStates));
    checkCudaErrors(cudaFreeHost(plan->h_CallValue));
    checkCudaErrors(cudaFreeHost(plan->h_OptionData));
    checkCudaErrors(cudaFree(plan->d_CallValue));
    checkCudaErrors(cudaFree(plan->d_OptionData));
}

// Main computations
// JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{
    __TOptionValue *h_CallValue = plan->h_CallValue;

    if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS) {
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }

    __TOptionData *h_OptionData = (__TOptionData *)plan->h_OptionData;

    for (int i = 0; i < plan->optionCount; i++) {
        const double T           = plan->optionData[i].T;
        const double R           = plan->optionData[i].R;
        const double V           = plan->optionData[i].V;
        const double MuByT       = (R - 0.5 * V * V) * T;
        const double VBySqrtT    = V * sqrt(T);
        h_OptionData[i].S        = (real)plan->optionData[i].S;
        h_OptionData[i].X        = (real)plan->optionData[i].X;
        h_OptionData[i].MuByT    = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
    }

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(
        plan->d_OptionData, h_OptionData, plan->optionCount * sizeof(__TOptionData), cudaMemcpyHostToDevice, stream));

    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(plan->rngStates,
                                                                         (__TOptionData *)(plan->d_OptionData),
                                                                         (__TOptionValue *)(plan->d_CallValue),
                                                                         plan->pathN,
                                                                         plan->optionCount);
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");

    // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(
        h_CallValue, plan->d_CallValue, plan->optionCount * sizeof(__TOptionValue), cudaMemcpyDeviceToHost, stream));

    // cudaDeviceSynchronize();
}
