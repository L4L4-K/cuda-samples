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

#include <assert.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#include "histogram_common.h"

////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////

#define TAG_MASK 0xFFFFFFFFU
inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag) { atomicAdd(s_WarpHist + data, 1); }

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag)
{
    addByte(s_WarpHist, (data >> 0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
    // Handle to thread block group
    // JP: indexing: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    cg::thread_block cta = cg::this_thread_block();
    // Per-warp subhistogram storage
    // JP: `__shared__`: shared memory は block 内 scratchpad です。別 thread が書いた値を読む前に同期が必要です。
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint           *s_WarpHist = s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++) {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    // Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    // JP: sync: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    cg::sync(cta);

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)) {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }

    // Merge per-warp histograms into per-block and write to global memory
    // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
    cg::sync(cta);

    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE) {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++) {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram256Kernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount)
{
    // Handle to thread block group
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    cg::thread_block cta = cg::this_thread_block();

    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
    }

    // JP: この anchor では shared memory の block-local scratchpad です。producer/consumer の順序と必要な barrier を確認します。
    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        // JP: この anchor では block/warp/group 内の device-side barrier です。参加 thread の範囲、shared memory visibility、次の反復に進む前の同期 を確認します。
        cg::sync(cta);

        // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        d_Histogram[blockIdx.x] = data[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
// histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
static uint      *d_PartialHistograms;

// Internal memory allocation
extern "C" void initHistogram256(void)
{
    checkCudaErrors(
        // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
        cudaMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
}

// Internal memory deallocation
// JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
extern "C" void closeHistogram256(void) { checkCudaErrors(cudaFree(d_PartialHistograms)); }

extern "C" void histogram256(uint *d_Histogram, void *d_Data, uint byteCount)
{
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    assert(byteCount % sizeof(uint) == 0);
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(
        d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    getLastCudaError("histogram256Kernel() execution failed\n");

    mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
        d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT);
    getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}
