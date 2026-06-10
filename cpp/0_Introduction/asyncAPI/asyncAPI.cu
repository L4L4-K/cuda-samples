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

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

__global__ void increment_kernel(int *g_data, int inc_value)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
        if (data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int main(int argc, char *argv[])
{
    int            devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    int n      = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value  = 26;

    // allocate host memory
    int *a = 0;
    // JP: `cudaMallocHost`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a = 0;
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    // JP: `cudaMemset`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    // JP: `cudaEvent_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    // JP: `cudaEventDestroy`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
