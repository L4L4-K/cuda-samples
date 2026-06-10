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
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel is used to
 * initialize an array to a specific value, after which the array is
 * copied to the host (CPU) memory.  To increase performance, multiple
 * kernel/memcopy pairs are launched asynchronously, each pair in its
 * own stream.  Devices with Compute Capability 1.1 can overlap a kernel
 * and a memcopy as long as they are issued in different streams.  Kernels
 * are serialized.  Thus, if n pairs are launched, streamed approach
 * can reduce the memcopy cost to the (1/n)th of a single copy of the entire
 * data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
 */

const char *sSDKsample = "simpleStreams";

// JP: `cudaEventDefault`, `cudaEventBlockingSync`, `cudaEventDisableTiming`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
const char *sEventSyncMethod[] = {"cudaEventDefault", "cudaEventBlockingSync", "cudaEventDisableTiming", NULL};

const char *sDeviceSyncMethod[] = {"cudaDeviceScheduleAuto",
                                   "cudaDeviceScheduleSpin",
                                   "cudaDeviceScheduleYield",
                                   "INVALID",
                                   "cudaDeviceScheduleBlockingSync",
                                   NULL};

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < num_iterations; i++) {
        g_data[idx] += *factor; // non-coalesced on purpose, to burn time
    }
}

bool correct_data(int *a, const int n, const int c)
{
    for (int i = 0; i < n; i++) {
        if (a[i] != c) {
            printf("%d: %d %d\n", i, a[i], c);
            return false;
        }
    }

    return true;
}

inline void AllocateHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    if (bPinGenericMemory) {
// allocate a generic page-aligned chunk of system memory
#ifdef WIN32
        printf("> VirtualAlloc() allocating %4.2f Mbytes of (generic page-aligned "
               "system memory)\n",
               (float)nbytes / 1048576.0f);
        *pp_a = (int *)VirtualAlloc(NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        printf("> mmap() allocating %4.2f Mbytes (generic page-aligned system "
               "memory)\n",
               (float)nbytes / 1048576.0f);
        *pp_a = (int *)mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif

        *ppAligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

        // JP: `cudaHostRegister`: page-locked host memory は DMA/async copy を安定させます。通常の free ではなく対応する CUDA API で解放します。
        printf("> cudaHostRegister() registering %4.2f Mbytes of generic allocated "
               "system memory\n",
               (float)nbytes / 1048576.0f);
        // pin allocate memory
        // JP: この連続する anchor 群では pinned host memory の登録/確保/解放です。async transfer や overlap の条件と lifetime を確認します。
        checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
    }
    else
#endif
#endif
    {
        printf("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n", (float)nbytes / 1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
        *ppAligned_a = *pp_a;
    }
}

inline void FreeHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory) {
        // unpin and delete host memory
        // JP: この anchor では pinned host memory の登録/確保/解放です。async transfer や overlap の条件と lifetime を確認します。
        checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
        munmap(*pp_a, nbytes);
#endif
    }
    else
#endif
#endif
    {
        // JP: `cudaFreeHost`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
        cudaFreeHost(*pp_a);
    }
}

static const char *sSyncMethod[] = {"0 (Automatic Blocking)",
                                    "1 (Spin Blocking)",
                                    "2 (Yield Blocking)",
                                    "3 (Undefined Blocking Method)",
                                    "4 (Blocking Sync Event) = low CPU utilization",
                                    NULL};

void printHelp()
{
    printf("Usage: %s [options below]\n", sSDKsample);
    printf("\t--sync_method=n for CPU/GPU synchronization\n");
    printf("\t             n=%s\n", sSyncMethod[0]);
    printf("\t             n=%s\n", sSyncMethod[1]);
    printf("\t             n=%s\n", sSyncMethod[2]);
    printf("\t   <Default> n=%s\n", sSyncMethod[4]);
    printf("\t--use_generic_memory (default) use generic page-aligned for system "
           "memory\n");
    printf("\t--use_cuda_malloc_host (optional) use cudaMallocHost to allocate "
           "system memory\n");
}

#if defined(__APPLE__) || defined(MACOSX)
#define DEFAULT_PINNED_GENERIC_MEMORY false
#else
#define DEFAULT_PINNED_GENERIC_MEMORY true
#endif

int main(int argc, char **argv)
{
    int   cuda_device = 0;
    int   nstreams    = 4;                        // number of streams for CUDA calls
    int   nreps       = 10;                       // number of times each experiment is repeated
    int   n           = 16 * 1024 * 1024;         // number of ints in the data set
    int   nbytes      = n * sizeof(int);          // number of data bytes
    dim3  threads, blocks;                        // kernel launch configuration
    float elapsed_time, time_memcpy, time_kernel; // timing variables
    float scale_factor = 1.0f;

    // allocate generic memory and pin it laster instead of using cudaHostAlloc()

    bool bPinGenericMemory  = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    int  device_sync_method = cudaDeviceBlockingSync;        // by default we use BlockingSync

    int niterations; // number of iterations for the loop inside the kernel

    printf("[ %s ]\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printHelp();
        return EXIT_SUCCESS;
    }

    if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method")) >= 0) {
        if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4) {
            printf("Device synchronization method set to = %s\n", sSyncMethod[device_sync_method]);
            printf("Setting reps to 100 to demonstrate steady state\n");
            nreps = 100;
        }
        else {
            printf("Invalid command line option sync_method=\"%d\"\n", device_sync_method);
            return EXIT_FAILURE;
        }
    }
    else {
        printHelp();
        return EXIT_SUCCESS;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)
        bPinGenericMemory = false; // Generic Pinning of System Paged memory not
                                   // currently supported on Mac OSX
#else
        bPinGenericMemory = true;
#endif
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host")) {
        bPinGenericMemory = false;
    }

    printf("\n> ");
    cuda_device = findCudaDevice(argc, (const char **)argv);

    // check the compute capability of the device
    int num_devices = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));

    if (0 == num_devices) {
        printf("your system does not have a CUDA capable device, waiving test...\n");
        return EXIT_WAIVED;
    }

    // check if the command-line chosen device ID is within range, exit if not
    if (cuda_device >= num_devices) {
        printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", cuda_device, num_devices - 1);
        return EXIT_FAILURE;
    }

    checkCudaErrors(cudaSetDevice(cuda_device));

    // Checking for compute capabilities
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    niterations = 5;

    // Check if GPU can map host memory (Generic Method), if not then we override
    // bPinGenericMemory to be false
    if (bPinGenericMemory) {
        printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");

        if (deviceProp.canMapHostMemory == 0) {
            printf("Using cudaMallocHost, CUDA device does not support mapping of "
                   "generic host memory\n");
            bPinGenericMemory = false;
        }
    }

    // Anything that is less than 32 Cores will have scaled down workload
    scale_factor =
        max((32.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)),
            1.0f);
    n = (int)rint((float)n / scale_factor);

    printf("> CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
    printf("> %d Multiprocessor(s) x %d (Cores/Multiprocessor) = %d (Cores)\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    printf("> scale_factor = %1.4f\n", 1.0f / scale_factor);
    printf("> array_size   = %d\n\n", n);

    // enable use of blocking sync, to reduce CPU usage
    printf("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
    checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));

    // allocate host memory
    int  c          = 5; // value to which the array will be initialized
    int *h_a        = 0; // pointer to the array data in host memory
    int *hAligned_a = 0; // pointer to the array data in host memory (aligned to
                         // MEMORY_ALIGNMENT)

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if
    // using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    // allocate device memory
    int *d_a = 0,
        *d_c = 0; // pointers to data and init value in the device memory
    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    // JP: `cudaMemset`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
    checkCudaErrors(cudaMemset(d_a, 0x0, nbytes));
    checkCudaErrors(cudaMalloc((void **)&d_c, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice));

    printf("\nStarting Test\n");

    // allocate and initialize an array of stream handles
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    // use blocking sync
    cudaEvent_t start_event, stop_event;
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync : cudaEventDefault);

    checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

    // time memcopy from device
    checkCudaErrors(cudaEventRecord(start_event, 0)); // record in stream-0, to
                                                      // ensure that all previous
                                                      // CUDA calls have
                                                      // completed
    checkCudaErrors(cudaMemcpyAsync(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    // JP: `cudaEventSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaEventSynchronize(stop_event)); // block until the event is actually recorded
    checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("memcopy:\t%.2f\n", time_memcpy);

    // time kernel
    threads = dim3(512, 1);
    blocks  = dim3(n / threads.x, 1);
    checkCudaErrors(cudaEventRecord(start_event, 0));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    init_array<<<blocks, threads, 0, streams[0]>>>(d_a, d_c, niterations);
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&time_kernel, start_event, stop_event));
    printf("kernel:\t\t%.2f\n", time_kernel);

    //////////////////////////////////////////////////////////////////////
    // time non-streamed execution for reference
    threads = dim3(512, 1);
    blocks  = dim3(n / threads.x, 1);
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(start_event, 0));

    for (int k = 0; k < nreps; k++) {
        // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
        init_array<<<blocks, threads>>>(d_a, d_c, niterations);
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        checkCudaErrors(cudaMemcpy(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    }

    // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

    //////////////////////////////////////////////////////////////////////
    // time execution with nstreams streams
    threads = dim3(512, 1);
    blocks  = dim3(n / (nstreams * threads.x), 1);
    memset(hAligned_a, 255,
           nbytes);                              // set host memory bits to all 1s, for testing correctness
    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemset(d_a, 0, nbytes)); // set device memory to all 0s, for testing correctness
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(start_event, 0));

    for (int k = 0; k < nreps; k++) {
        // asynchronously launch nstreams kernels, each operating on its own portion
        // of data
        for (int i = 0; i < nstreams; i++) {
            // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
            init_array<<<blocks, threads, 0, streams[i]>>>(d_a + i * n / nstreams, d_c, niterations);
        }

        // asynchronously launch nstreams memcopies.  Note that memcopy in stream x
        // will only
        //   commence executing when all previous CUDA calls in stream x have
        //   completed
        for (int i = 0; i < nstreams; i++) {
            // JP: この連続する anchor 群では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
            checkCudaErrors(cudaMemcpyAsync(hAligned_a + i * n / nstreams,
                                            d_a + i * n / nstreams,
                                            nbytes / nstreams,
                                            cudaMemcpyDeviceToHost,
                                            streams[i]));
        }
    }

    // JP: この連続する anchor 群では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("%d streams:\t%.2f\n", nstreams, elapsed_time / nreps);

    // check whether the output is correct
    printf("-------------------------------\n");
    bool bResults = correct_data(hAligned_a, n, c * nreps * niterations);

    // release resources
    for (int i = 0; i < nstreams; i++) {
        // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }

    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));

    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
    FreeHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_c));

    return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}
