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

// System includes
#include <assert.h>
#include <climits>
#include <stdio.h>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define THREADS_PER_BLOCK  512
#define ALLOWABLE_VARIANCE 1.e-6f
#define NUM_ELEMENTS       8000000

// Stores the square of each input element in output array
__global__ void squareArray(const float *input, float *output, int numElements)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        output[idx] = input[idx] * input[idx];
    }
}

// Stores the negative of each input element in output array
__global__ void negateArray(const float *input, float *output, int numElements)
{
    // JP: この anchor では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        output[idx] = input[idx] * -1;
    }
}

struct negSquareArrays
{
    float *input;
    float *square;
    float *negSquare;
    int    numElements;
    size_t bytes;
    size_t numBlocks;
};

void fillRandomly(float *array, int numElements)
{
    for (int n = 0; n < numElements; n++) {
        array[n] = rand() / (float)RAND_MAX;
    }
}

void resetOutputArrays(negSquareArrays *hostArrays)
{
    fillRandomly(hostArrays->square, hostArrays->numElements);
    fillRandomly(hostArrays->negSquare, hostArrays->numElements);
}

void prepareHostArrays(negSquareArrays *hostArrays)
{
    hostArrays->numElements = NUM_ELEMENTS;
    size_t bytes            = hostArrays->numElements * sizeof(float);

    size_t numBlocks = hostArrays->numElements / (size_t)THREADS_PER_BLOCK;
    if ((numBlocks % (size_t)THREADS_PER_BLOCK) != 0) {
        numBlocks++;
    }

    hostArrays->input     = (float *)malloc(bytes);
    hostArrays->square    = (float *)malloc(bytes);
    hostArrays->negSquare = (float *)malloc(bytes);
    hostArrays->bytes     = bytes;
    hostArrays->numBlocks = numBlocks;

    fillRandomly(hostArrays->input, hostArrays->numElements);
    fillRandomly(hostArrays->square, hostArrays->numElements);
    fillRandomly(hostArrays->negSquare, hostArrays->numElements);
}

// JP: `cudaGraphExec_t`: CUDA Graph は依存関係を記録して再実行する仕組みです。node 間の順序と使う buffer の寿命を確認します。
void createFreeGraph(cudaGraphExec_t *graphExec, float *dPtr)
{
    cudaGraph_t     graph;
    cudaGraphNode_t freeNode;

    checkCudaErrors(cudaGraphCreate(&graph, 0));

    checkCudaErrors(cudaGraphAddMemFreeNode(&freeNode, graph, NULL, 0, (void *)dPtr));

    checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
    checkCudaErrors(cudaGraphDestroy(graph));
}

/**
 * Demonstrates explicitly creating a CUDA graph including memory nodes.
 * createNegateSquaresGraphWithStreamCapture constructs an equivalent graph
 * using stream capture.
 *
 * If d_negSquare_out is non null, then:
 * 1) d_negSquare will not be freed;
 * 2) the value of d_negSquare_out will be set to d_negSquare.
 *
 * Diagram of the graph constructed by createNegateSquaresGraphExplicitly:
 *
 * alloc d_input
 *       |
 * alloc d_square
 *       |
 * Memcpy a to device
 *       |
 * launch kernel squareArray ------->---- Memcpy d_square to host
 *       |                                      |
 * free d_input                                 |
 *       |                                      |
 * allocate d_negSquare                         |
 *       |                                      |
 * launch kernel negateArray -------->--- free d_square
 *       |
 * Memcpy d_negSquare to host
 *       |
 * free d_negSquare
 */
// JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
void createNegateSquaresGraphExplicitly(cudaGraphExec_t *graphExec,
                                        int              device,
                                        negSquareArrays *hostArrays,
                                        float          **d_negSquare_out = NULL)
{
    // Array buffers on device
    float *d_input, *d_square, *d_negSquare;

    // Memory allocation parameters
    cudaMemAllocNodeParams allocParams;
    memset(&allocParams, 0, sizeof(allocParams));
    allocParams.bytesize                = hostArrays->bytes;
    allocParams.poolProps.allocType     = cudaMemAllocationTypePinned;
    allocParams.poolProps.location.id   = device;
    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;

    // Kernel launch parameters
    cudaKernelNodeParams kernelNodeParams = {0};
    // JP: この連続する anchor 群では block/thread/warp index から data index や担当範囲を決めます。境界条件と problem size の単位 を確認します。
    kernelNodeParams.gridDim              = dim3(hostArrays->numBlocks, 1, 1);
    kernelNodeParams.blockDim             = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes       = 0;
    kernelNodeParams.extra                = NULL;

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    cudaGraph_t     graph;
    cudaGraphNode_t allocNodeInput, allocNodeSquare, allocNodeNegSquare;
    cudaGraphNode_t copyNodeInput, copyNodeSquare, copyNodeNegSquare;
    cudaGraphNode_t squareKernelNode, negateKernelNode;
    cudaGraphNode_t freeNodeInput, freeNodeSquare;

    // Buffer for storing graph node dependencies
    std::vector<cudaGraphNode_t> nodeDependencies;

    checkCudaErrors(cudaGraphCreate(&graph, 0));

    checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeInput, graph, NULL, 0, &allocParams));
    d_input = (float *)allocParams.dptr;

    // To keep the graph structure simple (fewer branching dependencies),
    // allocNodeSquare should depend on allocNodeInput
    checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeSquare, graph, &allocNodeInput, 1, &allocParams));
    d_square = (float *)allocParams.dptr;

    // copyNodeInput needs to depend on allocNodeInput because copyNodeInput
    // writes to d_input. It does so here indirectly through allocNodeSquare.
    checkCudaErrors(cudaGraphAddMemcpyNode1D(&copyNodeInput,
                                             graph,
                                             &allocNodeSquare,
                                             1,
                                             d_input,
                                             hostArrays->input,
                                             hostArrays->bytes,
                                             // JP: `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
                                             cudaMemcpyHostToDevice));

    void *squareKernelArgs[3]     = {(void *)&d_input, (void *)&d_square, (void *)&(hostArrays->numElements)};
    kernelNodeParams.func         = (void *)squareArray;
    kernelNodeParams.kernelParams = (void **)squareKernelArgs;

    // Square kernel depends on copyNodeInput to ensure all data is on the device
    // before kernel launch.
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphAddKernelNode(&squareKernelNode, graph, &copyNodeInput, 1, &kernelNodeParams));

    checkCudaErrors(cudaGraphAddMemcpyNode1D(&copyNodeSquare,
                                             graph,
                                             &squareKernelNode,
                                             1,
                                             hostArrays->square,
                                             d_square,
                                             hostArrays->bytes,
                                             // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
                                             cudaMemcpyDeviceToHost));

    // Free of d_input depends on the square kernel to ensure that d_input is not
    // freed while being read by the kernel. It also depends on the alloc of
    // d_input via squareKernelNode > copyNodeInput > allocNodeSquare >
    // allocNodeInput.
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphAddMemFreeNode(&freeNodeInput, graph, &squareKernelNode, 1, d_input));

    // Allocation of C depends on free of A so CUDA can reuse the virtual address.
    checkCudaErrors(cudaGraphAddMemAllocNode(&allocNodeNegSquare, graph, &freeNodeInput, 1, &allocParams));
    d_negSquare = (float *)allocParams.dptr;

    if (d_negSquare == d_input) {
        printf("Check verified that d_negSquare and d_input share a virtual "
               "address.\n");
    }

    void *negateKernelArgs[3]     = {(void *)&d_square, (void *)&d_negSquare, (void *)&(hostArrays->numElements)};
    kernelNodeParams.func         = (void *)negateArray;
    kernelNodeParams.kernelParams = (void **)negateKernelArgs;

    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphAddKernelNode(&negateKernelNode, graph, &allocNodeNegSquare, 1, &kernelNodeParams));

    nodeDependencies.push_back(copyNodeSquare);
    nodeDependencies.push_back(negateKernelNode);
    checkCudaErrors(
        cudaGraphAddMemFreeNode(&freeNodeSquare, graph, nodeDependencies.data(), nodeDependencies.size(), d_square));
    nodeDependencies.clear();

    checkCudaErrors(cudaGraphAddMemcpyNode1D(&copyNodeNegSquare,
                                             graph,
                                             &negateKernelNode,
                                             1,
                                             hostArrays->negSquare,
                                             d_negSquare,
                                             hostArrays->bytes,
                                             // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
                                             cudaMemcpyDeviceToHost));

    if (d_negSquare_out == NULL) {
        // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
        cudaGraphNode_t freeNodeNegSquare;
        checkCudaErrors(cudaGraphAddMemFreeNode(&freeNodeNegSquare, graph, &copyNodeNegSquare, 1, d_negSquare));
    }
    else {
        *d_negSquare_out = d_negSquare;
    }

    checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
    checkCudaErrors(cudaGraphDestroy(graph));
}

/**
 * Adds work to a CUDA stream which negates the square of values in the input
 * array.
 *
 * If d_negSquare_out is non null, then:
 * 1) d_negSquare will not be freed;
 * 2) the value of d_negSquare_out will be set to d_negSquare.
 *
 * Diagram of the stream operations in doNegateSquaresInStream
 * ---------------------------------------------------------------------
 * | STREAM                             | STREAM2                      |
 * ---------------------------------------------------------------------
 *
 * alloc d_input
 *       |
 * alloc d_square
 *       |
 * Memcpy a to device
 *       |
 * launch kernel squareArray
 *       |
 * record squareKernelCompleteEvent -->-- wait squareKernelCompleteEvent
 *       |                                      |
 * free d_input                                 |
 *       |                                      |
 * allocate d_negSquare                   Memcpy d_square to host
 *       |                                      |
 * launch kernel negateArray                    |
 *       |                                      |
 * record negateKernelCompleteEvent -->-- wait negateKernelCompleteEvent
 *       |                                      |
 * Memcpy d_negSquare to host                   |
 *       |                                free d_square
 * free d_negSquare                             |
 *       |                                      |
 * wait squareFreeEvent --------------<---- record squareFreeEvent
 */
// JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
void doNegateSquaresInStream(cudaStream_t stream1, negSquareArrays *hostArrays, float **d_negSquare_out = NULL)
{
    float       *d_input, *d_square, *d_negSquare;
    cudaStream_t stream2;
    cudaEvent_t  squareKernelCompleteEvent, negateKernelCompleteEvent, squareFreeEvent;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

    checkCudaErrors(cudaEventCreate(&squareKernelCompleteEvent));
    checkCudaErrors(cudaEventCreate(&negateKernelCompleteEvent));
    checkCudaErrors(cudaEventCreate(&squareFreeEvent));

    // Virtual addresses are assigned synchronously when cudaMallocAsync is
    // called, thus there is no performace benefit gained by separating the
    // allocations into two streams.
    checkCudaErrors(cudaMallocAsync(&d_input, hostArrays->bytes, stream1));
    checkCudaErrors(cudaMallocAsync(&d_square, hostArrays->bytes, stream1));

    // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
    checkCudaErrors(cudaMemcpyAsync(d_input, hostArrays->input, hostArrays->bytes, cudaMemcpyHostToDevice, stream1));
    squareArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(d_input, d_square, hostArrays->numElements);
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(squareKernelCompleteEvent, stream1));

    checkCudaErrors(cudaStreamWaitEvent(stream2, squareKernelCompleteEvent, 0));
    checkCudaErrors(cudaMemcpyAsync(hostArrays->square, d_square, hostArrays->bytes, cudaMemcpyDeviceToHost, stream2));

    // JP: `cudaFreeAsync`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    checkCudaErrors(cudaFreeAsync(d_input, stream1));
    checkCudaErrors(cudaMallocAsync(&d_negSquare, hostArrays->bytes, stream1));
    negateArray<<<hostArrays->numBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
        d_square, d_negSquare, hostArrays->numElements);
    checkCudaErrors(cudaEventRecord(negateKernelCompleteEvent, stream1));
    checkCudaErrors(
        // JP: この anchor では host/device/peer transfer です。転送方向、byte 数、stream ordering、producer/consumer を確認します。
        cudaMemcpyAsync(hostArrays->negSquare, d_negSquare, hostArrays->bytes, cudaMemcpyDeviceToHost, stream1));
    if (d_negSquare_out == NULL) {
        // JP: この anchor では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
        checkCudaErrors(cudaFreeAsync(d_negSquare, stream1));
    }
    else {
        *d_negSquare_out = d_negSquare;
    }

    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamWaitEvent(stream2, negateKernelCompleteEvent, 0));
    // JP: この anchor では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
    checkCudaErrors(cudaFreeAsync(d_square, stream2));
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaEventRecord(squareFreeEvent, stream2));

    checkCudaErrors(cudaStreamWaitEvent(stream1, squareFreeEvent, 0));

    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaEventDestroy(squareKernelCompleteEvent));
    checkCudaErrors(cudaEventDestroy(negateKernelCompleteEvent));
    checkCudaErrors(cudaEventDestroy(squareFreeEvent));
}

/**
 * Demonstrates creating a CUDA graph including memory nodes using stream
 * capture. createNegateSquaresGraphExplicitly constructs an equivalent graph
 * without stream capture.
 */
// JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
void createNegateSquaresGraphWithStreamCapture(cudaGraphExec_t *graphExec,
                                               negSquareArrays *hostArrays,
                                               float          **d_negSquare_out = NULL)
{
    cudaGraph_t  graph;
    // JP: この連続する anchor 群では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t stream;

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    doNegateSquaresInStream(stream, hostArrays, d_negSquare_out);
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));

    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0));
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamDestroy(stream));
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphDestroy(graph));
}

void prepareRefArrays(negSquareArrays *hostArrays, negSquareArrays *deviceRefArrays, bool **foundValidationFailure)
{
    deviceRefArrays->bytes       = hostArrays->bytes;
    deviceRefArrays->numElements = hostArrays->numElements;

    for (int i = 0; i < hostArrays->numElements; i++) {
        hostArrays->square[i]    = hostArrays->input[i] * hostArrays->input[i];
        hostArrays->negSquare[i] = hostArrays->square[i] * -1;
    }

    // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
    checkCudaErrors(cudaMalloc((void **)&deviceRefArrays->negSquare, deviceRefArrays->bytes));
    checkCudaErrors(
        cudaMemcpy(deviceRefArrays->negSquare, hostArrays->negSquare, hostArrays->bytes, cudaMemcpyHostToDevice));

    // JP: `cudaMallocManaged`: Unified Memory は CPU/GPU で同じ pointer を使います。prefetch や同期で移動タイミングを意識します。
    checkCudaErrors(cudaMallocManaged((void **)foundValidationFailure, sizeof(bool)));
}

int checkValidationFailure(bool *foundValidationFailure)
{
    if (*foundValidationFailure) {
        printf("Validation FAILURE!\n\n");
        *foundValidationFailure = false;
        return EXIT_FAILURE;
    }
    else {
        printf("Validation PASSED!\n\n");
        return EXIT_SUCCESS;
    }
}

// JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
__global__ void validateGPU(float *d_negSquare, negSquareArrays devRefArrays, bool *foundValidationFailure)
{
    int   idx = blockIdx.x * blockDim.x + threadIdx.x;
    float ref, diff;

    if (idx < devRefArrays.numElements) {
        ref  = devRefArrays.negSquare[idx];
        diff = d_negSquare[idx] - ref;
        diff *= diff;
        ref *= ref;
        if (diff / ref > ALLOWABLE_VARIANCE) {
            *foundValidationFailure = true;
        }
    }
}

// JP: この anchor では GPU result や file/image output の validation です。失敗時は transfer/indexing/sync の境界から疑います。
void validateHost(negSquareArrays *hostArrays, bool *foundValidationFailure)
{
    float ref, diff;

    for (int i = 0; i < hostArrays->numElements; i++) {
        ref  = hostArrays->input[i] * hostArrays->input[i] * -1;
        diff = hostArrays->negSquare[i] - ref;
        diff *= diff;
        ref *= ref;
        if (diff / ref > ALLOWABLE_VARIANCE) {
            *foundValidationFailure = true;
        }
    }
}

int main(int argc, char **argv)
{
    negSquareArrays hostArrays, deviceRefArrays;
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    cudaStream_t    stream;
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    cudaGraphExec_t graphExec, graphExecFreeC;

    // Declare pointers for GPU buffers
    float *d_negSquare            = NULL;
    bool  *foundValidationFailure = NULL;

    srand(time(0));
    int device = findCudaDevice(argc, (const char **)argv);

    int driverVersion             = 0;
    int deviceSupportsMemoryPools = 0;

    cudaDriverGetVersion(&driverVersion);
    printf("Driver version is: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);

    if (driverVersion < 11040) {
        printf("Waiving execution as driver does not support Graph Memory Nodes\n");
        exit(EXIT_WAIVED);
    }

    cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);
    if (!deviceSupportsMemoryPools) {
        printf("Waiving execution as device does not support Memory Pools\n");
        exit(EXIT_WAIVED);
    }
    else {
        printf("Setting up sample.\n");
    }

    prepareHostArrays(&hostArrays);
    prepareRefArrays(&hostArrays, &deviceRefArrays, &foundValidationFailure);
    // JP: この anchor では stream/event resource と timeline operation です。投入順、依存、timing 範囲、destroy 前の完了 を確認します。
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    printf("Setup complete.\n\n");

    printf("Running negateSquares in a stream.\n");
    doNegateSquaresInStream(stream, &hostArrays);
    // JP: `cudaStreamSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("Validating negateSquares in a stream...\n");
    validateHost(&hostArrays, foundValidationFailure);
    checkValidationFailure(foundValidationFailure);
    resetOutputArrays(&hostArrays);

    printf("Running negateSquares in a stream-captured graph.\n");
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    createNegateSquaresGraphWithStreamCapture(&graphExec, &hostArrays);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("Validating negateSquares in a stream-captured graph...\n");
    // JP: この anchor では GPU result や file/image output の validation です。失敗時は transfer/indexing/sync の境界から疑います。
    validateHost(&hostArrays, foundValidationFailure);
    checkValidationFailure(foundValidationFailure);
    resetOutputArrays(&hostArrays);

    printf("Running negateSquares in an explicitly constructed graph.\n");
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    createNegateSquaresGraphExplicitly(&graphExec, device, &hostArrays);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("Validating negateSquares in an explicitly constructed graph...\n");
    // JP: この anchor では GPU result や file/image output の validation です。失敗時は transfer/indexing/sync の境界から疑います。
    validateHost(&hostArrays, foundValidationFailure);
    checkValidationFailure(foundValidationFailure);
    resetOutputArrays(&hostArrays);

    // Each of the three examples below free d_negSquare outside the graph. As
    // demonstrated by validateGPU, d_negSquare can be accessed by outside the
    // graph before d_negSquare is freed.

    printf("Running negateSquares with d_negSquare freed outside the stream.\n");
    createNegateSquaresGraphExplicitly(&graphExec, device, &hostArrays, &d_negSquare);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    validateGPU<<<hostArrays.numBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_negSquare, deviceRefArrays, foundValidationFailure);
    // Since cudaFree is synchronous, the stream must synchronize before freeing
    // d_negSquare to ensure d_negSquare no longer being accessed.
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));
    // JP: この anchor では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaFree(d_negSquare));
    printf("Validating negateSquares with d_negSquare freed outside the "
           "stream...\n");
    // JP: この anchor では GPU result や file/image output の validation です。失敗時は transfer/indexing/sync の境界から疑います。
    validateHost(&hostArrays, foundValidationFailure);
    checkValidationFailure(foundValidationFailure);
    resetOutputArrays(&hostArrays);

    printf("Running negateSquares with d_negSquare freed outside the graph.\n");
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
    validateGPU<<<hostArrays.numBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_negSquare, deviceRefArrays, foundValidationFailure);
    // JP: この anchor では CUDA resource lifetime end です。未完了 work が残っていないか確認し、確保/作成/登録と対応する API で閉じます。
    checkCudaErrors(cudaFreeAsync(d_negSquare, stream));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("Validating negateSquares with d_negSquare freed outside the graph...\n");
    checkValidationFailure(foundValidationFailure);
    resetOutputArrays(&hostArrays);

    printf("Running negateSquares with d_negSquare freed in a different graph.\n");
    createFreeGraph(&graphExecFreeC, d_negSquare);
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    // JP: この anchor では kernel launch の grid/block/shared-memory/stream 指定です。後続の sync/error check と完了確認を対応させます。
    validateGPU<<<hostArrays.numBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_negSquare, deviceRefArrays, foundValidationFailure);
    // JP: この anchor では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphLaunch(graphExecFreeC, stream));
    // JP: この anchor では device/stream/event の完了待ち境界です。validation や resource 解放の前に待つ work を確認します。
    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("Validating negateSquares with d_negSquare freed in a different "
           "graph...\n");
    checkValidationFailure(foundValidationFailure);

    printf("Cleaning up sample.\n");
    // JP: この連続する anchor 群では CUDA Graph/graphics resource dependency です。capture/node/instantiate/launch と buffer lifetime を対応させます。
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphExecDestroy(graphExecFreeC));
    // JP: この連続する anchor 群では device memory ownership です。確保 size、pointer lifetime、対応する cleanup を確認します。
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(foundValidationFailure));
    checkCudaErrors(cudaFree(deviceRefArrays.negSquare));
    free(hostArrays.input);
    free(hostArrays.square);
    free(hostArrays.negSquare);
    printf("Cleanup complete. Exiting sample.\n");
}
