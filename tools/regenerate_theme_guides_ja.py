#!/usr/bin/env python3
"""Regenerate detailed Japanese CUDA theme guides."""

from __future__ import annotations

# JP: Theme guide generator. CUDA API names below are documentation data; the script itself only writes Markdown files.
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THEME_DIR = ROOT / "docs_ja" / "themes"


def api(name: str, meaning: str, check: str) -> tuple[str, str, str]:
    return name, meaning, check


def sample(path: str, reason: str) -> tuple[str, str, str]:
    label = path.rstrip("/").split("/")[-2] if path.endswith("/README.ja.md") else path.split("/")[-1]
    return label, path, reason


THEMES: dict[str, dict[str, object]] = {
    "execution_model.md": {
        "title": "CUDA Execution Model",
        "anchor": "CUDA samples launch work as grids of thread blocks, where each thread runs the same kernel code on different data.",
        "concept": "CUDA の execution model は、host が kernel を投入し、device が grid、block、thread の階層で device code を実行するという考え方です。sample を読むときは、host から device work に渡る境界、各 thread が担当する data、host が完了を待つ位置を分けます。",
        "mental": ["grid は全体の仕事、block は協調できる thread の単位、thread は個々の data を処理する単位です。", "block 間は通常同期できません。block 内だけが shared memory と barrier を共有できます。", "kernel launch は非同期なので、launch 行だけでは host から結果を読めるとは限りません。"],
        "apis": [api("kernel<<<grid, block, sharedMem, stream>>>()", "Runtime API の kernel launch 構文です。", "grid/block/shared memory/stream と後続同期を確認します。"), api("cuLaunchKernel", "Driver API の launch です。", "CUfunction、argument 配列、shared memory、stream を対応させます。"), api("cudaGetLastError", "直近の launch 設定 error を確認します。", "完了待ちではないため同期 API と分けます。"), api("cudaDeviceSynchronize / cudaStreamSynchronize", "host が device work を待つ境界です。", "correctness 待ちか timing 待ちかを区別します。")],
        "samples": [sample("../../cpp/0_Introduction/vectorAdd/README.ja.md", "one-thread-per-element pattern を読みます。"), sample("../../cpp/0_Introduction/matrixMul/README.ja.md", "2D grid と block 内協調を読みます。"), sample("../../cpp/0_Introduction/simpleOccupancy/README.ja.md", "launch shape と occupancy を読みます。"), sample("../../cpp/6_Performance/LargeKernelParameter/README.ja.md", "kernel parameter と launch 境界を読みます。"), sample("../../python/1_GettingStarted/vectorAdd/README.ja.md", "Python launch wrapper と比較します。")],
        "reading": ["kernel launch 行で grid、block、shared memory、stream をメモします。", "kernel 内の `blockIdx`、`threadIdx`、`blockDim` が data index に変わる式を探します。", "rounded-up grid の場合、out-of-range を防ぐ条件分岐を確認します。", "launch 後の error check と同期 API を分けて読みます。", "validation が同期後に置かれているか確認します。"],
        "mistakes": ["launch が非同期なのに直後に host で結果を読む。", "thread count と element count を同じものとして扱う。", "block 間で `__syncthreads()` 相当の同期ができると思う。", "occupancy が高ければ常に速いと判断する。"],
        "perf": ["block size は memory coalescing、register pressure、occupancy、shared memory 使用量に影響します。", "small kernel を何度も launch する sample では launch overhead が支配的になることがあります。", "測定前には warmup と同期範囲を確認します。"],
        "exercises": ["vectorAdd の block size を変えたときの grid 計算式を説明する。", "matrixMul の 2D thread/block mapping を図にする。", "launch error と実行時 error が見える API を書き分ける。"],
        "links": [("Kernel Launch And Indexing", "kernel_indexing.md", "index 計算を詳しく読みます。"), ("Streams And Events", "streams_events.md", "launch をどの queue に入れるかを読みます。"), ("Performance", "performance.md", "launch overhead と occupancy を読みます。")],
    },
    "memory.md": {
        "title": "CUDA Memory",
        "anchor": "CUDA samples move data among host memory, device memory, pinned memory, managed memory, arrays, mapped memory, and external resources.",
        "concept": "CUDA memory を読む目的は、pointer がどこを指し、誰が所有し、どの processor からいつ見えるかを明確にすることです。`malloc`、`cudaMalloc`、`cudaMallocHost`、`cudaMallocManaged`、external memory は似た pointer に見えても lifetime と access rule が違います。",
        "mental": ["Host memory は CPU から自然に読めますが、GPU から直接読めるとは限りません。", "Device memory は GPU work の主な storage で、host は CUDA API を通して扱います。", "Pinned memory は DMA と async copy の前提になり、managed memory は CPU/GPU が同じ pointer を共有します。"],
        "apis": [api("cudaMalloc / cudaFree", "device memory の確保と解放です。", "byte size と cleanup path を対応させます。"), api("cudaMallocHost / cudaFreeHost", "page-locked host memory を確保します。", "async copy と overlap の前提を確認します。"), api("cudaMemcpy / cudaMemcpyAsync", "host/device 間の visibility を作る転送です。", "direction enum と stream dependency を確認します。"), api("cudaMallocManaged / cudaMemPrefetchAsync", "Unified Memory allocation と migration hint です。", "同期と access order を読みます。"), api("cudaExternalMemory / cudaGraphics*", "外部 API が所有する resource を CUDA に見せます。", "map/unmap と owner lifetime を区別します。")],
        "samples": [sample("../../cpp/0_Introduction/vectorAdd/README.ja.md", "H2D、kernel、D2H、free の基本形です。"), sample("../../cpp/0_Introduction/UnifiedMemoryStreams/README.ja.md", "Unified Memory と stream を読みます。"), sample("../../cpp/0_Introduction/simpleZeroCopy/README.ja.md", "mapped host memory を読みます。"), sample("../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md", "managed memory performance を読みます。"), sample("../../python/2_CoreConcepts/memoryResources/README.ja.md", "Python Buffer lifetime を読みます。")],
        "reading": ["allocation API と解放 API を対応表にします。", "copy/mapping/migration が input visibility を作る場所を探します。", "kernel/library call が読む pointer と書く pointer を分けます。", "validation 前に結果が host から見えるか確認します。", "cleanup 前に非同期 work が完了しているか確認します。"],
        "mistakes": ["element count を byte count として API に渡す。", "D2H と H2D の direction enum を取り違える。", "managed memory なら同期や prefetch が不要だと思う。", "external resource の owner を CUDA 側だと誤解する。"],
        "perf": ["pageable host memory では async copy の overlap が制限されることがあります。", "coalesced global memory access は thread index と data layout の両方で決まります。", "Unified Memory の page fault は timing に混ざるため warmup と prefetch を確認します。"],
        "exercises": ["vectorAdd の host/device pointer を表にする。", "UnifiedMemoryPerf の data movement path を図にする。", "external memory sample で CUDA と外部 API の owner を分ける。"],
        "links": [("Unified Memory", "unified_memory.md", "managed memory を詳しく読みます。"), ("Streams And Events", "streams_events.md", "Async copy と stream ordering を読みます。"), ("Performance", "performance.md", "memory bandwidth と measurement を読みます。")],
    },
    "kernel_indexing.md": {
        "title": "Kernel Launch And Indexing",
        "anchor": "CUDA kernels map grid, block, and thread coordinates to data indexes.",
        "concept": "kernel indexing は、CUDA の並列実行を data layout に結び付ける場所です。1D vector、2D image、matrix tile、3D volume では index 式が変わりますが、目的は「この thread がどの要素を読む/書くか」を明確にすることです。",
        "mental": ["`threadIdx` は block 内 coordinate、`blockIdx` は grid 内 block coordinate です。", "`blockDim` と `gridDim` は shape で、data size そのものではありません。", "global index は coordinate 変換であり、memory address とは stride や pitch を通して対応します。"],
        "apis": [api("blockIdx / threadIdx / blockDim / gridDim", "thread coordinate から global data index を作ります。", "1D/2D/3D の式と boundary check を一緒に読みます。"), api("dim3", "grid/block を multi-dimensional shape として表します。", "x/y/z の意味が data layout と一致しているか確認します。"), api("cudaMallocPitch / cudaMemcpy2D", "pitch を持つ 2D memory を扱います。", "width と pitch の単位が byte か element かを分けます。"), api("__launch_bounds__", "compiler に launch 上限や occupancy hint を与えます。", "performance hint であり correctness とは分けます。")],
        "samples": [sample("../../cpp/0_Introduction/vectorAdd/README.ja.md", "1D indexing の基本です。"), sample("../../cpp/0_Introduction/matrixMul/README.ja.md", "2D matrix coordinate を読みます。"), sample("../../cpp/6_Performance/transpose/README.ja.md", "coalescing と tile indexing を読みます。"), sample("../../cpp/2_Concepts_and_Techniques/reduction/README.ja.md", "multiple elements per thread を読みます。"), sample("../../cpp/5_Domain_Specific/volumeFiltering/README.ja.md", "3D volume coordinate を読みます。")],
        "reading": ["kernel の最初にある index 計算式を見つけます。", "data shape、stride、pitch、leading dimension と index 式を対応させます。", "write address が thread ごとに一意か、atomic が必要かを確認します。", "boundary check が read と write の両方を守るか確認します。", "CPU reference が同じ layout 前提か確認します。"],
        "mistakes": ["1D 式を 2D data にそのまま使う。", "pitch が byte 単位なのに element 単位で足す。", "read は guard しているが write の guard を忘れる。", "grid size を切り下げて末尾 data を処理しない。"],
        "perf": ["adjacent threads が adjacent addresses を読むと coalescing しやすくなります。", "2D tile では x/y の割り当てが bank conflict に影響します。", "boundary branch は必要ですが、全 thread が複雑に分岐する設計は見直します。"],
        "exercises": ["vectorAdd の global index を別 block size で手計算する。", "transpose の read/write address を thread 4 個分追う。", "matrixMul の row/col と A/B/C address 式を図にする。"],
        "links": [("Execution Model", "execution_model.md", "grid/block/thread の実行単位を確認します。"), ("Shared Memory", "shared_memory.md", "tile indexing と barrier を確認します。"), ("Performance", "performance.md", "coalescing と occupancy を確認します。")],
    },
}


def add_theme(name: str, title: str, anchor: str, concept: str, apis: list[tuple[str, str, str]], samples: list[tuple[str, str, str]], focus: str, links: list[tuple[str, str, str]]) -> None:
    THEMES[name] = {
        "title": title,
        "anchor": anchor,
        "concept": concept,
        "mental": [f"{focus} では、どの thread/process/API が resource を所有するかを先に分けます。", f"{focus} の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。", "English API names and output strings remain authoritative; Japanese notes explain the reading strategy."],
        "apis": apis,
        "samples": samples,
        "reading": [f"{focus} に関係する API call を探し、作成、設定、利用、破棄を対応させます。", "入力 data が GPU work から見える状態になる場所を探します。", "非同期 work と host-side validation の間にある synchronization boundary を確認します。", "error check、reference comparison、timing range を分けて読みます。", "関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。"],
        "mistakes": [f"{focus} の scope を大きく見積もり、実際には保証されない ordering を期待する。", "helper/wrapper が内部で CUDA resource を所有していることを見落とす。", "validation 前の同期や data visibility を確認しない。", "performance number だけを見て、測定範囲と setup cost を確認しない。"],
        "perf": [f"{focus} は correctness の仕組みであると同時に overhead や bottleneck になり得ます。", "timing では setup、GPU work、transfer、sync、validation を分けます。", "architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。"],
        "exercises": [f"{focus} に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。", "sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。", "関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。"],
        "links": links,
    }


add_theme("streams_events.md", "CUDA Streams And Events", "Streams order asynchronous CUDA work, and events mark points in stream timelines.", "stream は CUDA work を投入する queue、event はその queue 上の時点を表す marker です。同じ stream 内では順序が保たれますが、別 stream 間は event や explicit sync がなければ順序を期待できません。", [api("cudaStreamCreate / cudaStreamDestroy", "非同期 work の queue を作成/破棄します。", "default stream との違いと cleanup を確認します。"), api("cudaMemcpyAsync", "stream に転送を投入します。", "pinned memory、direction、同一 stream ordering を確認します。"), api("cudaEventRecord / cudaEventElapsedTime", "stream 上の時点を記録し timing します。", "record stream と wait/sync の位置を確認します。"), api("cudaStreamWaitEvent", "cross-stream dependency を作ります。", "producer event と consumer stream を対応させます。")], [sample("../../cpp/0_Introduction/asyncAPI/README.ja.md", "Async API の入口です。"), sample("../../cpp/0_Introduction/simpleStreams/README.ja.md", "stream と pinned memory overlap を読みます。"), sample("../../cpp/0_Introduction/simpleMultiCopy/README.ja.md", "copy/compute overlap を読みます。"), sample("../../cpp/3_CUDA_Features/StreamPriorities/README.ja.md", "priority stream を読みます。"), sample("../../python/2_CoreConcepts/streamingCopyComputeOverlap/README.ja.md", "Python stream overlap を読みます。")], "streams/events", [("Memory", "memory.md", "pinned memory と transfer を読みます。"), ("Performance", "performance.md", "overlap と timing を読みます。"), ("Graphs", "graphs.md", "stream capture を読みます。")])

add_theme("sync_atomics.md", "Synchronization And Atomics", "Synchronization creates ordering, while atomics protect shared updates to one memory location.", "sync は work の完了や visibility の境界を作り、atomic は複数 thread が同じ address を更新する競合を安全に扱います。host/device、stream、block、warp、group で同期の範囲が違います。", [api("__syncthreads / __syncwarp", "block/warp 内の待ち合わせです。", "全参加 thread が到達するか確認します。"), api("cudaDeviceSynchronize / cudaStreamSynchronize", "host が device/stream work を待ちます。", "validation 前の boundary か確認します。"), api("atomicAdd / atomicCAS", "同じ address の read-modify-write を保護します。", "contention と result order を確認します。"), api("__threadfence", "write visibility の順序を制御します。", "barrier ではないことを確認します。")], [sample("../../cpp/0_Introduction/simpleAtomicIntrinsics/README.ja.md", "atomic intrinsic の基本です。"), sample("../../cpp/0_Introduction/systemWideAtomics/README.ja.md", "system-scope atomic を読みます。"), sample("../../cpp/2_Concepts_and_Techniques/threadFenceReduction/README.ja.md", "fence と reduction completion を読みます。"), sample("../../cpp/3_CUDA_Features/warpAggregatedAtomicsCG/README.ja.md", "atomic contention reduction を読みます。"), sample("../../python/2_CoreConcepts/parallelReduction/README.ja.md", "Python reduction boundary を読みます。")], "synchronization/atomics", [("Shared Memory", "shared_memory.md", "barrier が必要な block 内共有を読みます。"), ("Cooperative Groups", "cooperative_groups.md", "group sync を読みます。"), ("Performance", "performance.md", "atomic contention を読みます。")])

add_theme("shared_memory.md", "CUDA Shared Memory", "Shared memory is a fast per-block scratchpad shared by threads in the same block.", "shared memory は block 内 thread が協調して使う scratchpad です。global memory から tile を読み込み、barrier 後に再利用する pattern が matrix、transpose、histogram などに出ます。", [api("__shared__", "静的 shared memory を宣言します。", "array shape と thread/block mapping を確認します。"), api("extern __shared__", "dynamic shared memory を宣言します。", "launch 時 byte 数と型変換を確認します。"), api("__syncthreads", "producer/consumer をそろえる barrier です。", "全 thread が到達するか確認します。"), api("cudaFuncSetAttribute", "shared memory carveout などを設定します。", "architecture 依存制限を確認します。")], [sample("../../cpp/0_Introduction/matrixMul/README.ja.md", "classic tiling を読みます。"), sample("../../cpp/6_Performance/transpose/README.ja.md", "bank conflict と coalescing を読みます。"), sample("../../cpp/2_Concepts_and_Techniques/histogram/README.ja.md", "block-local accumulation を読みます。"), sample("../../cpp/3_CUDA_Features/globalToShmemAsyncCopy/README.ja.md", "async copy staging を読みます。"), sample("../../python/2_CoreConcepts/matrixMulSharedMem/README.ja.md", "Python shared memory kernel を読みます。")], "shared memory", [("Kernel Launch And Indexing", "kernel_indexing.md", "tile coordinate を読みます。"), ("Synchronization And Atomics", "sync_atomics.md", "barrier scope を読みます。"), ("Performance", "performance.md", "reuse と bank conflict を読みます。")])

add_theme("unified_memory.md", "CUDA Unified Memory", "Unified Memory lets CPU and GPU use one managed pointer while the driver migrates pages.", "Unified Memory は CPU と GPU が同じ pointer を使える memory model です。ただし、migration、prefetch、advice、同期を読まなければ performance と visibility を誤解します。", [api("cudaMallocManaged / cudaFree", "managed allocation と解放です。", "CPU/GPU の両方で使う pointer lifetime を確認します。"), api("cudaMemPrefetchAsync", "指定 device へ pages を移動する hint です。", "stream ordering と access 直前の配置を確認します。"), api("cudaMemAdvise", "access pattern hint です。", "read-mostly や preferred location の意図を読みます。"), api("ManagedMemoryResource", "Python CUDA の managed allocator です。", "Buffer close と stream sync を対応させます。")], [sample("../../cpp/0_Introduction/UnifiedMemoryStreams/README.ja.md", "UM と stream ordering を読みます。"), sample("../../cpp/4_CUDA_Libraries/conjugateGradientUM/README.ja.md", "solver flow と UM を読みます。"), sample("../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md", "UM performance modes を読みます。"), sample("../../python/1_GettingStarted/blurImageUnifiedMemory/README.ja.md", "Python UM workflow を読みます。"), sample("../../python/2_CoreConcepts/memoryResources/README.ja.md", "ManagedMemoryResource を読みます。")], "Unified Memory", [("Memory", "memory.md", "memory ownership を読みます。"), ("Streams And Events", "streams_events.md", "prefetch ordering を読みます。"), ("Performance", "performance.md", "migration cost を読みます。")])

add_theme("cooperative_groups.md", "CUDA Cooperative Groups", "Cooperative Groups makes the group of threads participating in collective operations explicit.", "Cooperative Groups は、どの thread 集合が一緒に動くかを object と type で明示します。block、tile、warp、grid の scope を code から読めるようにします。", [api("this_thread_block", "現在の block group を取得します。", "block 内 rank と sync を読みます。"), api("tiled_partition", "block を小さい tile group に分けます。", "tile size と collective scope を確認します。"), api("group.sync / cg::sync", "group 内同期です。", "group の範囲を確認します。"), api("cudaLaunchCooperativeKernel", "cooperative kernel launch です。", "device support と grid size 制限を確認します。")], [sample("../../cpp/0_Introduction/simpleCooperativeGroups/README.ja.md", "basic group と sync です。"), sample("../../cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/README.ja.md", "multi-block reduction を読みます。"), sample("../../cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/README.ja.md", "solver flow と CG を読みます。"), sample("../../cpp/3_CUDA_Features/binaryPartitionCG/README.ja.md", "partitioned group を読みます。"), sample("../../python/2_CoreConcepts/reductionMultiBlockCG/README.ja.md", "Python CG kernel を読みます。")], "Cooperative Groups", [("Synchronization And Atomics", "sync_atomics.md", "group sync を読みます。"), ("Execution Model", "execution_model.md", "cooperative launch 条件を読みます。"), ("Performance", "performance.md", "aggregation と occupancy を読みます。")])

add_theme("graphs.md", "CUDA Graphs", "CUDA Graphs record a dependency graph of GPU work and replay it efficiently.", "CUDA Graph は copy、kernel、memset、library work などを node と依存関係として表し、instantiate した graph executable を繰り返し launch する仕組みです。", [api("cudaStreamBeginCapture / cudaStreamEndCapture", "stream work を graph に capture します。", "capture 対応 API と dependency を確認します。"), api("cudaGraphAdd*Node", "manual graph construction です。", "node parameter と dependency を確認します。"), api("cudaGraphInstantiate / cudaGraphLaunch", "GraphExec を作成し launch します。", "instantiate cost と replay を分けます。"), api("cudaGraphExecUpdate", "GraphExec を更新します。", "topology 変更の制限を確認します。")], [sample("../../cpp/3_CUDA_Features/simpleCudaGraphs/README.ja.md", "basic graph capture/replay を読みます。"), sample("../../cpp/3_CUDA_Features/graphMemoryNodes/README.ja.md", "memory nodes を読みます。"), sample("../../cpp/3_CUDA_Features/graphConditionalNodes/README.ja.md", "conditional node を読みます。"), sample("../../cpp/3_CUDA_Features/jacobiCudaGraphs/README.ja.md", "iterative solver graph を読みます。"), sample("../../python/2_CoreConcepts/cudaGraphs/README.ja.md", "Python graph を読みます。")], "CUDA Graphs", [("Streams And Events", "streams_events.md", "stream capture を読みます。"), ("Memory", "memory.md", "buffer lifetime を読みます。"), ("Performance", "performance.md", "launch overhead を読みます。")])

add_theme("runtime_driver_nvrtc.md", "CUDA Runtime, Driver, And NVRTC", "Samples use Runtime API, Driver API, NVRTC, JIT linking, PTX, and NVVM at different abstraction levels.", "Runtime API は便利な host API、Driver API は context/module/function handle を明示する API、NVRTC/JIT/NVVM は device code を compile/link する仕組みです。compile、load、lookup、launch を分けて読みます。", [api("cudaMalloc / kernel<<<>>>", "Runtime API の基本形です。", "暗黙 context と error check を確認します。"), api("cuInit / cuCtxCreate", "Driver API 初期化と context 作成です。", "CU* handle lifetime を追います。"), api("cuModuleLoad / cuModuleGetFunction", "module と function handle を得ます。", "PTX/cubin と symbol name を確認します。"), api("nvrtcCreateProgram / nvrtcCompileProgram", "runtime compilation です。", "compile option、log、generated code を確認します。")], [sample("../../cpp/0_Introduction/vectorAddDrv/README.ja.md", "Driver API vectorAdd を読みます。"), sample("../../cpp/0_Introduction/matrixMulDrv/README.ja.md", "Driver API matrixMul を読みます。"), sample("../../cpp/0_Introduction/vectorAdd_nvrtc/README.ja.md", "NVRTC compile から launch までを読みます。"), sample("../../cpp/3_CUDA_Features/ptxjit/README.ja.md", "PTX JIT を読みます。"), sample("../../cpp/7_libNVVM/simple/README.ja.md", "libNVVM flow を読みます。")], "Runtime/Driver/NVRTC", [("Execution Model", "execution_model.md", "launch shape を読みます。"), ("Memory", "memory.md", "allocation lifetime を読みます。"), ("Debugging, Profiling, And Testing", "debugging_profiling_testing.md", "compile log と error code を読みます。")])

add_theme("libraries.md", "CUDA Libraries", "CUDA libraries submit optimized device work through handles, descriptors, plans, and workspaces.", "CUDA library sample では、kernel を自分で書かなくても library call が device work を投入します。handle、descriptor、plan、workspace、data layout、stream association、cleanup を対応させます。", [api("cublasCreate / cublasDestroy", "cuBLAS handle の lifetime です。", "stream、math mode、pointer mode を確認します。"), api("cufftPlan* / cufftExec*", "FFT plan と実行です。", "batch、layout、in-place/out-of-place を確認します。"), api("cusparse descriptor APIs", "sparse descriptor を作ります。", "index base、format、workspace を確認します。"), api("cusolverDn / cusolverSp", "solver routines です。", "workspace、info、pivot、validation を確認します。")], [sample("../../cpp/4_CUDA_Libraries/simpleCUBLAS/README.ja.md", "handle と operation を読みます。"), sample("../../cpp/4_CUDA_Libraries/matrixMulCUBLAS/README.ja.md", "manual GEMM と cuBLAS を比較します。"), sample("../../cpp/4_CUDA_Libraries/simpleCUFFT/README.ja.md", "FFT plan を読みます。"), sample("../../cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/README.ja.md", "solver workspace を読みます。"), sample("../../cpp/4_CUDA_Libraries/nvJPEG/README.ja.md", "decoder state を読みます。")], "CUDA Libraries", [("Memory", "memory.md", "input/output buffer を読みます。"), ("Streams And Events", "streams_events.md", "library stream association を読みます。"), ("Performance", "performance.md", "workspace と timing を読みます。")])

add_theme("tensor_cores_wmma.md", "Tensor Cores And WMMA", "Tensor Core samples use specialized matrix instructions, tile APIs, and precision formats for high-throughput math.", "Tensor Core/WMMA sample は、通常の scalar multiply-add ではなく warp/tile 単位の matrix operation を specialized hardware に渡します。precision、layout、alignment、tile size、accumulator type が重要です。", [api("wmma::fragment", "WMMA tile fragment です。", "matrix_a、matrix_b、accumulator と layout を確認します。"), api("wmma::load_matrix_sync / mma_sync / store_matrix_sync", "tile load、multiply、store です。", "leading dimension と precision を確認します。"), api("cublasSetMathMode / cuBLASLt", "library 経由の Tensor Core path です。", "math mode、compute type、workspace を確認します。"), api("CUDA Tile APIs", "tile-level programming model です。", "tile shape と backend selection を確認します。")], [sample("../../cpp/3_CUDA_Features/cudaTensorCoreGemm/README.ja.md", "WMMA/Tensor Core GEMM を読みます。"), sample("../../cpp/3_CUDA_Features/tf32TensorCoreGemm/README.ja.md", "TF32 を読みます。"), sample("../../cpp/3_CUDA_Features/bf16TensorCoreGemm/README.ja.md", "BF16 を読みます。"), sample("../../cpp/3_CUDA_Features/immaTensorCoreGemm/README.ja.md", "INT matrix path を読みます。"), sample("../../cpp/9_CUDA_Tile/tileMatmul/README.ja.md", "CUDA Tile matmul を読みます。")], "Tensor Cores/WMMA", [("Shared Memory", "shared_memory.md", "tile staging を読みます。"), ("Libraries", "libraries.md", "cuBLAS/cuBLASLt path を読みます。"), ("Performance", "performance.md", "throughput と memory bottleneck を読みます。")])

add_theme("multi_gpu_p2p_ipc.md", "Multi-GPU, P2P, And IPC", "Multi-GPU samples coordinate memory, contexts, peer access, processes, and interconnect topology across devices.", "multi-GPU/P2P/IPC sample は、複数 device、process、context の間で memory と work をどう分担するかを示します。device owner、peer capability、IPC handle、topology を先に分けます。", [api("cudaGetDeviceCount / cudaSetDevice", "device enumeration と selection です。", "current device を確認します。"), api("cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess", "P2P capability と有効化です。", "device pair と error path を確認します。"), api("cudaMemcpyPeerAsync", "device 間転送です。", "source/destination device と stream を確認します。"), api("cudaIpcGetMemHandle / cudaIpcOpenMemHandle", "process 間 memory handle です。", "owner process と close/free を確認します。")], [sample("../../cpp/0_Introduction/simpleP2P/README.ja.md", "peer access を読みます。"), sample("../../cpp/0_Introduction/simpleIPC/README.ja.md", "IPC handle を読みます。"), sample("../../cpp/0_Introduction/simpleMultiGPU/README.ja.md", "work split を読みます。"), sample("../../cpp/5_Domain_Specific/MonteCarloMultiGPU/README.ja.md", "domain workload 分割を読みます。"), sample("../../python/4_DistributedComputing/multiGPUGradientAverage/README.ja.md", "Python distributed GPU を読みます。")], "multi-GPU/P2P/IPC", [("Memory", "memory.md", "device ごとの owner を読みます。"), ("Streams And Events", "streams_events.md", "device 間 copy ordering を読みます。"), ("Performance", "performance.md", "P2P bandwidth/latency を読みます。")])

add_theme("performance.md", "CUDA Performance", "Performance samples measure memory traffic, launch overhead, occupancy, overlap, bandwidth, latency, and math throughput.", "performance を読むときは、何を速くしたいか、何を測っているか、何を測っていないかを分けます。setup、transfer、kernel/library execution、sync、validation の範囲を明確にします。", [api("cudaEventRecord / cudaEventElapsedTime", "GPU stream 上の elapsed time を測ります。", "測定範囲に transfer や sync が含まれるか確認します。"), api("cudaOccupancyMaxPotentialBlockSize", "occupancy の launch hint を得ます。", "occupancy は十分条件ではないと理解します。"), api("cudaProfilerStart / cudaProfilerStop", "profiling 範囲を指定します。", "warmup や validation を含むか確認します。"), api("cudaMemcpyAsync + streams", "copy/compute overlap を作ります。", "pinned memory と dependency を確認します。")], [sample("../../cpp/6_Performance/transpose/README.ja.md", "coalescing と bank conflict を読みます。"), sample("../../cpp/6_Performance/alignedTypes/README.ja.md", "alignment を読みます。"), sample("../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md", "migration cost を読みます。"), sample("../../cpp/6_Performance/cudaGraphsPerfScaling/README.ja.md", "graph overhead を読みます。"), sample("../../python/2_CoreConcepts/launchConfigTuning/README.ja.md", "launch config tuning を読みます。")], "performance", [("Execution Model", "execution_model.md", "launch shape と occupancy を読みます。"), ("Memory", "memory.md", "memory traffic を読みます。"), ("Streams And Events", "streams_events.md", "overlap と event timing を読みます。")])

add_theme("debugging_profiling_testing.md", "Debugging, Profiling, And Testing", "Samples use error checks, reference validation, profiling ranges, assertions, printf, and device capability checks.", "debugging/profiling/testing は CUDA sample を安全に読むための観察点です。error check は API boundary、validation は result boundary、profiling は measurement boundary を示します。", [api("checkCudaErrors / getLastCudaError", "sample helper の error check macro です。", "どの API family を包むか確認します。"), api("cudaGetLastError / cudaDeviceSynchronize", "launch error と実行完了 error を確認します。", "直近 error check と完了待ちを分けます。"), api("assert / printf in kernel", "device-side debug 出力です。", "buffering と synchronization を確認します。"), api("np.allclose / sdkCompare*", "reference validation です。", "tolerance、layout、precision を確認します。")], [sample("../../cpp/0_Introduction/simpleAssert/README.ja.md", "device assert を読みます。"), sample("../../cpp/0_Introduction/simplePrintf/README.ja.md", "kernel printf を読みます。"), sample("../../cpp/1_Utilities/deviceQuery/README.ja.md", "capability check を読みます。"), sample("../../python/1_GettingStarted/kernelNsysProfile/README.ja.md", "profiling range を読みます。"), sample("../../cpp/3_CUDA_Features/ptxjit/README.ja.md", "JIT/Driver error handling を読みます。")], "debugging/profiling/testing", [("Performance", "performance.md", "測定範囲を読みます。"), ("Runtime, Driver, And NVRTC", "runtime_driver_nvrtc.md", "compile/runtime error を読みます。"), ("Execution Model", "execution_model.md", "launch error と execution error を分けます。")])

add_theme("python_cuda.md", "Python CUDA", "Python CUDA samples use CUDA Python, CuPy, framework interop, DLPack, streams, memory resources, and distributed tools.", "Python CUDA sample では、Python object が CUDA device、context、stream、module、memory buffer、framework tensor を包みます。見た目は Python でも、非同期実行、lifetime、hidden transfer/sync は CUDA の規則に従います。", [api("cuda.core.Device / Stream / Event", "Python object で CUDA resource を扱います。", "object lifetime と current device を確認します。"), api("Program / ProgramOptions / Linker", "Python から device code を compile/link します。", "architecture、log、kernel name を確認します。"), api("LaunchConfig / launch", "Python から kernel を起動します。", "grid/block/shared memory/stream と argument lifetime を確認します。"), api("CuPy / DLPack / torch / tensorflow", "framework interop の境界です。", "view か copy か、owner と lifetime を確認します。")], [sample("../../python/1_GettingStarted/vectorAdd/README.ja.md", "basic Python launch を読みます。"), sample("../../python/2_CoreConcepts/memoryResources/README.ja.md", "MemoryResource を読みます。"), sample("../../python/2_CoreConcepts/cudaGraphs/README.ja.md", "Python graphs を読みます。"), sample("../../python/3_FrameworkInterop/customPyTorchKernel/README.ja.md", "PyTorch interop を読みます。"), sample("../../python/4_DistributedComputing/multiGPUGradientAverage/README.ja.md", "distributed multi-GPU を読みます。")], "Python CUDA", [("Runtime, Driver, And NVRTC", "runtime_driver_nvrtc.md", "Python からの JIT/launch を読みます。"), ("Memory", "memory.md", "Buffer と framework tensor owner を読みます。"), ("Streams And Events", "streams_events.md", "stream.sync と event ordering を読みます。")])


REQUIRED = [
    "## Concept",
    "## Why It Matters",
    "## Mental Model",
    "## API Map",
    "## Sample References",
    "## Reading Steps",
    "## Common Mistakes",
    "## Performance Notes",
    "## Exercises",
    "## Cross-Theme Links",
    "## Review Checklist",
]


def render(filename: str, data: dict[str, object]) -> str:
    lines: list[str] = [
        f"# {data['title']}",
        "",
        f"English anchor: {data['anchor']}",
        "",
        "> **日本語**",
        f"> {data['concept']}",
        ">",
        "> **学習メモ**",
        "> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。",
        "",
        "## Concept",
        "",
        str(data["concept"]),
        "",
        "> **日本語**",
        "> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。",
        ">",
        "> **学習メモ**",
        "> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。",
        "",
        "## Why It Matters",
        "",
    ]
    lines.extend(f"- {item}" for item in [
        f"{data['title']} は correctness、resource lifetime、performance の読み方に直接関係します。",
        "API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。",
        "同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。",
    ])
    lines.extend(["", "## Mental Model", ""])
    lines.extend(f"- {item}" for item in data["mental"])  # type: ignore[index]
    lines.extend(["", "## API Map", "", "| API or concept | Meaning | What to check |", "| - | - | - |"])
    for name, meaning, check in data["apis"]:  # type: ignore[index]
        lines.append(f"| `{name}` | {meaning} | {check} |")
    lines.extend([
        "",
        "> **日本語**",
        "> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。",
        ">",
        "> **学習メモ**",
        "> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。",
        "",
        "## Sample References",
        "",
    ])
    for label, path, reason in data["samples"]:  # type: ignore[index]
        lines.append(f"- [{label}]({path}): {reason}")
    lines.extend(["", "## Reading Steps", ""])
    for index, step in enumerate(data["reading"], start=1):  # type: ignore[index]
        lines.append(f"{index}. {step}")
    lines.extend(["", "## Common Mistakes", ""])
    lines.extend(f"- {item}" for item in data["mistakes"])  # type: ignore[index]
    lines.extend(["", "## Performance Notes", ""])
    lines.extend(f"- {item}" for item in data["perf"])  # type: ignore[index]
    lines.extend(["", "## Exercises", ""])
    lines.extend(f"- {item}" for item in data["exercises"])  # type: ignore[index]
    lines.extend(["", "## Cross-Theme Links", ""])
    for label, file, reason in data["links"]:  # type: ignore[index]
        lines.append(f"- [{label}]({file}): {reason}")
    lines.extend([
        "",
        "## Review Checklist",
        "",
        "- Can you name the owner and lifetime of each CUDA resource used by the theme?",
        "- Can you point to the synchronization boundary before validation?",
        "- Can you explain which work is measured and which setup/cleanup is outside the measurement?",
        "- Can you compare one C++ sample and one Python or library sample that use the same theme?",
        "",
    ])
    return "\n".join(lines)


def render_index() -> str:
    lines = [
        "# Japanese CUDA Theme Guides",
        "",
        "English anchor: this directory groups Japanese learning guides by CUDA concept.",
        "",
        "> **日本語**",
        "> sample を個別に読む前に、概念ごとの mental model、API、よくある間違い、演習を確認するための入口です。",
        ">",
        "> **学習メモ**",
        "> 迷ったら `memory`、`execution_model`、`debugging_profiling_testing` から読み、sample の `Related Themes` で戻ってきます。",
        "",
        "## Guides",
        "",
    ]
    for filename, data in THEMES.items():
        lines.append(f"- [{data['title']}]({filename}): {data['anchor']}")
    lines.extend([
        "",
        "## Suggested Paths",
        "",
        "- First CUDA kernel: `execution_model` -> `kernel_indexing` -> `memory` -> `debugging_profiling_testing`.",
        "- Memory and overlap: `memory` -> `streams_events` -> `performance` -> `unified_memory`.",
        "- Advanced launch/runtime: `runtime_driver_nvrtc` -> `graphs` -> `cooperative_groups`.",
        "- Libraries and math: `libraries` -> `tensor_cores_wmma` -> `performance`.",
        "- Python workflow: `python_cuda` -> `memory` -> `streams_events` -> `debugging_profiling_testing`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    THEME_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in THEMES.items():
        (THEME_DIR / filename).write_text(render(filename, data), encoding="utf-8", newline="\n")
    (THEME_DIR / "README.md").write_text(render_index(), encoding="utf-8", newline="\n")
    print(f"Regenerated {len(THEMES)} theme guides and index.")


if __name__ == "__main__":
    main()
