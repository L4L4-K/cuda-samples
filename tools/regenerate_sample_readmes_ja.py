#!/usr/bin/env python3
"""Regenerate sample-specific Japanese learning guides for cpp/ and python/."""

from __future__ import annotations

# JP: This maintenance script reads source/README metadata and writes Japanese guide files; CUDA API names here are data, not executable CUDA work.
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXTS = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py"}
WALKTHROUGH_EXTS = {
    ".cu",
    ".cuh",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".py",
    ".cmake",
    ".sh",
    ".bat",
    ".cmd",
    ".ps1",
    ".hxx",
    ".hlsl",
    ".glsl",
    ".frag",
    ".vert",
    ".ptx",
    ".ll",
    ".bash",
}
DOC_DATA_EXTS = {".md", ".txt", ".json", ".dat", ".bin", ".ppm", ".pgm", ".bmp", ".png", ".jpg", ".jpeg"}
SKIP_TOKENS = {"CUDA", "CUDART", "CUDAToolkit", "CMAKE_CUDA", "cudaSuccess"}
VENDOR_OR_GENERATED_PARTS = {
    "boost",
    "d3dx11effect",
    "Common/GL",
    "cuda_drvapi_dynlink",
    "cuda_drvapi_dynlink_cuda",
    "ptxdump",
}


THEMES = {
    "execution_model": ("Execution Model", "docs_ja/themes/execution_model.md", "kernel launch、grid/block/thread の実行階層を読むための基礎です。"),
    "memory": ("Memory", "docs_ja/themes/memory.md", "host/device/managed/external memory の所有権と lifetime を追うための基礎です。"),
    "kernel_indexing": ("Kernel Launch And Indexing", "docs_ja/themes/kernel_indexing.md", "thread index から data index への対応と境界チェックを読むための基礎です。"),
    "streams_events": ("Streams And Events", "docs_ja/themes/streams_events.md", "非同期 work、overlap、timing event の順序を読むための基礎です。"),
    "sync_atomics": ("Synchronization And Atomics", "docs_ja/themes/sync_atomics.md", "barrier、fence、atomic update の必要性を判断するための基礎です。"),
    "shared_memory": ("Shared Memory", "docs_ja/themes/shared_memory.md", "block 内 scratchpad、tiling、`__syncthreads()` を読むための基礎です。"),
    "unified_memory": ("Unified Memory", "docs_ja/themes/unified_memory.md", "managed memory、migration、prefetch の意味を読むための基礎です。"),
    "cooperative_groups": ("Cooperative Groups", "docs_ja/themes/cooperative_groups.md", "block/grid/warp 単位の協調と同期を読むための基礎です。"),
    "graphs": ("CUDA Graphs", "docs_ja/themes/graphs.md", "capture、node dependency、replay、graph update を読むための基礎です。"),
    "runtime_driver_nvrtc": ("Runtime, Driver, And NVRTC", "docs_ja/themes/runtime_driver_nvrtc.md", "Runtime API、Driver API、NVRTC/JIT の境界を読むための基礎です。"),
    "libraries": ("CUDA Libraries", "docs_ja/themes/libraries.md", "handle、descriptor、workspace、library call の所有と実行順序を読むための基礎です。"),
    "tensor_cores_wmma": ("Tensor Cores And WMMA", "docs_ja/themes/tensor_cores_wmma.md", "Tensor Core、tile、precision、fragment の制約を読むための基礎です。"),
    "multi_gpu_p2p_ipc": ("Multi-GPU, P2P, And IPC", "docs_ja/themes/multi_gpu_p2p_ipc.md", "device topology、peer access、IPC handle、multi-process 境界を読むための基礎です。"),
    "performance": ("Performance", "docs_ja/themes/performance.md", "memory traffic、occupancy、overlap、launch overhead、timing を読むための基礎です。"),
    "debugging_profiling_testing": ("Debugging, Profiling, And Testing", "docs_ja/themes/debugging_profiling_testing.md", "error check、reference validation、profiling/timing の範囲を読むための基礎です。"),
    "python_cuda": ("Python CUDA", "docs_ja/themes/python_cuda.md", "Python object が CUDA resource を包む境界と hidden sync を読むための基礎です。"),
}


TOKEN_RE = re.compile(
    r"\b(?:"
    r"cuda[A-Z][A-Za-z0-9_]*|cu[A-Z][A-Za-z0-9_]*|CU[A-Za-z0-9_]+|"
    r"nvrtc[A-Za-z0-9_]*|nvJitLink[A-Za-z0-9_]*|"
    r"cublas[A-Za-z0-9_]*|cufft[A-Za-z0-9_]*|cusparse[A-Za-z0-9_]*|cusolver[A-Za-z0-9_]*|"
    r"curand[A-Za-z0-9_]*|nppi[A-Za-z0-9_]*|npps[A-Za-z0-9_]*|npp[A-Za-z0-9_]*|"
    r"nvjpeg[A-Za-z0-9_]*|nccl[A-Za-z0-9_]*|cub::[A-Za-z0-9_:]+|thrust::[A-Za-z0-9_:]+|"
    r"__syncthreads|__syncwarp|__shared__|blockIdx|threadIdx|blockDim|gridDim|atomic[A-Za-z0-9_]*|"
    r"DeviceMemoryResource|PinnedMemoryResource|ManagedMemoryResource|LaunchConfig|ProgramOptions|Program|Device|Stream|Event|"
    r"cupy|cp\.[A-Za-z0-9_]+|torch|tensorflow|mpi4py|numba\.cuda|launch"
    r")\b"
)
SNIPPET_ANCHOR_RE = re.compile(
    r"\b("
    r"cudaMalloc\w*|cudaFree\w*|cudaMemcpy\w*|cudaMemset\w*|cudaStream\w*|cudaEvent\w*|"
    r"cudaGraph\w*|cudaLaunch\w*|cuLaunchKernel|cu[A-Z][A-Za-z0-9_]*|"
    r"nvrtc\w*|nvJitLink\w*|cublas\w*|cufft\w*|cusparse\w*|cusolver\w*|curand\w*|"
    r"nppi\w*|npps\w*|nvjpeg\w*|nccl\w*|cub::[A-Za-z0-9_:]+|thrust::[A-Za-z0-9_:]+|"
    r"<<<[^>]*>>>|__global__|__device__|__shared__|__syncthreads|__syncwarp|atomic[A-Za-z0-9_]*|"
    r"blockIdx|threadIdx|blockDim|gridDim|tiled_partition|this_thread_block|this_grid|"
    r"LaunchConfig|ProgramOptions|Program\(|Linker\(|launch\(|Device\(|Stream\(|Event\(|"
    r"ManagedMemoryResource|DeviceMemoryResource|PinnedMemoryResource|cupy|cp\.|torch|tensorflow|"
    r"add_executable|target_link_libraries|find_package|CUDA_ARCHITECTURES|CMAKE_CUDA"
    r")\b"
)
SNIPPET_FOCUS = {
    "setup": re.compile(r"\b(main\s*\(|argparse|find_package|add_executable|cudaSetDevice|cuInit|Device\(|Program\(|cublasCreate|cufftPlan|cusolver|cusparse|nvjpegCreate)\b"),
    "allocation": re.compile(r"\b(cudaMalloc\w*|cudaHostAlloc|cudaMallocHost|cudaMallocManaged|malloc\(|new\s+|DeviceMemoryResource|ManagedMemoryResource|mem_alloc|workspace)\b"),
    "transfer": re.compile(r"\b(cudaMemcpy\w*|cudaMemset\w*|copy_to|copy_from|cudaGraphicsMapResources|cudaExternalMemory|cudaIpc|cudaMemcpyPeer)\b"),
    "work": re.compile(r"(<<<[^>]*>>>|cuLaunchKernel|cudaLaunchKernel|launch\(|cublas\w*\(|cufftExec\w*|cusolver\w*\(|cusparse\w*\(|nppi\w*\(|nvjpeg\w*\(|blockIdx|threadIdx|__shared__|__syncthreads)"),
    "sync_validation": re.compile(r"\b(cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize|__syncthreads|assert|compare|validate|allclose|PASS|FAIL|sdkCompare)\b"),
    "cleanup": re.compile(r"\b(cudaFree\w*|cudaDestroy\w*|cudaStreamDestroy|cudaEventDestroy|Destroy|destroy|free\(|cuMemFree\w*)\b"),
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sample_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in (ROOT / "cpp", ROOT / "python"):
        if base.exists():
            dirs.extend(sorted({p.parent for p in base.rglob("README.md")}))
    return sorted(dirs, key=rel)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def first_heading(readme: str, fallback: str) -> str:
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def english_overview(readme: str) -> str:
    lines = readme.splitlines()
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(("-", "*")) and not current:
            continue
        current.append(stripped)
        if len(" ".join(current)) > 420:
            paragraphs.append(" ".join(current))
            current = []
        if len(paragraphs) >= 2:
            break
    if current and len(paragraphs) < 2:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs[:2]) or "See the original README for the authoritative English description."


def heading_list(readme: str) -> list[str]:
    result: list[str] = []
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if text:
                result.append(text)
    return result[:12]


def files_in_sample(path: Path) -> list[Path]:
    result: list[Path] = []
    for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if child.is_file() and child.name != "README.ja.md":
            if ".git" not in child.parts and "__pycache__" not in child.parts:
                result.append(child)
    return result


def source_files(path: Path) -> list[Path]:
    return [p for p in files_in_sample(path) if p.suffix.lower() in SOURCE_EXTS]


def is_walkthrough_source(path: Path) -> bool:
    relative = rel(path)
    if path.name == "CMakeLists.txt":
        return True
    if path.suffix.lower() not in WALKTHROUGH_EXTS:
        return False
    return not any(part in relative for part in VENDOR_OR_GENERATED_PARTS)


def has_child_sample_dirs(sample: Path) -> bool:
    return any(child.is_dir() and (child / "README.md").exists() for child in sample.iterdir())


def walkthrough_files(sample: Path) -> list[Path]:
    # JP: category README は配下の全サンプルを代表するので、直下の CMake だけを抜粋対象にして重複を避けます。
    if has_child_sample_dirs(sample):
        candidates = [p for p in sample.iterdir() if p.is_file()]
    else:
        candidates = [p for p in sample.rglob("*") if p.is_file()]
    return sorted(
        [p for p in candidates if p.name != "README.ja.md" and is_walkthrough_source(p) and read_text(p).strip()],
        key=rel,
    )


def code_language(path: Path) -> str:
    if path.name == "CMakeLists.txt" or path.suffix.lower() == ".cmake":
        return "cmake"
    return {
        ".cu": "cuda",
        ".cuh": "cuda",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".py": "python",
        ".sh": "bash",
        ".bash": "bash",
        ".bat": "bat",
        ".cmd": "bat",
        ".ps1": "powershell",
        ".hlsl": "hlsl",
        ".glsl": "glsl",
        ".frag": "glsl",
        ".vert": "glsl",
        ".ptx": "ptx",
        ".ll": "llvm",
    }.get(path.suffix.lower(), "text")


def meaningful_start(lines: list[str]) -> int:
    # JP: 長い license/header を抜粋しないため、最初の実コードらしい行まで進めます。
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "/*", "*", "# Copyright", "# SPDX", "#!/")):
            continue
        if "Copyright" in stripped or "NVIDIA Corporation" in stripped:
            continue
        return index + 1
    return 1


def snippet_window(line_number: int, total: int, radius_before: int = 5, radius_after: int = 14) -> tuple[int, int]:
    start = max(1, line_number - radius_before)
    end = min(total, line_number + radius_after)
    return start, end


def line_is_diff_clean(line: str) -> bool:
    return line.rstrip(" \t") == line and " \t" not in line


def window_is_diff_clean(lines: list[str], start: int, end: int) -> bool:
    return all(line_is_diff_clean(line) for line in lines[start - 1 : end])


def first_clean_span(lines: list[str], start: int, max_len: int = 20) -> tuple[int, int] | None:
    # JP: Markdown に source line を写すと git diff --check も見るため、空白違反のない連続範囲だけを選びます。
    index = start
    total = len(lines)
    while index <= total:
        while index <= total and not line_is_diff_clean(lines[index - 1]):
            index += 1
        if index > total:
            return None
        end = index
        while end <= total and line_is_diff_clean(lines[end - 1]) and end - index < max_len:
            end += 1
        if end > index:
            return index, end - 1
        index += 1
    return None


def clean_window_near(lines: list[str], line_number: int) -> tuple[int, int] | None:
    total = len(lines)
    for before, after in ((5, 14), (3, 10), (1, 8), (0, 6), (0, 3)):
        start, end = snippet_window(line_number, total, before, after)
        if window_is_diff_clean(lines, start, end):
            return start, end
    search_start = max(1, line_number - 10)
    return first_clean_span(lines, search_start, max_len=14)


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1] + 3:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def select_snippets(path: Path) -> list[tuple[int, int]]:
    lines = read_text(path).splitlines()
    if not lines:
        return []
    start = meaningful_start(lines)
    total = len(lines)
    if total - start <= 70:
        if window_is_diff_clean(lines, start, total):
            return [(start, total)]
        clean = first_clean_span(lines, start)
        return [clean] if clean else []

    initial = first_clean_span(lines, start, max_len=19)
    windows: list[tuple[int, int]] = [initial] if initial else []
    for pattern in SNIPPET_FOCUS.values():
        for index, line in enumerate(lines[start - 1 :], start=start):
            if pattern.search(line):
                clean = clean_window_near(lines, index)
                if clean:
                    windows.append(clean)
                break
    if len(windows) == 1:
        for index, line in enumerate(lines[start - 1 :], start=start):
            if SNIPPET_ANCHOR_RE.search(line):
                clean = clean_window_near(lines, index)
                if clean:
                    windows.append(clean)
                break
    limit = 4 if total > 180 else 3
    return merge_windows(windows)[:limit]


def render_snippet(path: Path, start: int, end: int) -> list[str]:
    lines = read_text(path).splitlines()
    code = lines[start - 1 : end]
    result = [
        f"Source: {rel(path)}:{start}-{end}",
        f"```{code_language(path)}",
        *code,
        "```",
        "",
        f"> JP: この抜粋は `{rel(path)}` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。",
        "",
    ]
    return result


def render_code_walkthrough(sample: Path) -> list[str]:
    files = walkthrough_files(sample)
    if not files:
        return [
            "このディレクトリには検証対象の source/build/script file がありません。英語 README と親ディレクトリの build 設定を参照します。",
            "",
        ]
    result: list[str] = [
        "この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。",
        "",
    ]
    for path in files:
        result.append(f"### `{path.relative_to(sample).as_posix()}`")
        result.append("")
        snippets = select_snippets(path)
        if not snippets:
            result.append("このファイルは空、または抜粋できる実コード行がありません。")
            result.append("")
            continue
        for start, end in snippets:
            result.extend(render_snippet(path, start, end))
    return result


def token_counter(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for src in source_files(path):
        text = read_text(src)
        for token in TOKEN_RE.findall(text):
            if token in SKIP_TOKENS:
                continue
            if token in {"Device", "Stream", "Event", "Program", "launch"}:
                counter[token] += 1
            elif len(token) > 2:
                counter[token] += 1
    readme = path / "README.md"
    if readme.exists():
        for token in TOKEN_RE.findall(read_text(readme)):
            if token in SKIP_TOKENS:
                continue
            counter[token] += 2
    return counter


def has_any(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def detect_concepts(path: Path, tokens: Counter[str], readme: str) -> set[str]:
    value = f"{rel(path)} {path.name} {readme} {' '.join(tokens)}"
    concepts = {"execution_model", "memory", "debugging_profiling_testing"}
    if has_any(value, "blockIdx", "threadIdx", "blockDim", "gridDim", "launch", "vectorAdd", "kernel"):
        concepts.add("kernel_indexing")
    if has_any(value, "cudaStream", "cudaEvent", "stream", "event", "async", "overlap", "hyperq"):
        concepts.add("streams_events")
    if has_any(value, "__shared__", "shared memory", "sharedmem", "shmem", "tile", "matrixmul", "transpose"):
        concepts.add("shared_memory")
    if has_any(value, "__syncthreads", "__syncwarp", "atomic", "fence", "barrier", "cooperative"):
        concepts.add("sync_atomics")
    if has_any(value, "cudaMallocManaged", "unified", "managed", "prefetch", "uvm"):
        concepts.add("unified_memory")
    if has_any(value, "cooperative", "cg::", "this_grid", "this_thread_block"):
        concepts.add("cooperative_groups")
    if has_any(value, "graph", "cudaGraph", "graphExec", "conditional node"):
        concepts.add("graphs")
    if has_any(value, "driver", "drv", "cuModule", "cuLaunchKernel", "nvrtc", "jit", "ptx", "nvvm", "ProgramOptions", "Linker"):
        concepts.add("runtime_driver_nvrtc")
    if has_any(value, "cublas", "cufft", "cusparse", "cusolver", "curand", "npp", "nvjpeg", "cub::", "thrust::", "nccl"):
        concepts.add("libraries")
    if has_any(value, "wmma", "tensorcore", "tensor core", "imma", "dmma", "bf16", "tf32", "tileMatmul", "tileBmm", "cuda_tile"):
        concepts.add("tensor_cores_wmma")
    if has_any(value, "multigpu", "multi-gpu", "p2p", "peer", "ipc", "mpi", "nccl", "mmap", "crossgpu"):
        concepts.add("multi_gpu_p2p_ipc")
    if has_any(value, "perf", "bandwidth", "latency", "occupancy", "profile", "nsys", "benchmark", "timing", "elapsed"):
        concepts.add("performance")
    if rel(path).startswith("python/") or any(token.startswith(("cp.", "cupy", "torch", "tensorflow", "mpi4py")) for token in tokens):
        concepts.add("python_cuda")
    if has_any(value, "gl", "opengl", "d3d", "direct3d", "vulkan", "egl", "nvsci", "nvmedia", "interop"):
        concepts.update({"streams_events", "memory", "sync_atomics"})
    return concepts


def api_note(token: str) -> str:
    lower = token.lower()
    if "mallocmanaged" in lower or "managedmemory" in lower:
        return "Unified Memory の所有と CPU/GPU access の移動タイミングを見る API です。"
    if "mallochost" in lower or "hostalloc" in lower or "pinned" in lower:
        return "pinned host memory を作り、async copy や DMA の前提を作る API です。"
    if "malloc" in lower or "memalloc" in lower or "devicememory" in lower:
        return "device 側 storage を確保する API です。対応する cleanup と byte size を確認します。"
    if "free" in lower or "destroy" in lower or "close" in lower:
        return "resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。"
    if "memcpy" in lower or "copy" in lower or "memset" in lower:
        return "host/device 間の転送、初期化、または visibility を作る API です。方向と Async の順序を確認します。"
    if "stream" in lower or "event" in lower:
        return "非同期 work の順序、overlap、計測範囲を表す API です。"
    if "graph" in lower:
        return "CUDA Graph の node、capture、instantiate、launch、update の境界を表します。"
    if lower.startswith("cu") and not lower.startswith("cuda"):
        return "Driver API の handle 境界です。context/module/function と error code を追います。"
    if "nvrtc" in lower or "jit" in lower or token in {"Program", "ProgramOptions"}:
        return "実行時 compile/link の境界です。log、module、kernel name の対応を確認します。"
    if any(prefix in lower for prefix in ("cublas", "cufft", "cusparse", "cusolver", "curand", "npp", "nvjpeg", "cub::", "thrust::", "nccl")):
        return "CUDA library call です。handle/descriptor/workspace と data layout を確認します。"
    if token in {"blockIdx", "threadIdx", "blockDim", "gridDim"}:
        return "thread/block index から担当 data を決める記号です。境界チェックと一緒に読みます。"
    if token in {"__shared__", "__syncthreads", "__syncwarp"}:
        return "block 内共有 memory または同期境界です。producer/consumer の順序を確認します。"
    if "atomic" in lower:
        return "複数 thread が同じ address を更新する箇所です。競合と順序の意味を確認します。"
    if token in {"LaunchConfig", "launch", "Device", "Stream", "Event"} or token.startswith(("cp.", "cupy")):
        return "Python object から CUDA resource や device work を扱う境界です。hidden sync に注意します。"
    return "この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。"


def file_role(path: Path, sample: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    relname = path.relative_to(sample).as_posix()
    if name == "README.md":
        return "Original English purpose, prerequisites, run notes, and expected behavior."
    if name == "CMakeLists.txt":
        return "CMake target, CUDA architecture, and library dependency wiring."
    if name == "requirements.txt":
        return "Python package prerequisites for the sample."
    if ".vscode" in path.parts:
        return "Editor configuration. It is useful for navigation, but not part of CUDA runtime behavior."
    if suffix == ".cu":
        return "CUDA source containing kernels, Runtime API calls, or device-side helper code."
    if suffix == ".cuh":
        return "CUDA header with device functions, kernels, templates, or shared constants."
    if suffix in {".cpp", ".cc", ".c"}:
        return "Host-side setup, API calls, validation, and cleanup."
    if suffix in {".h", ".hpp"}:
        return "Host/device declarations, helper types, constants, or library wrappers."
    if suffix == ".py":
        return "Python entry point or helper using CUDA Python, CuPy, framework interop, or subprocess logic."
    if suffix in DOC_DATA_EXTS:
        return "Input, reference, generated-data description, or documentation used by the sample."
    return f"Supporting file used by `{relname}`."


def target_name(sample: Path) -> str:
    cmake = sample / "CMakeLists.txt"
    if cmake.exists():
        text = read_text(cmake)
        match = re.search(r"set\s*\(\s*sample_name\s+([A-Za-z0-9_+-]+)\s*\)", text, re.I)
        if match:
            return match.group(1)
        match = re.search(r"add_executable\s*\(\s*([A-Za-z0-9_+-]+)", text)
        if match and not match.group(1).startswith("$"):
            return match.group(1)
    return sample.name


def python_entry(sample: Path) -> str:
    preferred = sample / f"{sample.name}.py"
    if preferred.exists():
        return preferred.name
    py_files = [p for p in files_in_sample(sample) if p.suffix == ".py"]
    return py_files[0].name if py_files else f"{sample.name}.py"


def concept_notes(concepts: set[str]) -> list[str]:
    notes = []
    mapping = {
        "kernel_indexing": "kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。",
        "memory": "allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。",
        "streams_events": "stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。",
        "shared_memory": "shared memory を使う kernel では、tile を読み込む thread、使う thread、barrier の位置を対応させます。",
        "sync_atomics": "同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。",
        "unified_memory": "Unified Memory は pointer を共有しますが、migration、prefetch、同期の理解は必要です。",
        "cooperative_groups": "Cooperative Groups は協調する単位を明示します。grid/block/warp のどれを同期しているかを確認します。",
        "graphs": "CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。",
        "runtime_driver_nvrtc": "Runtime/Driver/NVRTC sample では、compile/load した module と launch する kernel 名の対応が重要です。",
        "libraries": "library sample では handle、descriptor、plan、workspace が GPU work の外側の resource です。",
        "tensor_cores_wmma": "Tensor Core sample では tile size、alignment、precision、accumulator の型が正しさと性能を決めます。",
        "multi_gpu_p2p_ipc": "multi-GPU sample では、device 選択、peer capability、context/IPC handle の寿命を分けて読みます。",
        "performance": "performance sample では、何を timing に含めるかと warmup/repeat の扱いを必ず確認します。",
        "python_cuda": "Python sample では、Python object の lifetime と CUDA stream/context の lifetime が別であることを意識します。",
    }
    for key in sorted(concepts):
        if key in mapping:
            notes.append(mapping[key])
    return notes


def common_mistakes(concepts: set[str]) -> list[str]:
    mistakes = [
        "API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。",
        "allocation size を byte で渡す API と element count で考える loop を混同する。",
        "kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。",
    ]
    if "shared_memory" in concepts:
        mistakes.append("shared memory を書いた thread と読む thread の間に必要な barrier を見落とす。")
    if "streams_events" in concepts:
        mistakes.append("different stream 間に依存があるのに event や explicit sync を置かない。")
    if "graphs" in concepts:
        mistakes.append("graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。")
    if "libraries" in concepts:
        mistakes.append("leading dimension、stride、descriptor、workspace size を host 配列の見た目だけで判断する。")
    if "runtime_driver_nvrtc" in concepts:
        mistakes.append("JIT compile log や mangled kernel name を確認せず、launch failure だけを見る。")
    if "multi_gpu_p2p_ipc" in concepts:
        mistakes.append("peer access が有効な device pair と、単に複数 GPU が存在することを混同する。")
    if "python_cuda" in concepts:
        mistakes.append("CuPy/NumPy/DLPack 変換で hidden copy や hidden sync が起きる可能性を見落とす。")
    return mistakes


def exercises(concepts: set[str], api_tokens: list[str]) -> list[str]:
    first_api = api_tokens[0] if api_tokens else "the main CUDA API"
    items = [
        f"`{first_api}` の直前と直後で、どの memory/resource が有効になったかをメモする。",
        "source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。",
        "problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。",
    ]
    if "shared_memory" in concepts:
        items.append("shared memory tile の producer、consumer、barrier を図にする。")
    if "streams_events" in concepts:
        items.append("stream timeline を描き、copy、kernel、event、host wait の位置を分ける。")
    if "graphs" in concepts:
        items.append("graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。")
    if "libraries" in concepts:
        items.append("handle/descriptor/workspace の作成、利用、破棄を対応表にする。")
    if "multi_gpu_p2p_ipc" in concepts:
        items.append("device ごとの ownership と、peer/IPC で共有される resource を分けて書く。")
    if "python_cuda" in concepts:
        items.append("Python object、DLPack/CuPy view、CUDA buffer の lifetime を別々に書く。")
    items.append("関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。")
    return items


def spotlight(sample: Path, concepts: set[str]) -> list[str]:
    name = sample.name.lower()
    path = rel(sample).lower()
    notes: list[str] = []
    if "vectoradd" in name:
        notes.append("vectorAdd では「1 thread が 1 element を担当する」形が最小単位です。rounded-up grid と `i < numElements` の境界チェックを必ず一緒に読みます。")
        notes.append("host 配列、device 配列、copy direction、kernel launch、D2H copy、validation、free の順序が CUDA Runtime API の基本形です。")
    if "devicequery" in name or "topologyquery" in name:
        notes.append("deviceQuery/topologyQuery は計算 kernel ではなく capability を読む sample です。SM version、memory size、feature flag は他 sample が実行可能かを判断する材料になります。")
    if "matrixmul" in name or "matmul" in name:
        notes.append("matrixMul 系では global memory の値を shared memory tile に移し、barrier 後に再利用します。性能の主役は arithmetic だけでなく memory reuse です。")
        notes.append("cuBLAS/Tensor Core 版がある場合は、同じ数学でも API、data layout、precision、workspace の責任分担が変わります。")
    if "stream" in name or "async" in name or "overlap" in path:
        notes.append("stream sample では、copy と kernel が本当に重なるには pinned memory、non-default stream、依存 event の条件がそろう必要があります。")
    if "graph" in name:
        notes.append("graph sample では、最初の capture/instantiate と繰り返し launch を分けて読みます。launch overhead を減らす代わりに依存関係と buffer lifetime が固定されます。")
    if "nvrtc" in name or "jit" in name or "ptx" in name or "drv" in name or "nvvm" in path:
        notes.append("NVRTC/Driver/JIT 系では、compile/load/link と kernel launch が別の段階です。compile log、module、function handle、launch parameter の対応が重要です。")
    if "multigpu" in name or "p2p" in name or "ipc" in name or "mpi" in name or "mmap" in name:
        notes.append("multi-GPU/P2P/IPC 系では、どの process/thread/device が resource を所有しているかを先に分けると読みやすくなります。")
    if "4_cuda_libraries" in path or "library" in path or "libraries" in concepts:
        notes.append("library sample では、CUDA kernel を直接書かなくても library call が device work を投入します。handle/descriptor/workspace の lifetime を kernel launch と同じ厳しさで追います。")
    if rel(sample).startswith("python/"):
        notes.append("Python sample では、Python object が CUDA pointer、stream、module を包んでいます。見た目は Python でも、同期と lifetime は CUDA の規則で決まります。")
    return notes


def section_block(japanese: str, memo: str) -> list[str]:
    return [
        "> **日本語**",
        f"> {japanese}",
        ">",
        "> **学習メモ**",
        f"> {memo}",
    ]


def relative_theme_link(sample: Path, theme_key: str) -> str:
    label, target, note = THEMES[theme_key]
    rel_path = Path(target)
    link = Path(*([".."] * len(sample.relative_to(ROOT).parts))) / rel_path
    return f"- [{label}]({link.as_posix()}): {note}"


def render(sample: Path) -> str:
    readme_path = sample / "README.md"
    readme = read_text(readme_path) if readme_path.exists() else ""
    title = first_heading(readme, sample.name)
    overview = english_overview(readme)
    headings = heading_list(readme)
    files = files_in_sample(sample)
    srcs = source_files(sample)
    tokens = token_counter(sample)
    api_tokens = [token for token, _ in tokens.most_common(18)]
    concepts = detect_concepts(sample, tokens, readme)
    primary_sources = [p.name for p in srcs[:5]]
    if not primary_sources:
        primary_sources = [p.name for p in files[:3]]
    language = "Python" if rel(sample).startswith("python/") else "C++/CUDA"
    target = target_name(sample)
    entry = python_entry(sample)
    theme_order = [
        "python_cuda",
        "runtime_driver_nvrtc",
        "libraries",
        "graphs",
        "multi_gpu_p2p_ipc",
        "tensor_cores_wmma",
        "cooperative_groups",
        "shared_memory",
        "streams_events",
        "unified_memory",
        "sync_atomics",
        "performance",
        "memory",
        "kernel_indexing",
        "execution_model",
        "debugging_profiling_testing",
    ]
    related = [key for key in theme_order if key in concepts][:8]

    lines: list[str] = [
        f"# {title} - Japanese Learning Guide",
        "",
        f"English source: [README.md](README.md)",
    ]
    if (sample / "CMakeLists.txt").exists():
        lines.append("Build file: [CMakeLists.txt](CMakeLists.txt)")
    if (sample / "requirements.txt").exists():
        lines.append("Python requirements: [requirements.txt](requirements.txt)")
    lines.extend(
        [
            "",
            "## English Overview",
            "",
            overview,
        ]
    )
    if headings:
        lines.extend(["", "Original README headings: " + ", ".join(f"`{heading}`" for heading in headings)])
    lines.extend([""] + section_block(
        f"`{rel(sample)}` は `{language}` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。",
        "API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。",
    ))

    lines.extend(
        [
            "",
            "## Purpose",
            "",
            f"English anchor: read `{sample.name}` as a focused example of the CUDA concepts used in `{rel(sample)}`.",
        ]
    )
    concept_names = ", ".join(THEMES[key][0] for key in related[:5])
    lines.extend(section_block(
        f"この sample の目的は、`{sample.name}` の小さな実装を通して {concept_names or 'CUDA API'} を具体的に追うことです。",
        f"最初に `{', '.join(primary_sources)}` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。",
    ))

    prereq = [
        "CUDA Toolkit matching this repository branch",
        "NVIDIA driver and a CUDA-capable GPU supported by the original README",
    ]
    if language == "C++/CUDA":
        prereq.append("CMake 3.20 or newer and a host compiler supported by the toolkit")
    else:
        prereq.extend(["Python 3 environment", "`requirements.txt` packages when the file is present"])
    if "libraries" in concepts:
        prereq.append("CUDA library components used by this sample, such as cuBLAS, cuFFT, cuSolver, NPP, CUB, or nvJPEG")
    if "multi_gpu_p2p_ipc" in concepts:
        prereq.append("The device topology required by the README, such as multiple GPUs, peer access, IPC, MPI, or process support")
    if has_any(rel(sample), "gl", "d3d", "vulkan", "egl", "nvsci", "nvmedia"):
        prereq.append("The graphics, display, or platform stack named by the English README")
    lines.extend(["", "## Prerequisites", ""])
    lines.extend(f"- {item}" for item in prereq)
    lines.extend([""] + section_block(
        "必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。",
        "実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。",
    ))

    lines.extend(["", "## Files", ""])
    for file in files:
        local = file.relative_to(sample).as_posix()
        lines.append(f"- `{local}`: {file_role(file, sample)}")
    lines.extend([""] + section_block(
        "各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。",
        "data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。",
    ))

    flow = [
        "Read the original README and identify the supported device, OS, and library assumptions.",
        f"Open `{primary_sources[0]}` first and locate the host-side setup or Python entry point.",
    ]
    if "python_cuda" in concepts:
        flow.append("Create or select the CUDA device/context and construct Python objects that wrap CUDA resources.")
    else:
        flow.append("Select the CUDA device and allocate host/device resources used by the sample.")
    if "libraries" in concepts:
        flow.append("Create library handles, descriptors, plans, or workspaces before the library call.")
    if "runtime_driver_nvrtc" in concepts:
        flow.append("Compile, link, load, or look up device code before launch, and keep compile logs visible while debugging.")
    if "graphs" in concepts:
        flow.append("Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.")
    if "multi_gpu_p2p_ipc" in concepts:
        flow.append("Enumerate devices, enable peer or IPC access, and record which device/process owns each resource.")
    if "shared_memory" in concepts:
        flow.append("Inside the kernel, map thread/block indexes to tile elements and check the barrier around shared memory reuse.")
    flow.extend(
        [
            "Move, map, or expose input data so GPU work can read the intended values.",
            "Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.",
            "Synchronize only at the required correctness or timing boundary.",
            "Validate results against the CPU/reference path, generated artifact, or expected status message.",
            "Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.",
        ]
    )
    lines.extend(["", "## Execution Flow", ""])
    lines.extend(f"- {item}" for item in flow)
    lines.extend([""] + section_block(
        "実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。",
        "CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。",
    ))

    lines.extend(["", "## Concrete Reading Path", ""])
    for source in srcs[:10]:
        local = source.relative_to(sample).as_posix()
        source_tokens = Counter(TOKEN_RE.findall(read_text(source))).most_common(5)
        token_text = ", ".join(f"`{token}`" for token, _ in source_tokens) if source_tokens else "control flow and helper functions"
        lines.append(f"- `{local}`: focus on {token_text}.")
    if len(srcs) > 10:
        lines.append(f"- Additional source files: {len(srcs) - 10} more support files. Use the same setup/work/sync/cleanup lens.")
    lines.extend([""] + section_block(
        "読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。",
        "まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。",
    ))

    lines.extend(["", "## Code Walkthrough", ""])
    lines.extend(render_code_walkthrough(sample))

    lines.extend(["", "## Key APIs And Concepts", ""])
    if api_tokens:
        lines.extend(["| API or concept | Why it matters |", "| - | - |"])
        for token in api_tokens[:14]:
            lines.append(f"| `{token}` | {api_note(token)} |")
    else:
        lines.extend(
            [
                "| API or concept | Why it matters |",
                "| - | - |",
                "| Source control flow | No CUDA API token was detected automatically, so read setup, generated output, and helper boundaries manually. |",
                "| Build file | The build wiring still documents the target name, input files, and environment assumptions. |",
            ]
        )
    lines.extend([""] + section_block(
        "API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。",
        "helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。",
    ))

    lines.extend(["", "## Memory, Synchronization, And Performance Notes", ""])
    for note in concept_notes(concepts):
        lines.append(f"- {note}")
    lines.extend([""] + section_block(
        "memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。",
        "correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。",
    ))

    spot = spotlight(sample, concepts)
    if spot:
        lines.extend(["", "## Sample-Specific Notes", ""])
        lines.extend(f"- {item}" for item in spot)
        lines.extend([""] + section_block(
            "この section は同じ template ではなく、sample 名、path、検出した API から読みどころを絞っています。",
            "似た名前の sample は Runtime 版、Driver 版、NVRTC 版、library 版の違いを比較すると学習効果が高くなります。",
        ))

    lines.extend(["", "## Build And Run", ""])
    if language == "C++/CUDA":
        lines.extend(
            [
                "English commands remain authoritative. Typical repository-root CMake flow:",
                "",
                "```bash",
                "cmake -S . -B build",
                f"cmake --build build --target {target}",
                f"ctest --test-dir build -R {target}",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "English commands remain authoritative. Typical local Python flow:",
                "",
                "```bash",
                "pip install -r requirements.txt",
                f"python {entry}",
                "```",
            ]
        )
    lines.extend([""] + section_block(
        "実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。",
        "build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。",
    ))

    expected = "Run the sample as documented and compare its output with the original README, validation message, generated file, or reference result."
    if "devicequery" in sample.name.lower():
        expected = "The sample prints device capability and topology-style properties rather than computing a numerical kernel result."
    elif "performance" in concepts:
        expected = "The sample prints timing, bandwidth, latency, throughput, or comparison data; exact values depend on hardware and driver."
    elif has_any(rel(sample), "image", "filter", "texture", "volume", "render", "gl", "d3d", "vulkan", "egl"):
        expected = "The sample may display a window or produce/validate image-like output; exact visuals depend on platform support."
    elif "libraries" in concepts:
        expected = "The sample validates the library result against a CPU/reference path or reports the documented success status."
    lines.extend(["", "## Expected Behavior", "", expected])
    lines.extend(section_block(
        "期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。",
        "`PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。",
    ))

    lines.extend(["", "## Common Mistakes", ""])
    lines.extend(f"- {item}" for item in common_mistakes(concepts))
    lines.extend([""] + section_block(
        "失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。",
        "source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。",
    ))

    lines.extend(["", "## Exercises", ""])
    lines.extend(f"- {item}" for item in exercises(concepts, api_tokens))
    lines.extend([""] + section_block(
        "演習では code を動かす前に、読み取った仮説を memo として整理します。",
        "変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。",
    ))

    lines.extend(["", "## Related Themes", ""])
    lines.extend(relative_theme_link(sample, key) for key in related)
    lines.extend([""] + section_block(
        "関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。",
        "同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。",
    ))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    count = 0
    for sample in sample_dirs():
        output = sample / "README.ja.md"
        output.write_text(render(sample), encoding="utf-8", newline="\n")
        count += 1
    print(f"Regenerated {count} sample README.ja.md files.")


if __name__ == "__main__":
    main()
