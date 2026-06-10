#!/usr/bin/env python3
"""Inventory Japanese translation and annotation coverage for this fork."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS_JA = ROOT / "docs_ja"
TRANSLATED = DOCS_JA / "translated"
STATUS = DOCS_JA / "_translation_status.md"
ANNOTATION_JSON = DOCS_JA / "_annotation_inventory.json"
SAMPLE_README_JSON = DOCS_JA / "_sample_readme_inventory.json"
THEME_GUIDE_JSON = DOCS_JA / "_theme_inventory.json"
COMPANION_JSON = DOCS_JA / "_companion_inventory.json"
GLOSSARY_JSON = DOCS_JA / "_glossary_inventory.json"

DOC_EXTS = {".md", ".txt", ".pdf", ".doc", ".docx"}
CODE_EXTS = {
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
}
VENDOR_PARTS = {
    ".git",
    "bin",
    "docs_ja",
    "__pycache__",
    "cmake/CPM.cmake",
    "Common/GL",
    "cpp/2_Concepts_and_Techniques/interval/boost",
    "cpp/5_Domain_Specific/simpleD3D11Texture/d3dx11effect",
}
GENERATED_NAME_PARTS = {
    "cuda_drvapi_dynlink_cuda",
    "ptxdump",
}

NEARBY_BEFORE = 4
NEARBY_AFTER = 2
TOP_LINE_LIMIT = 45
GROUP_LINE_GAP = 8
MISSING_EXAMPLE_LIMIT = 30
REQUIRED_SAMPLE_SECTIONS = (
    "## Purpose",
    "## Prerequisites",
    "## Files",
    "## Execution Flow",
    "## Concrete Reading Path",
    "## Key APIs And Concepts",
    "## Memory, Synchronization, And Performance Notes",
    "## Build And Run",
    "## Expected Behavior",
    "## Common Mistakes",
    "## Exercises",
    "## Related Themes",
)
REQUIRED_THEME_SECTIONS = (
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
)
MAJOR_COMPANION_TARGETS = (
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CMakeLists.txt",
    DOCS_JA / "README.md",
)
GLOSSARY_TARGETS = (
    DOCS_JA / "glossary" / "README.md",
    DOCS_JA / "glossary" / "terms.md",
    DOCS_JA / "glossary" / "api.md",
    DOCS_JA / "glossary" / "memory_transfer.md",
    DOCS_JA / "glossary" / "build_run.md",
)
REQUIRED_GLOSSARY_SECTIONS = {
    "README.md": ("## Files", "## How To Use"),
    "terms.md": ("## Reading Notes", "## Common Confusions", "## Exercises"),
    "api.md": ("## API Reading Pattern", "## Common Mistakes", "## Exercises"),
    "memory_transfer.md": ("## Direction Checklist", "## Common Mistakes", "## Exercises"),
    "build_run.md": ("## Build Reading Pattern", "## Common Mistakes", "## Exercises"),
}
GLOSSARY_MIN_LINES = {
    "README.md": 25,
    "terms.md": 35,
    "api.md": 40,
    "memory_transfer.md": 38,
    "build_run.md": 38,
}


@dataclass(frozen=True)
class AnchorRule:
    name: str
    pattern: re.Pattern[str]
    label: str


ANCHOR_RULES: tuple[AnchorRule, ...] = (
    AnchorRule(
        "device_memory",
        # JP: device allocation/free の anchor。確保と解放が対応しているかを近傍コメントで確認します。
        re.compile(r"\b(cudaMalloc(?:Pitch|3D|Array)?|cudaFree(?:Array)?|cuMemAlloc\w*|cuMemFree\w*|mem_alloc|DeviceMemory|device_buffer)\b"),
        "device allocation/free",
    ),
    AnchorRule(
        "pinned_memory",
        # JP: pinned host memory の anchor。非同期転送と lifetime の説明が近くにあるかを見ます。
        re.compile(r"\b(cudaMallocHost|cudaFreeHost|cudaHostAlloc|cudaHostRegister|cudaHostUnregister|pinned|pagelocked)\b", re.I),
        "pinned host memory",
    ),
    AnchorRule(
        "managed_memory",
        # JP: Unified Memory の anchor。移動タイミング、prefetch、同期の説明を求めます。
        re.compile(r"\b(cudaMallocManaged|cudaMemPrefetchAsync|cudaMemAdvise|ManagedMemoryResource|managed_mr|managed memory|UnifiedMemory|Unified Memory)\b"),
        "unified memory",
    ),
    AnchorRule(
        "transfer",
        # JP: host/device transfer の anchor。方向、Async の順序、同期境界の説明を求めます。
        re.compile(r"\b(cudaMemcpy(?:2D|3D)?(?:Async)?|cuMemcpy\w*|cudaMemset(?:Async)?|copy_to_host|copy_to_device|copy_from|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|HostToDevice|DeviceToHost)\b"),
        "host/device transfer",
    ),
    AnchorRule(
        "kernel_launch",
        # JP: kernel launch の anchor。grid/block/shared-memory/stream と完了確認の説明を求めます。
        re.compile(r"(<<<[^>]*>>>|\.launch\s*\(|cuLaunchKernel|cudaLaunchKernel|LaunchKernel)"),
        "kernel launch",
    ),
    AnchorRule(
        "indexing",
        # JP: indexing の anchor。担当要素の計算と境界チェックの説明を求めます。
        re.compile(r"\b(blockIdx|threadIdx|blockDim|gridDim|laneId|warpSize|cg::this_thread_block|this_grid)\b"),
        "thread/block indexing",
    ),
    AnchorRule(
        "shared_memory",
        # JP: shared memory の anchor。block 内共有、tile、必要な同期の説明を求めます。
        re.compile(r"\b(__shared__|extern\s+__shared__|sharedMem|shared memory|SharedMemory|cuda.shared)\b"),
        "shared memory",
    ),
    AnchorRule(
        "sync",
        # JP: synchronization の anchor。host が何を待つのか、block 内か device/stream 全体かを区別します。
        re.compile(r"\b(__syncthreads|__syncwarp|cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize|cudaThreadSynchronize|cooperative_groups::sync|cg::sync|synchronize\s*\(|Synchronize)\b"),
        "synchronization boundary",
    ),
    AnchorRule(
        "streams_events",
        # JP: streams/events の anchor。非同期順序、overlap、計測範囲の説明を求めます。
        re.compile(r"\b(cudaStream\w*|cudaEvent\w*|CUstream|CUevent|Stream\(|Event\(|stream\s*=)\b", re.I),
        "streams/events",
    ),
    AnchorRule(
        "graphs",
        # JP: CUDA Graph の anchor。node 依存、capture/replay、buffer lifetime の説明を求めます。
        re.compile(r"\b(cudaGraph\w*|cudaUserObject\w*|graphExec|Graph\(|graph\s*=)\b"),
        "CUDA Graph dependency",
    ),
    AnchorRule(
        "driver_api",
        # JP: Driver API の anchor。CU* handle の所有、context/module/function の境界を説明します。
        re.compile(r"\b(cu[A-Z][A-Za-z0-9_]*\s*\(|CUdevice|CUcontext|CUmodule|CUfunction|CUstream|CUevent)\b"),
        "Driver API boundary",
    ),
    AnchorRule(
        "nvrtc",
        # JP: NVRTC/JIT の anchor。compile/link した module と kernel launch の対応を説明します。
        re.compile(r"\b(nvrtc\w*|NVRTC|nvJitLink\w*|jitify|JIT compilation|Program\(|Linker\()\b"),
        "runtime compilation/JIT",
    ),
    AnchorRule(
        "library_resources",
        # JP: CUDA library resource の anchor。handle/descriptor/workspace の作成と破棄を対応させます。
        re.compile(r"\b(cublas\w*|cufft\w*|cusparse\w*|cusolver\w*|curand\w*|nppi\w*|npps\w*|npp[A-Z]\w*|nvjpeg\w*|nccl\w*|cudnn\w*|Create\w*Handle|Destroy\w*Handle)\b"),
        "CUDA library resources",
    ),
    AnchorRule(
        "validation",
        # JP: validation の anchor。CPU/reference 比較と、失敗時に疑う境界を説明します。
        re.compile(r"\b(sdkCompare\w*|compare\w*|checkResult|validate\w*|np\.allclose|cupy\.allclose|Result\s*=\s*PASS|PASS|FAIL)\b"),
        "validation/checking",
    ),
    AnchorRule(
        "cleanup",
        # JP: cleanup の anchor。確保、作成、登録した resource の lifetime end を確認します。
        re.compile(r"\b(cudaFree\w*|cudaDestroy\w*|cudaEventDestroy|cudaStreamDestroy|Destroy\w*|cuMemFree\w*)\b"),
        "cleanup/lifetime end",
    ),
    AnchorRule(
        "build_cuda",
        # JP: CUDA build wiring の anchor。target、architecture、link dependency の意味を説明します。
        re.compile(r"\b(CUDA::|CUDAToolkit|cuda_add|target_link_libraries|add_executable|set_target_properties|CMAKE_CUDA|CUDA_ARCHITECTURES|find_package)\b"),
        "CMake CUDA build wiring",
    ),
    AnchorRule(
        "python_cuda",
        # JP: Python CUDA の anchor。Python object と CUDA resource/context/stream の境界を説明します。
        re.compile(r"\b(cuda\.|cupy|cp\.|numba\.cuda|Device\(|Context\(|Module\(|Kernel\(|MemoryResource|mpi4py|torch|tensorflow)\b"),
        "Python CUDA boundary",
    ),
)

COMMENT_ONLY_RE = re.compile(r"^\s*(//|#|\*|/\*|\*/|<!--)")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def has_vendor_part(path: Path) -> bool:
    value = rel(path)
    parts = set(path.relative_to(ROOT).parts)
    if parts & {".git", "bin", "docs_ja", "__pycache__"}:
        return True
    return any(value.startswith(prefix + "/") or value == prefix for prefix in VENDOR_PARTS if "/" in prefix)


def is_generated(path: Path) -> bool:
    name = path.name.lower()
    return any(part in name for part in GENERATED_NAME_PARTS)


def source_docs() -> list[Path]:
    docs: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or has_vendor_part(path):
            continue
        if path.name.endswith(".ja.md"):
            continue
        if path.suffix.lower() in DOC_EXTS:
            docs.append(path)
    return sorted(docs, key=rel)


def companion_for(path: Path) -> Path:
    relative = rel(path)
    safe = relative.replace("/", "__")
    return TRANSLATED / f"{safe}.ja.md"


def sample_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in (ROOT / "cpp", ROOT / "python"):
        if not base.exists():
            continue
        for readme in base.rglob("README.md"):
            if has_vendor_part(readme):
                continue
            dirs.append(readme.parent)
    return sorted(dirs, key=rel)


def code_files() -> list[Path]:
    files: set[Path] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or has_vendor_part(path) or is_generated(path):
            continue
        if path.name == "CMakeLists.txt" or path.suffix.lower() in CODE_EXTS:
            files.add(path)
    return sorted(files, key=rel)


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def source_paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    suffix = path.suffix.lower()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if in_fence:
            continue
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(("-", "*", "+")) and suffix in {".md", ".txt"}:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if len(stripped) > 28:
                paragraphs.append(stripped)
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return [paragraph for paragraph in paragraphs if paragraph.strip()]


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def is_comment_only(line: str) -> bool:
    return bool(COMMENT_ONLY_RE.match(line))


def python_triple_string_lines(lines: list[str]) -> set[int]:
    in_triple = False
    quote = ""
    inside: set[int] = set()
    for index, line in enumerate(lines):
        position = 0
        if in_triple:
            inside.add(index + 1)
            end = line.find(quote, position)
            if end < 0:
                continue
            in_triple = False
            position = end + 3
        while True:
            candidates = [(line.find("'''", position), "'''"), (line.find('"""', position), '"""')]
            candidates = [(offset, marker) for offset, marker in candidates if offset >= 0]
            if not candidates:
                break
            start, marker = min(candidates, key=lambda item: item[0])
            comment_at = line.find("#")
            if comment_at >= 0 and comment_at < start:
                break
            end = line.find(marker, start + 3)
            if end >= 0:
                position = end + 3
                continue
            in_triple = True
            quote = marker
            inside.add(index + 1)
            break
    return inside


def strip_python_string_literals(line: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(line):
        start = index
        while index < len(line) and line[index] in "rRuUbBfF":
            index += 1
        if index < len(line) and line[index] in {"'", '"'}:
            quote = line[index]
            if index + 2 < len(line) and line[index : index + 3] == quote * 3:
                index += 3
                end_marker = quote * 3
                while index < len(line) and line[index : index + 3] != end_marker:
                    index += 1
                index = min(len(line), index + 3)
                result.append(" ")
                continue
            index += 1
            escaped = False
            while index < len(line):
                char = line[index]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    index += 1
                    break
                index += 1
            result.append(" ")
            continue
        result.append(line[start])
        index = start + 1
    return "".join(result)


def strip_c_like_string_literals(line: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char in {"'", '"'}:
            quote = char
            index += 1
            escaped = False
            while index < len(line):
                current = line[index]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    index += 1
                    break
                index += 1
            result.append(" ")
            continue
        if char == "R" and index + 1 < len(line) and line[index + 1] == '"':
            # Raw string literals often contain sample output text; anchor search
            # should stay on executable/source tokens.
            result.append(" ")
            break
        result.append(char)
        index += 1
    return "".join(result)


def strip_inline_comment_text(line: str, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return line.split("#", 1)[0]
    if suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
        line = line.split("//", 1)[0]
        while "/*" in line and "*/" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            line = line[:start] + " " + line[end + 2 :]
    if suffix in {".cmake", ".sh", ".bat", ".cmd", ".ps1"} or path.name == "CMakeLists.txt":
        return line.split("#", 1)[0]
    return line


def searchable_code_line(line: str, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        line = strip_python_string_literals(line)
    elif suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
        line = strip_c_like_string_literals(line)
    return strip_inline_comment_text(line, path)


def is_macro_continuation(lines: list[str], index: int) -> bool:
    stripped = lines[index].rstrip()
    if stripped.endswith("\\"):
        return True
    return index > 0 and lines[index - 1].rstrip().endswith("\\")


def is_probably_generic_jp(line: str) -> bool:
    text = line.lower()
    generic_terms = (
        "この file では",
        "この build file",
        "英語の識別子",
        "保持します",
    )
    concrete_terms = (
        "所有",
        "解放",
        "転送",
        "方向",
        "host",
        "device",
        "stream",
        "event",
        "resource",
        "timeline",
        "timing",
        "memory",
        "lifetime",
        "context",
        "module",
        "function",
        "thread",
        "index",
        "api",
        "cuda",
        "同期境界",
        "同期",
        "依存関係",
        "依存",
        "完了",
        "境界",
        "launch shape",
        "grid",
        "block",
        "shared",
        "workspace",
        "descriptor",
        "handle",
        "検証",
        "破棄",
    )
    return any(term in line for term in generic_terms) and not any(term in text for term in concrete_terms)


def line_has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def inaccurate_comment_flags(path: Path, lines: list[str], jp_line_numbers: list[int]) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    relative = rel(path)
    for line_number in jp_line_numbers:
        text = lines[line_number - 1].strip()
        lower = text.lower()
        if "__syncthreads" in text and any(term in lower for term in ("host 処理", "cpu", "gpu work", "device-wide")):
            flags.append(
                {
                    "line": line_number,
                    "reason": "__syncthreads is block-local device synchronization, not host completion",
                    "text": text,
                }
            )
        if "__syncthreads" in text and "検証" in text:
            flags.append(
                {
                    "line": line_number,
                    "reason": "__syncthreads comment refers to host-side validation",
                    "text": text,
                }
            )
        if relative.endswith("cpp/1_Utilities/deviceQuery/deviceQuery.cpp") and line_number <= TOP_LINE_LIMIT:
            negative_property_note = any(term in text for term in ("行いません", "ではありません", "しません"))
            if not negative_property_note and (
                any(term in lower for term in ("stream/event", "unified memory", "migration")) or "block 内同期" in text
            ):
                flags.append(
                    {
                        "line": line_number,
                        "reason": "deviceQuery reports device properties; it should not be described as a streams/shared/unified-memory execution sample",
                        "text": text,
                    }
                )
        if relative.endswith("cpp/1_Utilities/deviceQuery/deviceQuery.cpp") and "GPU result" in text:
            flags.append(
                {
                    "line": line_number,
                    "reason": "deviceQuery validates API/device availability, not GPU numerical results against a CPU reference",
                    "text": text,
                }
            )
    return flags


def useful_jp_line_numbers(
    lines: list[str], jp_line_numbers: list[int], inaccurate_lines: set[int]
) -> list[int]:
    useful: list[int] = []
    for line_number in jp_line_numbers:
        text = lines[line_number - 1]
        if line_number in inaccurate_lines:
            continue
        if not line_has_japanese(text):
            continue
        if is_probably_generic_jp(text):
            continue
        useful.append(line_number)
    return useful


def nearby_jp_lines(anchor_line: int, jp_lines: Iterable[int]) -> list[int]:
    return [jp for jp in jp_lines if anchor_line - NEARBY_BEFORE <= jp <= anchor_line + NEARBY_AFTER]


def detect_anchor_lines(lines: list[str], path: Path | None = None) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    ignored_python_lines = python_triple_string_lines(lines) if path and path.suffix.lower() == ".py" else set()
    for index, line in enumerate(lines):
        number = index + 1
        if "JP:" in line:
            continue
        if number in ignored_python_lines:
            continue
        if path and path.suffix.lower() in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"} and is_macro_continuation(lines, index):
            continue
        if is_comment_only(line):
            continue
        search_line = searchable_code_line(line, path) if path else line
        if not search_line.strip():
            continue
        categories: list[str] = []
        labels: list[str] = []
        for rule in ANCHOR_RULES:
            if rule.pattern.search(search_line):
                categories.append(rule.name)
                labels.append(rule.label)
        if categories:
            anchors.append(
                {
                    "id": len(anchors),
                    "line": number,
                    "categories": sorted(set(categories)),
                    "labels": sorted(set(labels)),
                    "text": line.strip()[:180],
                }
            )
    return anchors


def anchor_has_nearby_jp(anchor_line: int, jp_lines: Iterable[int]) -> bool:
    return any(anchor_line - NEARBY_BEFORE <= jp <= anchor_line + NEARBY_AFTER for jp in jp_lines)


def jp_allows_grouping(text: str, categories: set[str]) -> bool:
    lower = text.lower()
    generic_group_terms = ("連続", "複数", "まとめ", "それぞれ", "対応", "group", "all", "pairs")
    if any(term in lower or term in text for term in generic_group_terms):
        return True
    if categories & {"device_memory", "pinned_memory", "managed_memory"}:
        return any(term in lower or term in text for term in ("ownership", "lifetime", "resource", "確保", "解放", "所有"))
    if "transfer" in categories:
        return any(term in lower or term in text for term in ("direction", "方向", "visibility", "async", "同期"))
    if categories & {"streams_events", "graphs"}:
        return any(term in lower or term in text for term in ("ordering", "順序", "依存", "計測", "lifetime"))
    if "cleanup" in categories:
        return any(term in lower or term in text for term in ("cleanup", "lifetime", "解放", "破棄", "閉じ"))
    if categories & {"library_resources", "driver_api", "nvrtc"}:
        return any(term in lower or term in text for term in ("handle", "descriptor", "workspace", "resource", "lifetime", "所有"))
    if categories & {"indexing", "shared_memory", "sync", "kernel_launch", "validation", "python_cuda", "build_cuda"}:
        return True
    return False


def group_cover_anchors(
    anchors: list[dict[str, object]],
    directly_covered: dict[int, int],
    lines: list[str],
) -> tuple[dict[int, int], list[dict[str, object]]]:
    grouped: dict[int, int] = {}
    grouped_records: list[dict[str, object]] = []
    if not anchors:
        return grouped, grouped_records

    current: list[dict[str, object]] = []
    for anchor in anchors:
        if not current:
            current = [anchor]
            continue
        previous = current[-1]
        previous_categories = set(str(item) for item in previous["categories"])
        current_categories = set(str(item) for item in anchor["categories"])
        gap = int(anchor["line"]) - int(previous["line"])
        if gap <= GROUP_LINE_GAP and previous_categories & current_categories:
            current.append(anchor)
        else:
            grouped.update(_cover_anchor_group(current, directly_covered, lines, grouped_records))
            current = [anchor]
    grouped.update(_cover_anchor_group(current, directly_covered, lines, grouped_records))
    return grouped, grouped_records


def _cover_anchor_group(
    group: list[dict[str, object]],
    directly_covered: dict[int, int],
    lines: list[str],
    grouped_records: list[dict[str, object]],
) -> dict[int, int]:
    result: dict[int, int] = {}
    if len(group) < 2:
        return result
    covered_members = [anchor for anchor in group if int(anchor["id"]) in directly_covered]
    if not covered_members:
        return result
    group_categories = set.intersection(*(set(str(item) for item in anchor["categories"]) for anchor in group))
    if not group_categories:
        return result
    for covered in covered_members:
        jp_line = directly_covered[int(covered["id"])]
        jp_text = lines[jp_line - 1]
        if not jp_allows_grouping(jp_text, group_categories):
            continue
        for anchor in group:
            anchor_id = int(anchor["id"])
            if anchor_id in directly_covered or anchor_id in result:
                continue
            result[anchor_id] = jp_line
            grouped_records.append(
                {
                    "line": int(anchor["line"]),
                    "categories": anchor["categories"],
                    "covered_by_jp_line": jp_line,
                    "rule": "adjacent repeated anchor group",
                    "text": anchor["text"],
                }
            )
        break
    return result


def group_cover_build_file(
    path: Path,
    anchors: list[dict[str, object]],
    directly_covered: dict[int, int],
    useful_jp_lines: list[int],
    lines: list[str],
) -> tuple[dict[int, int], list[dict[str, object]]]:
    if path.name != "CMakeLists.txt" and path.suffix.lower() != ".cmake":
        return {}, []
    overview_lines = [line for line in useful_jp_lines if line <= TOP_LINE_LIMIT]
    if not overview_lines:
        return {}, []
    overview = overview_lines[0]
    overview_text = lines[overview - 1].lower()
    if not any(term in overview_text for term in ("cmake", "target", "build", "link", "cuda", "architecture")):
        return {}, []
    grouped: dict[int, int] = {}
    records: list[dict[str, object]] = []
    for anchor in anchors:
        anchor_id = int(anchor["id"])
        categories = set(str(item) for item in anchor["categories"])
        if anchor_id in directly_covered or "build_cuda" not in categories:
            continue
        grouped[anchor_id] = overview
        records.append(
            {
                "line": int(anchor["line"]),
                "categories": anchor["categories"],
                "covered_by_jp_line": overview,
                "rule": "file-level CMake build overview",
                "text": anchor["text"],
            }
        )
    return grouped, records


def analyze_code_file(path: Path) -> dict[str, object]:
    lines = read_lines(path)
    jp_line_numbers = [i for i, line in enumerate(lines, start=1) if "JP:" in line]
    jp_lines = [lines[i - 1].strip() for i in jp_line_numbers]
    inaccurate_flags = inaccurate_comment_flags(path, lines, jp_line_numbers)
    inaccurate_lines = {int(flag["line"]) for flag in inaccurate_flags}
    useful_jp_lines = useful_jp_line_numbers(lines, jp_line_numbers, inaccurate_lines)
    anchors = detect_anchor_lines(lines, path)
    categories: dict[str, int] = {}
    categories_with_jp: set[str] = set()
    directly_covered: dict[int, int] = {}

    for anchor in anchors:
        line_number = int(anchor["line"])
        nearby_lines = nearby_jp_lines(line_number, useful_jp_lines)
        for category in anchor["categories"]:
            categories[category] = categories.get(category, 0) + 1
        if nearby_lines:
            jp_line = min(nearby_lines, key=lambda jp: abs(jp - line_number))
            directly_covered[int(anchor["id"])] = jp_line

    grouped_coverage, grouped_records = group_cover_anchors(anchors, directly_covered, lines)
    build_grouped, build_grouped_records = group_cover_build_file(path, anchors, directly_covered, useful_jp_lines, lines)
    for anchor_id, jp_line in build_grouped.items():
        grouped_coverage.setdefault(anchor_id, jp_line)
    grouped_records.extend(build_grouped_records)
    covered_ids = set(directly_covered) | set(grouped_coverage)
    missing_details = [anchor for anchor in anchors if int(anchor["id"]) not in covered_ids]
    for anchor in anchors:
        if int(anchor["id"]) in covered_ids:
            for category in anchor["categories"]:
                categories_with_jp.add(str(category))

    detected = sorted(categories)
    missing_categories = sorted({str(category) for anchor in missing_details for category in anchor["categories"]})
    top_only = bool(jp_line_numbers) and all(line <= TOP_LINE_LIMIT for line in jp_line_numbers)
    generic_line_count = sum(1 for line in jp_lines if is_probably_generic_jp(line))
    jp_near_anchor = any(anchor_has_nearby_jp(int(anchor["line"]), useful_jp_lines) for anchor in anchors)
    generic_only = bool(jp_line_numbers) and generic_line_count == len(jp_line_numbers) and anchors

    status = "DONE"
    reasons: list[str] = []
    if not jp_line_numbers:
        status = "PARTIAL"
        reasons.append("missing JP comments")
    if anchors and not jp_near_anchor:
        status = "PARTIAL"
        reasons.append("JP comments are only top-level or away from anchors")
    if missing_details:
        status = "PARTIAL"
        reasons.append("anchor instances lack nearby JP or valid grouping")
    if generic_only and anchors:
        status = "PARTIAL"
        reasons.append("generic-only JP comments")
    if top_only and generic_only and anchors:
        status = "PARTIAL"
        reasons.append("generic top-level JP comments for files with anchors")
    if inaccurate_flags:
        status = "PARTIAL"
        reasons.append("misleading or inaccurate JP comments")

    return {
        "path": rel(path),
        "status": status,
        "jp_count": len(jp_line_numbers),
        "jp_lines": jp_line_numbers,
        "top_only": top_only,
        "generic_only": generic_only,
        "inaccurate_comment_flags": inaccurate_flags,
        "detected_anchors": [{"name": name, "count": categories[name]} for name in detected],
        "anchors_with_nearby_jp": sorted(categories_with_jp),
        "missing_anchors": missing_categories,
        "missing_anchor_examples": missing_details[:MISSING_EXAMPLE_LIMIT],
        "grouped_anchors": grouped_records[:MISSING_EXAMPLE_LIMIT],
        "anchor_count": len(anchors),
        "covered_anchor_count": len(covered_ids),
        "missing_anchor_count": len(missing_details),
        "grouped_anchor_count": len(grouped_coverage),
        "reasons": reasons,
    }


def analyze_sample_readme(sample_dir: Path) -> dict[str, object]:
    path = sample_dir / "README.ja.md"
    reasons: list[str] = []
    if not path.exists():
        return {
            "path": rel(path),
            "sample": rel(sample_dir),
            "status": "PARTIAL",
            "line_count": 0,
            "jp_blocks": 0,
            "memo_blocks": 0,
            "missing_sections": list(REQUIRED_SAMPLE_SECTIONS),
            "reasons": ["missing README.ja.md"],
        }

    text = path.read_text(encoding="utf-8", errors="ignore")
    missing_sections = [section for section in REQUIRED_SAMPLE_SECTIONS if section not in text]
    if missing_sections:
        reasons.append("missing required learning-guide sections")
    if "English anchor: this sample demonstrates" in text:
        reasons.append("old template marker remains")
    if "| API or concept |" not in text:
        reasons.append("missing API/concept table")
    if "> **日本語**" not in text or "> **学習メモ**" not in text:
        reasons.append("missing Japanese comparison blocks")
    if "Sample-Specific Notes" not in text and "Concrete Reading Path" not in text:
        reasons.append("missing sample-specific reading path")
    line_count = len(text.splitlines())
    if line_count < 110:
        reasons.append("guide is too short for required coverage")

    return {
        "path": rel(path),
        "sample": rel(sample_dir),
        "status": "DONE" if not reasons else "PARTIAL",
        "line_count": line_count,
        "jp_blocks": text.count("> **日本語**"),
        "memo_blocks": text.count("> **学習メモ**"),
        "missing_sections": missing_sections,
        "reasons": reasons,
    }


def analyze_theme_guide(path: Path) -> dict[str, object]:
    reasons: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    missing_sections = [section for section in REQUIRED_THEME_SECTIONS if section not in text]
    if missing_sections:
        reasons.append("missing required theme sections")
    if "| API or concept |" not in text:
        reasons.append("missing API/concept table")
    if "## Sample References" not in text or "../../cpp/" not in text and "../../python/" not in text:
        reasons.append("missing sample references")
    if text.count("> **日本語**") < 2 or text.count("> **学習メモ**") < 2:
        reasons.append("missing Japanese learning blocks")
    line_count = len(text.splitlines())
    if line_count < 85:
        reasons.append("theme guide is too short for required coverage")
    return {
        "path": rel(path),
        "status": "DONE" if not reasons else "PARTIAL",
        "line_count": line_count,
        "jp_blocks": text.count("> **日本語**"),
        "memo_blocks": text.count("> **学習メモ**"),
        "missing_sections": missing_sections,
        "reasons": reasons,
    }


def analyze_major_companion(path: Path) -> dict[str, object]:
    companion = path if path == DOCS_JA / "README.md" else companion_for(path)
    text = companion.read_text(encoding="utf-8", errors="ignore") if companion.exists() else ""
    paragraphs = [] if path == DOCS_JA / "README.md" else source_paragraphs(path)
    source_paragraph_count = len(paragraphs)
    english_blocks = (
        text.count("English paragraph")
        + text.count("English anchor:")
        + text.count("## English Reference")
        + text.count("> **English**")
    )
    jp_blocks = text.count("> **日本語**")
    memo_blocks = text.count("> **学習メモ**")
    reasons: list[str] = []
    if not companion.exists():
        reasons.append("missing companion")
    if "English anchor:" not in text and "## English Reference" not in text:
        reasons.append("missing English anchor/reference")
    if "> **日本語**" not in text or "> **学習メモ**" not in text:
        reasons.append("missing Japanese learning blocks")
    if text.count("## ") < 5:
        reasons.append("not section-level enough")
    if source_paragraph_count:
        if english_blocks < source_paragraph_count:
            reasons.append("not enough English paragraph/reference blocks for source paragraphs")
        if jp_blocks < source_paragraph_count:
            reasons.append("not enough Japanese blocks for source paragraphs")
    line_count = len(text.splitlines())
    if line_count < 45:
        reasons.append("too short for major companion")
    return {
        "source": rel(path),
        "path": rel(companion),
        "status": "DONE" if not reasons else "PARTIAL",
        "line_count": line_count,
        "source_paragraph_count": source_paragraph_count,
        "english_reference_blocks": english_blocks,
        "jp_blocks": jp_blocks,
        "memo_blocks": memo_blocks,
        "section_count": text.count("## "),
        "reasons": reasons,
    }


def analyze_glossary_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    reasons: list[str] = []
    required_sections = REQUIRED_GLOSSARY_SECTIONS.get(path.name, ())
    missing_sections = [section for section in required_sections if section not in text]
    line_count = len(text.splitlines())

    if not path.exists():
        reasons.append("missing glossary file")
    if missing_sections:
        reasons.append("missing required glossary sections")
    if line_count < GLOSSARY_MIN_LINES.get(path.name, 30):
        reasons.append("glossary entry is too short for quick-reference coverage")
    if not line_has_japanese(text):
        reasons.append("missing Japanese learning text")
    if path.name != "README.md":
        header_line = text.splitlines()[2] if line_count > 2 else ""
        if "English" not in header_line or "日本語" not in header_line:
            reasons.append("missing English/Japanese comparison table header")
        if "| ---" not in text and "--- | --- | ---" not in text:
            reasons.append("missing markdown table divider")
        if text.count("|") < 12:
            reasons.append("table has too few reference rows")

    return {
        "path": rel(path),
        "status": "DONE" if not reasons else "PARTIAL",
        "line_count": line_count,
        "section_count": text.count("## "),
        "table_pipes": text.count("|"),
        "missing_sections": missing_sections,
        "reasons": reasons,
    }


def theme_guides() -> list[Path]:
    theme_dir = DOCS_JA / "themes"
    if not theme_dir.exists():
        return []
    return sorted([p for p in theme_dir.glob("*.md") if p.name != "README.md"], key=rel)


def japanese_files() -> list[Path]:
    files = (
        [p for p in DOCS_JA.rglob("*") if p.is_file()]
        + [p for base in (ROOT / "cpp", ROOT / "python") if base.exists() for p in base.rglob("README.ja.md")]
        + [p for p in ROOT.rglob("*.ja.md") if p.is_file() and "docs_ja" not in p.parts and p.name != "README.ja.md"]
    )
    return sorted(set(files), key=rel)


def summarize() -> dict[str, object]:
    docs = source_docs()
    samples = sample_dirs()
    code = code_files()
    annotation_records = [analyze_code_file(path) for path in code]
    partial_annotations = [record for record in annotation_records if record["status"] != "DONE"]
    sample_readme_records = [analyze_sample_readme(path) for path in samples]
    partial_sample_readmes = [record for record in sample_readme_records if record["status"] != "DONE"]
    theme_records = [analyze_theme_guide(path) for path in theme_guides()]
    partial_theme_guides = [record for record in theme_records if record["status"] != "DONE"]
    companion_records = [analyze_major_companion(path) for path in MAJOR_COMPANION_TARGETS]
    partial_companions = [record for record in companion_records if record["status"] != "DONE"]
    glossary_records = [analyze_glossary_file(path) for path in GLOSSARY_TARGETS]
    partial_glossary = [record for record in glossary_records if record["status"] != "DONE"]
    missing_companions = [p for p in docs if not companion_for(p).exists()]
    status = "DONE"
    if (
        missing_companions
        or partial_companions
        or partial_sample_readmes
        or partial_annotations
        or partial_theme_guides
        or partial_glossary
    ):
        status = "PARTIAL"

    return {
        "status": status,
        "branch": git_value(["branch", "--show-current"]),
        "head": git_value(["rev-parse", "HEAD"]),
        "base": (ROOT / ".ja_translation_base").read_text(encoding="utf-8").strip()
        if (ROOT / ".ja_translation_base").exists()
        else "MISSING",
        "counts": {
            "source_docs": len(docs),
            "doc_companions_done": len(docs) - len(missing_companions),
            "major_companions_done": len(companion_records) - len(partial_companions),
            "major_companions_partial": len(partial_companions),
            "major_companion_source_paragraphs": sum(
                int(record.get("source_paragraph_count", 0)) for record in companion_records
            ),
            "major_companion_english_blocks": sum(
                int(record.get("english_reference_blocks", 0)) for record in companion_records
            ),
            "major_companion_jp_blocks": sum(int(record.get("jp_blocks", 0)) for record in companion_records),
            "sample_dirs": len(samples),
            "sample_readme_ja_done": len(samples) - len(partial_sample_readmes),
            "sample_readme_ja_partial": len(partial_sample_readmes),
            "theme_guides": len(theme_records),
            "theme_guides_done": len(theme_records) - len(partial_theme_guides),
            "theme_guides_partial": len(partial_theme_guides),
            "glossary_files": len(glossary_records),
            "glossary_files_done": len(glossary_records) - len(partial_glossary),
            "glossary_files_partial": len(partial_glossary),
            "annotation_files": len(code),
            "annotation_files_done": len(code) - len(partial_annotations),
            "annotation_files_partial": len(partial_annotations),
            "annotation_anchor_files": sum(1 for record in annotation_records if int(record["anchor_count"]) > 0),
            "annotation_anchor_instances": sum(int(record["anchor_count"]) for record in annotation_records),
            "annotation_anchor_instances_covered": sum(int(record["covered_anchor_count"]) for record in annotation_records),
            "annotation_anchor_instances_missing": sum(int(record["missing_anchor_count"]) for record in annotation_records),
            "annotation_grouped_anchor_instances": sum(int(record["grouped_anchor_count"]) for record in annotation_records),
            "annotation_inaccurate_comment_flags": sum(
                len(record.get("inaccurate_comment_flags", [])) for record in annotation_records
            ),
            "japanese_files": len(japanese_files()),
        },
        "missing": {
            "doc_companions": [rel(p) for p in missing_companions[:200]],
            "major_companions": [str(record["path"]) for record in partial_companions[:200]],
            "sample_readme_ja": [str(record["path"]) for record in partial_sample_readmes[:200]],
            "theme_guides": [str(record["path"]) for record in partial_theme_guides[:200]],
            "glossary": [str(record["path"]) for record in partial_glossary[:200]],
            "annotations": [str(record["path"]) for record in partial_annotations[:200]],
        },
        "overflow": {
            "doc_companions": max(0, len(missing_companions) - 200),
            "major_companions": max(0, len(partial_companions) - 200),
            "sample_readme_ja": max(0, len(partial_sample_readmes) - 200),
            "theme_guides": max(0, len(partial_theme_guides) - 200),
            "glossary": max(0, len(partial_glossary) - 200),
            "annotations": max(0, len(partial_annotations) - 200),
        },
        "annotation_records": annotation_records,
        "sample_readme_records": sample_readme_records,
        "theme_records": theme_records,
        "companion_records": companion_records,
        "glossary_records": glossary_records,
    }


def annotation_reason(record: dict[str, object]) -> str:
    reasons = record.get("reasons") or []
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(reason) for reason in reasons)
    missing = record.get("missing_anchors") or []
    if isinstance(missing, list) and missing:
        return "missing nearby JP for " + ", ".join(str(item) for item in missing[:5])
    return "needs review"


def render_markdown(summary: dict[str, object]) -> str:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    counts = summary["counts"]
    missing = summary["missing"]
    overflow = summary["overflow"]
    records = list(summary["annotation_records"])
    partial_records = [record for record in records if record["status"] != "DONE"]
    done_records = [record for record in records if record["status"] == "DONE"]
    sample_records = list(summary["sample_readme_records"])
    partial_sample_records = [record for record in sample_records if record["status"] != "DONE"]
    theme_records = list(summary["theme_records"])
    partial_theme_records = [record for record in theme_records if record["status"] != "DONE"]
    companion_records = list(summary["companion_records"])
    partial_companion_records = [record for record in companion_records if record["status"] != "DONE"]
    glossary_records = list(summary["glossary_records"])
    partial_glossary_records = [record for record in glossary_records if record["status"] != "DONE"]
    lines = [
        "# Japanese Translation Status",
        "",
        "English inventory for the local Japanese learning overlay.",
        "",
        "> **日本語**",
        "> CUDA Samples の日本語学習用 README、companion docs、`JP:` 注釈を、アンカー位置まで含めて棚卸しします。",
        ">",
        "> **学習メモ**",
        "> `DONE` は対象ファイルがそろい、重要な CUDA/API アンカーの近くに具体的な `JP:` 注釈がある状態です。`PARTIAL` は、トップだけの一般コメントやアンカー不足が残っている状態です。",
        "",
        f"- Status: {summary['status']}",
        f"- Updated: {now}",
        f"- Branch: `{summary['branch']}`",
        f"- Base commit: `{summary['base']}`",
        f"- Current commit: `{summary['head']}`",
        f"- Detailed annotation record: `{rel(ANNOTATION_JSON)}`",
        f"- Detailed sample README record: `{rel(SAMPLE_README_JSON)}`",
        f"- Detailed theme guide record: `{rel(THEME_GUIDE_JSON)}`",
        f"- Detailed companion record: `{rel(COMPANION_JSON)}`",
        f"- Detailed glossary record: `{rel(GLOSSARY_JSON)}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Missing Or Partial Items", ""])
    for key, values in missing.items():
        lines.append(f"### {key}")
        if values:
            for item in values:
                lines.append(f"- `{item}`")
            if overflow[key]:
                lines.append(f"- ... plus {overflow[key]} more")
        else:
            lines.append("- None")
        lines.append("")

    lines.extend(
        [
            "## Annotation Quality",
            "",
            f"- DONE files: {len(done_records)}",
            f"- PARTIAL files: {len(partial_records)}",
            f"- Anchor search window: {NEARBY_BEFORE} lines before to {NEARBY_AFTER} lines after each anchor.",
            f"- Grouping window: adjacent repeated anchors up to {GROUP_LINE_GAP} lines apart, with a useful JP comment explaining the grouped resource/direction/cleanup pattern.",
            f"- Covered anchor instances: {counts['annotation_anchor_instances_covered']}",
            f"- Missing anchor instances: {counts['annotation_anchor_instances_missing']}",
            f"- Grouped anchor instances: {counts['annotation_grouped_anchor_instances']}",
            f"- Inaccurate JP flags: {counts['annotation_inaccurate_comment_flags']}",
            "",
        ]
    )
    if partial_records:
        lines.extend(["### First PARTIAL Annotation Records", ""])
        for record in partial_records[:120]:
            missing_anchors = record.get("missing_anchors") or []
            missing_text = ", ".join(str(item) for item in missing_anchors[:8]) if missing_anchors else "none"
            inaccurate = record.get("inaccurate_comment_flags") or []
            lines.append(
                f"- `{record['path']}`: jp_count={record['jp_count']}, "
                f"anchors={record['anchor_count']}, covered={record.get('covered_anchor_count', 0)}, "
                f"missing_count={record.get('missing_anchor_count', 0)}, grouped={record.get('grouped_anchor_count', 0)}, "
                f"inaccurate_flags={len(inaccurate)}, missing=[{missing_text}], reason={annotation_reason(record)}"
            )
        if len(partial_records) > 120:
            lines.append(f"- ... plus {len(partial_records) - 120} more PARTIAL files")
        lines.append("")

    lines.extend(
        [
            "## Sample README Quality",
            "",
            f"- DONE guides: {len(sample_records) - len(partial_sample_records)}",
            f"- PARTIAL guides: {len(partial_sample_records)}",
            "- Required sections: " + ", ".join(f"`{section}`" for section in REQUIRED_SAMPLE_SECTIONS),
            "",
        ]
    )
    if partial_sample_records:
        lines.extend(["### First PARTIAL Sample README Records", ""])
        for record in partial_sample_records[:80]:
            reason_text = "; ".join(str(reason) for reason in record.get("reasons", [])) or "needs review"
            lines.append(
                f"- `{record['path']}`: lines={record['line_count']}, "
                f"jp_blocks={record['jp_blocks']}, memo_blocks={record['memo_blocks']}, reason={reason_text}"
            )
        if len(partial_sample_records) > 80:
            lines.append(f"- ... plus {len(partial_sample_records) - 80} more PARTIAL guides")
        lines.append("")

    lines.extend(
        [
            "## Theme Guide Quality",
            "",
            f"- DONE guides: {len(theme_records) - len(partial_theme_records)}",
            f"- PARTIAL guides: {len(partial_theme_records)}",
            "- Required sections: " + ", ".join(f"`{section}`" for section in REQUIRED_THEME_SECTIONS),
            "",
        ]
    )
    if partial_theme_records:
        lines.extend(["### First PARTIAL Theme Guide Records", ""])
        for record in partial_theme_records:
            reason_text = "; ".join(str(reason) for reason in record.get("reasons", [])) or "needs review"
            lines.append(
                f"- `{record['path']}`: lines={record['line_count']}, "
                f"jp_blocks={record['jp_blocks']}, memo_blocks={record['memo_blocks']}, reason={reason_text}"
            )
        lines.append("")

    lines.extend(
        [
            "## Major Companion Quality",
            "",
            f"- DONE companions: {len(companion_records) - len(partial_companion_records)}",
            f"- PARTIAL companions: {len(partial_companion_records)}",
            "",
        ]
    )
    if partial_companion_records:
        lines.extend(["### PARTIAL Major Companion Records", ""])
        for record in partial_companion_records:
            reason_text = "; ".join(str(reason) for reason in record.get("reasons", [])) or "needs review"
            lines.append(
                f"- `{record['path']}`: lines={record['line_count']}, "
                f"sections={record['section_count']}, source_paragraphs={record.get('source_paragraph_count', 0)}, "
                f"english_blocks={record.get('english_reference_blocks', 0)}, jp_blocks={record['jp_blocks']}, "
                f"reason={reason_text}"
            )
        lines.append("")

    lines.extend(
        [
            "## Glossary Quality",
            "",
            f"- DONE glossary files: {len(glossary_records) - len(partial_glossary_records)}",
            f"- PARTIAL glossary files: {len(partial_glossary_records)}",
            "- Required files: " + ", ".join(f"`{rel(path)}`" for path in GLOSSARY_TARGETS),
            "",
        ]
    )
    if partial_glossary_records:
        lines.extend(["### PARTIAL Glossary Records", ""])
        for record in partial_glossary_records:
            reason_text = "; ".join(str(reason) for reason in record.get("reasons", [])) or "needs review"
            lines.append(
                f"- `{record['path']}`: lines={record['line_count']}, "
                f"sections={record['section_count']}, table_pipes={record['table_pipes']}, reason={reason_text}"
            )
        lines.append("")

    lines.extend(
        [
            "## Policy Notes",
            "",
            "- English source text, file names, commands, APIs, expected output, license text, and attribution are preserved.",
            "- Japanese comments use `JP:` and are intended as learning annotations only.",
            "- Vendor/generated support paths are excluded from annotation coverage to avoid changing third-party or generated material.",
            "- Japanese license explanations, if present, are unofficial learning references only; original English license text remains authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="refresh docs_ja/_translation_status.md and annotation JSON")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args()

    summary = summarize()
    if args.write:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        ANNOTATION_JSON.write_text(
            json.dumps(summary["annotation_records"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        SAMPLE_README_JSON.write_text(
            json.dumps(summary["sample_readme_records"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        THEME_GUIDE_JSON.write_text(
            json.dumps(summary["theme_records"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        COMPANION_JSON.write_text(
            json.dumps(summary["companion_records"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        GLOSSARY_JSON.write_text(
            json.dumps(summary["glossary_records"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        STATUS.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(summary))
    return 0 if summary["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
