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
    "Common/GL",
    "cpp/2_Concepts_and_Techniques/interval/boost",
    "cpp/5_Domain_Specific/simpleD3D11Texture/d3dx11effect",
}
GENERATED_NAME_PARTS = {
    "ptxdump",
}

NEARBY_BEFORE = 4
NEARBY_AFTER = 2
TOP_LINE_LIMIT = 45


@dataclass(frozen=True)
class AnchorRule:
    name: str
    pattern: re.Pattern[str]
    label: str


ANCHOR_RULES: tuple[AnchorRule, ...] = (
    AnchorRule(
        "device_memory",
        re.compile(r"\b(cudaMalloc(?:Pitch|3D|Array)?|cudaFree(?:Array)?|cuMemAlloc\w*|cuMemFree\w*|mem_alloc|DeviceMemory|device_buffer)\b"),
        "device allocation/free",
    ),
    AnchorRule(
        "pinned_memory",
        re.compile(r"\b(cudaMallocHost|cudaFreeHost|cudaHostAlloc|cudaHostRegister|cudaHostUnregister|pinned|pagelocked)\b", re.I),
        "pinned host memory",
    ),
    AnchorRule(
        "managed_memory",
        re.compile(r"\b(cudaMallocManaged|cudaMemPrefetchAsync|cudaMemAdvise|managed memory|UnifiedMemory|Unified Memory)\b"),
        "unified memory",
    ),
    AnchorRule(
        "transfer",
        re.compile(r"\b(cudaMemcpy(?:2D|3D)?(?:Async)?|cuMemcpy\w*|cudaMemset(?:Async)?|copy_to_host|copy_to_device|copy_from|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|HostToDevice|DeviceToHost)\b"),
        "host/device transfer",
    ),
    AnchorRule(
        "kernel_launch",
        re.compile(r"(<<<[^>]*>>>|\.launch\s*\(|launch\s*\(|cuLaunchKernel|LaunchKernel)"),
        "kernel launch",
    ),
    AnchorRule(
        "indexing",
        re.compile(r"\b(blockIdx|threadIdx|blockDim|gridDim|laneId|warpSize|cg::this_thread_block|this_grid)\b"),
        "thread/block indexing",
    ),
    AnchorRule(
        "shared_memory",
        re.compile(r"\b(__shared__|extern\s+__shared__|sharedMem|shared memory|SharedMemory|cuda.shared)\b"),
        "shared memory",
    ),
    AnchorRule(
        "sync",
        re.compile(r"\b(__syncthreads|__syncwarp|cudaDeviceSynchronize|cudaStreamSynchronize|cudaEventSynchronize|cudaThreadSynchronize|cooperative_groups::sync|cg::sync|synchronize\s*\(|Synchronize)\b"),
        "synchronization boundary",
    ),
    AnchorRule(
        "streams_events",
        re.compile(r"\b(cudaStream\w*|cudaEvent\w*|CUstream|CUevent|Stream\(|Event\(|stream\s*=|streaming|event)\b", re.I),
        "streams/events",
    ),
    AnchorRule(
        "graphs",
        re.compile(r"\b(cudaGraph\w*|cudaUserObject\w*|graphExec|Graph\(|graph\s*=)\b"),
        "CUDA Graph dependency",
    ),
    AnchorRule(
        "driver_api",
        re.compile(r"\b(cu[A-Z][A-Za-z0-9_]*|CUdevice|CUcontext|CUmodule|CUfunction|CUresult)\b"),
        "Driver API boundary",
    ),
    AnchorRule(
        "nvrtc",
        re.compile(r"\b(nvrtc\w*|NVRTC|nvJitLink\w*|jitify|JIT compilation|compile\s*\(|Program\(|Linker\()\b"),
        "runtime compilation/JIT",
    ),
    AnchorRule(
        "library_resources",
        re.compile(r"\b(cublas\w*|cufft\w*|cusparse\w*|cusolver\w*|curand\w*|npp\w*|nvjpeg\w*|nccl\w*|cudnn\w*|Create\w*Handle|Destroy\w*Handle|descriptor|workspace)\b", re.I),
        "CUDA library resources",
    ),
    AnchorRule(
        "validation",
        re.compile(r"\b(assert|sdkCompare\w*|compare\w*|checkResult|validate\w*|np\.allclose|cupy\.allclose|Result\s*=\s*PASS|PASS|FAIL)\b"),
        "validation/checking",
    ),
    AnchorRule(
        "cleanup",
        re.compile(r"\b(cudaFree\w*|cudaDestroy\w*|cudaEventDestroy|cudaStreamDestroy|Destroy\w*|destroy\s*\(|free\s*\(|delete\s+|cuMemFree\w*|release\s*\(|close\s*\()\b"),
        "cleanup/lifetime end",
    ),
    AnchorRule(
        "build_cuda",
        re.compile(r"\b(CUDA::|CUDAToolkit|cuda_add|target_link_libraries|add_executable|set_target_properties|CMAKE_CUDA|CUDA_ARCHITECTURES|find_package)\b"),
        "CMake CUDA build wiring",
    ),
    AnchorRule(
        "python_cuda",
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


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def is_comment_only(line: str) -> bool:
    return bool(COMMENT_ONLY_RE.match(line))


def is_probably_generic_jp(line: str) -> bool:
    text = line.lower()
    generic_terms = (
        "この file では",
        "この build file",
        "英語の識別子",
        "確認します",
        "保持します",
    )
    concrete_terms = (
        "所有",
        "解放",
        "転送",
        "方向",
        "host",
        "device",
        "同期境界",
        "依存関係",
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


def detect_anchor_lines(lines: list[str]) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if "JP:" in line:
            continue
        if is_comment_only(line):
            continue
        categories: list[str] = []
        labels: list[str] = []
        for rule in ANCHOR_RULES:
            if rule.pattern.search(line):
                categories.append(rule.name)
                labels.append(rule.label)
        if categories:
            anchors.append(
                {
                    "line": number,
                    "categories": sorted(set(categories)),
                    "labels": sorted(set(labels)),
                    "text": line.strip()[:180],
                }
            )
    return anchors


def anchor_has_nearby_jp(anchor_line: int, jp_lines: Iterable[int]) -> bool:
    return any(anchor_line - NEARBY_BEFORE <= jp <= anchor_line + NEARBY_AFTER for jp in jp_lines)


def analyze_code_file(path: Path) -> dict[str, object]:
    lines = read_lines(path)
    jp_line_numbers = [i for i, line in enumerate(lines, start=1) if "JP:" in line]
    jp_lines = [lines[i - 1].strip() for i in jp_line_numbers]
    anchors = detect_anchor_lines(lines)
    categories: dict[str, int] = {}
    categories_with_jp: set[str] = set()
    missing_details: list[dict[str, object]] = []

    for anchor in anchors:
        line_number = int(anchor["line"])
        nearby = anchor_has_nearby_jp(line_number, jp_line_numbers)
        for category in anchor["categories"]:
            categories[category] = categories.get(category, 0) + 1
            if nearby:
                categories_with_jp.add(str(category))
        if not nearby:
            missing_details.append(anchor)

    detected = sorted(categories)
    missing_categories = [name for name in detected if name not in categories_with_jp]
    top_only = bool(jp_line_numbers) and all(line <= TOP_LINE_LIMIT for line in jp_line_numbers)
    generic_line_count = sum(1 for line in jp_lines if is_probably_generic_jp(line))
    jp_near_anchor = any(anchor_has_nearby_jp(int(anchor["line"]), jp_line_numbers) for anchor in anchors)
    generic_only = bool(jp_line_numbers) and generic_line_count == len(jp_line_numbers) and (not anchors or not jp_near_anchor)

    status = "DONE"
    reasons: list[str] = []
    if not jp_line_numbers:
        status = "PARTIAL"
        reasons.append("missing JP comments")
    if anchors and not jp_near_anchor:
        status = "PARTIAL"
        reasons.append("JP comments are only top-level or away from anchors")
    if missing_categories:
        status = "PARTIAL"
        reasons.append("important anchor categories lack nearby JP")
    if generic_only and anchors:
        status = "PARTIAL"
        reasons.append("generic-only JP comments")

    return {
        "path": rel(path),
        "status": status,
        "jp_count": len(jp_line_numbers),
        "jp_lines": jp_line_numbers,
        "top_only": top_only,
        "generic_only": generic_only,
        "detected_anchors": [{"name": name, "count": categories[name]} for name in detected],
        "anchors_with_nearby_jp": sorted(categories_with_jp),
        "missing_anchors": missing_categories,
        "missing_anchor_examples": missing_details[:12],
        "anchor_count": len(anchors),
        "reasons": reasons,
    }


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
    missing_companions = [p for p in docs if not companion_for(p).exists()]
    missing_sample_readmes = [p for p in samples if not (p / "README.ja.md").exists()]
    status = "DONE"
    if missing_companions or missing_sample_readmes or partial_annotations:
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
            "sample_dirs": len(samples),
            "sample_readme_ja_done": len(samples) - len(missing_sample_readmes),
            "annotation_files": len(code),
            "annotation_files_done": len(code) - len(partial_annotations),
            "annotation_files_partial": len(partial_annotations),
            "annotation_anchor_files": sum(1 for record in annotation_records if int(record["anchor_count"]) > 0),
            "annotation_anchor_instances": sum(int(record["anchor_count"]) for record in annotation_records),
            "japanese_files": len(japanese_files()),
        },
        "missing": {
            "doc_companions": [rel(p) for p in missing_companions[:200]],
            "sample_readme_ja": [rel(p) for p in missing_sample_readmes[:200]],
            "annotations": [str(record["path"]) for record in partial_annotations[:200]],
        },
        "overflow": {
            "doc_companions": max(0, len(missing_companions) - 200),
            "sample_readme_ja": max(0, len(missing_sample_readmes) - 200),
            "annotations": max(0, len(partial_annotations) - 200),
        },
        "annotation_records": annotation_records,
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
            "",
        ]
    )
    if partial_records:
        lines.extend(["### First PARTIAL Annotation Records", ""])
        for record in partial_records[:120]:
            missing_anchors = record.get("missing_anchors") or []
            missing_text = ", ".join(str(item) for item in missing_anchors[:8]) if missing_anchors else "none"
            lines.append(
                f"- `{record['path']}`: jp_count={record['jp_count']}, "
                f"anchors={record['anchor_count']}, missing=[{missing_text}], reason={annotation_reason(record)}"
            )
        if len(partial_records) > 120:
            lines.append(f"- ... plus {len(partial_records) - 120} more PARTIAL files")
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
        STATUS.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(summary))
    return 0 if summary["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
