#!/usr/bin/env python3
"""Inventory Japanese translation and annotation coverage for this fork."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_JA = ROOT / "docs_ja"
TRANSLATED = DOCS_JA / "translated"
STATUS = DOCS_JA / "_translation_status.md"

DOC_EXTS = {".md", ".txt", ".pdf", ".doc", ".docx"}
CODE_EXTS = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".py", ".cmake"}
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


def contains_jp(path: Path) -> bool:
    try:
        return "JP:" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def summarize() -> dict[str, object]:
    docs = source_docs()
    samples = sample_dirs()
    code = code_files()
    missing_companions = [p for p in docs if not companion_for(p).exists()]
    missing_sample_readmes = [p for p in samples if not (p / "README.ja.md").exists()]
    missing_annotations = [p for p in code if not contains_jp(p)]
    japanese_files = sorted(
        [p for p in DOCS_JA.rglob("*") if p.is_file()]
        + [p for base in (ROOT / "cpp", ROOT / "python") if base.exists() for p in base.rglob("README.ja.md")]
        + [p for p in ROOT.rglob("*.ja.md") if p.is_file() and "docs_ja" not in p.parts and p.name != "README.ja.md"],
        key=rel,
    )
    status = "DONE"
    if missing_companions or missing_sample_readmes or missing_annotations:
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
            "annotation_files_done": len(code) - len(missing_annotations),
            "japanese_files": len(japanese_files),
        },
        "missing": {
            "doc_companions": [rel(p) for p in missing_companions[:200]],
            "sample_readme_ja": [rel(p) for p in missing_sample_readmes[:200]],
            "annotations": [rel(p) for p in missing_annotations[:200]],
        },
        "overflow": {
            "doc_companions": max(0, len(missing_companions) - 200),
            "sample_readme_ja": max(0, len(missing_sample_readmes) - 200),
            "annotations": max(0, len(missing_annotations) - 200),
        },
    }


def render_markdown(summary: dict[str, object]) -> str:
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    counts = summary["counts"]
    missing = summary["missing"]
    overflow = summary["overflow"]
    lines = [
        "# Japanese Translation Status",
        "",
        "English inventory for the local Japanese learning overlay.",
        "",
        "> **日本語**",
        "> CUDA Samples の日本語学習用 README、companion docs、`JP:` 注釈の棚卸しです。",
        ">",
        "> **学習メモ**",
        "> `DONE` は対象ファイルがそろい、`PARTIAL` は未対応が残っている状態です。完了時に `PARTIAL` を残さないよう確認します。",
        "",
        f"- Status: {summary['status']}",
        f"- Updated: {now}",
        f"- Branch: `{summary['branch']}`",
        f"- Base commit: `{summary['base']}`",
        f"- Current commit: `{summary['head']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Missing Items", ""])
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
            "## Policy Notes",
            "",
            "- English source text, file names, commands, APIs, expected output, license text, and attribution are preserved.",
            "- Japanese comments use `JP:` and are intended as learning annotations only.",
            "- Vendor/generated support paths are excluded from annotation coverage to avoid changing third-party or generated material.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="refresh docs_ja/_translation_status.md")
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = parser.parse_args()

    summary = summarize()
    if args.write:
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(summary))
    return 0 if summary["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
