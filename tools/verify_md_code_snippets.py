#!/usr/bin/env python3
"""Verify Markdown Source: path:start-end code snippets."""

from __future__ import annotations

# JP: Markdown に埋め込んだ実コード抜粋を、現在の source line と照合して stale 化を検出します。
import datetime as _dt
import datetime as _dt
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from regenerate_sample_readmes_ja import rel, walkthrough_files


ROOT = Path(__file__).resolve().parents[1]
DOCS_JA = ROOT / "docs_ja"
REVIEW_DIR = DOCS_JA / "reviews"
MANIFEST_MD = REVIEW_DIR / "markdown_code_snippet_manifest.md"
MANIFEST_JSON = REVIEW_DIR / "markdown_code_snippet_manifest.json"
HEADER_RE = re.compile(r"^Source:\s+(.+):(\d+)-(\d+)\s*$")
FENCE_RE = re.compile(r"^```([A-Za-z0-9_+.-]*)\s*$")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
MAJOR_COMPANION_DOCS = (
    DOCS_JA / "README.md",
    DOCS_JA / "translated" / "README.md.ja.md",
    DOCS_JA / "translated" / "CMakeLists.txt.ja.md",
    DOCS_JA / "translated" / "CONTRIBUTING.md.ja.md",
    DOCS_JA / "translated" / "CHANGELOG.md.ja.md",
)


@dataclass(frozen=True)
class Snippet:
    doc: Path
    header_line: int
    source: Path
    source_text: str
    start: int
    end: int
    language: str
    code: list[str]
    explanation: str


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def markdown_files() -> list[Path]:
    files: set[Path] = set()
    for base in (ROOT / "cpp", ROOT / "python", DOCS_JA):
        if base.exists():
            files.update(p for p in base.rglob("*.md") if p.is_file())
    return sorted(files, key=rel)


def sample_readmes() -> list[Path]:
    result: list[Path] = []
    for base in (ROOT / "cpp", ROOT / "python"):
        if base.exists():
            result.extend(p for p in base.rglob("README.ja.md") if p.is_file())
    return sorted(result, key=rel)


def theme_guides() -> list[Path]:
    theme_dir = DOCS_JA / "themes"
    if not theme_dir.exists():
        return []
    return sorted((p for p in theme_dir.glob("*.md") if p.name != "README.md"), key=rel)


def parse_snippets(path: Path) -> tuple[list[Snippet], list[dict[str, object]]]:
    # JP: header の直後に fenced code block がある、という単純な形式だけを正とします。
    lines = read_lines(path)
    snippets: list[Snippet] = []
    malformed: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("Source:"):
            index += 1
            continue
        match = HEADER_RE.match(line)
        if not match:
            malformed.append({"doc": rel(path), "line": index + 1, "reason": "invalid Source header"})
            index += 1
            continue
        source_text, start_text, end_text = match.groups()
        start = int(start_text)
        end = int(end_text)
        fence_index = index + 1
        while fence_index < len(lines) and not lines[fence_index].strip():
            fence_index += 1
        if fence_index >= len(lines) or not FENCE_RE.match(lines[fence_index]):
            malformed.append({"doc": rel(path), "line": index + 1, "reason": "missing opening fence"})
            index += 1
            continue
        language = FENCE_RE.match(lines[fence_index]).group(1)  # type: ignore[union-attr]
        code_start = fence_index + 1
        code_end = code_start
        while code_end < len(lines) and not lines[code_end].startswith("```"):
            code_end += 1
        if code_end >= len(lines):
            malformed.append({"doc": rel(path), "line": index + 1, "reason": "missing closing fence"})
            index += 1
            continue
        explanation_index = code_end + 1
        while explanation_index < len(lines) and not lines[explanation_index].strip():
            explanation_index += 1
        explanation = lines[explanation_index].strip() if explanation_index < len(lines) else ""
        source = (ROOT / source_text.replace("\\", "/")).resolve()
        snippets.append(
            Snippet(
                doc=path,
                header_line=index + 1,
                source=source,
                source_text=source_text.replace("\\", "/"),
                start=start,
                end=end,
                language=language,
                code=lines[code_start:code_end],
                explanation=explanation,
            )
        )
        index = code_end + 1
    return snippets, malformed


def validate_snippet(snippet: Snippet) -> list[str]:
    # JP: ここが Markdown snippet と実 source line を照合する validation 境界です。
    reasons: list[str] = []
    if not snippet.source.exists():
        reasons.append("source path does not exist")
        return reasons
    try:
        source_lines = read_lines(snippet.source)
    except OSError as exc:
        reasons.append(f"cannot read source path: {exc}")
        return reasons
    if snippet.start < 1 or snippet.end < snippet.start or snippet.end > len(source_lines):
        reasons.append("line range is outside source file")
        return reasons
    expected = source_lines[snippet.start - 1 : snippet.end]
    if expected != snippet.code:
        reasons.append("fenced code does not match source lines")
    if "JP:" not in snippet.explanation and not JAPANESE_RE.search(snippet.explanation):
        reasons.append("snippet is not followed by a Japanese explanation")
    return reasons


def snippet_sources_by_doc(snippets: list[Snippet]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for snippet in snippets:
        result.setdefault(rel(snippet.doc), set()).add(snippet.source_text)
    return result


def required_sample_coverage(snippets: list[Snippet]) -> list[dict[str, object]]:
    source_map = snippet_sources_by_doc(snippets)
    missing: list[dict[str, object]] = []
    for readme in sample_readmes():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        doc_key = rel(readme)
        if "## Code Walkthrough" not in text:
            missing.append({"doc": doc_key, "reason": "missing ## Code Walkthrough section"})
        files = [p for p in walkthrough_files(readme.parent) if p.exists() and read_lines(p)]
        if not files:
            continue
        present = source_map.get(doc_key, set())
        for source in files:
            source_key = rel(source)
            if source_key not in present:
                missing.append({"doc": doc_key, "source": source_key, "reason": "missing required source snippet"})
    return missing


def required_theme_coverage(snippets: list[Snippet]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for snippet in snippets:
        counts[rel(snippet.doc)] = counts.get(rel(snippet.doc), 0) + 1
    missing: list[dict[str, object]] = []
    for guide in theme_guides():
        text = guide.read_text(encoding="utf-8", errors="ignore")
        doc_key = rel(guide)
        if "## Representative Code" not in text:
            missing.append({"doc": doc_key, "reason": "missing ## Representative Code section"})
        if counts.get(doc_key, 0) < 3:
            missing.append({"doc": doc_key, "reason": "fewer than 3 representative snippets"})
    return missing


def required_major_doc_coverage(snippets: list[Snippet]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for snippet in snippets:
        counts[rel(snippet.doc)] = counts.get(rel(snippet.doc), 0) + 1
    missing: list[dict[str, object]] = []
    for doc in MAJOR_COMPANION_DOCS:
        if not doc.exists():
            missing.append({"doc": rel(doc), "reason": "major companion doc does not exist"})
            continue
        if counts.get(rel(doc), 0) < 1:
            missing.append({"doc": rel(doc), "reason": "missing command/CMake/source snippet"})
    return missing


def summarize() -> dict[str, object]:
    all_snippets: list[Snippet] = []
    malformed: list[dict[str, object]] = []
    for path in markdown_files():
        snippets, bad = parse_snippets(path)
        all_snippets.extend(snippets)
        malformed.extend(bad)

    stale: list[dict[str, object]] = []
    for snippet in all_snippets:
        # JP: 各 snippet の validation 結果を manifest の stale record に変換します。
        reasons = validate_snippet(snippet)
        if reasons:
            stale.append(
                {
                    "doc": rel(snippet.doc),
                    "line": snippet.header_line,
                    "source": snippet.source_text,
                    "range": f"{snippet.start}-{snippet.end}",
                    "reasons": reasons,
                }
            )

    missing = (
        required_sample_coverage(all_snippets)
        + required_theme_coverage(all_snippets)
        + required_major_doc_coverage(all_snippets)
    )
    status = "DONE" if not stale and not missing and not malformed else "PARTIAL"
    docs_with_snippets = {rel(snippet.doc) for snippet in all_snippets}
    return {
        "schema_version": 1,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "branch": git_value(["branch", "--show-current"]),
        "head": git_value(["rev-parse", "HEAD"]),
        "counts": {
            "markdown_files_scanned": len(markdown_files()),
            "markdown_files_with_snippets": len(docs_with_snippets),
            "total_snippets": len(all_snippets),
            "malformed_snippets": len(malformed),
            "stale_snippets": len(stale),
            "missing_required_snippets": len(missing),
            "sample_readmes": len(sample_readmes()),
            "theme_guides": len(theme_guides()),
            "major_companion_docs": len(MAJOR_COMPANION_DOCS),
        },
        "malformed_snippets": malformed,
        "stale_snippets": stale,
        "missing_required_snippets": missing,
    }


def render_markdown(summary: dict[str, object]) -> str:
    counts = summary["counts"]
    lines = [
        "# Markdown Code Snippet Manifest",
        "",
        "English manifest for verified Markdown code snippets embedded in Japanese learning documents.",
        "",
        "> JP: `Source: path:start-end` の直後に置いた fenced code block を、現在の source line と照合します。Markdown だけを読んでも実コードの流れを学べる状態を確認します。",
        "",
        f"- Status: {summary['status']}",
        f"- Updated: {summary['generated_at_utc']}",
        f"- Branch: `{summary['branch']}`",
        f"- Source state checked: `{summary['head']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Problems", ""])
    for key in ("malformed_snippets", "stale_snippets", "missing_required_snippets"):
        values = summary[key]
        lines.append(f"### {key}")
        if values:
            for item in values[:200]:  # type: ignore[index]
                lines.append(f"- `{json.dumps(item, ensure_ascii=False)}`")
            if len(values) > 200:  # type: ignore[arg-type]
                lines.append(f"- ... plus {len(values) - 200} more")  # type: ignore[arg-type]
        else:
            lines.append("- None")
        lines.append("")

    lines.extend(
        [
            "## Policy Notes",
            "",
            "- Snippets are copied from live source/build/script files and checked byte-for-line after newline normalization.",
            "- License headers are intentionally avoided by the generator unless the selected source file has no other meaningful lines.",
            "- Each snippet must be followed by Japanese explanatory text.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    summary = summarize()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    MANIFEST_MD.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    print(render_markdown(summary))
    return 0 if summary["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
