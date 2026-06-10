# Markdown Code Snippet Manifest

English manifest for verified Markdown code snippets embedded in Japanese learning documents.

> JP: `Source: path:start-end` の直後に置いた fenced code block を、現在の source line と照合します。Markdown だけを読んでも実コードの流れを学べる状態を確認します。

- Status: DONE
- Updated: 2026-06-10T08:13:45Z
- Branch: `ja-study/local-annotations`
- Source state checked: `b7db33299d3303855027b7c7098bc3a3796c02a9`

## Counts

- markdown_files_scanned: 1094
- markdown_files_with_snippets: 276
- total_snippets: 1776
- malformed_snippets: 0
- stale_snippets: 0
- missing_required_snippets: 0
- sample_readmes: 255
- theme_guides: 16
- major_companion_docs: 5

## Problems

### malformed_snippets
- None

### stale_snippets
- None

### missing_required_snippets
- None

## Policy Notes

- Snippets are copied from live source/build/script files and checked byte-for-line after newline normalization.
- License headers are intentionally avoided by the generator unless the selected source file has no other meaningful lines.
- Each snippet must be followed by Japanese explanatory text.
