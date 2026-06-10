# Japanese Translation Status

English inventory for the local Japanese learning overlay.

> **日本語**
> CUDA Samples の日本語学習用 README、companion docs、`JP:` 注釈を、アンカー位置まで含めて棚卸しします。
>
> **学習メモ**
> `DONE` は対象ファイルがそろい、重要な CUDA/API アンカーの近くに具体的な `JP:` 注釈がある状態です。`PARTIAL` は、トップだけの一般コメントやアンカー不足が残っている状態です。

- Status: DONE
- Updated: 2026-06-10 07:18:15 UTC
- Branch: `ja-study/local-annotations`
- Base commit: `b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- Current commit: `c69bb98fa5b9bc959c7f7b0b77bc70a6a7540aa7`
- Detailed annotation record: `docs_ja/_annotation_inventory.json`
- Detailed sample README record: `docs_ja/_sample_readme_inventory.json`
- Detailed theme guide record: `docs_ja/_theme_inventory.json`
- Detailed companion record: `docs_ja/_companion_inventory.json`
- Detailed glossary record: `docs_ja/_glossary_inventory.json`

## Counts

- source_docs: 557
- doc_companions_done: 557
- major_companions_done: 5
- major_companions_partial: 0
- major_companion_source_paragraphs: 240
- major_companion_english_blocks: 251
- major_companion_jp_blocks: 254
- sample_dirs: 255
- sample_readme_ja_done: 255
- sample_readme_ja_partial: 0
- theme_guides: 16
- theme_guides_done: 16
- theme_guides_partial: 0
- glossary_files: 5
- glossary_files_done: 5
- glossary_files_partial: 0
- annotation_files: 922
- annotation_files_done: 922
- annotation_files_partial: 0
- annotation_anchor_files: 644
- annotation_anchor_instances: 9815
- annotation_anchor_instances_covered: 9815
- annotation_anchor_instances_missing: 0
- annotation_grouped_anchor_instances: 2613
- annotation_inaccurate_comment_flags: 0
- japanese_files: 844

## Missing Or Partial Items

### doc_companions
- None

### major_companions
- None

### sample_readme_ja
- None

### theme_guides
- None

### glossary
- None

### annotations
- None

## Annotation Quality

- DONE files: 922
- PARTIAL files: 0
- Anchor search window: 4 lines before to 2 lines after each anchor.
- Grouping window: adjacent repeated anchors up to 8 lines apart, with a useful JP comment explaining the grouped resource/direction/cleanup pattern.
- Covered anchor instances: 9815
- Missing anchor instances: 0
- Grouped anchor instances: 2613
- Inaccurate JP flags: 0

## Sample README Quality

- DONE guides: 255
- PARTIAL guides: 0
- Required sections: `## Purpose`, `## Prerequisites`, `## Files`, `## Execution Flow`, `## Concrete Reading Path`, `## Key APIs And Concepts`, `## Memory, Synchronization, And Performance Notes`, `## Build And Run`, `## Expected Behavior`, `## Common Mistakes`, `## Exercises`, `## Related Themes`

## Theme Guide Quality

- DONE guides: 16
- PARTIAL guides: 0
- Required sections: `## Concept`, `## Why It Matters`, `## Mental Model`, `## API Map`, `## Sample References`, `## Reading Steps`, `## Common Mistakes`, `## Performance Notes`, `## Exercises`, `## Cross-Theme Links`, `## Review Checklist`

## Major Companion Quality

- DONE companions: 5
- PARTIAL companions: 0

## Glossary Quality

- DONE glossary files: 5
- PARTIAL glossary files: 0
- Required files: `docs_ja/glossary/README.md`, `docs_ja/glossary/terms.md`, `docs_ja/glossary/api.md`, `docs_ja/glossary/memory_transfer.md`, `docs_ja/glossary/build_run.md`

## Policy Notes

- English source text, file names, commands, APIs, expected output, license text, and attribution are preserved.
- Japanese comments use `JP:` and are intended as learning annotations only.
- Vendor/generated support paths are excluded from annotation coverage to avoid changing third-party or generated material.
- Japanese license explanations, if present, are unofficial learning references only; original English license text remains authoritative.
