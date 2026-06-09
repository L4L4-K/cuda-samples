# CUDA Samples Japanese Learning Notes

> **日本語**
> このディレクトリは、CUDA Samples を日本語で学ぶための補助資料です。英語の原文、ファイル名、API 名、コマンド、出力文字列はそのまま残し、横に日本語の説明と学習メモを追加します。
>
> **学習メモ**
> CUDA のサンプルは「何を計算するか」だけでなく、「CPU と GPU のどちらがメモリを所有するか」「いつ転送するか」「どの同期が必要か」「性能上どこが効くか」を読むと理解しやすくなります。

## English Orientation

This fork adds Japanese companion material for local study while preserving the original CUDA sample behavior. The original English files remain the source of truth for commands, identifiers, APIs, expected output, license text, and attribution.

> **日本語**
> この fork では、実行結果やビルド設定を変えずに、日本語の伴走資料を追加します。英語の README、コード、API 名、ターゲット名、出力文字列は比較しやすいように維持します。
>
> **学習メモ**
> 翻訳は「置き換え」ではなく「並走」です。CUDA API やサンプル名は検索しやすさのため英語のまま読み、周辺に日本語で意図を補います。

## Map

- `themes/`: concept guides for CUDA execution, memory, streams, graphs, libraries, multi-GPU work, performance, testing, and Python CUDA.
- `glossary/`: terms, API, memory-transfer, and build/run cheat sheets.
- `translated/`: Japanese companion pages for repository documents, including binary document references.
- `../cpp/**/README.ja.md` and `../python/**/README.ja.md`: per-sample study guides.
- `_translation_status.md`: generated inventory and completion status.
- `reviews/final_report.md`: final audit record for the local translation work.

> **日本語**
> まず `themes/` で全体像をつかみ、個別サンプルでは `README.ja.md` を読み、最後に元の README とコードを横断すると効率よく学習できます。
>
> **学習メモ**
> 同じ API でも、単純な `vectorAdd` とライブラリサンプル、マルチ GPU サンプルでは「同期」「所有権」「性能のボトルネック」が変わります。

## Verification

Run the inventory script from the repository root:

```bash
python tools/inventory_ja.py --write
```

> **日本語**
> このコマンドは、サンプルごとの日本語 README、翻訳 companion、`JP:` コメントの有無を数え、`docs_ja/_translation_status.md` を更新します。
>
> **学習メモ**
> 翻訳作業では「見たつもり」を避けるため、対象ファイル数と未対応ファイル数を毎回機械的に確認します。
