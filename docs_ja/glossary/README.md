# Glossary And Cheatsheets

English anchor: quick references for recurring CUDA terms, APIs, memory transfers, and build/run workflows.

> **日本語**
> CUDA Samples を読みながら参照する用語集と cheat sheet です。英語の API 名、file 名、command はそのまま残し、日本語で意味、注意点、読み方を補います。
>
> **学習メモ**
> glossary は翻訳辞書ではなく、source を読むための index です。分からない語を引いたら、関連 theme と sample README に戻って確認します。

## Files

- [Terms](terms.md): execution model、memory、sync、performance の基本語。
- [API](api.md): Runtime、Driver、NVRTC、library、Python CUDA の代表 API。
- [Memory Transfer](memory_transfer.md): H2D、D2H、D2D、pinned、mapped、managed、peer の読み方。
- [Build And Run](build_run.md): CMake、Python requirements、test runner、debug/profiling の入口。

## How To Use

1. Keep the English source open and search for the exact API or term.
2. Use this glossary to identify ownership, ordering, visibility, validation, and cleanup concerns.
3. Open the matching theme guide for the longer mental model.
4. Return to the sample `README.ja.md` and source `JP:` comments for local context.

> **日本語**
> 同じ語でも sample によって意味の重心が変わります。`stream` は overlap sample では性能、graph sample では capture、library sample では library work の投入先として読みます。
>
> **学習メモ**
> 迷ったら `terms.md` -> theme guide -> sample guide -> source の順に戻ると、抽象語と実装行がつながります。
