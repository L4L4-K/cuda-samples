# CUDA Memory

English anchor: CUDA samples move data among host memory, device memory, pinned memory, managed memory, arrays, mapped memory, and external resources.

> **日本語**
> CUDA memory を読む目的は、pointer がどこを指し、誰が所有し、どの processor からいつ見えるかを明確にすることです。`malloc`、`cudaMalloc`、`cudaMallocHost`、`cudaMallocManaged`、external memory は似た pointer に見えても lifetime と access rule が違います。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

CUDA memory を読む目的は、pointer がどこを指し、誰が所有し、どの processor からいつ見えるかを明確にすることです。`malloc`、`cudaMalloc`、`cudaMallocHost`、`cudaMallocManaged`、external memory は似た pointer に見えても lifetime と access rule が違います。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Memory は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- Host memory は CPU から自然に読めますが、GPU から直接読めるとは限りません。
- Device memory は GPU work の主な storage で、host は CUDA API を通して扱います。
- Pinned memory は DMA と async copy の前提になり、managed memory は CPU/GPU が同じ pointer を共有します。

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `cudaMalloc / cudaFree` | device memory の確保と解放です。 | byte size と cleanup path を対応させます。 |
| `cudaMallocHost / cudaFreeHost` | page-locked host memory を確保します。 | async copy と overlap の前提を確認します。 |
| `cudaMemcpy / cudaMemcpyAsync` | host/device 間の visibility を作る転送です。 | direction enum と stream dependency を確認します。 |
| `cudaMallocManaged / cudaMemPrefetchAsync` | Unified Memory allocation と migration hint です。 | 同期と access order を読みます。 |
| `cudaExternalMemory / cudaGraphics*` | 外部 API が所有する resource を CUDA に見せます。 | map/unmap と owner lifetime を区別します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [vectorAdd](../../cpp/0_Introduction/vectorAdd/README.ja.md): H2D、kernel、D2H、free の基本形です。
- [UnifiedMemoryStreams](../../cpp/0_Introduction/UnifiedMemoryStreams/README.ja.md): Unified Memory と stream を読みます。
- [simpleZeroCopy](../../cpp/0_Introduction/simpleZeroCopy/README.ja.md): mapped host memory を読みます。
- [UnifiedMemoryPerf](../../cpp/6_Performance/UnifiedMemoryPerf/README.ja.md): managed memory performance を読みます。
- [memoryResources](../../python/2_CoreConcepts/memoryResources/README.ja.md): Python Buffer lifetime を読みます。

## Reading Steps

1. allocation API と解放 API を対応表にします。
2. copy/mapping/migration が input visibility を作る場所を探します。
3. kernel/library call が読む pointer と書く pointer を分けます。
4. validation 前に結果が host から見えるか確認します。
5. cleanup 前に非同期 work が完了しているか確認します。

## Common Mistakes

- element count を byte count として API に渡す。
- D2H と H2D の direction enum を取り違える。
- managed memory なら同期や prefetch が不要だと思う。
- external resource の owner を CUDA 側だと誤解する。

## Performance Notes

- pageable host memory では async copy の overlap が制限されることがあります。
- coalesced global memory access は thread index と data layout の両方で決まります。
- Unified Memory の page fault は timing に混ざるため warmup と prefetch を確認します。

## Exercises

- vectorAdd の host/device pointer を表にする。
- UnifiedMemoryPerf の data movement path を図にする。
- external memory sample で CUDA と外部 API の owner を分ける。

## Cross-Theme Links

- [Unified Memory](unified_memory.md): managed memory を詳しく読みます。
- [Streams And Events](streams_events.md): Async copy と stream ordering を読みます。
- [Performance](performance.md): memory bandwidth と measurement を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
