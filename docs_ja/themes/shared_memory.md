# CUDA Shared Memory

English anchor: Shared memory is a fast per-block scratchpad shared by threads in the same block.

> **日本語**
> shared memory は block 内 thread が協調して使う scratchpad です。global memory から tile を読み込み、barrier 後に再利用する pattern が matrix、transpose、histogram などに出ます。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

shared memory は block 内 thread が協調して使う scratchpad です。global memory から tile を読み込み、barrier 後に再利用する pattern が matrix、transpose、histogram などに出ます。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- CUDA Shared Memory は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- shared memory では、どの thread/process/API が resource を所有するかを先に分けます。
- shared memory の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `__shared__` | 静的 shared memory を宣言します。 | array shape と thread/block mapping を確認します。 |
| `extern __shared__` | dynamic shared memory を宣言します。 | launch 時 byte 数と型変換を確認します。 |
| `__syncthreads` | producer/consumer をそろえる barrier です。 | 全 thread が到達するか確認します。 |
| `cudaFuncSetAttribute` | shared memory carveout などを設定します。 | architecture 依存制限を確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [matrixMul](../../cpp/0_Introduction/matrixMul/README.ja.md): classic tiling を読みます。
- [transpose](../../cpp/6_Performance/transpose/README.ja.md): bank conflict と coalescing を読みます。
- [histogram](../../cpp/2_Concepts_and_Techniques/histogram/README.ja.md): block-local accumulation を読みます。
- [globalToShmemAsyncCopy](../../cpp/3_CUDA_Features/globalToShmemAsyncCopy/README.ja.md): async copy staging を読みます。
- [matrixMulSharedMem](../../python/2_CoreConcepts/matrixMulSharedMem/README.ja.md): Python shared memory kernel を読みます。

## Reading Steps

1. shared memory に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- shared memory の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- shared memory は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- shared memory に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Kernel Launch And Indexing](kernel_indexing.md): tile coordinate を読みます。
- [Synchronization And Atomics](sync_atomics.md): barrier scope を読みます。
- [Performance](performance.md): reuse と bank conflict を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
