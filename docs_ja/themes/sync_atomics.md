# Synchronization And Atomics

English anchor: Synchronization creates ordering, while atomics protect shared updates to one memory location.

> **日本語**
> sync は work の完了や visibility の境界を作り、atomic は複数 thread が同じ address を更新する競合を安全に扱います。host/device、stream、block、warp、group で同期の範囲が違います。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

sync は work の完了や visibility の境界を作り、atomic は複数 thread が同じ address を更新する競合を安全に扱います。host/device、stream、block、warp、group で同期の範囲が違います。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- Synchronization And Atomics は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- synchronization/atomics では、どの thread/process/API が resource を所有するかを先に分けます。
- synchronization/atomics の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `__syncthreads / __syncwarp` | block/warp 内の待ち合わせです。 | 全参加 thread が到達するか確認します。 |
| `cudaDeviceSynchronize / cudaStreamSynchronize` | host が device/stream work を待ちます。 | validation 前の boundary か確認します。 |
| `atomicAdd / atomicCAS` | 同じ address の read-modify-write を保護します。 | contention と result order を確認します。 |
| `__threadfence` | write visibility の順序を制御します。 | barrier ではないことを確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [simpleAtomicIntrinsics](../../cpp/0_Introduction/simpleAtomicIntrinsics/README.ja.md): atomic intrinsic の基本です。
- [systemWideAtomics](../../cpp/0_Introduction/systemWideAtomics/README.ja.md): system-scope atomic を読みます。
- [threadFenceReduction](../../cpp/2_Concepts_and_Techniques/threadFenceReduction/README.ja.md): fence と reduction completion を読みます。
- [warpAggregatedAtomicsCG](../../cpp/3_CUDA_Features/warpAggregatedAtomicsCG/README.ja.md): atomic contention reduction を読みます。
- [parallelReduction](../../python/2_CoreConcepts/parallelReduction/README.ja.md): Python reduction boundary を読みます。

## Reading Steps

1. synchronization/atomics に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- synchronization/atomics の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- synchronization/atomics は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- synchronization/atomics に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Shared Memory](shared_memory.md): barrier が必要な block 内共有を読みます。
- [Cooperative Groups](cooperative_groups.md): group sync を読みます。
- [Performance](performance.md): atomic contention を読みます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
