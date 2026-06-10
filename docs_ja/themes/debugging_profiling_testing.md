# Debugging, Profiling, And Testing

English anchor: Samples use error checks, reference validation, profiling ranges, assertions, printf, and device capability checks.

> **日本語**
> debugging/profiling/testing は CUDA sample を安全に読むための観察点です。error check は API boundary、validation は result boundary、profiling は measurement boundary を示します。
>
> **学習メモ**
> この guide は code を開く前に読む前提知識です。API 名、sample 名、command、出力文字列は英語のまま保持し、意味と読み方を日本語で補います。

## Concept

debugging/profiling/testing は CUDA sample を安全に読むための観察点です。error check は API boundary、validation は result boundary、profiling は measurement boundary を示します。

> **日本語**
> まず「どの resource があり、誰が所有し、いつ同期されるか」を説明できる状態にしてから source を読みます。
>
> **学習メモ**
> sample は production code ではなく、1 つの CUDA concept を切り出した教材です。簡潔さのために省かれた汎用化や error path も意識します。

## Why It Matters

- Debugging, Profiling, And Testing は correctness、resource lifetime、performance の読み方に直接関係します。
- API の戻り値、非同期 ordering、validation の位置を分けることで、sample の意図が見えます。
- 同じ concept でも Runtime API、Driver API、library、Python wrapper で責任分担が変わります。

## Mental Model

- debugging/profiling/testing では、どの thread/process/API が resource を所有するかを先に分けます。
- debugging/profiling/testing の API は便利な wrapper に見えても、lifetime、ordering、visibility の境界を持ちます。
- English API names and output strings remain authoritative; Japanese notes explain the reading strategy.

## API Map

| API or concept | Meaning | What to check |
| - | - | - |
| `checkCudaErrors / getLastCudaError` | sample helper の error check macro です。 | どの API family を包むか確認します。 |
| `cudaGetLastError / cudaDeviceSynchronize` | launch error と実行完了 error を確認します。 | 直近 error check と完了待ちを分けます。 |
| `assert / printf in kernel` | device-side debug 出力です。 | buffering と synchronization を確認します。 |
| `np.allclose / sdkCompare*` | reference validation です。 | tolerance、layout、precision を確認します。 |

> **日本語**
> table の API 名は翻訳せず、ownership、ordering、visibility、validation、cleanup のどれに関係するかを確認します。
>
> **学習メモ**
> helper macro や wrapper の中にも CUDA API が隠れていることがあります。source annotation の `JP:` と照合しながら読みます。

## Sample References

- [simpleAssert](../../cpp/0_Introduction/simpleAssert/README.ja.md): device assert を読みます。
- [simplePrintf](../../cpp/0_Introduction/simplePrintf/README.ja.md): kernel printf を読みます。
- [deviceQuery](../../cpp/1_Utilities/deviceQuery/README.ja.md): capability check を読みます。
- [kernelNsysProfile](../../python/1_GettingStarted/kernelNsysProfile/README.ja.md): profiling range を読みます。
- [ptxjit](../../cpp/3_CUDA_Features/ptxjit/README.ja.md): JIT/Driver error handling を読みます。

## Reading Steps

1. debugging/profiling/testing に関係する API call を探し、作成、設定、利用、破棄を対応させます。
2. 入力 data が GPU work から見える状態になる場所を探します。
3. 非同期 work と host-side validation の間にある synchronization boundary を確認します。
4. error check、reference comparison、timing range を分けて読みます。
5. 関連 sample と比較して、同じ概念が別 API family でどう変わるか確認します。

## Common Mistakes

- debugging/profiling/testing の scope を大きく見積もり、実際には保証されない ordering を期待する。
- helper/wrapper が内部で CUDA resource を所有していることを見落とす。
- validation 前の同期や data visibility を確認しない。
- performance number だけを見て、測定範囲と setup cost を確認しない。

## Performance Notes

- debugging/profiling/testing は correctness の仕組みであると同時に overhead や bottleneck になり得ます。
- timing では setup、GPU work、transfer、sync、validation を分けます。
- architecture、driver、problem size に依存するため、sample の数値は相対比較として読みます。

## Exercises

- debugging/profiling/testing に関係する API を 5 つ選び、owner、input、output、cleanup を表にする。
- sample の timeline を setup、GPU work、sync、validation、cleanup に分ける。
- 関連 theme を 1 つ読み、同じ API pattern がどこで再利用されているか探す。

## Cross-Theme Links

- [Performance](performance.md): 測定範囲を読みます。
- [Runtime, Driver, And NVRTC](runtime_driver_nvrtc.md): compile/runtime error を読みます。
- [Execution Model](execution_model.md): launch error と execution error を分けます。

## Review Checklist

- Can you name the owner and lifetime of each CUDA resource used by the theme?
- Can you point to the synchronization boundary before validation?
- Can you explain which work is measured and which setup/cleanup is outside the measurement?
- Can you compare one C++ sample and one Python or library sample that use the same theme?
