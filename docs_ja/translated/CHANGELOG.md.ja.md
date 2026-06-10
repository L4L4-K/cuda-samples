# CHANGELOG.md Japanese Companion

Original English document: [`CHANGELOG.md`](../../CHANGELOG.md)

## English Reference

This companion keeps every parsed English source paragraph and places Japanese study notes directly below it. Commands, paths, APIs, target names, output strings, LICENSE text, headers, and attribution remain in English.

> **日本語**
> この companion は、原文の各 paragraph を英語のまま残し、その直下に日本語の理解メモを追加します。実行名、API 名、path、target、出力文字列は翻訳しません。
> **学習メモ**
> 迷った場合は英語の原文を authoritative source とし、日本語は CUDA の前提、build/run の流れ、memory/sync/performance の読みどころを補う secondary material として使います。

## Paragraph Notes

## Source Paragraph 001

Context: `CUDA 13.3`

English paragraph 1:
> * Added **CUDA Tile C++** samples under `cpp/9_CUDA_Tile`.

> **日本語**
> この項目は `CUDA 13.3` の CUDA Tile sample に関する 追加 です。`CUDA Tile C++`、`cpp/9_CUDA_Tile` を見て、tile 単位の execution model、data movement、autotuning のどこを学ぶ sample か確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 002

Context: `CUDA 13.3`

English paragraph 2:
> * Added a set of **CCCL 3.3 feature samples** under `cpp/4_CUDA_Libraries/`, each built against CCCL fetched via CPM
> (pinned to v3.3.3, with an optional `CCCL_SOURCE_DIR` override):

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 追加 です。`CCCL 3.3 feature samples`、`cpp/4_CUDA_Libraries/`、`CCCL`、`CCCL_SOURCE_DIR` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 003

Context: `CUDA 13.3`

English paragraph 3:
> * `cubDeviceFind` - `cub::DeviceFind::FindIf`, `LowerBound`, and `UpperBound` device-wide search algorithms.

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 変更 です。`cubDeviceFind`、`cub::DeviceFind::FindIf`、`LowerBound`、`UpperBound` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 004

Context: `CUDA 13.3`

English paragraph 4:
> * `cubDeviceSegmentedScan` - `cub::DeviceSegmentedScan::ExclusiveSegmentedSum` and `InclusiveSegmentedScan` with a
> custom binary operator.

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 変更 です。`cubDeviceSegmentedScan`、`cub::DeviceSegmentedScan::ExclusiveSegmentedSum`、`InclusiveSegmentedScan` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 005

Context: `CUDA 13.3`

English paragraph 5:
> * `cubDeviceTransform` - N-to-M `cub::DeviceTransform::Transform` where the op returns a `cuda::std::tuple`.

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 変更 です。`cubDeviceTransform`、`cub::DeviceTransform::Transform`、`cuda::std::tuple` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 006

Context: `CUDA 13.3`

English paragraph 6:
> * `libcuxxRandom` - `cuda::pcg64` and `cuda::std::philox4x32` engines driving the uniform, normal, Poisson, and
> Bernoulli distributions from `<cuda/std/random>`.

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 変更 です。`libcuxxRandom`、`cuda::pcg64`、`cuda::std::philox4x32`、`<cuda/std/random>` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 007

Context: `CUDA 13.3`

English paragraph 7:
> * `libcuxxMdspan` - DLPack <-> `cuda::std::mdspan` bridging via `cuda::to_device_mdspan` / `cuda::to_dlpack_tensor`,
> plus `cuda::shared_memory_mdspan` for multi-dimensional views of shared memory.

> **日本語**
> この項目は `CUDA 13.3` の CUDA library または CCCL sample に関する 変更 です。`libcuxxMdspan`、`cuda::std::mdspan`、`cuda::to_device_mdspan`、`cuda::to_dlpack_tensor`、`cuda::shared_memory_mdspan` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 008

Context: `CUDA 13.3`

English paragraph 8:
> * Added **cuda.compute 1.0 Python samples** under `python/2_CoreConcepts/`:

> **日本語**
> この変更履歴項目は `CUDA 13.3` で Python CUDA 関連の 追加 があったことを示します。`cuda.compute 1.0 Python samples`、`python/2_CoreConcepts/` を見て、C++ sample と Python sample の build/run 手順の違いを区別します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 009

Context: `CUDA 13.3`

English paragraph 9:
> * `cudaComputeLambdas` - Python lambdas / regular callables driving `reduce_into`, `unary_transform`, and
> `inclusive_scan` in `cuda.compute` (from the `cuda-cccl` package).

> **日本語**
> この変更履歴項目は `CUDA 13.3` で Python CUDA 関連の 変更 があったことを示します。`cudaComputeLambdas`、`Python`、`reduce_into`、`unary_transform`、`inclusive_scan`、`cuda.compute` を見て、C++ sample と Python sample の build/run 手順の違いを区別します。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 010

Context: `CUDA 13.3`

English paragraph 10:
> * `binarySearch` - parallel `cuda.compute.upper_bound` / `lower_bound`, verified against `numpy.searchsorted`.

> **日本語**
> この変更履歴項目は `CUDA 13.3` で Python CUDA 関連の 変更 があったことを示します。`binarySearch`、`cuda.compute.upper_bound`、`lower_bound`、`numpy.searchsorted` を見て、C++ sample と Python sample の build/run 手順の違いを区別します。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 011

Context: `CUDA 13.2 (update)`

English paragraph 11:
> * Added **CUDA Python samples** under `python/`. These scripts use [CUDA
> Python](https://nvidia.github.io/cuda-python/) (including `cuda.core`) and are organized like the C++ tree:
> `1_GettingStarted`, `2_CoreConcepts`, `3_FrameworkInterop`, and `4_DistributedComputing`, plus shared helpers in
> `python/Utilities`. Each sample includes a `README.md` and `requirements.txt`. They are **not** built by the root
> CMake project; install dependencies with `pip install -r requirements.txt` in the sample directory, then run the
> corresponding `.py` file as documented in that sample’s README.

> **日本語**
> この変更履歴項目は `CUDA 13.2 (update)` で Python CUDA 関連の 追加 があったことを示します。`CUDA Python samples`、`python/`、`Python`、`cuda`、`cuda.core`、`1_GettingStarted` を見て、C++ sample と Python sample の build/run 手順の違いを区別します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 012

Context: `CUDA 13.2 (update)`

English paragraph 12:
> * Renamed top-level `Samples` directory to `cpp` to accommodate Python samples alongside existing C++ samples; updated
> path references in `CMakeLists.txt`, `README.md`, and `Common` headers accordingly.

> **日本語**
> この変更履歴項目は `CUDA 13.2 (update)` で Python CUDA 関連の 更新 があったことを示します。`Samples`、`cpp`、`Python`、`CMakeLists.txt`、`README.md`、`Common` を見て、C++ sample と Python sample の build/run 手順の違いを区別します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 013

Context: `CUDA 13.2`

English paragraph 13:
> * Added the MSVC compile flag `-Xcompiler=/Zc:preprocessor` in CMakeLists.txt to comply with CUDA13.2 CCCL.
> Previously, using the traditional preprocessor triggered the warning “MSVC/cl.exe with traditional preprocessor is
> used…”, which now leads to a build error.

> **日本語**
> この項目は `CUDA 13.2` の build system に関する 追加 です。`-Xcompiler=/Zc:preprocessor`、`CMakeLists.txt`、`CCCL` は configure、compile option、dependency 解決に直結するため、sample logic の変更とは分けて読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 014

Context: `CUDA 13.1`

English paragraph 14:
> * Minor bug fixes and enhancements, no structural or functional changes

> **日本語**
> この変更履歴項目は `CUDA 13.1` で 修正 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 015

Context: `CUDA 13.0`

English paragraph 15:
> * Updated the samples using the cudaDeviceProp fields which are deprecated and removed in CUDA 13.0, replacing the
> fields with the equivalents in "cudaDeviceGetAttribute":

> **日本語**
> この項目は `CUDA 13.0` の Runtime API device property 取得に関する 更新 です。`cudaDeviceProp`、`cudaDeviceGetAttribute` を見て、古い `cudaDeviceProp` field と現在の attribute API の対応を確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 016

Context: `CUDA 13.0`

English paragraph 16:
> * Deprecated "cudaDeviceProp" fields

> **日本語**
> この項目は `CUDA 13.0` の Runtime API device property 取得に関する 非推奨化への対応 です。`cudaDeviceProp` を見て、古い `cudaDeviceProp` field と現在の attribute API の対応を確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 017

Context: `CUDA 13.0`

English paragraph 17:
> `int clockRate; // - Replaced with "cudaDevAttrClockRate"` `int deviceOverlap; // - Replaced with
> "cudaDevAttrGpuOverlap */` `int kernelExecTimeoutEnabled; // - Replaced with "cudaDevAttrKernelExecTimeout` `int
> computeMode; // - Replaced with "cudaDevAttrComputeMode" */` `int memoryClockRate; // - Replaced with
> "cudaDevAttrMemoryClockRate"` `int cooperativeMultiDeviceLaunch; // - Deprecated,
> cudaLaunchCooperativeKernelMultiDevice is deprecated.`

> **日本語**
> この項目は `CUDA 13.0` の memory API または memory sample に関する 更新 です。`int clockRate; // - Replaced with "cudaDevAttrClockRate"`、`int deviceOverlap; // - Replaced with "cudaDevAttrGpuOverlap */`、`int kernelExecTimeoutEnabled; // - Replaced with "cudaDevAttrKernelExecTimeout`、`int computeMode; // - Replaced with "cudaDevAttrComputeMode" */`、`int memoryClockRate; // - Replaced with "cudaDevAttrMemoryClockRate"`、`int cooperativeMultiDeviceLaunch; // - Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 018

Context: `CUDA 13.0`

English paragraph 18:
> * `2_Concepts_and_Techniques`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`2_Concepts_and_Techniques` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 019

Context: `CUDA 13.0`

English paragraph 19:
> * `streamOrderedAllocationIPC`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`streamOrderedAllocationIPC` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 020

Context: `CUDA 13.0`

English paragraph 20:
> * Updated the samples using the CUDA driver API "cuCtxCreate" with adding the parameter "CUctxCreateParams" as
> "cuCtxCreate" is updated to "cuCtxCreate_v4" by default in CUDA 13.0:

> **日本語**
> この項目は `CUDA 13.0` の Driver API 境界に関する 更新 です。`cuCtxCreate`、`cuCtxCreate_v4` では context、module、function、handle lifetime の変更を意識します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 021

Context: `CUDA 13.0`

English paragraph 21:
> * `2_Concepts_and_Techniques`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`2_Concepts_and_Techniques` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 022

Context: `CUDA 13.0`

English paragraph 22:
> * `8_Platform_Specific/Tegra`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`8_Platform_Specific/Tegra` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 023

Context: `CUDA 13.0`

English paragraph 23:
> * `EGLSync_CUDAEvent_Interop`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`EGLSync_CUDAEvent_Interop` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 024

Context: `CUDA 13.0`

English paragraph 24:
> * Updated the sample using CUDA API "cudaGraphAddNode"/"cudaStreamGetCaptureInfo" with adding "cudaGraphEdgeData"
> pointer parameter as they are updated to "cudaGraphAddNode_v2"/"cudaStreamGetCaptureInfo_v3" by default in CUDA 13.0:

> **日本語**
> この項目は `CUDA 13.0` の CUDA Graph 関連の 更新 です。`cudaGraphAddNode`、`cudaStreamGetCaptureInfo`、`cudaGraphEdgeData`、`cudaGraphAddNode_v2`、`cudaStreamGetCaptureInfo_v3` について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 025

Context: `CUDA 13.0`

English paragraph 25:
> * Updated the samples using CUDA API "cudaMemAdvise"/"cudaMemPrefetchAsync" with changing the parameter "int device"
> to "cudaMemLocation location" as they are updated to "cudaMemAdvise_v2"/"cudaMemPrefetchAsyn_v2" by default in CUDA
> 13.0.

> **日本語**
> この項目は `CUDA 13.0` の memory API または memory sample に関する 更新 です。`cudaMemAdvise`、`cudaMemPrefetchAsync`、`cudaMemLocation`、`cudaMemAdvise_v2`、`cudaMemPrefetchAsyn_v2` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 026

Context: `CUDA 13.0`

English paragraph 26:
> * `conjugateGradientMultiDeviceCG`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`conjugateGradientMultiDeviceCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 027

Context: `CUDA 13.0`

English paragraph 27:
> * Replaced "thrust::identity<uint>()" with "cuda::std::identity()" as it is deprecated in CUDA 13.0.

> **日本語**
> この項目は `CUDA 13.0` で古い API、sample、または platform support が整理されたことを示します。`cuda` が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 028

Context: `CUDA 13.0`

English paragraph 28:
> * `2_Concepts_and_Techniques`

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。`2_Concepts_and_Techniques` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 029

Context: `CUDA 13.0`

English paragraph 29:
> * Updated the the headers file and samples for CUFFT error codes update.

> **日本語**
> この項目は `CUDA 13.0` の CUDA library または CCCL sample に関する 更新 です。`CUFFT` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 030

Context: `CUDA 13.0`

English paragraph 30:
> * `CUFFT_INCOMPLETE_PARAMETER_LIST`

> **日本語**
> この項目は `CUDA 13.0` の CUDA library または CCCL sample に関する 変更 です。`CUFFT_INCOMPLETE_PARAMETER_LIST` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 031

Context: `CUDA 13.0`

English paragraph 31:
> * Header files and samples that are related with this change:

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 変更 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 032

Context: `CUDA 13.0`

English paragraph 32:
> * Updated toolchain for cross-compilation for Tegra QNX platforms.

> **日本語**
> この変更履歴項目は `CUDA 13.0` で 更新 された内容です。`Tegra` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 033

Context: `CUDA 12.9`

English paragraph 33:
> * Updated toolchain for cross-compilation for Tegra Linux platforms.

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 更新 された内容です。`Tegra`、`Linux` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> build 関連の変更は runtime behavior ではなく toolchain contract の変更です。失敗時は generator と dependency を先に疑います。

## Source Paragraph 034

Context: `CUDA 12.9`

English paragraph 34:
> * Added `run_tests.py` utility to exercise all samples. See README.md for details

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 追加 された内容です。`run_tests.py`、`README.md` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 035

Context: `CUDA 12.9`

English paragraph 35:
> * Repository has been updated with consistent code formatting across all samples

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 更新 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 036

Context: `CUDA 12.9`

English paragraph 36:
> * Many small code tweaks and bug fixes (see commit history for details)

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 修正 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 037

Context: `CUDA 12.9`

English paragraph 37:
> * Removed the following outdated samples:

> **日本語**
> この項目は `CUDA 12.9` で古い API、sample、または platform support が整理されたことを示します。主要な語句 が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 038

Context: `CUDA 12.9`

English paragraph 38:
> * `bandwidthTest` - this sample was out of date and did not produce accurate results. For bandwidth

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 変更 された内容です。`bandwidthTest` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 039

Context: `CUDA 12.9`

English paragraph 39:
> testing of NVIDIA GPU platforms, please refer to [NVBandwidth](https://github.com/NVIDIA/nvbandwidth)

> **日本語**
> この変更履歴項目は `CUDA 12.9` で 変更 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 040

Context: `CUDA 12.8`

English paragraph 40:
> * Updated build system across the repository to CMake. Removed Visual Studio project files and Makefiles.

> **日本語**
> この項目は `CUDA 12.8` の build system に関する 更新 です。`CMake`、`Visual Studio` は configure、compile option、dependency 解決に直結するため、sample logic の変更とは分けて読みます。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 041

Context: `CUDA 12.8`

English paragraph 41:
> * Removed the following outdated samples:

> **日本語**
> この項目は `CUDA 12.8` で古い API、sample、または platform support が整理されたことを示します。主要な語句 が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 042

Context: `CUDA 12.8`

English paragraph 42:
> * `c++11_cuda` demonstrating CUDA and C++ 11 interoperability (reason: obsolete)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`c++11_cuda` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 043

Context: `CUDA 12.8`

English paragraph 43:
> * `concurrentKernels` demonstrating the ability to run multiple kernels simultaneously (reason: obsolete)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`concurrentKernels` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 044

Context: `CUDA 12.8`

English paragraph 44:
> * `cppIntegration` demonstrating calling between .cu and .cpp files (reason: obsolete)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`cppIntegration` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 045

Context: `CUDA 12.8`

English paragraph 45:
> * `cppOverload` demonstrating C++ function overloading (reason: obsolete)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`cppOverload` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 046

Context: `CUDA 12.8`

English paragraph 46:
> * `simpleSeparateCompilation` demonstrating NVCC compilation to a static library (reason: trivial)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`simpleSeparateCompilation` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 047

Context: `CUDA 12.8`

English paragraph 47:
> * `simpleTemplates_nvrtc` demonstrating NVRTC usage for `simpleTemplates` sample (reason: redundant)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`simpleTemplates_nvrtc`、`NVRTC`、`simpleTemplates` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 048

Context: `CUDA 12.8`

English paragraph 48:
> * `simpleVoteIntrinsics_nvrtc` demonstrating NVRTC usage for `simpleVoteIntrinsics` sample (reason: redundant)

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`simpleVoteIntrinsics_nvrtc`、`NVRTC`、`simpleVoteIntrinsics` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 049

Context: `CUDA 12.8`

English paragraph 49:
> * `2_Concepts_and_Techniques`

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`2_Concepts_and_Techniques` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 050

Context: `CUDA 12.8`

English paragraph 50:
> * `cuHook` demonstrating dlsym hooks. (reason: incompatible with modern `glibc`)

> **日本語**
> この項目は `CUDA 12.8` の Driver API 境界に関する 変更 です。`cuHook`、`glibc` では context、module、function、handle lifetime の変更を意識します。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 051

Context: `CUDA 12.8`

English paragraph 51:
> * `batchedLabelMarkersAndLabelCompressionNPP` demonstrating NPP features (reason: some functionality removed from
> library)

> **日本語**
> この項目は `CUDA 12.8` の CUDA library または CCCL sample に関する 削除 です。`batchedLabelMarkersAndLabelCompressionNPP`、`NPP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 052

Context: `CUDA 12.8`

English paragraph 52:
> * Legacy Direct3D 9 and 10 interoperability samples:

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 053

Context: `CUDA 12.8`

English paragraph 53:
> * `8_Platform_Specific/Tegra`

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`8_Platform_Specific/Tegra` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 054

Context: `CUDA 12.8`

English paragraph 54:
> * Temporarily removed the following two samples pending updates:

> **日本語**
> この項目は `CUDA 12.8` で古い API、sample、または platform support が整理されたことを示します。主要な語句 が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 055

Context: `CUDA 12.8`

English paragraph 55:
> * `nbody_screen` demonstrating the nbody sample in QNX

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`nbody_screen` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 056

Context: `CUDA 12.8`

English paragraph 56:
> * `simpleGLES_screen` demonstrating GLES interop in QNX

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`simpleGLES_screen` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 057

Context: `CUDA 12.8`

English paragraph 57:
> * Moved the following Tegra-specific samples to a dedicated subdirectory: `8_Platform_Specific/Tegra`

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`Tegra`、`8_Platform_Specific/Tegra` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 058

Context: `CUDA 12.8`

English paragraph 58:
> * `EGLSync_CUDAEvent_Interop`

> **日本語**
> この変更履歴項目は `CUDA 12.8` で 変更 された内容です。`EGLSync_CUDAEvent_Interop` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 059

Context: `CUDA 12.8`

English paragraph 59:
> * `cuDLALayerwiseStatsHybrid`

> **日本語**
> この項目は `CUDA 12.8` の Tegra/cuDLA platform sample に関する変更です。`cuDLALayerwiseStatsHybrid` では DLA accelerator、CUDA interop、platform-specific dependency、error/reporting mode を確認します。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 060

Context: `CUDA 12.8`

English paragraph 60:
> * `cuDLALayerwiseStatsStandalone`

> **日本語**
> この項目は `CUDA 12.8` の Tegra/cuDLA platform sample に関する変更です。`cuDLALayerwiseStatsStandalone` では DLA accelerator、CUDA interop、platform-specific dependency、error/reporting mode を確認します。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 061

Context: `CUDA 12.4`

English paragraph 61:
> * Added graphConditionalNodes Sample

> **日本語**
> この項目は `CUDA 12.4` の CUDA Graph 関連の 追加 です。主要な語句 について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 062

Context: `CUDA 12.2`

English paragraph 62:
> * libNVVM samples received updates

> **日本語**
> この変更履歴項目は `CUDA 12.2` で 更新 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 063

Context: `CUDA 12.2`

English paragraph 63:
> * Enabled HOST_COMPILER flag to the makefiles for GCC which is untested but may still work.

> **日本語**
> この項目は `CUDA 12.2` の build system に関する 変更 です。主要な語句 は configure、compile option、dependency 解決に直結するため、sample logic の変更とは分けて読みます。
> **学習メモ**
> build 関連の変更は runtime behavior ではなく toolchain contract の変更です。失敗時は generator と dependency を先に疑います。

## Source Paragraph 064

Context: `CUDA 12.1`

English paragraph 64:
> * Added new sample for Large Kernels

> **日本語**
> この変更履歴項目は `CUDA 12.1` で 追加 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 065

Context: `CUDA 12.0`

English paragraph 65:
> * Added new flags for JIT compiling

> **日本語**
> この変更履歴項目は `CUDA 12.0` で 追加 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 066

Context: `CUDA 12.0`

English paragraph 66:
> * Removed deprecated APIs in Hopper Architecture

> **日本語**
> この項目は `CUDA 12.0` で古い API、sample、または platform support が整理されたことを示します。主要な語句 が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 067

Context: `CUDA 11.6`

English paragraph 67:
> * Added new folder structure for samples

> **日本語**
> この変更履歴項目は `CUDA 11.6` で 追加 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 068

Context: `CUDA 11.6`

English paragraph 68:
> * Added support of Visual Studio 2022 to all samples supported on [Windows](#windows-1).

> **日本語**
> この変更履歴項目は `CUDA 11.6` で 追加 された内容です。`Visual Studio`、`Windows` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 069

Context: `CUDA 11.6`

English paragraph 69:
> * All CUDA samples are now only available on [GitHub](https://github.com/nvidia/cuda-samples). They are no longer
> available via CUDA toolkit.

> **日本語**
> この変更履歴項目は `CUDA 11.6` で 変更 された内容です。`cuda` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 070

Context: `CUDA 11.5`

English paragraph 70:
> * Added `cuDLAHybridMode`. Demonstrate usage of cuDLA in hybrid mode.

> **日本語**
> この項目は `CUDA 11.5` の Tegra/cuDLA platform sample に関する変更です。`cuDLAHybridMode`、`cuDLA` では DLA accelerator、CUDA interop、platform-specific dependency、error/reporting mode を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 071

Context: `CUDA 11.5`

English paragraph 71:
> * Added `cuDLAStandaloneMode`. Demonstrate usage of cuDLA in standalone mode.

> **日本語**
> この項目は `CUDA 11.5` の Tegra/cuDLA platform sample に関する変更です。`cuDLAStandaloneMode`、`cuDLA` では DLA accelerator、CUDA interop、platform-specific dependency、error/reporting mode を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 072

Context: `CUDA 11.5`

English paragraph 72:
> * Added `cuDLAErrorReporting`. Demonstrate DLA error detection via CUDA.

> **日本語**
> この項目は `CUDA 11.5` の Tegra/cuDLA platform sample に関する変更です。`cuDLAErrorReporting` では DLA accelerator、CUDA interop、platform-specific dependency、error/reporting mode を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 073

Context: `CUDA 11.5`

English paragraph 73:
> * Added `graphMemoryNodes`. Demonstrates memory allocations and frees within CUDA graphs using Graph APIs and Stream
> Capture APIs.

> **日本語**
> この項目は `CUDA 11.5` の CUDA Graph 関連の 追加 です。`graphMemoryNodes` について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 074

Context: `CUDA 11.5`

English paragraph 74:
> * Added `graphMemoryFootprint`. Demonstrates how graph memory nodes re-use virtual addresses and physical memory.

> **日本語**
> この項目は `CUDA 11.5` の CUDA Graph 関連の 追加 です。`graphMemoryFootprint` について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 075

Context: `CUDA 11.5`

English paragraph 75:
> * All samples from CUDA toolkit are now available on [GitHub](https://github.com/nvidia/cuda-samples).

> **日本語**
> この変更履歴項目は `CUDA 11.5` で 変更 された内容です。`cuda` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 076

Context: `CUDA 11.4 update 1`

English paragraph 76:
> * Added support for VS Code on linux platform.

> **日本語**
> この変更履歴項目は `CUDA 11.4 update 1` で 追加 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 077

Context: `CUDA 11.4`

English paragraph 77:
> * Added `cdpQuadtree`. Demonstrates Quad Trees implementation using CUDA Dynamic Parallelism.

> **日本語**
> この変更履歴項目は `CUDA 11.4` で 追加 された内容です。`cdpQuadtree` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 078

Context: `CUDA 11.4`

English paragraph 78:
> * Updated `simpleVulkan`, `simpleVulkanMMAP` and `vulkanImageCUDA`. Demonstrates use of SPIR-V shaders.

> **日本語**
> この変更履歴項目は `CUDA 11.4` で 更新 された内容です。`simpleVulkan`、`simpleVulkanMMAP`、`vulkanImageCUDA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 079

Context: `CUDA 11.3`

English paragraph 79:
> *  Added `streamOrderedAllocationIPC`. Demonstrates Inter Process Communication using one process per GPU for
> computation.

> **日本語**
> この変更履歴項目は `CUDA 11.3` で 追加 された内容です。`streamOrderedAllocationIPC` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 080

Context: `CUDA 11.3`

English paragraph 80:
> *  Added `simpleCUBLAS_LU`. Demonstrates batched matrix LU decomposition using cuBLAS API `cublas<t>getrfBatched()`

> **日本語**
> この項目は `CUDA 11.3` の CUDA library または CCCL sample に関する 追加 です。`simpleCUBLAS_LU`、`cuBLAS`、`cublas<t>getrfBatched()` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 081

Context: `CUDA 11.3`

English paragraph 81:
> *  Updated `simpleVulkan`. Demonstrates use of timeline semaphore.

> **日本語**
> この変更履歴項目は `CUDA 11.3` で 更新 された内容です。`simpleVulkan` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> CHANGELOG は実装の意図を短く残す索引です。詳しい挙動は該当 sample の README.ja.md と source の JP コメントで確認します。

## Source Paragraph 082

Context: `CUDA 11.3`

English paragraph 82:
> *  Updated multiple samples to use pinned memory using `cudaMallocHost()`.

> **日本語**
> この項目は `CUDA 11.3` の memory API または memory sample に関する 更新 です。`cudaMallocHost()` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 083

Context: `CUDA 11.2`

English paragraph 83:
> *  Added `streamOrderedAllocation`. Demonstrates stream ordered memory allocation on a GPU using cudaMallocAsync and
> cudaMemPool family of APIs.

> **日本語**
> この項目は `CUDA 11.2` の memory API または memory sample に関する 追加 です。`streamOrderedAllocation`、`cudaMallocAsync`、`cudaMemPool` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 084

Context: `CUDA 11.2`

English paragraph 84:
> *  Added `streamOrderedAllocationP2P`. Demonstrates peer-to-peer access of stream ordered memory allocated using
> cudaMallocAsync and cudaMemPool family of APIs.

> **日本語**
> この項目は `CUDA 11.2` の memory API または memory sample に関する 追加 です。`streamOrderedAllocationP2P`、`cudaMallocAsync`、`cudaMemPool` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 085

Context: `CUDA 11.2`

English paragraph 85:
> *  Dropped Visual Studio 2015 support from all the windows supported samples.

> **日本語**
> この変更履歴項目は `CUDA 11.2` で サポート条件の変更 された内容です。`Visual Studio` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 086

Context: `CUDA 11.2`

English paragraph 86:
> *  FreeImage is no longer distributed with the CUDA Samples. On Windows, see the [Dependencies](./README.md#freeimage)
> section for more details on how to set up FreeImage. On Linux, it is recommended to install FreeImage with your
> distribution's package manager.

> **日本語**
> この変更履歴項目は `CUDA 11.2` で 変更 された内容です。`Windows`、`README.md`、`Linux` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 087

Context: `CUDA 11.2`

English paragraph 87:
> *  All the samples using CUDA Pipeline & Arrive-wait barriers are been updated to use new `cuda::pipeline` and
> `cuda::barrier` interfaces.

> **日本語**
> この変更履歴項目は `CUDA 11.2` で 更新 された内容です。`cuda::pipeline`、`cuda::barrier` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 088

Context: `CUDA 11.2`

English paragraph 88:
> *  Updated all the samples to build with parallel build option `--threads` of `nvcc` cuda compiler.

> **日本語**
> この項目は `CUDA 11.2` の build system に関する 更新 です。`--threads`、`nvcc`、`cuda` は configure、compile option、dependency 解決に直結するため、sample logic の変更とは分けて読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 089

Context: `CUDA 11.2`

English paragraph 89:
> *  Added `cudaNvSciNvMedia`. Demonstrates CUDA-NvMedia interop via NvSciBuf/NvSciSync APIs.

> **日本語**
> この変更履歴項目は `CUDA 11.2` で 追加 された内容です。`cudaNvSciNvMedia` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 090

Context: `CUDA 11.2`

English paragraph 90:
> *  Added `simpleGL`. Demonstrates interoperability between CUDA and OpenGL.

> **日本語**
> この変更履歴項目は `CUDA 11.2` で 追加 された内容です。`simpleGL` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 091

Context: `CUDA 11.1`

English paragraph 91:
> *  Added `watershedSegmentationNPP`. Demonstrates how to use the NPP watershed segmentation function.

> **日本語**
> この項目は `CUDA 11.1` の CUDA library または CCCL sample に関する 追加 です。`watershedSegmentationNPP`、`NPP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 092

Context: `CUDA 11.1`

English paragraph 92:
> *  Added `batchedLabelMarkersAndLabelCompressionNPP`. Demonstrates how to use the NPP label markers generation and
> label compression functions based on a Union Find (UF) algorithm including both single image and batched image
> versions.

> **日本語**
> この項目は `CUDA 11.1` の CUDA library または CCCL sample に関する 追加 です。`batchedLabelMarkersAndLabelCompressionNPP`、`NPP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 093

Context: `CUDA 11.1`

English paragraph 93:
> *  Dropped Visual Studio 2012, 2013 support from all the windows supported samples.

> **日本語**
> この変更履歴項目は `CUDA 11.1` で サポート条件の変更 された内容です。`Visual Studio` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 094

Context: `CUDA 11.1`

English paragraph 94:
> *  Added kernel performing warp aggregated atomic max in multi buckets using cg::labeled_partition & cg::reduce in
> `warpAggregatedAtomicsCG`.

> **日本語**
> この変更履歴項目は `CUDA 11.1` で 追加 された内容です。`warpAggregatedAtomicsCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 095

Context: `CUDA 11.1`

English paragraph 95:
> *  Added extended CG shuffle mechanics to `shfl_scan` sample.

> **日本語**
> この変更履歴項目は `CUDA 11.1` で 追加 された内容です。`shfl_scan` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 096

Context: `CUDA 11.1`

English paragraph 96:
> *  Added `cudaOpenMP`. Demonstrates how to use OpenMP API to write an application for multiple GPUs.

> **日本語**
> この変更履歴項目は `CUDA 11.1` で 追加 された内容です。`cudaOpenMP` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 097

Context: `CUDA 11.1`

English paragraph 97:
> *  Added `simpleZeroCopy`. Demonstrates how to use zero copy, kernels can read and write directly to pinned system
> memory.

> **日本語**
> この項目は `CUDA 11.1` の memory API または memory sample に関する 追加 です。`simpleZeroCopy` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 098

Context: `CUDA 11.0`

English paragraph 98:
> *  Added `dmmaTensorCoreGemm`. Demonstrates double precision GEMM computation using the Double precision Warp Matrix
> Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`dmmaTensorCoreGemm`、`WMMA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 099

Context: `CUDA 11.0`

English paragraph 99:
> *  Added `bf16TensorCoreGemm`. Demonstrates __nv_bfloat16 (e8m7) GEMM computation using the __nv_bfloat16 WMMA API
> introduced with CUDA 11 in Ampere chip family tensor cores.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`bf16TensorCoreGemm`、`WMMA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 100

Context: `CUDA 11.0`

English paragraph 100:
> *  Added `tf32TensorCoreGemm`. Demonstrates tf32 (e8m10) GEMM computation using the tf32 WMMA API introduced with CUDA
> 11 in Ampere chip family tensor cores.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`tf32TensorCoreGemm`、`WMMA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 101

Context: `CUDA 11.0`

English paragraph 101:
> *  Added `globalToShmemAsyncCopy`. Demonstrates async copy of data from global to shared memory when on compute
> capability 8.0 or higher. Also demonstrates arrive-wait barrier for synchronization.

> **日本語**
> この項目は `CUDA 11.0` の memory API または memory sample に関する 追加 です。`globalToShmemAsyncCopy` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 102

Context: `CUDA 11.0`

English paragraph 102:
> *  Added `simpleAWBarrier`. Demonstrates arrive wait barriers.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`simpleAWBarrier` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 103

Context: `CUDA 11.0`

English paragraph 103:
> *  Added `simpleAttributes`. Demonstrates the stream attributes that affect L2 locality.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`simpleAttributes` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 104

Context: `CUDA 11.0`

English paragraph 104:
> *  Added warp aggregated atomic multi bucket increments kernel using labeled_partition cooperative groups in
> `warpAggregatedAtomicsCG` which can be used on compute capability 7.0 and above GPU architectures.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`warpAggregatedAtomicsCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 105

Context: `CUDA 11.0`

English paragraph 105:
> *  Added `binaryPartitionCG`. Demonstrates  binary partition cooperative groups and reduction within the thread block.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`binaryPartitionCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 106

Context: `CUDA 11.0`

English paragraph 106:
> *  Added two new reduction kernels in `reduction` one which demonstrates reduce_add_sync intrinstic supported on
> compute capability 8.0 and another which uses cooperative_groups::reduce function which does thread_block_tile level
> reduction introduced from CUDA 11.0.

> **日本語**
> この項目は `CUDA 11.0` の CUDA Tile sample に関する 追加 です。`reduction` を見て、tile 単位の execution model、data movement、autotuning のどこを学ぶ sample か確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 107

Context: `CUDA 11.0`

English paragraph 107:
> *  Added `cudaCompressibleMemory`. Demonstrates compressible memory allocation using cuMemMap API.

> **日本語**
> この項目は `CUDA 11.0` の memory API または memory sample に関する 追加 です。`cudaCompressibleMemory`、`cuMemMap` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 108

Context: `CUDA 11.0`

English paragraph 108:
> *  Added `simpleVulkanMMAP`. Demonstrates Vulkan CUDA Interop via cuMemMap APIs.

> **日本語**
> この項目は `CUDA 11.0` の memory API または memory sample に関する 追加 です。`simpleVulkanMMAP`、`cuMemMap` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 109

Context: `CUDA 11.0`

English paragraph 109:
> *  Added `concurrentKernels`. Demonstrates the use of CUDA streams for concurrent execution of several kernels on a
> GPU.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で 追加 された内容です。`concurrentKernels` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 110

Context: `CUDA 11.0`

English paragraph 110:
> *  Dropped Mac OSX support from all samples.

> **日本語**
> この変更履歴項目は `CUDA 11.0` で サポート条件の変更 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 111

Context: `CUDA 10.2`

English paragraph 111:
> *  Added `simpleD3D11`. Demonstrates CUDA-D3D11 External Resource Interoperability APIs for updating D3D11 buffers
> from CUDA and synchronization between D3D11 and CUDA with Keyed Mutexes.

> **日本語**
> この変更履歴項目は `CUDA 10.2` で 追加 された内容です。`simpleD3D11` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 112

Context: `CUDA 10.2`

English paragraph 112:
> *  Added `simpleDrvRuntime`. Demonstrates CUDA Driver and Runtime APIs working together to load fatbinary of a CUDA
> kernel.

> **日本語**
> この変更履歴項目は `CUDA 10.2` で 追加 された内容です。`simpleDrvRuntime` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 113

Context: `CUDA 10.2`

English paragraph 113:
> *  Added `vectorAddMMAP`. Demonstrates how cuMemMap API allows the user to specify the physical properties of their
> memory while retaining the contiguous nature of their access.

> **日本語**
> この項目は `CUDA 10.2` の memory API または memory sample に関する 追加 です。`vectorAddMMAP`、`cuMemMap` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 114

Context: `CUDA 10.2`

English paragraph 114:
> *  Added `memMapIPCDrv`. Demonstrates Inter Process Communication using cuMemMap APIs.

> **日本語**
> この項目は `CUDA 10.2` の memory API または memory sample に関する 追加 です。`memMapIPCDrv`、`cuMemMap` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 115

Context: `CUDA 10.2`

English paragraph 115:
> *  Added `cudaNvSci`. Demonstrates CUDA-NvSciBuf/NvSciSync Interop.

> **日本語**
> この変更履歴項目は `CUDA 10.2` で 追加 された内容です。`cudaNvSci` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 116

Context: `CUDA 10.2`

English paragraph 116:
> *  Added `jacobiCudaGraphs`. Demonstrates Instantiated CUDA Graph Update with Jacobi Iterative Method using different
> approaches.

> **日本語**
> この項目は `CUDA 10.2` の CUDA Graph 関連の 更新 です。`jacobiCudaGraphs` について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 117

Context: `CUDA 10.2`

English paragraph 117:
> *  Added `cuSolverSp_LinearSolver`. Demonstrates cuSolverSP's LU, QR and Cholesky factorization.

> **日本語**
> この項目は `CUDA 10.2` の CUDA library または CCCL sample に関する 追加 です。`cuSolverSp_LinearSolver`、`cuSolverSP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 118

Context: `CUDA 10.2`

English paragraph 118:
> *  Added `MersenneTwisterGP11213`. Demonstrates the Mersenne Twister random number generator GP11213 in cuRAND.

> **日本語**
> この項目は `CUDA 10.2` の CUDA library または CCCL sample に関する 追加 です。`MersenneTwisterGP11213`、`cuRAND` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 119

Context: `CUDA 10.1 Update 2`

English paragraph 119:
> *  Added `vulkanImageCUDA`. Demonstrates how to perform Vulkan image - CUDA Interop.

> **日本語**
> この変更履歴項目は `CUDA 10.1 Update 2` で 追加 された内容です。`vulkanImageCUDA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 120

Context: `CUDA 10.1 Update 2`

English paragraph 120:
> *  Added `nvJPEG_encoder`. Demonstrates encoding of jpeg images using NVJPEG Library.

> **日本語**
> この項目は `CUDA 10.1 Update 2` の CUDA library または CCCL sample に関する 追加 です。`nvJPEG_encoder` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 121

Context: `CUDA 10.1 Update 2`

English paragraph 121:
> *  Added Windows OS support to `nvJPEG` sample.

> **日本語**
> この項目は `CUDA 10.1 Update 2` の CUDA library または CCCL sample に関する 追加 です。`Windows`、`nvJPEG` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 122

Context: `CUDA 10.1 Update 2`

English paragraph 122:
> *  Added `boxFilterNPP`. Demonstrates how to use NPP FilterBox function to perform a box filter.

> **日本語**
> この項目は `CUDA 10.1 Update 2` の CUDA library または CCCL sample に関する 追加 です。`boxFilterNPP`、`NPP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 123

Context: `CUDA 10.1 Update 2`

English paragraph 123:
> *  Added `cannyEdgeDetectorNPP`. Demonstrates the nppiFilterCannyBorder_8u_C1R Canny Edge Detection image filter
> function.

> **日本語**
> この項目は `CUDA 10.1 Update 2` の CUDA library または CCCL sample に関する 追加 です。`cannyEdgeDetectorNPP` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 124

Context: `CUDA 10.1 Update 1`

English paragraph 124:
> *  Added `NV12toBGRandResize`. Demonstrates how to convert and resize NV12 frames to BGR planars frames using CUDA in
> batch.

> **日本語**
> この変更履歴項目は `CUDA 10.1 Update 1` で 追加 された内容です。`NV12toBGRandResize` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 125

Context: `CUDA 10.1 Update 1`

English paragraph 125:
> *  Added `EGLStream_CUDA_Interop`. Demonstrates data exchange between CUDA and EGL Streams.

> **日本語**
> この変更履歴項目は `CUDA 10.1 Update 1` で 追加 された内容です。`EGLStream_CUDA_Interop` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 126

Context: `CUDA 10.1 Update 1`

English paragraph 126:
> *  Added `cuSolverDn_LinearSolver`. Demonstrates cuSolverDN's LU, QR and Cholesky factorization.

> **日本語**
> この項目は `CUDA 10.1 Update 1` の CUDA library または CCCL sample に関する 追加 です。`cuSolverDn_LinearSolver`、`cuSolverDN` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 127

Context: `CUDA 10.1 Update 1`

English paragraph 127:
> *  Added support of Visual Studio 2019 to all samples supported on [Windows](./README.md#windows-1).

> **日本語**
> この変更履歴項目は `CUDA 10.1 Update 1` で 追加 された内容です。`Visual Studio`、`Windows`、`README.md` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 128

Context: `CUDA 10.1`

English paragraph 128:
> *  Added `immaTensorCoreGemm`. Demonstrates integer GEMM computation using the Warp Matrix Multiply and Accumulate
> (WMMA) API for integers employing the Tensor Cores.

> **日本語**
> この変更履歴項目は `CUDA 10.1` で 追加 された内容です。`immaTensorCoreGemm`、`WMMA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 129

Context: `CUDA 10.1`

English paragraph 129:
> *  Added `simpleIPC`. Demonstrates Inter Process Communication with one process per GPU for computation.

> **日本語**
> この変更履歴項目は `CUDA 10.1` で 追加 された内容です。`simpleIPC` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 130

Context: `CUDA 10.1`

English paragraph 130:
> *  Added `nvJPEG`. Demonstrates single and batched decoding of jpeg images using NVJPEG Library.

> **日本語**
> この項目は `CUDA 10.1` の CUDA library または CCCL sample に関する 追加 です。`nvJPEG` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 131

Context: `CUDA 10.1`

English paragraph 131:
> *  Added `bandwidthTest`. It measures the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.

> **日本語**
> この項目は `CUDA 10.1` の memory API または memory sample に関する 追加 です。`bandwidthTest` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 132

Context: `CUDA 10.1`

English paragraph 132:
> *  Added `reduction`. Demonstrates several important optimization strategies for Data-Parallel Algorithms like
> reduction.

> **日本語**
> この変更履歴項目は `CUDA 10.1` で 追加 された内容です。`reduction` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 133

Context: `CUDA 10.1`

English paragraph 133:
> *  Update all the samples to support CUDA 10.1.

> **日本語**
> この変更履歴項目は `CUDA 10.1` で 更新 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 134

Context: `CUDA 10.0`

English paragraph 134:
> *  Added `simpleCudaGraphs`. Demonstrates CUDA Graphs creation, instantiation and launch using Graphs APIs and Stream
> Capture APIs.

> **日本語**
> この項目は `CUDA 10.0` の CUDA Graph 関連の 追加 です。`simpleCudaGraphs` について、capture、node、instantiate、launch、buffer lifetime のどこが変わったかを追います。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 135

Context: `CUDA 10.0`

English paragraph 135:
> *  Added `conjugateGradientCudaGraphs`. Demonstrates conjugate gradient solver on GPU using CUBLAS and CUSPARSE
> library calls captured and called using CUDA Graph APIs.

> **日本語**
> この項目は `CUDA 10.0` の CUDA library または CCCL sample に関する 追加 です。`conjugateGradientCudaGraphs`、`CUBLAS` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 136

Context: `CUDA 10.0`

English paragraph 136:
> *  Added `simpleVulkan`. Demonstrates Vulkan - CUDA Interop.

> **日本語**
> この変更履歴項目は `CUDA 10.0` で 追加 された内容です。`simpleVulkan` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 137

Context: `CUDA 10.0`

English paragraph 137:
> *  Added `simpleD3D12`. Demonstrates DX12 - CUDA Interop.

> **日本語**
> この変更履歴項目は `CUDA 10.0` で 追加 された内容です。`simpleD3D12` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 138

Context: `CUDA 10.0`

English paragraph 138:
> *  Added `UnifiedMemoryPerf`. Demonstrates performance comparision of various memory types involved in system.

> **日本語**
> この項目は `CUDA 10.0` の memory API または memory sample に関する 追加 です。`UnifiedMemoryPerf` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 139

Context: `CUDA 10.0`

English paragraph 139:
> *  Added `p2pBandwidthLatencyTest`. Demonstrates Peer-To-Peer (P2P) data transfers between pairs of GPUs and computes
> latency and bandwidth.

> **日本語**
> この変更履歴項目は `CUDA 10.0` で 追加 された内容です。`p2pBandwidthLatencyTest` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 140

Context: `CUDA 10.0`

English paragraph 140:
> *  Added `systemWideAtomics`. Demonstrates system wide atomic instructions.

> **日本語**
> この変更履歴項目は `CUDA 10.0` で 追加 された内容です。`systemWideAtomics` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 141

Context: `CUDA 10.0`

English paragraph 141:
> *  Added `simpleCUBLASXT`. Demonstrates CUBLAS-XT library which performs GEMM operations over multiple GPUs.

> **日本語**
> この項目は `CUDA 10.0` の CUDA library または CCCL sample に関する 追加 です。`simpleCUBLASXT`、`CUBLAS` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 142

Context: `CUDA 10.0`

English paragraph 142:
> *  Added Windows OS support to `conjugateGradientMultiDeviceCG` sample.

> **日本語**
> この変更履歴項目は `CUDA 10.0` で 追加 された内容です。`Windows`、`conjugateGradientMultiDeviceCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 143

Context: `CUDA 10.0`

English paragraph 143:
> *  Removed support of Visual Studio 2010 from all samples.

> **日本語**
> この項目は `CUDA 10.0` で古い API、sample、または platform support が整理されたことを示します。`Visual Studio` が現在の build で使える前提か確認します。
> **学習メモ**
> 古い CUDA Toolkit 向けの説明と現在の source がずれる可能性があります。compatibility note と actual API call の両方を確認します。

## Source Paragraph 144

Context: `CUDA 9.2`

English paragraph 144:
> This is the first release of CUDA Samples on GitHub:

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 変更 された内容です。主要な語句 を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 145

Context: `CUDA 9.2`

English paragraph 145:
> *  Added `vectorAdd_nvrtc`. Demonstrates runtime compilation library using NVRTC of a simple vectorAdd kernel.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`vectorAdd_nvrtc`、`NVRTC` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 146

Context: `CUDA 9.2`

English paragraph 146:
> *  Added `warpAggregatedAtomicsCG`. Demonstrates warp aggregated atomics using Cooperative Groups.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`warpAggregatedAtomicsCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 147

Context: `CUDA 9.2`

English paragraph 147:
> *  Added `deviceQuery`. Enumerates the properties of the CUDA devices present in the system.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`deviceQuery` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 148

Context: `CUDA 9.2`

English paragraph 148:
> *  Added `matrixMul`. Demonstrates a matrix multiplication using shared memory through tiled approach.

> **日本語**
> この項目は `CUDA 9.2` の CUDA Tile sample に関する 追加 です。`matrixMul` を見て、tile 単位の execution model、data movement、autotuning のどこを学ぶ sample か確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 149

Context: `CUDA 9.2`

English paragraph 149:
> *  Added `matrixMulDrv`. Demonstrates a matrix multiplication using shared memory through tiled approach, uses CUDA
> Driver API.

> **日本語**
> この項目は `CUDA 9.2` の CUDA Tile sample に関する 追加 です。`matrixMulDrv` を見て、tile 単位の execution model、data movement、autotuning のどこを学ぶ sample か確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 150

Context: `CUDA 9.2`

English paragraph 150:
> *  Added `cudaTensorCoreGemm`. Demonstrates a GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA)
> API introduced in CUDA 9, as well as the new Tensor Cores introduced in the Volta chip family.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`cudaTensorCoreGemm`、`WMMA` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 151

Context: `CUDA 9.2`

English paragraph 151:
> *  Added `simpleVoteIntrinsics` which uses *_sync equivalent of the vote intrinsics _any, _all added since CUDA 9.0.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`simpleVoteIntrinsics` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 152

Context: `CUDA 9.2`

English paragraph 152:
> *  Added `shfl_scan` which uses *_sync equivalent of the shfl intrinsics added since CUDA 9.0.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`shfl_scan` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 153

Context: `CUDA 9.2`

English paragraph 153:
> *  Added `conjugateGradientMultiBlockCG`. Demonstrates a conjugate gradient solver on GPU using Multi Block
> Cooperative Groups.

> **日本語**
> この変更履歴項目は `CUDA 9.2` で 追加 された内容です。`conjugateGradientMultiBlockCG` を sample path、API 名、toolkit version の対応表として読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 154

Context: `CUDA 9.2`

English paragraph 154:
> *  Added `conjugateGradientMultiDeviceCG`. Demonstrates a conjugate gradient solver on multiple GPUs using Multi
> Device Cooperative Groups, also uses unified memory prefetching and usage hints APIs.

> **日本語**
> この項目は `CUDA 9.2` の memory API または memory sample に関する 追加 です。`conjugateGradientMultiDeviceCG` を見て、ownership、transfer direction、migration、解放責任を確認します。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 155

Context: `CUDA 9.2`

English paragraph 155:
> *  Added `simpleCUBLAS`. Demonstrates how perform GEMM operations using CUBLAS library.

> **日本語**
> この項目は `CUDA 9.2` の CUDA library または CCCL sample に関する 追加 です。`simpleCUBLAS`、`CUBLAS` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Source Paragraph 156

Context: `CUDA 9.2`

English paragraph 156:
> *  Added `simpleCUFFT`. Demonstrates how perform FFT operations using CUFFT library.

> **日本語**
> この項目は `CUDA 9.2` の CUDA library または CCCL sample に関する 追加 です。`simpleCUFFT`、`CUFFT` では handle、descriptor、workspace、algorithm contract、library version dependency を重点的に読みます。
> **学習メモ**
> 新規 sample は README、CMakeLists、source、expected output の四点を一緒に読むと、学習対象の CUDA concept が見えます。

## Cross References

English anchor: related Japanese study material for this repository.

> **日本語**
> 関連する sample ごとの `README.ja.md`、`docs_ja/themes/`、`docs_ja/glossary/`、source 内の `JP:` コメントを合わせて読むと、本文の build/run 手順と CUDA concept を接続できます。
> **学習メモ**
> この file は文書の伴走資料です。behavior、build graph、test output を変える目的の変更ではありません。
