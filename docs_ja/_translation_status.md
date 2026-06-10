# Japanese Translation Status

English inventory for the local Japanese learning overlay.

> **日本語**
> CUDA Samples の日本語学習用 README、companion docs、`JP:` 注釈を、アンカー位置まで含めて棚卸しします。
>
> **学習メモ**
> `DONE` は対象ファイルがそろい、重要な CUDA/API アンカーの近くに具体的な `JP:` 注釈がある状態です。`PARTIAL` は、トップだけの一般コメントやアンカー不足が残っている状態です。

- Status: PARTIAL
- Updated: 2026-06-10 05:26:01 UTC
- Branch: `ja-study/local-annotations`
- Base commit: `b7c5481c556c3fe98db060207ecaa41a4b9a9abc`
- Current commit: `ab7afc22d6d06ac5a57b8c30ece1eed0785825cc`
- Detailed annotation record: `docs_ja/_annotation_inventory.json`
- Detailed sample README record: `docs_ja/_sample_readme_inventory.json`
- Detailed theme guide record: `docs_ja/_theme_inventory.json`
- Detailed companion record: `docs_ja/_companion_inventory.json`
- Detailed glossary record: `docs_ja/_glossary_inventory.json`

## Counts

- source_docs: 557
- doc_companions_done: 557
- major_companions_done: 1
- major_companions_partial: 4
- major_companion_source_paragraphs: 240
- major_companion_english_blocks: 24
- major_companion_jp_blocks: 28
- sample_dirs: 255
- sample_readme_ja_done: 255
- sample_readme_ja_partial: 0
- theme_guides: 16
- theme_guides_done: 16
- theme_guides_partial: 0
- glossary_files: 5
- glossary_files_done: 5
- glossary_files_partial: 0
- annotation_files: 923
- annotation_files_done: 571
- annotation_files_partial: 352
- annotation_anchor_files: 645
- annotation_anchor_instances: 9822
- annotation_anchor_instances_covered: 4433
- annotation_anchor_instances_missing: 5389
- annotation_grouped_anchor_instances: 1344
- annotation_inaccurate_comment_flags: 46
- japanese_files: 842

## Missing Or Partial Items

### doc_companions
- None

### major_companions
- `docs_ja/translated/README.md.ja.md`
- `docs_ja/translated/CHANGELOG.md.ja.md`
- `docs_ja/translated/CONTRIBUTING.md.ja.md`
- `docs_ja/translated/CMakeLists.txt.ja.md`

### sample_readme_ja
- None

### theme_guides
- None

### glossary
- None

### annotations
- `Common/UtilNPP/ImageAllocatorsNPP.h`
- `Common/UtilNPP/SignalAllocatorsNPP.h`
- `Common/helper_cuda.h`
- `Common/helper_cuda_drvapi.h`
- `Common/helper_cusolver.h`
- `Common/helper_image.h`
- `Common/nvrtc_helper.h`
- `Common/rendercheck_gl.h`
- `Common/rendercheck_gles.h`
- `cmake/CPM.cmake`
- `cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu`
- `cpp/0_Introduction/asyncAPI/asyncAPI.cu`
- `cpp/0_Introduction/clock/clock.cu`
- `cpp/0_Introduction/clock_nvrtc/clock_kernel.cu`
- `cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu`
- `cpp/0_Introduction/matrixMul/matrixMul.cu`
- `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp`
- `cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu`
- `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.c`
- `cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h`
- `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp`
- `cpp/0_Introduction/matrixMul_nvrtc/matrixMul.cpp`
- `cpp/0_Introduction/matrixMul_nvrtc/matrixMul_kernel.cu`
- `cpp/0_Introduction/mergeSort/bitonic.cu`
- `cpp/0_Introduction/mergeSort/main.cpp`
- `cpp/0_Introduction/mergeSort/mergeSort.cu`
- `cpp/0_Introduction/mergeSort/mergeSort_validate.cpp`
- `cpp/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu`
- `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu`
- `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics.cpp`
- `cpp/0_Introduction/simpleAttributes/simpleAttributes.cu`
- `cpp/0_Introduction/simpleCUDA2GL/main.cpp`
- `cpp/0_Introduction/simpleCallback/simpleCallback.cu`
- `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`
- `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp`
- `cpp/0_Introduction/simpleHyperQ/simpleHyperQ.cu`
- `cpp/0_Introduction/simpleIPC/simpleIPC.cu`
- `cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu`
- `cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu`
- `cpp/0_Introduction/simpleMultiGPU/simpleMultiGPU.cu`
- `cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu`
- `cpp/0_Introduction/simpleP2P/simpleP2P.cu`
- `cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu`
- `cpp/0_Introduction/simpleStreams/simpleStreams.cu`
- `cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu`
- `cpp/0_Introduction/simpleTemplates/sharedmem.cuh`
- `cpp/0_Introduction/simpleTemplates/simpleTemplates.cu`
- `cpp/0_Introduction/simpleTexture/simpleTexture.cu`
- `cpp/0_Introduction/simpleTexture3D/simpleTexture3D.cpp`
- `cpp/0_Introduction/simpleTextureDrv/simpleTextureDrv.cpp`
- `cpp/0_Introduction/simpleVoteIntrinsics/simpleVoteIntrinsics.cu`
- `cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh`
- `cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu`
- `cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu`
- `cpp/0_Introduction/template/template.cu`
- `cpp/0_Introduction/vectorAdd/vectorAdd.cu`
- `cpp/0_Introduction/vectorAddDrv/vectorAddDrv.cpp`
- `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.cpp`
- `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp`
- `cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd.cpp`
- `cpp/1_Utilities/deviceQuery/deviceQuery.cpp`
- `cpp/1_Utilities/deviceQueryDrv/deviceQueryDrv.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.h`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.h`
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/main.cpp`
- `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers.cpp`
- `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers_kernels.cu`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/src/piestimator.cu`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/src/piestimator.cu`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu`
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/src/piestimator.cu`
- `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/pricingengine.cu`
- `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp`
- `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu`
- `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu`
- `cpp/2_Concepts_and_Techniques/convolutionSeparable/main.cpp`
- `cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture.cu`
- `cpp/2_Concepts_and_Techniques/convolutionTexture/main.cpp`
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu`
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh`
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh`
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_quantization.cuh`
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu`
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu`
- `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu`
- `cpp/2_Concepts_and_Techniques/histogram/histogram256.cu`
- `cpp/2_Concepts_and_Techniques/histogram/histogram64.cu`
- `cpp/2_Concepts_and_Techniques/histogram/main.cpp`
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp`
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_copy_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_knn_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm2_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/inlinePTX_nvrtc/inlinePTX_kernel.cu`
- `cpp/2_Concepts_and_Techniques/interval/interval.cu`
- `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp`
- `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu`
- `cpp/2_Concepts_and_Techniques/particles/particles.cpp`
- `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh`
- `cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu`
- `cpp/2_Concepts_and_Techniques/reduction/reduction.cpp`
- `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`
- `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu`
- `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd.cu`
- `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/scan/main.cpp`
- `cpp/2_Concepts_and_Techniques/scan/scan.cu`
- `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/kernels.cuh`
- `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu`
- `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh`
- `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu`
- `cpp/2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu`
- `cpp/2_Concepts_and_Techniques/sortingNetworks/main.cpp`
- `cpp/2_Concepts_and_Techniques/sortingNetworks/oddEvenMergeSort.cu`
- `cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_validate.cpp`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu`
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P/streamOrderedAllocationP2P.cu`
- `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction.cu`
- `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh`
- `cpp/2_Concepts_and_Techniques/threadMigration/threadMigration.cpp`
- `cpp/2_Concepts_and_Techniques/threadMigration/threadMigration_kernel.cu`
- `cpp/3_CUDA_Features/StreamPriorities/StreamPriorities.cu`
- `cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu`
- `cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu`
- `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp`
- `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu`
- `cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu`
- `cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpBitonicSort.cu`
- `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu`
- `cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu`
- `cpp/3_CUDA_Features/cdpSimplePrint/cdpSimplePrint.cu`
- `cpp/3_CUDA_Features/cdpSimpleQuicksort/cdpSimpleQuicksort.cu`
- `cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp`
- `cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu`
- `cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu`
- `cpp/3_CUDA_Features/dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu`
- `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- `cpp/3_CUDA_Features/graphConditionalNodes/graphConditionalNodes.cu`
- `cpp/3_CUDA_Features/graphMemoryFootprint/graphMemoryFootprint.cu`
- `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu`
- `cpp/3_CUDA_Features/immaTensorCoreGemm/immaTensorCoreGemm.cu`
- `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu`
- `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp`
- `cpp/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp`
- `cpp/3_CUDA_Features/newdelete/newdelete.cu`
- `cpp/3_CUDA_Features/ptxjit/ptxjit.cpp`
- `cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu`
- `cpp/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu`
- `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu`
- `cpp/4_CUDA_Libraries/FilterBorderControlNPP/FilterBorderControlNPP.cpp`
- `cpp/4_CUDA_Libraries/MersenneTwisterGP11213/MersenneTwister.cpp`
- `cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.cpp`
- `cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.h`
- `cpp/4_CUDA_Libraries/boxFilterNPP/boxFilterNPP.cpp`
- `cpp/4_CUDA_Libraries/cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp`
- `cpp/4_CUDA_Libraries/conjugateGradient/main.cpp`
- `cpp/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu`
- `cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu`
- `cpp/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu`
- `cpp/4_CUDA_Libraries/conjugateGradientPrecond/main.cpp`
- `cpp/4_CUDA_Libraries/conjugateGradientUM/main.cpp`
- `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp`
- `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/mmio_wrapper.cpp`
- `cpp/4_CUDA_Libraries/cuSolverRf/cuSolverRf.cpp`
- `cpp/4_CUDA_Libraries/cuSolverRf/mmio_wrapper.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/mmio_wrapper.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/cuSolverSp_LowlevelQR.cpp`
- `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/mmio_wrapper.cpp`
- `cpp/4_CUDA_Libraries/cubDeviceFind/cubDeviceFind.cu`
- `cpp/4_CUDA_Libraries/cubDeviceSegmentedScan/cubDeviceSegmentedScan.cu`
- `cpp/4_CUDA_Libraries/cubDeviceTransform/cubDeviceTransform.cu`
- `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp`
- `cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu`
- `cpp/4_CUDA_Libraries/freeImageInteropNPP/freeImageInteropNPP.cpp`
- `cpp/4_CUDA_Libraries/histEqualizationNPP/histEqualizationNPP.cpp`
- `cpp/4_CUDA_Libraries/jitLto/jitLto.cpp`
- `cpp/4_CUDA_Libraries/libcuxxMdspan/libcuxxMdspan.cu`
- `cpp/4_CUDA_Libraries/lineOfSight/lineOfSight.cu`
- `cpp/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp`
- `cpp/4_CUDA_Libraries/nvJPEG/nvJPEG.cpp`
- `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp`
- `cpp/4_CUDA_Libraries/oceanFFT/oceanFFT.cpp`
- `cpp/4_CUDA_Libraries/oceanFFT/oceanFFT_kernel.cu`
- ... plus 152 more

## Annotation Quality

- DONE files: 571
- PARTIAL files: 352
- Anchor search window: 4 lines before to 2 lines after each anchor.
- Grouping window: adjacent repeated anchors up to 8 lines apart, with a useful JP comment explaining the grouped resource/direction/cleanup pattern.
- Covered anchor instances: 4433
- Missing anchor instances: 5389
- Grouped anchor instances: 1344
- Inaccurate JP flags: 46

### First PARTIAL Annotation Records

- `Common/UtilNPP/ImageAllocatorsNPP.h`: jp_count=4, anchors=91, covered=2, missing_count=89, grouped=0, inaccurate_flags=0, missing=[library_resources, transfer], reason=anchor instances lack nearby JP or valid grouping
- `Common/UtilNPP/SignalAllocatorsNPP.h`: jp_count=3, anchors=65, covered=1, missing_count=64, grouped=0, inaccurate_flags=0, missing=[library_resources, transfer], reason=anchor instances lack nearby JP or valid grouping
- `Common/helper_cuda.h`: jp_count=3, anchors=7, covered=2, missing_count=5, grouped=0, inaccurate_flags=0, missing=[library_resources], reason=anchor instances lack nearby JP or valid grouping
- `Common/helper_cuda_drvapi.h`: jp_count=2, anchors=16, covered=1, missing_count=15, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `Common/helper_cusolver.h`: jp_count=2, anchors=4, covered=2, missing_count=2, grouped=1, inaccurate_flags=0, missing=[library_resources], reason=anchor instances lack nearby JP or valid grouping
- `Common/helper_image.h`: jp_count=3, anchors=11, covered=0, missing_count=11, grouped=0, inaccurate_flags=0, missing=[validation], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping
- `Common/nvrtc_helper.h`: jp_count=4, anchors=14, covered=6, missing_count=8, grouped=3, inaccurate_flags=0, missing=[driver_api, nvrtc], reason=anchor instances lack nearby JP or valid grouping
- `Common/rendercheck_gl.h`: jp_count=3, anchors=6, covered=1, missing_count=5, grouped=0, inaccurate_flags=0, missing=[validation], reason=anchor instances lack nearby JP or valid grouping
- `Common/rendercheck_gles.h`: jp_count=3, anchors=6, covered=1, missing_count=5, grouped=0, inaccurate_flags=0, missing=[validation], reason=anchor instances lack nearby JP or valid grouping
- `cmake/CPM.cmake`: jp_count=3, anchors=7, covered=2, missing_count=5, grouped=1, inaccurate_flags=0, missing=[build_cuda], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu`: jp_count=6, anchors=42, covered=10, missing_count=32, grouped=0, inaccurate_flags=0, missing=[cleanup, library_resources, managed_memory, streams_events, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/asyncAPI/asyncAPI.cu`: jp_count=9, anchors=19, covered=17, missing_count=2, grouped=0, inaccurate_flags=0, missing=[streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/clock/clock.cu`: jp_count=8, anchors=17, covered=14, missing_count=3, grouped=2, inaccurate_flags=2, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/clock_nvrtc/clock_kernel.cu`: jp_count=4, anchors=8, covered=5, missing_count=3, grouped=2, inaccurate_flags=2, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu`: jp_count=9, anchors=71, covered=18, missing_count=53, grouped=4, inaccurate_flags=2, missing=[device_memory, indexing, kernel_launch, shared_memory, sync, transfer], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/matrixMul/matrixMul.cu`: jp_count=11, anchors=40, covered=24, missing_count=16, grouped=7, inaccurate_flags=2, missing=[kernel_launch, pinned_memory, streams_events, sync, transfer], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp`: jp_count=7, anchors=21, covered=11, missing_count=10, grouped=4, inaccurate_flags=0, missing=[device_memory, driver_api, kernel_launch, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu`: jp_count=4, anchors=8, covered=6, missing_count=2, grouped=3, inaccurate_flags=2, missing=[sync], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.c`: jp_count=6, anchors=74, covered=26, missing_count=48, grouped=16, inaccurate_flags=0, missing=[cleanup, device_memory, driver_api, kernel_launch, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h`: jp_count=2, anchors=16, covered=0, missing_count=16, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp`: jp_count=6, anchors=34, covered=7, missing_count=27, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/matrixMul_nvrtc/matrixMul.cpp`: jp_count=7, anchors=12, covered=7, missing_count=5, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/matrixMul_nvrtc/matrixMul_kernel.cu`: jp_count=4, anchors=8, covered=7, missing_count=1, grouped=3, inaccurate_flags=0, missing=[sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/mergeSort/bitonic.cu`: jp_count=6, anchors=56, covered=6, missing_count=50, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/mergeSort/main.cpp`: jp_count=6, anchors=20, covered=18, missing_count=2, grouped=3, inaccurate_flags=0, missing=[transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/mergeSort/mergeSort.cu`: jp_count=9, anchors=78, covered=29, missing_count=49, grouped=13, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/mergeSort/mergeSort_validate.cpp`: jp_count=3, anchors=2, covered=1, missing_count=1, grouped=0, inaccurate_flags=0, missing=[validation], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu`: jp_count=9, anchors=28, covered=17, missing_count=11, grouped=2, inaccurate_flags=0, missing=[device_memory, indexing, shared_memory, streams_events, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu`: jp_count=8, anchors=10, covered=9, missing_count=1, grouped=0, inaccurate_flags=0, missing=[streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics.cpp`: jp_count=6, anchors=7, covered=6, missing_count=1, grouped=0, inaccurate_flags=0, missing=[driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleAttributes/simpleAttributes.cu`: jp_count=10, anchors=25, covered=21, missing_count=4, grouped=0, inaccurate_flags=2, missing=[streams_events, sync], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/simpleCUDA2GL/main.cpp`: jp_count=5, anchors=18, covered=5, missing_count=13, grouped=2, inaccurate_flags=0, missing=[device_memory, graphs], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleCallback/simpleCallback.cu`: jp_count=8, anchors=15, covered=12, missing_count=3, grouped=2, inaccurate_flags=0, missing=[streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`: jp_count=8, anchors=14, covered=8, missing_count=6, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp`: jp_count=10, anchors=28, covered=21, missing_count=7, grouped=3, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleHyperQ/simpleHyperQ.cu`: jp_count=10, anchors=30, covered=24, missing_count=6, grouped=9, inaccurate_flags=0, missing=[indexing, streams_events, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleIPC/simpleIPC.cu`: jp_count=10, anchors=24, covered=17, missing_count=7, grouped=2, inaccurate_flags=0, missing=[cleanup, device_memory, streams_events, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu`: jp_count=8, anchors=14, covered=8, missing_count=6, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu`: jp_count=9, anchors=49, covered=15, missing_count=34, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, kernel_launch, pinned_memory, streams_events, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleMultiGPU/simpleMultiGPU.cu`: jp_count=9, anchors=18, covered=15, missing_count=3, grouped=0, inaccurate_flags=0, missing=[cleanup, pinned_memory, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu`: jp_count=9, anchors=17, covered=14, missing_count=3, grouped=2, inaccurate_flags=0, missing=[indexing, streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleP2P/simpleP2P.cu`: jp_count=9, anchors=25, covered=21, missing_count=4, grouped=5, inaccurate_flags=0, missing=[kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu`: jp_count=9, anchors=33, covered=16, missing_count=17, grouped=2, inaccurate_flags=0, missing=[cleanup, device_memory, indexing, kernel_launch, streams_events, sync, transfer, validation], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleStreams/simpleStreams.cu`: jp_count=9, anchors=44, covered=15, missing_count=29, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, kernel_launch, pinned_memory, streams_events, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu`: jp_count=9, anchors=19, covered=14, missing_count=5, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleTemplates/sharedmem.cuh`: jp_count=2, anchors=23, covered=1, missing_count=22, grouped=0, inaccurate_flags=0, missing=[shared_memory], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleTemplates/simpleTemplates.cu`: jp_count=9, anchors=18, covered=9, missing_count=9, grouped=1, inaccurate_flags=2, missing=[device_memory, sync, transfer, validation], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/simpleTexture/simpleTexture.cu`: jp_count=9, anchors=14, covered=11, missing_count=3, grouped=0, inaccurate_flags=0, missing=[kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleTexture3D/simpleTexture3D.cpp`: jp_count=7, anchors=12, covered=6, missing_count=6, grouped=0, inaccurate_flags=0, missing=[graphs], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleTextureDrv/simpleTextureDrv.cpp`: jp_count=7, anchors=21, covered=11, missing_count=10, grouped=3, inaccurate_flags=0, missing=[driver_api, kernel_launch, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleVoteIntrinsics/simpleVoteIntrinsics.cu`: jp_count=6, anchors=20, covered=9, missing_count=11, grouped=1, inaccurate_flags=0, missing=[device_memory, kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh`: jp_count=2, anchors=3, covered=1, missing_count=2, grouped=0, inaccurate_flags=0, missing=[indexing], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu`: jp_count=6, anchors=15, covered=6, missing_count=9, grouped=1, inaccurate_flags=0, missing=[cleanup, pinned_memory], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu`: jp_count=7, anchors=6, covered=5, missing_count=1, grouped=1, inaccurate_flags=0, missing=[managed_memory], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/template/template.cu`: jp_count=9, anchors=13, covered=9, missing_count=4, grouped=1, inaccurate_flags=2, missing=[device_memory, sync, transfer], reason=anchor instances lack nearby JP or valid grouping; misleading or inaccurate JP comments
- `cpp/0_Introduction/vectorAdd/vectorAdd.cu`: jp_count=6, anchors=11, covered=8, missing_count=3, grouped=3, inaccurate_flags=0, missing=[device_memory, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/vectorAddDrv/vectorAddDrv.cpp`: jp_count=7, anchors=18, covered=13, missing_count=5, grouped=1, inaccurate_flags=0, missing=[driver_api, kernel_launch, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.cpp`: jp_count=2, anchors=6, covered=2, missing_count=4, grouped=1, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp`: jp_count=6, anchors=15, covered=7, missing_count=8, grouped=0, inaccurate_flags=0, missing=[driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd.cpp`: jp_count=6, anchors=12, covered=11, missing_count=1, grouped=2, inaccurate_flags=0, missing=[driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/1_Utilities/deviceQuery/deviceQuery.cpp`: jp_count=5, anchors=2, covered=2, missing_count=0, grouped=0, inaccurate_flags=2, missing=[none], reason=misleading or inaccurate JP comments
- `cpp/1_Utilities/deviceQueryDrv/deviceQueryDrv.cpp`: jp_count=5, anchors=18, covered=14, missing_count=4, grouped=10, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp`: jp_count=3, anchors=3, covered=0, missing_count=3, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp`: jp_count=5, anchors=9, covered=3, missing_count=6, grouped=0, inaccurate_flags=0, missing=[device_memory, driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h`: jp_count=3, anchors=2, covered=1, missing_count=1, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu`: jp_count=8, anchors=19, covered=12, missing_count=7, grouped=1, inaccurate_flags=0, missing=[indexing, kernel_launch, streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp`: jp_count=4, anchors=3, covered=0, missing_count=3, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp`: jp_count=4, anchors=5, covered=1, missing_count=4, grouped=0, inaccurate_flags=0, missing=[driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.h`: jp_count=2, anchors=2, covered=1, missing_count=1, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp`: jp_count=5, anchors=18, covered=13, missing_count=5, grouped=9, inaccurate_flags=0, missing=[driver_api, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.h`: jp_count=2, anchors=2, covered=1, missing_count=1, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/main.cpp`: jp_count=2, anchors=11, covered=0, missing_count=11, grouped=0, inaccurate_flags=0, missing=[driver_api], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers.cpp`: jp_count=7, anchors=11, covered=5, missing_count=6, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, graphs], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers_kernels.cu`: jp_count=8, anchors=39, covered=13, missing_count=26, grouped=3, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/src/piestimator.cu`: jp_count=9, anchors=29, covered=13, missing_count=16, grouped=1, inaccurate_flags=0, missing=[device_memory, indexing, library_resources, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/src/piestimator.cu`: jp_count=9, anchors=46, covered=17, missing_count=29, grouped=3, inaccurate_flags=0, missing=[device_memory, indexing, library_resources, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu`: jp_count=9, anchors=31, covered=24, missing_count=7, grouped=13, inaccurate_flags=0, missing=[device_memory, indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/src/piestimator.cu`: jp_count=9, anchors=35, covered=28, missing_count=7, grouped=18, inaccurate_flags=0, missing=[device_memory, indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/pricingengine.cu`: jp_count=9, anchors=35, covered=15, missing_count=20, grouped=4, inaccurate_flags=0, missing=[device_memory, indexing, library_resources, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp`: jp_count=7, anchors=17, covered=6, missing_count=11, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, graphs, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu`: jp_count=7, anchors=25, covered=15, missing_count=10, grouped=4, inaccurate_flags=0, missing=[indexing, kernel_launch, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu`: jp_count=6, anchors=20, covered=6, missing_count=14, grouped=1, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/convolutionSeparable/main.cpp`: jp_count=5, anchors=10, covered=8, missing_count=2, grouped=0, inaccurate_flags=0, missing=[sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture.cu`: jp_count=3, anchors=6, covered=3, missing_count=3, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/convolutionTexture/main.cpp`: jp_count=5, anchors=12, covered=7, missing_count=5, grouped=1, inaccurate_flags=0, missing=[sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu`: jp_count=6, anchors=37, covered=8, missing_count=29, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, kernel_launch, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh`: jp_count=4, anchors=18, covered=8, missing_count=10, grouped=2, inaccurate_flags=0, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh`: jp_count=4, anchors=20, covered=8, missing_count=12, grouped=2, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_quantization.cuh`: jp_count=3, anchors=10, covered=5, missing_count=5, grouped=2, inaccurate_flags=0, missing=[indexing], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh`: jp_count=4, anchors=18, covered=9, missing_count=9, grouped=3, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh`: jp_count=4, anchors=44, covered=12, missing_count=32, grouped=7, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh`: jp_count=4, anchors=27, covered=15, missing_count=12, grouped=10, inaccurate_flags=0, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh`: jp_count=4, anchors=11, covered=5, missing_count=6, grouped=0, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh`: jp_count=4, anchors=33, covered=17, missing_count=16, grouped=11, inaccurate_flags=0, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu`: jp_count=7, anchors=55, covered=17, missing_count=38, grouped=9, inaccurate_flags=0, missing=[device_memory, kernel_launch, sync, transfer, validation], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu`: jp_count=6, anchors=16, covered=14, missing_count=2, grouped=1, inaccurate_flags=0, missing=[transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu`: jp_count=3, anchors=22, covered=4, missing_count=18, grouped=0, inaccurate_flags=0, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu`: jp_count=5, anchors=6, covered=4, missing_count=2, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/histogram/histogram256.cu`: jp_count=8, anchors=24, covered=11, missing_count=13, grouped=2, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/histogram/histogram64.cu`: jp_count=8, anchors=27, covered=11, missing_count=16, grouped=4, inaccurate_flags=0, missing=[indexing, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/histogram/main.cpp`: jp_count=5, anchors=11, covered=7, missing_count=4, grouped=1, inaccurate_flags=0, missing=[sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp`: jp_count=6, anchors=10, covered=5, missing_count=5, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, graphs], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_copy_kernel.cuh`: jp_count=2, anchors=3, covered=1, missing_count=2, grouped=0, inaccurate_flags=0, missing=[indexing], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_knn_kernel.cuh`: jp_count=3, anchors=6, covered=3, missing_count=3, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm2_kernel.cuh`: jp_count=5, anchors=22, covered=11, missing_count=11, grouped=4, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm_kernel.cuh`: jp_count=3, anchors=6, covered=3, missing_count=3, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/inlinePTX_nvrtc/inlinePTX_kernel.cu`: jp_count=1, anchors=1, covered=0, missing_count=1, grouped=0, inaccurate_flags=0, missing=[indexing], reason=JP comments are only top-level or away from anchors; anchor instances lack nearby JP or valid grouping; generic-only JP comments; generic top-level JP comments for files with anchors
- `cpp/2_Concepts_and_Techniques/interval/interval.cu`: jp_count=7, anchors=16, covered=14, missing_count=2, grouped=0, inaccurate_flags=0, missing=[streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp`: jp_count=5, anchors=5, covered=4, missing_count=1, grouped=0, inaccurate_flags=0, missing=[device_memory], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu`: jp_count=7, anchors=19, covered=15, missing_count=4, grouped=8, inaccurate_flags=0, missing=[kernel_launch, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/particles/particles.cpp`: jp_count=5, anchors=6, covered=4, missing_count=2, grouped=1, inaccurate_flags=0, missing=[indexing], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh`: jp_count=4, anchors=10, covered=5, missing_count=5, grouped=0, inaccurate_flags=0, missing=[indexing], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu`: jp_count=4, anchors=9, covered=8, missing_count=1, grouped=0, inaccurate_flags=0, missing=[streams_events], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/reduction/reduction.cpp`: jp_count=5, anchors=20, covered=5, missing_count=15, grouped=0, inaccurate_flags=0, missing=[cleanup, device_memory, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`: jp_count=5, anchors=172, covered=23, missing_count=149, grouped=18, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu`: jp_count=7, anchors=19, covered=8, missing_count=11, grouped=0, inaccurate_flags=0, missing=[indexing, shared_memory, sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd.cu`: jp_count=6, anchors=12, covered=11, missing_count=1, grouped=0, inaccurate_flags=0, missing=[transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_kernel.cuh`: jp_count=4, anchors=8, covered=4, missing_count=4, grouped=0, inaccurate_flags=0, missing=[indexing, sync], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/scan/main.cpp`: jp_count=5, anchors=11, covered=7, missing_count=4, grouped=1, inaccurate_flags=0, missing=[sync, transfer], reason=anchor instances lack nearby JP or valid grouping
- `cpp/2_Concepts_and_Techniques/scan/scan.cu`: jp_count=8, anchors=21, covered=9, missing_count=12, grouped=0, inaccurate_flags=0, missing=[indexing, kernel_launch, shared_memory, sync], reason=anchor instances lack nearby JP or valid grouping
- ... plus 232 more PARTIAL files

## Sample README Quality

- DONE guides: 255
- PARTIAL guides: 0
- Required sections: `## Purpose`, `## Prerequisites`, `## Files`, `## Execution Flow`, `## Concrete Reading Path`, `## Key APIs And Concepts`, `## Memory, Synchronization, And Performance Notes`, `## Build And Run`, `## Expected Behavior`, `## Common Mistakes`, `## Exercises`, `## Related Themes`

## Theme Guide Quality

- DONE guides: 16
- PARTIAL guides: 0
- Required sections: `## Concept`, `## Why It Matters`, `## Mental Model`, `## API Map`, `## Sample References`, `## Reading Steps`, `## Common Mistakes`, `## Performance Notes`, `## Exercises`, `## Cross-Theme Links`, `## Review Checklist`

## Major Companion Quality

- DONE companions: 1
- PARTIAL companions: 4

### PARTIAL Major Companion Records

- `docs_ja/translated/README.md.ja.md`: lines=78, sections=7, source_paragraphs=48, english_blocks=7, jp_blocks=7, reason=not enough English paragraph/reference blocks for source paragraphs; not enough Japanese blocks for source paragraphs
- `docs_ja/translated/CHANGELOG.md.ja.md`: lines=53, sections=5, source_paragraphs=156, english_blocks=5, jp_blocks=5, reason=not enough English paragraph/reference blocks for source paragraphs; not enough Japanese blocks for source paragraphs
- `docs_ja/translated/CONTRIBUTING.md.ja.md`: lines=53, sections=5, source_paragraphs=25, english_blocks=5, jp_blocks=5, reason=not enough English paragraph/reference blocks for source paragraphs; not enough Japanese blocks for source paragraphs
- `docs_ja/translated/CMakeLists.txt.ja.md`: lines=57, sections=5, source_paragraphs=11, english_blocks=4, jp_blocks=5, reason=not enough English paragraph/reference blocks for source paragraphs; not enough Japanese blocks for source paragraphs

## Glossary Quality

- DONE glossary files: 5
- PARTIAL glossary files: 0
- Required files: `docs_ja/glossary/README.md`, `docs_ja/glossary/terms.md`, `docs_ja/glossary/api.md`, `docs_ja/glossary/memory_transfer.md`, `docs_ja/glossary/build_run.md`

## Policy Notes

- English source text, file names, commands, APIs, expected output, license text, and attribution are preserved.
- Japanese comments use `JP:` and are intended as learning annotations only.
- Vendor/generated support paths are excluded from annotation coverage to avoid changing third-party or generated material.
- Japanese license explanations, if present, are unofficial learning references only; original English license text remains authoritative.
