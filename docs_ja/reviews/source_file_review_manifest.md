# One-by-One Source File Review Manifest

English manifest for the required individual source/build/script review.

> **日本語**
> この manifest は、git-tracked source/build/script file を deterministic path order で 1 file ずつ読み、各 file の review record を `source_file_review.json` に保存したことを示します。
> **学習メモ**
> `tools/inventory_ja.py` の anchor coverage は必要条件ですが、この manifest では各 file の full read、summary、JP comment assessment、edit/no-edit verdict を個別に記録します。

## Summary

- total_review_targets: 1068
- done: 1068
- partial: 0
- blocked: 0
- whole_file_read_true: 1068
- files_with_jp_comments: 923
- jp_comments_assessed_total: 6036
- files_edited_in_this_pass: 4
- files_reviewed_with_no_edit: 1064
- read_only_vendor_generated_targets: 127
- non_utf8_decode_notes: 28
- branch: `ja-study/local-annotations`
- review_pass_base: `8ccd87823bea840e52bd75051a6163fc827cd758`
- source_state_head_at_generation: `c69bb98fa5b9bc959c7f7b0b77bc70a6a7540aa7`
- commit hash note: no `PENDING_SOURCE_REVIEW_COMMIT` values remain in `source_file_review.json`.

## Target Selection

- Included requested extensions: `.cu .cuh .cpp .cc .c .h .hpp .py .cmake CMakeLists.txt .sh .bat .cmd .ps1`.
- Additional source-like helpers included: `.hxx .hlsl .glsl .frag .vert .ptx .ll .bash`.
- Order: deterministic lexicographic order from git-tracked paths.
- Excluded: binary/data/doc/config files that are not source/build/script review targets.

## Edited Files In This Pass

- `cmake/CPM.cmake`: Removed local JP learning comments from vendored CPM helper; behavior/build semantics restored/preserved.
- `cpp/1_Utilities/deviceQueryDrv/deviceQueryDrv.cpp`: Replaced generic kernel/shared-memory/validation JP comments with Driver API device-property query notes.
- `python/1_GettingStarted/deviceQuery/deviceQuery.py`: Replaced generic Python kernel/shared-memory JP comments with CUDA Python device-property query notes.
- `tools/inventory_ja.py`: Excluded vendored CPM helper from anchor inventory so third-party code remains read-only after source review.

## Read-Only Vendor/Generated Targets

- `Common/GL/freeglut.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/freeglut_ext.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/freeglut_std.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/glew.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/glext.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/glut.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/glxew.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/glxext.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/wglew.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `Common/GL/wglext.h`: vendor/read-only OpenGL, GLEW, or freeglut compatibility header; no local JP edits applied.
- `cmake/CPM.cmake`: third-party CPM helper script; local JP learning comments were removed and file is left read-only.
- `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.c`: CUDA Driver API dynamic-link helper/generated-style compatibility table; reviewed read-only unless JP comments were inappropriate.
- `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.h`: CUDA Driver API dynamic-link helper/generated-style compatibility table; reviewed read-only unless JP comments were inappropriate.
- `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink_cuda.h`: CUDA Driver API dynamic-link helper/generated-style compatibility table; reviewed read-only unless JP comments were inappropriate.
- `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/2_Concepts_and_Techniques/interval/boost/config.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/borland_prefix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/borland_suffix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/msvc_prefix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/msvc_suffix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi_prefix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/abi_suffix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/auto_link.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/borland.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/codegear.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/comeau.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/common_edg.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/compaq_cxx.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/digitalmars.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/gcc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/gcc_xml.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/greenhills.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/hp_acc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/intel.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/kai.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/metrowerks.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/mpw.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/pgi.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/sgi_mipspro.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/sunpro_cc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/vacpp.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/visualc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/cmath.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/complex.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/functional.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/memory.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/utility.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/aix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/amigaos.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/beos.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/bsd.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/cygwin.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/hpux.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/irix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/linux.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/macos.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/qnxnto.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/solaris.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/vxworks.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/win32.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/posix_features.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/requires_threads.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/select_compiler_config.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/select_platform_config.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/select_stdlib_config.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/dinkumware.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/libcomo.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/libstdcpp3.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/modena.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/msl.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/roguewave.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/sgi.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/stlport.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/vacpp.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/suffix.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/user.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/config/warning_disable.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/limits.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith2.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith3.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/checking.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/certain.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/explicit.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/lexicographic.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/possible.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/set.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/tribool.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/constants.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/alpha_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/bcc_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/bugs.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/c99_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/c99sub_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/division.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/ia64_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/interval_prototype.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/msvc_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/ppc_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/sparc_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/test_input.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/x86_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/x86gcc_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/ext/integer.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/ext/x86_fast_rounding_control.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/hw_rounding.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/interval.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/io.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/limits.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/policies.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounded_arith.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounded_transc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounding.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/transc.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/utility.hpp`: vendored Boost interval header; reviewed read-only, original non-UTF bytes may appear in attribution/comments.
- `cpp/5_Domain_Specific/simpleD3D12/d3dx12.h`: third-party Direct3D 12 helper header; reviewed read-only.
- `cpp/7_libNVVM/cuda-shared-memory/extern_shared_memory.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/cuda-shared-memory/shared_memory.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/device-side-launch/dsl-gpu64.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/ptxgen/test.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/simple/simple-gpu64.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/syscalls/malloc-free.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/syscalls/vprintf.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.
- `cpp/7_libNVVM/uvmlite/uvmlite64.ll`: sample IR/PTX input file; reviewed as source-like intermediate text and left read-only.

## Executable Non-Source Exclusions

- `bin/win64/Debug/freeglut.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/Debug/glew64.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/RelWithDebInfo/freeglut.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/RelWithDebInfo/glew64.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/Release/freeglut.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/Release/glew64.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/minsizeRel/freeglut.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `bin/win64/minsizeRel/glew64.dll`: tracked executable binary DLL artifact; excluded from source/build/script review
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_f_1.yuv`: tracked raw YUV data artifact; excluded from source/build/script review
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_yuv_f_1.yuv`: tracked raw YUV data artifact; excluded from source/build/script review
- `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_yuv_f_2.yuv`: tracked raw YUV data artifact; excluded from source/build/script review

## Review Records

| # | Status | Path | Lines | JP | Edited | Concepts | Verdict |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | DONE | `CMakeLists.txt` | 36 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 2 | DONE | `Common/GL/freeglut.h` | 22 | 0 | no | host/helper source structure | DONE |
| 3 | DONE | `Common/GL/freeglut_ext.h` | 115 | 0 | no | graphics/interop | DONE |
| 4 | DONE | `Common/GL/freeglut_std.h` | 547 | 0 | no | graphics/interop | DONE |
| 5 | DONE | `Common/GL/glew.h` | 14457 | 0 | no | graphics/interop, shader source | DONE |
| 6 | DONE | `Common/GL/glext.h` | 7125 | 0 | no | graphics/interop, shader source | DONE |
| 7 | DONE | `Common/GL/glut.h` | 597 | 0 | no | graphics/interop | DONE |
| 8 | DONE | `Common/GL/glxew.h` | 1121 | 0 | no | graphics/interop | DONE |
| 9 | DONE | `Common/GL/glxext.h` | 805 | 0 | no | graphics/interop | DONE |
| 10 | DONE | `Common/GL/wglew.h` | 958 | 0 | no | graphics/interop | DONE |
| 11 | DONE | `Common/GL/wglext.h` | 696 | 0 | no | graphics/interop | DONE |
| 12 | DONE | `Common/UtilNPP/Exceptions.h` | 199 | 2 | no | Driver API, CUDA libraries | DONE |
| 13 | DONE | `Common/UtilNPP/Image.h` | 157 | 2 | no | CUDA libraries | DONE |
| 14 | DONE | `Common/UtilNPP/ImageAllocatorsCPU.h` | 82 | 2 | no | CUDA libraries | DONE |
| 15 | DONE | `Common/UtilNPP/ImageAllocatorsNPP.h` | 1232 | 93 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries, multi-GPU/P2P/IPC | DONE |
| 16 | DONE | `Common/UtilNPP/ImageIO.h` | 151 | 2 | no | CUDA libraries | DONE |
| 17 | DONE | `Common/UtilNPP/ImagePacked.h` | 173 | 2 | no | CUDA libraries | DONE |
| 18 | DONE | `Common/UtilNPP/ImagesCPU.h` | 123 | 2 | no | CUDA libraries | DONE |
| 19 | DONE | `Common/UtilNPP/ImagesNPP.h` | 151 | 2 | no | CUDA libraries | DONE |
| 20 | DONE | `Common/UtilNPP/Pixel.h` | 128 | 2 | no | CUDA libraries | DONE |
| 21 | DONE | `Common/UtilNPP/Signal.h` | 170 | 2 | no | CUDA libraries | DONE |
| 22 | DONE | `Common/UtilNPP/SignalAllocatorsCPU.h` | 68 | 2 | no | CUDA libraries | DONE |
| 23 | DONE | `Common/UtilNPP/SignalAllocatorsNPP.h` | 751 | 67 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries, multi-GPU/P2P/IPC | DONE |
| 24 | DONE | `Common/UtilNPP/SignalsCPU.h` | 109 | 2 | no | CUDA libraries | DONE |
| 25 | DONE | `Common/UtilNPP/SignalsNPP.h` | 115 | 2 | no | CUDA libraries | DONE |
| 26 | DONE | `Common/drvapi_error_string.h` | 472 | 2 | no | shared memory, streams/events, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 27 | DONE | `Common/dynlink_d3d11.h` | 161 | 1 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 28 | DONE | `Common/exception.h` | 152 | 1 | no | host/helper source structure | DONE |
| 29 | DONE | `Common/helper_cuda.h` | 996 | 8 | no | shared memory, Driver API, CUDA libraries, Python CUDA | DONE |
| 30 | DONE | `Common/helper_cuda_drvapi.h` | 425 | 12 | no | Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 31 | DONE | `Common/helper_cusolver.h` | 169 | 3 | no | Driver API, CUDA libraries | DONE |
| 32 | DONE | `Common/helper_functions.h` | 60 | 1 | no | host/helper source structure | DONE |
| 33 | DONE | `Common/helper_gl.h` | 269 | 2 | no | graphics/interop | DONE |
| 34 | DONE | `Common/helper_image.h` | 1015 | 14 | no | host/helper source structure | DONE |
| 35 | DONE | `Common/helper_math.h` | 1470 | 1 | no | host/helper source structure | DONE |
| 36 | DONE | `Common/helper_multiprocess.cpp` | 562 | 2 | no | multi-GPU/P2P/IPC | DONE |
| 37 | DONE | `Common/helper_multiprocess.h` | 142 | 1 | no | multi-GPU/P2P/IPC | DONE |
| 38 | DONE | `Common/helper_nvJPEG.hxx` | 428 | 0 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries | DONE |
| 39 | DONE | `Common/helper_string.h` | 435 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 40 | DONE | `Common/helper_timer.h` | 466 | 1 | no | host/helper source structure | DONE |
| 41 | DONE | `Common/multithreading.cpp` | 79 | 1 | no | Driver API | DONE |
| 42 | DONE | `Common/multithreading.h` | 77 | 1 | no | Driver API | DONE |
| 43 | DONE | `Common/nvMath.h` | 113 | 2 | no | graphics/interop | DONE |
| 44 | DONE | `Common/nvMatrix.h` | 542 | 2 | no | graphics/interop | DONE |
| 45 | DONE | `Common/nvQuaternion.h` | 532 | 2 | no | graphics/interop | DONE |
| 46 | DONE | `Common/nvShaderUtils.h` | 261 | 1 | no | NVRTC/JIT/libNVVM/PTX, shader source | DONE |
| 47 | DONE | `Common/nvVector.h` | 1076 | 2 | no | graphics/interop | DONE |
| 48 | DONE | `Common/nvrtc_helper.h` | 218 | 6 | no | Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 49 | DONE | `Common/param.h` | 237 | 1 | no | host/helper source structure | DONE |
| 50 | DONE | `Common/paramgl.h` | 308 | 1 | no | graphics/interop | DONE |
| 51 | DONE | `Common/rendercheck_d3d11.cpp` | 126 | 2 | no | host/helper source structure | DONE |
| 52 | DONE | `Common/rendercheck_d3d11.h` | 53 | 1 | no | graphics/interop | DONE |
| 53 | DONE | `Common/rendercheck_gl.h` | 1355 | 8 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 54 | DONE | `Common/rendercheck_gles.h` | 1320 | 8 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 55 | DONE | `cmake/CPM.cmake` | 1298 | 0 | yes | CMake/build wiring | DONE |
| 56 | DONE | `cmake/InstallSamples.cmake` | 344 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 57 | DONE | `cmake/Modules/FindEGL.cmake` | 19 | 1 | no | graphics/interop | DONE |
| 58 | DONE | `cmake/Modules/FindFreeImage.cmake` | 19 | 1 | no | host/helper source structure | DONE |
| 59 | DONE | `cmake/Modules/FindNVSCI.cmake` | 63 | 1 | no | graphics/interop | DONE |
| 60 | DONE | `cmake/toolchains/toolchain-aarch64-linux.cmake` | 70 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 61 | DONE | `cmake/toolchains/toolchain-aarch64-qnx.cmake` | 39 | 1 | no | host/helper source structure | DONE |
| 62 | DONE | `cpp/0_Introduction/CMakeLists.txt` | 48 | 1 | no | CMake/build wiring | DONE |
| 63 | DONE | `cpp/0_Introduction/UnifiedMemoryStreams/CMakeLists.txt` | 57 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 64 | DONE | `cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu` | 364 | 25 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 65 | DONE | `cpp/0_Introduction/asyncAPI/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 66 | DONE | `cpp/0_Introduction/asyncAPI/asyncAPI.cu` | 154 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 67 | DONE | `cpp/0_Introduction/clock/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 68 | DONE | `cpp/0_Introduction/clock/clock.cu` | 162 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 69 | DONE | `cpp/0_Introduction/clock_nvrtc/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 70 | DONE | `cpp/0_Introduction/clock_nvrtc/clock.cpp` | 143 | 6 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 71 | DONE | `cpp/0_Introduction/clock_nvrtc/clock_kernel.cu` | 83 | 6 | no | thread/block indexing, shared memory, synchronization | DONE |
| 72 | DONE | `cpp/0_Introduction/cudaOpenMP/CMakeLists.txt` | 46 | 2 | no | CMake/build wiring | DONE |
| 73 | DONE | `cpp/0_Introduction/cudaOpenMP/cudaOpenMP.cu` | 162 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 74 | DONE | `cpp/0_Introduction/fp16ScalarProduct/CMakeLists.txt` | 36 | 2 | no | CMake/build wiring | DONE |
| 75 | DONE | `cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu` | 254 | 42 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 76 | DONE | `cpp/0_Introduction/matrixMul/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 77 | DONE | `cpp/0_Introduction/matrixMul/matrixMul.cu` | 361 | 19 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 78 | DONE | `cpp/0_Introduction/matrixMulDrv/CMakeLists.txt` | 75 | 2 | no | CMake/build wiring | DONE |
| 79 | DONE | `cpp/0_Introduction/matrixMulDrv/matrixMul.h` | 41 | 1 | no | host/helper source structure | DONE |
| 80 | DONE | `cpp/0_Introduction/matrixMulDrv/matrixMulDrv.cpp` | 349 | 14 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 81 | DONE | `cpp/0_Introduction/matrixMulDrv/matrixMul_kernel.cu` | 133 | 5 | no | thread/block indexing, shared memory, synchronization | DONE |
| 82 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/CMakeLists.txt` | 54 | 2 | no | CMake/build wiring | DONE |
| 83 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.c` | 683 | 15 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 84 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink.h` | 19 | 1 | no | host/helper source structure | DONE |
| 85 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/cuda_drvapi_dynlink_cuda.h` | 2038 | 8 | no | CUDA kernel launch, shared memory, streams/events, Driver API | DONE |
| 86 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_32.ptx` | 727 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 87 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/extras/matrixMul_kernel_64.ptx` | 737 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 88 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/extras/ptx2c.py` | 94 | 2 | no | Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 89 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/helper_cuda_drvapi.h` | 367 | 13 | no | Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 90 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul.h` | 42 | 1 | no | host/helper source structure | DONE |
| 91 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp` | 364 | 17 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 92 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_gold.cpp` | 57 | 1 | no | host/helper source structure | DONE |
| 93 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_kernel_32_ptxdump.c` | 1391 | 0 | no | host/helper source structure | DONE |
| 94 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_kernel_32_ptxdump.h` | 45 | 0 | no | host/helper source structure | DONE |
| 95 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_kernel_64_ptxdump.c` | 1428 | 0 | no | host/helper source structure | DONE |
| 96 | DONE | `cpp/0_Introduction/matrixMulDynlinkJIT/matrixMul_kernel_64_ptxdump.h` | 45 | 0 | no | host/helper source structure | DONE |
| 97 | DONE | `cpp/0_Introduction/matrixMul_nvrtc/CMakeLists.txt` | 65 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 98 | DONE | `cpp/0_Introduction/matrixMul_nvrtc/matrixMul.cpp` | 259 | 10 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 99 | DONE | `cpp/0_Introduction/matrixMul_nvrtc/matrixMul_kernel.cu` | 135 | 5 | no | thread/block indexing, shared memory, synchronization | DONE |
| 100 | DONE | `cpp/0_Introduction/mergeSort/CMakeLists.txt` | 40 | 2 | no | CMake/build wiring | DONE |
| 101 | DONE | `cpp/0_Introduction/mergeSort/bitonic.cu` | 304 | 23 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 102 | DONE | `cpp/0_Introduction/mergeSort/main.cpp` | 126 | 7 | no | CUDA Runtime memory/transfer, synchronization, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 103 | DONE | `cpp/0_Introduction/mergeSort/mergeSort.cu` | 565 | 28 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 104 | DONE | `cpp/0_Introduction/mergeSort/mergeSort_common.h` | 62 | 2 | no | host/helper source structure | DONE |
| 105 | DONE | `cpp/0_Introduction/mergeSort/mergeSort_host.cpp` | 366 | 3 | no | host/helper source structure | DONE |
| 106 | DONE | `cpp/0_Introduction/mergeSort/mergeSort_validate.cpp` | 132 | 4 | no | host/helper source structure | DONE |
| 107 | DONE | `cpp/0_Introduction/simpleAWBarrier/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 108 | DONE | `cpp/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu` | 266 | 18 | no | CUDA Runtime memory/transfer, thread/block indexing, shared memory, synchronization | DONE |
| 109 | DONE | `cpp/0_Introduction/simpleAssert/CMakeLists.txt` | 40 | 2 | no | CMake/build wiring | DONE |
| 110 | DONE | `cpp/0_Introduction/simpleAssert/simpleAssert.cu` | 135 | 5 | no | CUDA kernel launch, thread/block indexing, synchronization, Driver API | DONE |
| 111 | DONE | `cpp/0_Introduction/simpleAssert_nvrtc/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 112 | DONE | `cpp/0_Introduction/simpleAssert_nvrtc/simpleAssert.cpp` | 125 | 4 | no | CUDA kernel launch, synchronization, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 113 | DONE | `cpp/0_Introduction/simpleAssert_nvrtc/simpleAssert_kernel.cu` | 42 | 3 | no | thread/block indexing | DONE |
| 114 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics/CMakeLists.txt` | 38 | 2 | no | CMake/build wiring | DONE |
| 115 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu` | 143 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 116 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_cpu.cpp` | 184 | 1 | no | host/helper source structure | DONE |
| 117 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh` | 85 | 2 | no | thread/block indexing | DONE |
| 118 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/CMakeLists.txt` | 48 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 119 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics.cpp` | 160 | 7 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 120 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics_cpu.cpp` | 183 | 1 | no | host/helper source structure | DONE |
| 121 | DONE | `cpp/0_Introduction/simpleAtomicIntrinsics_nvrtc/simpleAtomicIntrinsics_kernel.cuh` | 83 | 2 | no | thread/block indexing | DONE |
| 122 | DONE | `cpp/0_Introduction/simpleAttributes/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 123 | DONE | `cpp/0_Introduction/simpleAttributes/simpleAttributes.cu` | 220 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 124 | DONE | `cpp/0_Introduction/simpleCUDA2GL/CMakeLists.txt` | 95 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 125 | DONE | `cpp/0_Introduction/simpleCUDA2GL/main.cpp` | 762 | 14 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 126 | DONE | `cpp/0_Introduction/simpleCUDA2GL/simpleCUDA2GL.cu` | 68 | 4 | no | CUDA kernel launch, thread/block indexing, shared memory, Driver API | DONE |
| 127 | DONE | `cpp/0_Introduction/simpleCallback/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 128 | DONE | `cpp/0_Introduction/simpleCallback/multithreading.cpp` | 153 | 1 | no | Driver API | DONE |
| 129 | DONE | `cpp/0_Introduction/simpleCallback/multithreading.h` | 102 | 1 | no | Driver API | DONE |
| 130 | DONE | `cpp/0_Introduction/simpleCallback/simpleCallback.cu` | 226 | 11 | no | CUDA Runtime memory/transfer, thread/block indexing, streams/events, Driver API | DONE |
| 131 | DONE | `cpp/0_Introduction/simpleCooperativeGroups/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 132 | DONE | `cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu` | 181 | 4 | no | CUDA kernel launch, shared memory, synchronization, Driver API | DONE |
| 133 | DONE | `cpp/0_Introduction/simpleCubemapTexture/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 134 | DONE | `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu` | 274 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 135 | DONE | `cpp/0_Introduction/simpleDrvRuntime/CMakeLists.txt` | 77 | 2 | no | CMake/build wiring | DONE |
| 136 | DONE | `cpp/0_Introduction/simpleDrvRuntime/simpleDrvRuntime.cpp` | 240 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 137 | DONE | `cpp/0_Introduction/simpleDrvRuntime/vectorAdd_kernel.cu` | 45 | 2 | no | thread/block indexing | DONE |
| 138 | DONE | `cpp/0_Introduction/simpleHyperQ/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 139 | DONE | `cpp/0_Introduction/simpleHyperQ/simpleHyperQ.cu` | 250 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 140 | DONE | `cpp/0_Introduction/simpleIPC/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 141 | DONE | `cpp/0_Introduction/simpleIPC/simpleIPC.cu` | 371 | 14 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 142 | DONE | `cpp/0_Introduction/simpleLayeredTexture/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 143 | DONE | `cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu` | 221 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 144 | DONE | `cpp/0_Introduction/simpleMPI/CMakeLists.txt` | 53 | 2 | no | CMake/build wiring, multi-GPU/P2P/IPC | DONE |
| 145 | DONE | `cpp/0_Introduction/simpleMPI/simpleMPI.cpp` | 130 | 1 | no | multi-GPU/P2P/IPC | DONE |
| 146 | DONE | `cpp/0_Introduction/simpleMPI/simpleMPI.cu` | 110 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 147 | DONE | `cpp/0_Introduction/simpleMPI/simpleMPI.h` | 46 | 1 | no | multi-GPU/P2P/IPC | DONE |
| 148 | DONE | `cpp/0_Introduction/simpleMultiCopy/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 149 | DONE | `cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu` | 389 | 23 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 150 | DONE | `cpp/0_Introduction/simpleMultiGPU/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 151 | DONE | `cpp/0_Introduction/simpleMultiGPU/simpleMultiGPU.cu` | 243 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 152 | DONE | `cpp/0_Introduction/simpleMultiGPU/simpleMultiGPU.h` | 64 | 2 | no | streams/events, multi-GPU/P2P/IPC | DONE |
| 153 | DONE | `cpp/0_Introduction/simpleOccupancy/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 154 | DONE | `cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu` | 251 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 155 | DONE | `cpp/0_Introduction/simpleP2P/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 156 | DONE | `cpp/0_Introduction/simpleP2P/simpleP2P.cu` | 276 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 157 | DONE | `cpp/0_Introduction/simplePitchLinearTexture/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 158 | DONE | `cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu` | 314 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 159 | DONE | `cpp/0_Introduction/simplePrintf/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 160 | DONE | `cpp/0_Introduction/simplePrintf/simplePrintf.cu` | 78 | 4 | no | CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 161 | DONE | `cpp/0_Introduction/simpleStreams/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 162 | DONE | `cpp/0_Introduction/simpleStreams/simpleStreams.cu` | 444 | 22 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 163 | DONE | `cpp/0_Introduction/simpleSurfaceWrite/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 164 | DONE | `cpp/0_Introduction/simpleSurfaceWrite/simpleSurfaceWrite.cu` | 303 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 165 | DONE | `cpp/0_Introduction/simpleTemplates/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 166 | DONE | `cpp/0_Introduction/simpleTemplates/sharedmem.cuh` | 188 | 3 | no | shared memory | DONE |
| 167 | DONE | `cpp/0_Introduction/simpleTemplates/simpleTemplates.cu` | 278 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 168 | DONE | `cpp/0_Introduction/simpleTexture/CMakeLists.txt` | 51 | 2 | no | CMake/build wiring | DONE |
| 169 | DONE | `cpp/0_Introduction/simpleTexture/simpleTexture.cu` | 260 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 170 | DONE | `cpp/0_Introduction/simpleTexture3D/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 171 | DONE | `cpp/0_Introduction/simpleTexture3D/simpleTexture3D.cpp` | 408 | 11 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 172 | DONE | `cpp/0_Introduction/simpleTexture3D/simpleTexture3D_kernel.cu` | 144 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 173 | DONE | `cpp/0_Introduction/simpleTextureDrv/CMakeLists.txt` | 88 | 2 | no | CMake/build wiring | DONE |
| 174 | DONE | `cpp/0_Introduction/simpleTextureDrv/simpleTextureDrv.cpp` | 366 | 14 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 175 | DONE | `cpp/0_Introduction/simpleTextureDrv/simpleTexture_kernel.cu` | 56 | 2 | no | thread/block indexing, Python CUDA | DONE |
| 176 | DONE | `cpp/0_Introduction/simpleVoteIntrinsics/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 177 | DONE | `cpp/0_Introduction/simpleVoteIntrinsics/simpleVoteIntrinsics.cu` | 306 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 178 | DONE | `cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh` | 85 | 4 | no | thread/block indexing | DONE |
| 179 | DONE | `cpp/0_Introduction/simpleZeroCopy/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 180 | DONE | `cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu` | 258 | 8 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 181 | DONE | `cpp/0_Introduction/systemWideAtomics/CMakeLists.txt` | 45 | 2 | no | CMake/build wiring | DONE |
| 182 | DONE | `cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu` | 362 | 8 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 183 | DONE | `cpp/0_Introduction/template/CMakeLists.txt` | 38 | 2 | no | CMake/build wiring | DONE |
| 184 | DONE | `cpp/0_Introduction/template/template.cu` | 176 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 185 | DONE | `cpp/0_Introduction/template/template_cpu.cpp` | 46 | 1 | no | host/helper source structure | DONE |
| 186 | DONE | `cpp/0_Introduction/vectorAdd/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 187 | DONE | `cpp/0_Introduction/vectorAdd/vectorAdd.cu` | 205 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 188 | DONE | `cpp/0_Introduction/vectorAddDrv/CMakeLists.txt` | 74 | 2 | no | CMake/build wiring | DONE |
| 189 | DONE | `cpp/0_Introduction/vectorAddDrv/vectorAddDrv.cpp` | 236 | 11 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 190 | DONE | `cpp/0_Introduction/vectorAddDrv/vectorAdd_kernel.cu` | 45 | 2 | no | thread/block indexing | DONE |
| 191 | DONE | `cpp/0_Introduction/vectorAddMMAP/CMakeLists.txt` | 70 | 2 | no | CMake/build wiring | DONE |
| 192 | DONE | `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.cpp` | 205 | 5 | no | Driver API, multi-GPU/P2P/IPC | DONE |
| 193 | DONE | `cpp/0_Introduction/vectorAddMMAP/multidevicealloc_memmap.hpp` | 81 | 2 | no | Driver API, multi-GPU/P2P/IPC, Python CUDA | DONE |
| 194 | DONE | `cpp/0_Introduction/vectorAddMMAP/vectorAddMMAP.cpp` | 272 | 11 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 195 | DONE | `cpp/0_Introduction/vectorAddMMAP/vectorAdd_kernel.cu` | 45 | 2 | no | thread/block indexing | DONE |
| 196 | DONE | `cpp/0_Introduction/vectorAdd_nvrtc/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 197 | DONE | `cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd.cpp` | 164 | 7 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 198 | DONE | `cpp/0_Introduction/vectorAdd_nvrtc/vectorAdd_kernel.cu` | 44 | 2 | no | thread/block indexing | DONE |
| 199 | DONE | `cpp/1_Utilities/CMakeLists.txt` | 5 | 1 | no | CMake/build wiring | DONE |
| 200 | DONE | `cpp/1_Utilities/deviceQuery/CMakeLists.txt` | 45 | 2 | no | CMake/build wiring | DONE |
| 201 | DONE | `cpp/1_Utilities/deviceQuery/deviceQuery.cpp` | 341 | 6 | no | thread/block indexing, shared memory, Driver API, multi-GPU/P2P/IPC | DONE |
| 202 | DONE | `cpp/1_Utilities/deviceQueryDrv/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 203 | DONE | `cpp/1_Utilities/deviceQueryDrv/deviceQueryDrv.cpp` | 332 | 8 | yes | thread/block indexing, shared memory, Driver API, multi-GPU/P2P/IPC | DONE |
| 204 | DONE | `cpp/1_Utilities/topologyQuery/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 205 | DONE | `cpp/1_Utilities/topologyQuery/topologyQuery.cu` | 79 | 1 | no | multi-GPU/P2P/IPC | DONE |
| 206 | DONE | `cpp/2_Concepts_and_Techniques/CMakeLists.txt` | 34 | 1 | no | CMake/build wiring | DONE |
| 207 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/CMakeLists.txt` | 57 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 208 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.cpp` | 270 | 5 | no | Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 209 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_consumer.h` | 75 | 3 | no | streams/events, Driver API, Python CUDA | DONE |
| 210 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.cpp` | 304 | 10 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop, multi-GPU/P2P/IPC | DONE |
| 211 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/cuda_producer.h` | 74 | 3 | no | streams/events, Driver API, graphics/interop, Python CUDA | DONE |
| 212 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.cpp` | 427 | 3 | no | graphics/interop | DONE |
| 213 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/eglstrm_common.h` | 110 | 1 | no | graphics/interop, Python CUDA | DONE |
| 214 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/helper.h` | 232 | 4 | no | Driver API, graphics/interop, Python CUDA | DONE |
| 215 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/kernel.cu` | 162 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 216 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_CrossGPU/main.cpp` | 385 | 7 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 217 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/CMakeLists.txt` | 58 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 218 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.cpp` | 336 | 7 | no | Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 219 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_consumer.h` | 66 | 3 | no | Driver API | DONE |
| 220 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.cpp` | 428 | 9 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop, multi-GPU/P2P/IPC | DONE |
| 221 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/cuda_producer.h` | 74 | 3 | no | Driver API, graphics/interop | DONE |
| 222 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.cpp` | 135 | 1 | no | graphics/interop | DONE |
| 223 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/eglstrm_common.h` | 101 | 1 | no | graphics/interop, Python CUDA | DONE |
| 224 | DONE | `cpp/2_Concepts_and_Techniques/EGLStream_CUDA_Interop/main.cpp` | 242 | 9 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 225 | DONE | `cpp/2_Concepts_and_Techniques/FunctionPointers/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 226 | DONE | `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers.cpp` | 517 | 11 | no | CUDA Runtime memory/transfer, shared memory, synchronization, CUDA Graphs | DONE |
| 227 | DONE | `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers_kernels.cu` | 428 | 14 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 228 | DONE | `cpp/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers_kernels.h` | 59 | 1 | no | host/helper source structure | DONE |
| 229 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/CMakeLists.txt` | 42 | 2 | no | CMake/build wiring | DONE |
| 230 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/inc/cudasharedmem.h` | 104 | 2 | no | shared memory | DONE |
| 231 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/inc/piestimator.h` | 45 | 1 | no | host/helper source structure | DONE |
| 232 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/inc/test.h` | 61 | 1 | no | host/helper source structure | DONE |
| 233 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/src/main.cpp` | 259 | 2 | no | Driver API | DONE |
| 234 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/src/piestimator.cu` | 299 | 16 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 235 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineP/src/test.cpp` | 114 | 1 | no | Driver API | DONE |
| 236 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 237 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/inc/cudasharedmem.h` | 104 | 2 | no | shared memory | DONE |
| 238 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/inc/piestimator.h` | 44 | 1 | no | host/helper source structure | DONE |
| 239 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/inc/test.h` | 59 | 1 | no | host/helper source structure | DONE |
| 240 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/src/main.cpp` | 241 | 2 | no | host/helper source structure | DONE |
| 241 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/src/piestimator.cu` | 397 | 21 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 242 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiInlineQ/src/test.cpp` | 114 | 1 | no | Driver API | DONE |
| 243 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 244 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/cudasharedmem.h` | 104 | 2 | no | shared memory | DONE |
| 245 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/piestimator.h` | 45 | 1 | no | host/helper source structure | DONE |
| 246 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/inc/test.h` | 61 | 1 | no | host/helper source structure | DONE |
| 247 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/main.cpp` | 257 | 2 | no | host/helper source structure | DONE |
| 248 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu` | 304 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 249 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/test.cpp` | 111 | 1 | no | Driver API | DONE |
| 250 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 251 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/inc/cudasharedmem.h` | 104 | 2 | no | shared memory | DONE |
| 252 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/inc/piestimator.h` | 44 | 1 | no | host/helper source structure | DONE |
| 253 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/inc/test.h` | 59 | 1 | no | host/helper source structure | DONE |
| 254 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/src/main.cpp` | 239 | 2 | no | host/helper source structure | DONE |
| 255 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/src/piestimator.cu` | 319 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 256 | DONE | `cpp/2_Concepts_and_Techniques/MC_EstimatePiQ/src/test.cpp` | 113 | 1 | no | Driver API | DONE |
| 257 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/CMakeLists.txt` | 42 | 2 | no | CMake/build wiring | DONE |
| 258 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/inc/asianoption.h` | 52 | 1 | no | host/helper source structure | DONE |
| 259 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/inc/cudasharedmem.h` | 104 | 2 | no | shared memory | DONE |
| 260 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/inc/pricingengine.h` | 47 | 1 | no | host/helper source structure | DONE |
| 261 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/inc/test.h` | 59 | 1 | no | host/helper source structure | DONE |
| 262 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/main.cpp` | 256 | 2 | no | host/helper source structure | DONE |
| 263 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/pricingengine.cu` | 389 | 20 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 264 | DONE | `cpp/2_Concepts_and_Techniques/MC_SingleAsianOptionP/src/test.cpp` | 139 | 1 | no | Driver API | DONE |
| 265 | DONE | `cpp/2_Concepts_and_Techniques/boxFilter/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 266 | DONE | `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter.cpp` | 645 | 16 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 267 | DONE | `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_cpu.cpp` | 127 | 1 | no | host/helper source structure | DONE |
| 268 | DONE | `cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu` | 521 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 269 | DONE | `cpp/2_Concepts_and_Techniques/convolutionSeparable/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 270 | DONE | `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu` | 207 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 271 | DONE | `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable_common.h` | 51 | 1 | no | host/helper source structure | DONE |
| 272 | DONE | `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable_gold.cpp` | 69 | 1 | no | host/helper source structure | DONE |
| 273 | DONE | `cpp/2_Concepts_and_Techniques/convolutionSeparable/main.cpp` | 172 | 7 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 274 | DONE | `cpp/2_Concepts_and_Techniques/convolutionTexture/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 275 | DONE | `cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture.cu` | 158 | 5 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 276 | DONE | `cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture_common.h` | 57 | 1 | no | host/helper source structure | DONE |
| 277 | DONE | `cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture_gold.cpp` | 79 | 1 | no | host/helper source structure | DONE |
| 278 | DONE | `cpp/2_Concepts_and_Techniques/convolutionTexture/main.cpp` | 204 | 7 | no | CUDA Runtime memory/transfer, shared memory, synchronization, Driver API | DONE |
| 279 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.cpp` | 471 | 2 | no | host/helper source structure | DONE |
| 280 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/BmpUtil.h` | 128 | 1 | no | host/helper source structure | DONE |
| 281 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/CMakeLists.txt` | 62 | 2 | no | CMake/build wiring | DONE |
| 282 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/Common.h` | 88 | 1 | no | host/helper source structure | DONE |
| 283 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.cpp` | 397 | 1 | no | host/helper source structure | DONE |
| 284 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/DCT8x8_Gold.h` | 51 | 1 | no | host/helper source structure | DONE |
| 285 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8.cu` | 690 | 25 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 286 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel1.cuh` | 242 | 10 | no | thread/block indexing, shared memory, synchronization, Driver API | DONE |
| 287 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel2.cuh` | 285 | 11 | no | thread/block indexing, shared memory, Driver API | DONE |
| 288 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_quantization.cuh` | 130 | 4 | no | thread/block indexing, Driver API | DONE |
| 289 | DONE | `cpp/2_Concepts_and_Techniques/dct8x8/dct8x8_kernel_short.cuh` | 544 | 8 | no | thread/block indexing, shared memory, Driver API | DONE |
| 290 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/CMakeLists.txt` | 58 | 2 | no | CMake/build wiring | DONE |
| 291 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large.cuh` | 997 | 32 | no | thread/block indexing, shared memory | DONE |
| 292 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_multi.cuh` | 281 | 13 | no | thread/block indexing, shared memory | DONE |
| 293 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_large_onei.cuh` | 175 | 8 | no | thread/block indexing, shared memory | DONE |
| 294 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_kernel_small.cuh` | 286 | 15 | no | thread/block indexing, shared memory | DONE |
| 295 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cu` | 396 | 28 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 296 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_large.cuh` | 86 | 1 | no | host/helper source structure | DONE |
| 297 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cu` | 193 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 298 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_small.cuh` | 83 | 1 | no | host/helper source structure | DONE |
| 299 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/bisect_util.cu` | 613 | 17 | no | thread/block indexing, shared memory | DONE |
| 300 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/config.h` | 42 | 1 | no | host/helper source structure | DONE |
| 301 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.cpp` | 85 | 1 | no | host/helper source structure | DONE |
| 302 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/gerschgorin.h` | 44 | 1 | no | host/helper source structure | DONE |
| 303 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/main.cu` | 331 | 6 | no | CUDA Runtime memory/transfer, Driver API | DONE |
| 304 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/matlab.cpp` | 71 | 1 | no | host/helper source structure | DONE |
| 305 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/matlab.h` | 121 | 1 | no | host/helper source structure | DONE |
| 306 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/structs.h` | 133 | 1 | no | host/helper source structure | DONE |
| 307 | DONE | `cpp/2_Concepts_and_Techniques/eigenvalues/util.h` | 140 | 2 | no | host/helper source structure | DONE |
| 308 | DONE | `cpp/2_Concepts_and_Techniques/histogram/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 309 | DONE | `cpp/2_Concepts_and_Techniques/histogram/histogram256.cu` | 170 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 310 | DONE | `cpp/2_Concepts_and_Techniques/histogram/histogram64.cu` | 217 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 311 | DONE | `cpp/2_Concepts_and_Techniques/histogram/histogram_common.h` | 85 | 1 | no | thread/block indexing, shared memory | DONE |
| 312 | DONE | `cpp/2_Concepts_and_Techniques/histogram/histogram_gold.cpp` | 64 | 2 | no | host/helper source structure | DONE |
| 313 | DONE | `cpp/2_Concepts_and_Techniques/histogram/main.cpp` | 236 | 8 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 314 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/CMakeLists.txt` | 100 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 315 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/bmploader.cpp` | 136 | 1 | no | host/helper source structure | DONE |
| 316 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.cu` | 116 | 4 | no | CUDA Runtime memory/transfer, shared memory, Driver API | DONE |
| 317 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising.h` | 83 | 1 | no | Driver API | DONE |
| 318 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoisingGL.cpp` | 558 | 9 | no | CUDA Runtime memory/transfer, shared memory, synchronization, CUDA Graphs | DONE |
| 319 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_copy_kernel.cuh` | 51 | 3 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 320 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_knn_kernel.cuh` | 149 | 5 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 321 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm2_kernel.cuh` | 224 | 11 | no | CUDA kernel launch, thread/block indexing, shared memory, Driver API | DONE |
| 322 | DONE | `cpp/2_Concepts_and_Techniques/imageDenoising/imageDenoising_nlm_kernel.cuh` | 157 | 5 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 323 | DONE | `cpp/2_Concepts_and_Techniques/inlinePTX/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 324 | DONE | `cpp/2_Concepts_and_Techniques/inlinePTX/inlinePTX.cu` | 116 | 8 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 325 | DONE | `cpp/2_Concepts_and_Techniques/inlinePTX_nvrtc/CMakeLists.txt` | 49 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 326 | DONE | `cpp/2_Concepts_and_Techniques/inlinePTX_nvrtc/inlinePTX.cpp` | 114 | 6 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 327 | DONE | `cpp/2_Concepts_and_Techniques/inlinePTX_nvrtc/inlinePTX_kernel.cu` | 42 | 2 | no | thread/block indexing | DONE |
| 328 | DONE | `cpp/2_Concepts_and_Techniques/interval/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 329 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config.hpp` | 59 | 0 | no | host/helper source structure | DONE |
| 330 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/borland_prefix.hpp` | 24 | 0 | no | host/helper source structure | DONE |
| 331 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/borland_suffix.hpp` | 7 | 0 | no | host/helper source structure | DONE |
| 332 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/msvc_prefix.hpp` | 20 | 0 | no | host/helper source structure | DONE |
| 333 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi/msvc_suffix.hpp` | 6 | 0 | no | host/helper source structure | DONE |
| 334 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi_prefix.hpp` | 24 | 0 | no | host/helper source structure | DONE |
| 335 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/abi_suffix.hpp` | 25 | 0 | no | host/helper source structure | DONE |
| 336 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/auto_link.hpp` | 362 | 0 | no | host/helper source structure | DONE |
| 337 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/borland.hpp` | 264 | 0 | no | host/helper source structure | DONE |
| 338 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/codegear.hpp` | 162 | 0 | no | host/helper source structure | DONE |
| 339 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/comeau.hpp` | 55 | 0 | no | host/helper source structure | DONE |
| 340 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/common_edg.hpp` | 94 | 0 | no | host/helper source structure | DONE |
| 341 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/compaq_cxx.hpp` | 16 | 0 | no | host/helper source structure | DONE |
| 342 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/digitalmars.hpp` | 93 | 0 | no | host/helper source structure | DONE |
| 343 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/gcc.hpp` | 202 | 0 | no | host/helper source structure | DONE |
| 344 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/gcc_xml.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 345 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/greenhills.hpp` | 26 | 0 | no | host/helper source structure | DONE |
| 346 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/hp_acc.hpp` | 127 | 0 | no | host/helper source structure | DONE |
| 347 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/intel.hpp` | 179 | 0 | no | host/helper source structure | DONE |
| 348 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/kai.hpp` | 30 | 0 | no | host/helper source structure | DONE |
| 349 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/metrowerks.hpp` | 132 | 0 | no | host/helper source structure | DONE |
| 350 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/mpw.hpp` | 79 | 0 | no | host/helper source structure | DONE |
| 351 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/pgi.hpp` | 61 | 0 | no | host/helper source structure | DONE |
| 352 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/sgi_mipspro.hpp` | 27 | 0 | no | host/helper source structure | DONE |
| 353 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/sunpro_cc.hpp` | 130 | 0 | no | host/helper source structure | DONE |
| 354 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/vacpp.hpp` | 85 | 0 | no | host/helper source structure | DONE |
| 355 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/compiler/visualc.hpp` | 258 | 0 | no | host/helper source structure | DONE |
| 356 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/cmath.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 357 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/complex.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 358 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/functional.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 359 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/memory.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 360 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/no_tr1/utility.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 361 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/aix.hpp` | 29 | 0 | no | host/helper source structure | DONE |
| 362 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/amigaos.hpp` | 13 | 0 | no | host/helper source structure | DONE |
| 363 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/beos.hpp` | 23 | 0 | no | host/helper source structure | DONE |
| 364 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/bsd.hpp` | 76 | 0 | no | host/helper source structure | DONE |
| 365 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/cygwin.hpp` | 46 | 0 | no | host/helper source structure | DONE |
| 366 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/hpux.hpp` | 86 | 0 | no | host/helper source structure | DONE |
| 367 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/irix.hpp` | 28 | 0 | no | host/helper source structure | DONE |
| 368 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/linux.hpp` | 96 | 0 | no | host/helper source structure | DONE |
| 369 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/macos.hpp` | 83 | 0 | no | host/helper source structure | DONE |
| 370 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/qnxnto.hpp` | 26 | 0 | no | host/helper source structure | DONE |
| 371 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/solaris.hpp` | 24 | 0 | no | host/helper source structure | DONE |
| 372 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/vxworks.hpp` | 30 | 0 | no | host/helper source structure | DONE |
| 373 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/platform/win32.hpp` | 59 | 0 | no | host/helper source structure | DONE |
| 374 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/posix_features.hpp` | 91 | 0 | no | host/helper source structure | DONE |
| 375 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/requires_threads.hpp` | 100 | 0 | no | host/helper source structure | DONE |
| 376 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/select_compiler_config.hpp` | 120 | 0 | no | host/helper source structure | DONE |
| 377 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/select_platform_config.hpp` | 88 | 0 | no | host/helper source structure | DONE |
| 378 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/select_stdlib_config.hpp` | 74 | 0 | no | host/helper source structure | DONE |
| 379 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/dinkumware.hpp` | 131 | 0 | no | host/helper source structure | DONE |
| 380 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/libcomo.hpp` | 69 | 0 | no | host/helper source structure | DONE |
| 381 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/libstdcpp3.hpp` | 123 | 0 | no | host/helper source structure | DONE |
| 382 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/modena.hpp` | 50 | 0 | no | host/helper source structure | DONE |
| 383 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/msl.hpp` | 74 | 0 | no | host/helper source structure | DONE |
| 384 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/roguewave.hpp` | 179 | 0 | no | host/helper source structure | DONE |
| 385 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/sgi.hpp` | 132 | 0 | no | host/helper source structure | DONE |
| 386 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/stlport.hpp` | 237 | 0 | no | host/helper source structure | DONE |
| 387 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/stdlib/vacpp.hpp` | 40 | 0 | no | host/helper source structure | DONE |
| 388 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/suffix.hpp` | 584 | 0 | no | host/helper source structure | DONE |
| 389 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/user.hpp` | 121 | 0 | no | host/helper source structure | DONE |
| 390 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/config/warning_disable.hpp` | 47 | 0 | no | host/helper source structure | DONE |
| 391 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/limits.hpp` | 138 | 0 | no | host/helper source structure | DONE |
| 392 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval.hpp` | 29 | 0 | no | host/helper source structure | DONE |
| 393 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith.hpp` | 280 | 0 | no | host/helper source structure | DONE |
| 394 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith2.hpp` | 312 | 0 | no | host/helper source structure | DONE |
| 395 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/arith3.hpp` | 65 | 0 | no | host/helper source structure | DONE |
| 396 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/checking.hpp` | 109 | 0 | no | host/helper source structure | DONE |
| 397 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare.hpp` | 19 | 0 | no | host/helper source structure | DONE |
| 398 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/certain.hpp` | 119 | 0 | no | host/helper source structure | DONE |
| 399 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/explicit.hpp` | 224 | 0 | no | host/helper source structure | DONE |
| 400 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/lexicographic.hpp` | 128 | 0 | no | host/helper source structure | DONE |
| 401 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/possible.hpp` | 119 | 0 | no | host/helper source structure | DONE |
| 402 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/set.hpp` | 95 | 0 | no | host/helper source structure | DONE |
| 403 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/compare/tribool.hpp` | 168 | 0 | no | host/helper source structure | DONE |
| 404 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/constants.hpp` | 79 | 0 | no | host/helper source structure | DONE |
| 405 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/alpha_rounding_control.hpp` | 110 | 0 | no | host/helper source structure | DONE |
| 406 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/bcc_rounding_control.hpp` | 62 | 0 | no | host/helper source structure | DONE |
| 407 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/bugs.hpp` | 80 | 0 | no | host/helper source structure | DONE |
| 408 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/c99_rounding_control.hpp` | 50 | 0 | no | host/helper source structure | DONE |
| 409 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/c99sub_rounding_control.hpp` | 45 | 0 | no | host/helper source structure | DONE |
| 410 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/division.hpp` | 198 | 0 | no | host/helper source structure | DONE |
| 411 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/ia64_rounding_control.hpp` | 79 | 0 | no | host/helper source structure | DONE |
| 412 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/interval_prototype.hpp` | 39 | 0 | no | host/helper source structure | DONE |
| 413 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/msvc_rounding_control.hpp` | 152 | 0 | no | host/helper source structure | DONE |
| 414 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/ppc_rounding_control.hpp` | 93 | 0 | no | host/helper source structure | DONE |
| 415 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/sparc_rounding_control.hpp` | 107 | 0 | no | host/helper source structure | DONE |
| 416 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/test_input.hpp` | 73 | 0 | no | host/helper source structure | DONE |
| 417 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/x86_rounding_control.hpp` | 110 | 0 | no | host/helper source structure | DONE |
| 418 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/detail/x86gcc_rounding_control.hpp` | 48 | 0 | no | host/helper source structure | DONE |
| 419 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/ext/integer.hpp` | 62 | 0 | no | host/helper source structure | DONE |
| 420 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/ext/x86_fast_rounding_control.hpp` | 66 | 0 | no | host/helper source structure | DONE |
| 421 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/hw_rounding.hpp` | 67 | 0 | no | host/helper source structure | DONE |
| 422 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/interval.hpp` | 484 | 0 | no | host/helper source structure | DONE |
| 423 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/io.hpp` | 40 | 0 | no | host/helper source structure | DONE |
| 424 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/limits.hpp` | 50 | 0 | no | host/helper source structure | DONE |
| 425 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/policies.hpp` | 74 | 0 | no | host/helper source structure | DONE |
| 426 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounded_arith.hpp` | 158 | 0 | no | host/helper source structure | DONE |
| 427 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounded_transc.hpp` | 164 | 0 | no | host/helper source structure | DONE |
| 428 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/rounding.hpp` | 91 | 0 | no | host/helper source structure | DONE |
| 429 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/transc.hpp` | 205 | 0 | no | host/helper source structure | DONE |
| 430 | DONE | `cpp/2_Concepts_and_Techniques/interval/boost/numeric/interval/utility.hpp` | 327 | 0 | no | host/helper source structure | DONE |
| 431 | DONE | `cpp/2_Concepts_and_Techniques/interval/cpu_interval.h` | 311 | 1 | no | host/helper source structure | DONE |
| 432 | DONE | `cpp/2_Concepts_and_Techniques/interval/cuda_interval.h` | 344 | 2 | no | thread/block indexing | DONE |
| 433 | DONE | `cpp/2_Concepts_and_Techniques/interval/cuda_interval_lib.h` | 326 | 1 | no | host/helper source structure | DONE |
| 434 | DONE | `cpp/2_Concepts_and_Techniques/interval/cuda_interval_rounded_arith.h` | 137 | 1 | no | host/helper source structure | DONE |
| 435 | DONE | `cpp/2_Concepts_and_Techniques/interval/interval.cu` | 161 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 436 | DONE | `cpp/2_Concepts_and_Techniques/interval/interval.h` | 53 | 1 | no | Driver API | DONE |
| 437 | DONE | `cpp/2_Concepts_and_Techniques/particles/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 438 | DONE | `cpp/2_Concepts_and_Techniques/particles/particleSystem.cpp` | 500 | 6 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 439 | DONE | `cpp/2_Concepts_and_Techniques/particles/particleSystem.cuh` | 73 | 2 | no | CUDA Graphs, Driver API, graphics/interop | DONE |
| 440 | DONE | `cpp/2_Concepts_and_Techniques/particles/particleSystem.h` | 148 | 2 | no | CUDA Graphs, NVRTC/JIT/libNVVM/PTX, graphics/interop, shader source | DONE |
| 441 | DONE | `cpp/2_Concepts_and_Techniques/particles/particleSystem_cuda.cu` | 229 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, CUDA Graphs | DONE |
| 442 | DONE | `cpp/2_Concepts_and_Techniques/particles/particles.cpp` | 760 | 7 | no | thread/block indexing, synchronization, Driver API, graphics/interop | DONE |
| 443 | DONE | `cpp/2_Concepts_and_Techniques/particles/particles_kernel.cuh` | 60 | 1 | no | host/helper source structure | DONE |
| 444 | DONE | `cpp/2_Concepts_and_Techniques/particles/particles_kernel_impl.cuh` | 349 | 7 | no | thread/block indexing, shared memory | DONE |
| 445 | DONE | `cpp/2_Concepts_and_Techniques/particles/render_particles.cpp` | 175 | 1 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 446 | DONE | `cpp/2_Concepts_and_Techniques/particles/render_particles.h` | 76 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 447 | DONE | `cpp/2_Concepts_and_Techniques/particles/shaders.cpp` | 66 | 1 | no | shader source | DONE |
| 448 | DONE | `cpp/2_Concepts_and_Techniques/particles/shaders.h` | 30 | 1 | no | host/helper source structure | DONE |
| 449 | DONE | `cpp/2_Concepts_and_Techniques/radixSortThrust/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 450 | DONE | `cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu` | 222 | 5 | no | synchronization, streams/events, Driver API | DONE |
| 451 | DONE | `cpp/2_Concepts_and_Techniques/reduction/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 452 | DONE | `cpp/2_Concepts_and_Techniques/reduction/reduction.cpp` | 608 | 14 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 453 | DONE | `cpp/2_Concepts_and_Techniques/reduction/reduction.h` | 35 | 1 | no | host/helper source structure | DONE |
| 454 | DONE | `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu` | 1063 | 72 | no | CUDA kernel launch, thread/block indexing, shared memory, synchronization | DONE |
| 455 | DONE | `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 456 | DONE | `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu` | 405 | 16 | no | CUDA Runtime memory/transfer, thread/block indexing, shared memory, synchronization | DONE |
| 457 | DONE | `cpp/2_Concepts_and_Techniques/scalarProd/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 458 | DONE | `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd.cu` | 175 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 459 | DONE | `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_cpu.cpp` | 46 | 1 | no | host/helper source structure | DONE |
| 460 | DONE | `cpp/2_Concepts_and_Techniques/scalarProd/scalarProd_kernel.cuh` | 105 | 8 | no | thread/block indexing, shared memory | DONE |
| 461 | DONE | `cpp/2_Concepts_and_Techniques/scan/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 462 | DONE | `cpp/2_Concepts_and_Techniques/scan/main.cpp` | 190 | 8 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 463 | DONE | `cpp/2_Concepts_and_Techniques/scan/scan.cu` | 282 | 16 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 464 | DONE | `cpp/2_Concepts_and_Techniques/scan/scan_common.h` | 63 | 1 | no | host/helper source structure | DONE |
| 465 | DONE | `cpp/2_Concepts_and_Techniques/scan/scan_gold.cpp` | 39 | 1 | no | host/helper source structure | DONE |
| 466 | DONE | `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/CMakeLists.txt` | 58 | 2 | no | CMake/build wiring | DONE |
| 467 | DONE | `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/common.cuh` | 36 | 1 | no | host/helper source structure | DONE |
| 468 | DONE | `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/kernels.cuh` | 204 | 10 | no | thread/block indexing | DONE |
| 469 | DONE | `cpp/2_Concepts_and_Techniques/segmentationTreeThrust/segmentationTree.cu` | 916 | 20 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 470 | DONE | `cpp/2_Concepts_and_Techniques/shfl_scan/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 471 | DONE | `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_integral_image.cuh` | 334 | 15 | no | thread/block indexing, shared memory, synchronization | DONE |
| 472 | DONE | `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu` | 450 | 31 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 473 | DONE | `cpp/2_Concepts_and_Techniques/shfl_scan/util.h` | 62 | 1 | no | synchronization, Driver API | DONE |
| 474 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 475 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu` | 299 | 28 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 476 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/main.cpp` | 164 | 8 | no | CUDA Runtime memory/transfer, synchronization, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 477 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/oddEvenMergeSort.cu` | 199 | 13 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 478 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_common.cuh` | 55 | 1 | no | host/helper source structure | DONE |
| 479 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_common.h` | 57 | 2 | no | host/helper source structure | DONE |
| 480 | DONE | `cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_validate.cpp` | 145 | 4 | no | host/helper source structure | DONE |
| 481 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 482 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu` | 252 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 483 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC/CMakeLists.txt` | 49 | 2 | no | CMake/build wiring | DONE |
| 484 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu` | 494 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 485 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 486 | DONE | `cpp/2_Concepts_and_Techniques/streamOrderedAllocationP2P/streamOrderedAllocationP2P.cu` | 254 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 487 | DONE | `cpp/2_Concepts_and_Techniques/threadFenceReduction/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 488 | DONE | `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction.cu` | 505 | 12 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 489 | DONE | `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction.h` | 34 | 1 | no | host/helper source structure | DONE |
| 490 | DONE | `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh` | 421 | 18 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 491 | DONE | `cpp/2_Concepts_and_Techniques/threadMigration/CMakeLists.txt` | 79 | 2 | no | CMake/build wiring | DONE |
| 492 | DONE | `cpp/2_Concepts_and_Techniques/threadMigration/threadMigration.cpp` | 451 | 13 | no | CUDA kernel launch, shared memory, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 493 | DONE | `cpp/2_Concepts_and_Techniques/threadMigration/threadMigration_kernel.cu` | 30 | 2 | no | thread/block indexing | DONE |
| 494 | DONE | `cpp/3_CUDA_Features/CMakeLists.txt` | 26 | 1 | no | CMake/build wiring | DONE |
| 495 | DONE | `cpp/3_CUDA_Features/StreamPriorities/CMakeLists.txt` | 45 | 2 | no | CMake/build wiring | DONE |
| 496 | DONE | `cpp/3_CUDA_Features/StreamPriorities/StreamPriorities.cu` | 205 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 497 | DONE | `cpp/3_CUDA_Features/bf16TensorCoreGemm/CMakeLists.txt` | 42 | 2 | no | CMake/build wiring | DONE |
| 498 | DONE | `cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu` | 880 | 35 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 499 | DONE | `cpp/3_CUDA_Features/binaryPartitionCG/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 500 | DONE | `cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu` | 168 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 501 | DONE | `cpp/3_CUDA_Features/bindlessTexture/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 502 | DONE | `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.cpp` | 430 | 10 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 503 | DONE | `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture.h` | 67 | 1 | no | host/helper source structure | DONE |
| 504 | DONE | `cpp/3_CUDA_Features/bindlessTexture/bindlessTexture_kernel.cu` | 427 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 505 | DONE | `cpp/3_CUDA_Features/cdpAdvancedQuicksort/CMakeLists.txt` | 47 | 2 | no | CMake/build wiring | DONE |
| 506 | DONE | `cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpAdvancedQuicksort.cu` | 609 | 25 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 507 | DONE | `cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpBitonicSort.cu` | 305 | 15 | no | thread/block indexing, shared memory, NVRTC/JIT/libNVVM/PTX | DONE |
| 508 | DONE | `cpp/3_CUDA_Features/cdpAdvancedQuicksort/cdpQuicksort.h` | 78 | 1 | no | host/helper source structure | DONE |
| 509 | DONE | `cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu` | 214 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 510 | DONE | `cpp/3_CUDA_Features/cdpBezierTessellation/CMakeLists.txt` | 46 | 2 | no | CMake/build wiring | DONE |
| 511 | DONE | `cpp/3_CUDA_Features/cdpQuadtree/CMakeLists.txt` | 46 | 2 | no | CMake/build wiring | DONE |
| 512 | DONE | `cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu` | 761 | 27 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 513 | DONE | `cpp/3_CUDA_Features/cdpSimplePrint/CMakeLists.txt` | 46 | 2 | no | CMake/build wiring | DONE |
| 514 | DONE | `cpp/3_CUDA_Features/cdpSimplePrint/cdpSimplePrint.cu` | 169 | 8 | no | CUDA kernel launch, thread/block indexing, shared memory, synchronization | DONE |
| 515 | DONE | `cpp/3_CUDA_Features/cdpSimpleQuicksort/CMakeLists.txt` | 46 | 2 | no | CMake/build wiring | DONE |
| 516 | DONE | `cpp/3_CUDA_Features/cdpSimpleQuicksort/cdpSimpleQuicksort.cu` | 255 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 517 | DONE | `cpp/3_CUDA_Features/cudaCompressibleMemory/CMakeLists.txt` | 45 | 2 | no | CMake/build wiring | DONE |
| 518 | DONE | `cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp` | 118 | 3 | no | Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 519 | DONE | `cpp/3_CUDA_Features/cudaCompressibleMemory/compMalloc.h` | 35 | 1 | no | host/helper source structure | DONE |
| 520 | DONE | `cpp/3_CUDA_Features/cudaCompressibleMemory/saxpy.cu` | 202 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 521 | DONE | `cpp/3_CUDA_Features/cudaTensorCoreGemm/CMakeLists.txt` | 38 | 2 | no | CMake/build wiring | DONE |
| 522 | DONE | `cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu` | 640 | 23 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 523 | DONE | `cpp/3_CUDA_Features/dmmaTensorCoreGemm/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 524 | DONE | `cpp/3_CUDA_Features/dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu` | 1084 | 42 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 525 | DONE | `cpp/3_CUDA_Features/globalToShmemAsyncCopy/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 526 | DONE | `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu` | 1103 | 75 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 527 | DONE | `cpp/3_CUDA_Features/graphConditionalNodes/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 528 | DONE | `cpp/3_CUDA_Features/graphConditionalNodes/graphConditionalNodes.cu` | 624 | 70 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 529 | DONE | `cpp/3_CUDA_Features/graphMemoryFootprint/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 530 | DONE | `cpp/3_CUDA_Features/graphMemoryFootprint/graphMemoryFootprint.cu` | 424 | 34 | no | CUDA Runtime memory/transfer, thread/block indexing, synchronization, streams/events | DONE |
| 531 | DONE | `cpp/3_CUDA_Features/graphMemoryNodes/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 532 | DONE | `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu` | 611 | 56 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 533 | DONE | `cpp/3_CUDA_Features/immaTensorCoreGemm/CMakeLists.txt` | 38 | 2 | no | CMake/build wiring | DONE |
| 534 | DONE | `cpp/3_CUDA_Features/immaTensorCoreGemm/immaTensorCoreGemm.cu` | 655 | 23 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 535 | DONE | `cpp/3_CUDA_Features/jacobiCudaGraphs/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 536 | DONE | `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.cu` | 449 | 54 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 537 | DONE | `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobi.h` | 34 | 1 | no | host/helper source structure | DONE |
| 538 | DONE | `cpp/3_CUDA_Features/jacobiCudaGraphs/main.cpp` | 234 | 8 | no | CUDA Runtime memory/transfer, streams/events, CUDA Graphs, Driver API | DONE |
| 539 | DONE | `cpp/3_CUDA_Features/memMapIPCDrv/CMakeLists.txt` | 78 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 540 | DONE | `cpp/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp` | 680 | 17 | no | CUDA kernel launch, shared memory, streams/events, Driver API | DONE |
| 541 | DONE | `cpp/3_CUDA_Features/memMapIPCDrv/memMapIpc_kernel.cu` | 39 | 2 | no | thread/block indexing | DONE |
| 542 | DONE | `cpp/3_CUDA_Features/newdelete/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 543 | DONE | `cpp/3_CUDA_Features/newdelete/container.hpp` | 109 | 1 | no | host/helper source structure | DONE |
| 544 | DONE | `cpp/3_CUDA_Features/newdelete/newdelete.cu` | 347 | 30 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 545 | DONE | `cpp/3_CUDA_Features/ptxjit/CMakeLists.txt` | 71 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 546 | DONE | `cpp/3_CUDA_Features/ptxjit/ptxjit.cpp` | 258 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 547 | DONE | `cpp/3_CUDA_Features/ptxjit/ptxjit_kernel.cu` | 38 | 2 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX | DONE |
| 548 | DONE | `cpp/3_CUDA_Features/simpleCudaGraphs/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 549 | DONE | `cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu` | 461 | 54 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 550 | DONE | `cpp/3_CUDA_Features/tf32TensorCoreGemm/CMakeLists.txt` | 44 | 2 | no | CMake/build wiring | DONE |
| 551 | DONE | `cpp/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu` | 888 | 35 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 552 | DONE | `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 553 | DONE | `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu` | 330 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 554 | DONE | `cpp/4_CUDA_Libraries/CMakeLists.txt` | 42 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 555 | DONE | `cpp/4_CUDA_Libraries/FilterBorderControlNPP/CMakeLists.txt` | 78 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 556 | DONE | `cpp/4_CUDA_Libraries/FilterBorderControlNPP/FilterBorderControlNPP.cpp` | 620 | 18 | no | streams/events, Driver API, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 557 | DONE | `cpp/4_CUDA_Libraries/MersenneTwisterGP11213/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 558 | DONE | `cpp/4_CUDA_Libraries/MersenneTwisterGP11213/MersenneTwister.cpp` | 191 | 12 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 559 | DONE | `cpp/4_CUDA_Libraries/batchCUBLAS/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 560 | DONE | `cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.cpp` | 749 | 22 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 561 | DONE | `cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.h` | 182 | 3 | no | Driver API, CUDA libraries | DONE |
| 562 | DONE | `cpp/4_CUDA_Libraries/boxFilterNPP/CMakeLists.txt` | 75 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 563 | DONE | `cpp/4_CUDA_Libraries/boxFilterNPP/boxFilterNPP.cpp` | 221 | 8 | no | streams/events, Driver API, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 564 | DONE | `cpp/4_CUDA_Libraries/cannyEdgeDetectorNPP/CMakeLists.txt` | 75 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 565 | DONE | `cpp/4_CUDA_Libraries/cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp` | 270 | 10 | no | CUDA Runtime memory/transfer, streams/events, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 566 | DONE | `cpp/4_CUDA_Libraries/conjugateGradient/CMakeLists.txt` | 48 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 567 | DONE | `cpp/4_CUDA_Libraries/conjugateGradient/main.cpp` | 317 | 15 | no | CUDA Runtime memory/transfer, synchronization, Driver API, CUDA libraries | DONE |
| 568 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientCudaGraphs/CMakeLists.txt` | 48 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 569 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu` | 517 | 57 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 570 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/CMakeLists.txt` | 51 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 571 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu` | 547 | 20 | no | CUDA Runtime memory/transfer, thread/block indexing, shared memory, synchronization | DONE |
| 572 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/CMakeLists.txt` | 50 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 573 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu` | 832 | 20 | no | CUDA Runtime memory/transfer, shared memory, synchronization, streams/events | DONE |
| 574 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientPrecond/CMakeLists.txt` | 48 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 575 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientPrecond/main.cpp` | 658 | 32 | no | CUDA Runtime memory/transfer, Driver API, NVRTC/JIT/libNVVM/PTX, CUDA libraries | DONE |
| 576 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientUM/CMakeLists.txt` | 48 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 577 | DONE | `cpp/4_CUDA_Libraries/conjugateGradientUM/main.cpp` | 310 | 14 | no | CUDA Runtime memory/transfer, synchronization, Driver API, CUDA libraries | DONE |
| 578 | DONE | `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/CMakeLists.txt` | 60 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 579 | DONE | `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp` | 627 | 40 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 580 | DONE | `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/mmio.c` | 483 | 2 | no | host/helper source structure | DONE |
| 581 | DONE | `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/mmio.h` | 139 | 1 | no | host/helper source structure | DONE |
| 582 | DONE | `cpp/4_CUDA_Libraries/cuSolverDn_LinearSolver/mmio_wrapper.cpp` | 482 | 8 | no | Driver API, CUDA libraries | DONE |
| 583 | DONE | `cpp/4_CUDA_Libraries/cuSolverRf/CMakeLists.txt` | 62 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 584 | DONE | `cpp/4_CUDA_Libraries/cuSolverRf/cuSolverRf.cpp` | 942 | 43 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 585 | DONE | `cpp/4_CUDA_Libraries/cuSolverRf/mmio.c` | 476 | 2 | no | host/helper source structure | DONE |
| 586 | DONE | `cpp/4_CUDA_Libraries/cuSolverRf/mmio.h` | 139 | 1 | no | host/helper source structure | DONE |
| 587 | DONE | `cpp/4_CUDA_Libraries/cuSolverRf/mmio_wrapper.cpp` | 500 | 8 | no | Driver API, CUDA libraries | DONE |
| 588 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/CMakeLists.txt` | 62 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 589 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/cuSolverSp_LinearSolver.cpp` | 772 | 34 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 590 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.c` | 479 | 2 | no | host/helper source structure | DONE |
| 591 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio.h` | 139 | 1 | no | host/helper source structure | DONE |
| 592 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LinearSolver/mmio_wrapper.cpp` | 482 | 8 | no | Driver API, CUDA libraries | DONE |
| 593 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/CMakeLists.txt` | 62 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 594 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp` | 476 | 30 | no | CUDA Runtime memory/transfer, streams/events, Driver API, CUDA libraries | DONE |
| 595 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/mmio.c` | 479 | 2 | no | host/helper source structure | DONE |
| 596 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/mmio.h` | 139 | 1 | no | host/helper source structure | DONE |
| 597 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/mmio_wrapper.cpp` | 482 | 8 | no | Driver API, CUDA libraries | DONE |
| 598 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/CMakeLists.txt` | 65 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 599 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/cuSolverSp_LowlevelQR.cpp` | 520 | 30 | no | CUDA Runtime memory/transfer, streams/events, Driver API, CUDA libraries | DONE |
| 600 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/mmio.c` | 479 | 2 | no | host/helper source structure | DONE |
| 601 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/mmio.h` | 139 | 1 | no | host/helper source structure | DONE |
| 602 | DONE | `cpp/4_CUDA_Libraries/cuSolverSp_LowlevelQR/mmio_wrapper.cpp` | 500 | 8 | no | Driver API, CUDA libraries | DONE |
| 603 | DONE | `cpp/4_CUDA_Libraries/cubDeviceFind/CMakeLists.txt` | 65 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 604 | DONE | `cpp/4_CUDA_Libraries/cubDeviceFind/cubDeviceFind.cu` | 253 | 5 | no | synchronization, CUDA libraries | DONE |
| 605 | DONE | `cpp/4_CUDA_Libraries/cubDeviceSegmentedScan/CMakeLists.txt` | 66 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 606 | DONE | `cpp/4_CUDA_Libraries/cubDeviceSegmentedScan/cubDeviceSegmentedScan.cu` | 192 | 4 | no | synchronization, CUDA libraries | DONE |
| 607 | DONE | `cpp/4_CUDA_Libraries/cubDeviceTransform/CMakeLists.txt` | 65 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 608 | DONE | `cpp/4_CUDA_Libraries/cubDeviceTransform/cubDeviceTransform.cu` | 159 | 4 | no | synchronization, CUDA libraries | DONE |
| 609 | DONE | `cpp/4_CUDA_Libraries/cudaNvSci/CMakeLists.txt` | 70 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 610 | DONE | `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.cpp` | 631 | 11 | no | CUDA Runtime memory/transfer, streams/events, Driver API, graphics/interop | DONE |
| 611 | DONE | `cpp/4_CUDA_Libraries/cudaNvSci/cudaNvSci.h` | 114 | 2 | no | streams/events, Driver API, graphics/interop, multi-GPU/P2P/IPC | DONE |
| 612 | DONE | `cpp/4_CUDA_Libraries/cudaNvSci/imageKernels.cu` | 130 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 613 | DONE | `cpp/4_CUDA_Libraries/cudaNvSci/main.cpp` | 104 | 1 | no | Driver API, Python CUDA | DONE |
| 614 | DONE | `cpp/4_CUDA_Libraries/freeImageInteropNPP/CMakeLists.txt` | 75 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 615 | DONE | `cpp/4_CUDA_Libraries/freeImageInteropNPP/freeImageInteropNPP.cpp` | 336 | 11 | no | CUDA Runtime memory/transfer, streams/events, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 616 | DONE | `cpp/4_CUDA_Libraries/histEqualizationNPP/CMakeLists.txt` | 76 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 617 | DONE | `cpp/4_CUDA_Libraries/histEqualizationNPP/histEqualizationNPP.cpp` | 341 | 15 | no | CUDA Runtime memory/transfer, streams/events, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 618 | DONE | `cpp/4_CUDA_Libraries/jitLto/CMakeLists.txt` | 58 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 619 | DONE | `cpp/4_CUDA_Libraries/jitLto/jitLto.cpp` | 270 | 10 | no | CUDA kernel launch, thread/block indexing, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 620 | DONE | `cpp/4_CUDA_Libraries/libcuxxMdspan/CMakeLists.txt` | 95 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 621 | DONE | `cpp/4_CUDA_Libraries/libcuxxMdspan/libcuxxMdspan.cu` | 259 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 622 | DONE | `cpp/4_CUDA_Libraries/libcuxxRandom/CMakeLists.txt` | 68 | 2 | no | CMake/build wiring, CUDA libraries | DONE |
| 623 | DONE | `cpp/4_CUDA_Libraries/libcuxxRandom/libcuxxRandom.cu` | 187 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 624 | DONE | `cpp/4_CUDA_Libraries/lineOfSight/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 625 | DONE | `cpp/4_CUDA_Libraries/lineOfSight/lineOfSight.cu` | 358 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 626 | DONE | `cpp/4_CUDA_Libraries/matrixMulCUBLAS/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 627 | DONE | `cpp/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp` | 403 | 15 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 628 | DONE | `cpp/4_CUDA_Libraries/nvJPEG/CMakeLists.txt` | 54 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 629 | DONE | `cpp/4_CUDA_Libraries/nvJPEG/nvJPEG.cpp` | 669 | 29 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 630 | DONE | `cpp/4_CUDA_Libraries/nvJPEG_encoder/CMakeLists.txt` | 60 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 631 | DONE | `cpp/4_CUDA_Libraries/nvJPEG_encoder/nvJPEG_encoder.cpp` | 485 | 19 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 632 | DONE | `cpp/4_CUDA_Libraries/oceanFFT/CMakeLists.txt` | 103 | 3 | no | CMake/build wiring, CUDA libraries, graphics/interop | DONE |
| 633 | DONE | `cpp/4_CUDA_Libraries/oceanFFT/data/ocean.frag` | 30 | 0 | no | shader source | DONE |
| 634 | DONE | `cpp/4_CUDA_Libraries/oceanFFT/data/ocean.vert` | 24 | 0 | no | shader source | DONE |
| 635 | DONE | `cpp/4_CUDA_Libraries/oceanFFT/oceanFFT.cpp` | 929 | 26 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, CUDA libraries | DONE |
| 636 | DONE | `cpp/4_CUDA_Libraries/oceanFFT/oceanFFT_kernel.cu` | 165 | 7 | no | CUDA kernel launch, thread/block indexing, Driver API, CUDA libraries | DONE |
| 637 | DONE | `cpp/4_CUDA_Libraries/randomFog/CMakeLists.txt` | 121 | 3 | no | CMake/build wiring, CUDA libraries, graphics/interop | DONE |
| 638 | DONE | `cpp/4_CUDA_Libraries/randomFog/randomFog.cpp` | 757 | 5 | no | CUDA libraries, graphics/interop, shader source | DONE |
| 639 | DONE | `cpp/4_CUDA_Libraries/randomFog/rng.cpp` | 324 | 10 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries | DONE |
| 640 | DONE | `cpp/4_CUDA_Libraries/randomFog/rng.h` | 74 | 2 | no | CUDA libraries | DONE |
| 641 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLAS/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 642 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLAS/simpleCUBLAS.cpp` | 264 | 11 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries | DONE |
| 643 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLASXT/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 644 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLASXT/simpleCUBLASXT.cpp` | 316 | 8 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries | DONE |
| 645 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLAS_LU/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 646 | DONE | `cpp/4_CUDA_Libraries/simpleCUBLAS_LU/simpleCUBLAS_LU.cpp` | 428 | 10 | no | CUDA Runtime memory/transfer, Driver API, CUDA libraries, multi-GPU/P2P/IPC | DONE |
| 647 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 648 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT/simpleCUFFT.cu` | 295 | 14 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 649 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_2d_MGPU/CMakeLists.txt` | 51 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 650 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_2d_MGPU/simpleCUFFT_2d_MGPU.cu` | 390 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 651 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_MGPU/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 652 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_MGPU/simpleCUFFT_MGPU.cu` | 400 | 13 | no | CUDA kernel launch, thread/block indexing, synchronization, Driver API | DONE |
| 653 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_callback/CMakeLists.txt` | 47 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 654 | DONE | `cpp/4_CUDA_Libraries/simpleCUFFT_callback/simpleCUFFT_callback.cu` | 333 | 16 | no | CUDA Runtime memory/transfer, shared memory, Driver API, CUDA libraries | DONE |
| 655 | DONE | `cpp/4_CUDA_Libraries/watershedSegmentationNPP/CMakeLists.txt` | 75 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 656 | DONE | `cpp/4_CUDA_Libraries/watershedSegmentationNPP/watershedSegmentationNPP.cpp` | 629 | 33 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 657 | DONE | `cpp/5_Domain_Specific/BlackScholes/BlackScholes.cu` | 254 | 8 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 658 | DONE | `cpp/5_Domain_Specific/BlackScholes/BlackScholes_gold.cpp` | 99 | 1 | no | host/helper source structure | DONE |
| 659 | DONE | `cpp/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh` | 121 | 2 | no | thread/block indexing | DONE |
| 660 | DONE | `cpp/5_Domain_Specific/BlackScholes/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 661 | DONE | `cpp/5_Domain_Specific/BlackScholes_nvrtc/BlackScholes.cpp` | 288 | 7 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 662 | DONE | `cpp/5_Domain_Specific/BlackScholes_nvrtc/BlackScholes_gold.cpp` | 101 | 1 | no | host/helper source structure | DONE |
| 663 | DONE | `cpp/5_Domain_Specific/BlackScholes_nvrtc/BlackScholes_kernel.cuh` | 118 | 2 | no | thread/block indexing | DONE |
| 664 | DONE | `cpp/5_Domain_Specific/BlackScholes_nvrtc/CMakeLists.txt` | 49 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 665 | DONE | `cpp/5_Domain_Specific/CMakeLists.txt` | 38 | 1 | no | CMake/build wiring | DONE |
| 666 | DONE | `cpp/5_Domain_Specific/FDTD3d/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 667 | DONE | `cpp/5_Domain_Specific/FDTD3d/inc/FDTD3d.h` | 55 | 1 | no | host/helper source structure | DONE |
| 668 | DONE | `cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dGPU.h` | 57 | 1 | no | host/helper source structure | DONE |
| 669 | DONE | `cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh` | 165 | 5 | no | thread/block indexing, shared memory | DONE |
| 670 | DONE | `cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dReference.h` | 61 | 2 | no | host/helper source structure | DONE |
| 671 | DONE | `cpp/5_Domain_Specific/FDTD3d/src/FDTD3d.cpp` | 234 | 2 | no | host/helper source structure | DONE |
| 672 | DONE | `cpp/5_Domain_Specific/FDTD3d/src/FDTD3dGPU.cu` | 271 | 11 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 673 | DONE | `cpp/5_Domain_Specific/FDTD3d/src/FDTD3dReference.cpp` | 194 | 3 | no | host/helper source structure | DONE |
| 674 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/CMakeLists.txt` | 47 | 2 | no | CMake/build wiring | DONE |
| 675 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/addKernel.cuh` | 65 | 3 | no | CUDA kernel launch, thread/block indexing | DONE |
| 676 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/common.h` | 77 | 1 | no | host/helper source structure | DONE |
| 677 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/derivativesKernel.cuh` | 150 | 3 | no | CUDA kernel launch, thread/block indexing | DONE |
| 678 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/downscaleKernel.cuh` | 100 | 3 | no | CUDA kernel launch, thread/block indexing | DONE |
| 679 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.cu` | 261 | 8 | no | CUDA Runtime memory/transfer, Driver API, multi-GPU/P2P/IPC | DONE |
| 680 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/flowCUDA.h` | 43 | 1 | no | host/helper source structure | DONE |
| 681 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/flowGold.cpp` | 524 | 1 | no | host/helper source structure | DONE |
| 682 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/flowGold.h` | 44 | 1 | no | host/helper source structure | DONE |
| 683 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/main.cpp` | 240 | 1 | no | host/helper source structure | DONE |
| 684 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/solverKernel.cuh` | 206 | 7 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 685 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/upscaleKernel.cuh` | 105 | 3 | no | CUDA kernel launch, thread/block indexing | DONE |
| 686 | DONE | `cpp/5_Domain_Specific/HSOpticalFlow/warpingKernel.cuh` | 108 | 3 | no | CUDA kernel launch, thread/block indexing | DONE |
| 687 | DONE | `cpp/5_Domain_Specific/Mandelbrot/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 688 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot.cpp` | 1372 | 19 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 689 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_cuda.cu` | 540 | 11 | no | CUDA kernel launch, thread/block indexing | DONE |
| 690 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.cpp` | 343 | 1 | no | host/helper source structure | DONE |
| 691 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_gold.h` | 87 | 1 | no | host/helper source structure | DONE |
| 692 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.cuh` | 485 | 1 | no | host/helper source structure | DONE |
| 693 | DONE | `cpp/5_Domain_Specific/Mandelbrot/Mandelbrot_kernel.h` | 67 | 1 | no | host/helper source structure | DONE |
| 694 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/CMakeLists.txt` | 46 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 695 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp` | 490 | 7 | no | synchronization, streams/events, Driver API, multi-GPU/P2P/IPC | DONE |
| 696 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_common.h` | 104 | 3 | no | streams/events, CUDA libraries, multi-GPU/P2P/IPC | DONE |
| 697 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_gold.cpp` | 144 | 3 | no | Driver API, CUDA libraries | DONE |
| 698 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_kernel.cu` | 243 | 19 | no | CUDA Runtime memory/transfer, thread/block indexing, shared memory, synchronization | DONE |
| 699 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/MonteCarlo_reduction.cuh` | 83 | 4 | no | thread/block indexing, shared memory | DONE |
| 700 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.cpp` | 75 | 1 | no | Driver API | DONE |
| 701 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/multithreading.h` | 73 | 1 | no | Driver API | DONE |
| 702 | DONE | `cpp/5_Domain_Specific/MonteCarloMultiGPU/realtype.h` | 40 | 1 | no | host/helper source structure | DONE |
| 703 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/CMakeLists.txt` | 47 | 2 | no | CMake/build wiring | DONE |
| 704 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/bgr_resize.cu` | 180 | 6 | no | CUDA kernel launch, thread/block indexing, streams/events, Driver API | DONE |
| 705 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/nv12_resize.cu` | 128 | 6 | no | CUDA kernel launch, thread/block indexing, streams/events, Driver API | DONE |
| 706 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/nv12_to_bgr_planar.cu` | 164 | 5 | no | CUDA kernel launch, thread/block indexing, streams/events, Python CUDA | DONE |
| 707 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert.h` | 77 | 4 | no | streams/events | DONE |
| 708 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/resize_convert_main.cpp` | 494 | 20 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 709 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/utils.cu` | 154 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, multi-GPU/P2P/IPC | DONE |
| 710 | DONE | `cpp/5_Domain_Specific/NV12toBGRandResize/utils.h` | 35 | 1 | no | host/helper source structure | DONE |
| 711 | DONE | `cpp/5_Domain_Specific/SobelFilter/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 712 | DONE | `cpp/5_Domain_Specific/SobelFilter/SobelFilter.cpp` | 498 | 11 | no | CUDA Runtime memory/transfer, shared memory, synchronization, CUDA Graphs | DONE |
| 713 | DONE | `cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.cu` | 296 | 13 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 714 | DONE | `cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.h` | 44 | 1 | no | host/helper source structure | DONE |
| 715 | DONE | `cpp/5_Domain_Specific/SobolQRNG/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 716 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol.cpp` | 312 | 6 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 717 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol.h` | 63 | 1 | no | host/helper source structure | DONE |
| 718 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_gold.cpp` | 179 | 1 | no | host/helper source structure | DONE |
| 719 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_gold.h` | 64 | 1 | no | host/helper source structure | DONE |
| 720 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.cu` | 215 | 7 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 721 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.h` | 63 | 1 | no | host/helper source structure | DONE |
| 722 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.cpp` | 10273 | 1 | no | host/helper source structure | DONE |
| 723 | DONE | `cpp/5_Domain_Specific/SobolQRNG/sobol_primitives.h` | 78 | 1 | no | host/helper source structure | DONE |
| 724 | DONE | `cpp/5_Domain_Specific/bicubicTexture/CMakeLists.txt` | 99 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 725 | DONE | `cpp/5_Domain_Specific/bicubicTexture/bicubicTexture.cpp` | 776 | 12 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 726 | DONE | `cpp/5_Domain_Specific/bicubicTexture/bicubicTexture_cuda.cu` | 140 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, Driver API | DONE |
| 727 | DONE | `cpp/5_Domain_Specific/bicubicTexture/bicubicTexture_kernel.cuh` | 376 | 5 | no | thread/block indexing, Driver API | DONE |
| 728 | DONE | `cpp/5_Domain_Specific/bilateralFilter/CMakeLists.txt` | 100 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 729 | DONE | `cpp/5_Domain_Specific/bilateralFilter/bilateralFilter.cpp` | 696 | 14 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 730 | DONE | `cpp/5_Domain_Specific/bilateralFilter/bilateralFilter_cpu.cpp` | 161 | 1 | no | host/helper source structure | DONE |
| 731 | DONE | `cpp/5_Domain_Specific/bilateralFilter/bilateral_kernel.cu` | 273 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 732 | DONE | `cpp/5_Domain_Specific/bilateralFilter/bmploader.cpp` | 139 | 1 | no | host/helper source structure | DONE |
| 733 | DONE | `cpp/5_Domain_Specific/binomialOptions/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 734 | DONE | `cpp/5_Domain_Specific/binomialOptions/binomialOptions.cpp` | 191 | 2 | no | synchronization | DONE |
| 735 | DONE | `cpp/5_Domain_Specific/binomialOptions/binomialOptions_common.h` | 54 | 1 | no | host/helper source structure | DONE |
| 736 | DONE | `cpp/5_Domain_Specific/binomialOptions/binomialOptions_gold.cpp` | 127 | 1 | no | host/helper source structure | DONE |
| 737 | DONE | `cpp/5_Domain_Specific/binomialOptions/binomialOptions_kernel.cu` | 165 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 738 | DONE | `cpp/5_Domain_Specific/binomialOptions/realtype.h` | 40 | 1 | no | host/helper source structure | DONE |
| 739 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/CMakeLists.txt` | 67 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 740 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/binomialOptions.cpp` | 199 | 1 | no | Python CUDA | DONE |
| 741 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/binomialOptions_common.h` | 55 | 1 | no | host/helper source structure | DONE |
| 742 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/binomialOptions_gold.cpp` | 133 | 1 | no | host/helper source structure | DONE |
| 743 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/binomialOptions_gpu.cpp` | 138 | 6 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 744 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/binomialOptions_kernel.cu` | 115 | 5 | no | thread/block indexing, shared memory, synchronization | DONE |
| 745 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/common_gpu_header.h` | 32 | 1 | no | host/helper source structure | DONE |
| 746 | DONE | `cpp/5_Domain_Specific/binomialOptions_nvrtc/realtype.h` | 42 | 1 | no | host/helper source structure | DONE |
| 747 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/CMakeLists.txt` | 46 | 3 | no | CMake/build wiring, CUDA libraries | DONE |
| 748 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cu` | 367 | 13 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 749 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D.cuh` | 461 | 7 | no | thread/block indexing | DONE |
| 750 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_common.h` | 91 | 1 | no | host/helper source structure | DONE |
| 751 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/convolutionFFT2D_gold.cpp` | 72 | 1 | no | host/helper source structure | DONE |
| 752 | DONE | `cpp/5_Domain_Specific/convolutionFFT2D/main.cpp` | 564 | 34 | no | CUDA Runtime memory/transfer, synchronization, Driver API, CUDA libraries | DONE |
| 753 | DONE | `cpp/5_Domain_Specific/dwtHaar1D/CMakeLists.txt` | 43 | 2 | no | CMake/build wiring | DONE |
| 754 | DONE | `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D.cu` | 396 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, shared memory, Driver API | DONE |
| 755 | DONE | `cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh` | 256 | 8 | no | thread/block indexing, shared memory | DONE |
| 756 | DONE | `cpp/5_Domain_Specific/dxtc/CMakeLists.txt` | 43 | 2 | no | CMake/build wiring | DONE |
| 757 | DONE | `cpp/5_Domain_Specific/dxtc/CudaMath.h` | 114 | 7 | no | thread/block indexing, shared memory | DONE |
| 758 | DONE | `cpp/5_Domain_Specific/dxtc/dds.h` | 87 | 1 | no | host/helper source structure | DONE |
| 759 | DONE | `cpp/5_Domain_Specific/dxtc/dxtc.cu` | 820 | 28 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 760 | DONE | `cpp/5_Domain_Specific/dxtc/permutations.h` | 147 | 2 | no | host/helper source structure | DONE |
| 761 | DONE | `cpp/5_Domain_Specific/fastWalshTransform/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 762 | DONE | `cpp/5_Domain_Specific/fastWalshTransform/fastWalshTransform.cu` | 170 | 6 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 763 | DONE | `cpp/5_Domain_Specific/fastWalshTransform/fastWalshTransform_gold.cpp` | 100 | 1 | no | host/helper source structure | DONE |
| 764 | DONE | `cpp/5_Domain_Specific/fastWalshTransform/fastWalshTransform_kernel.cuh` | 200 | 13 | no | CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 765 | DONE | `cpp/5_Domain_Specific/fluidsGL/CMakeLists.txt` | 101 | 3 | no | CMake/build wiring, CUDA libraries, graphics/interop | DONE |
| 766 | DONE | `cpp/5_Domain_Specific/fluidsGL/defines.h` | 48 | 1 | no | host/helper source structure | DONE |
| 767 | DONE | `cpp/5_Domain_Specific/fluidsGL/fluidsGL.cpp` | 509 | 13 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, CUDA libraries | DONE |
| 768 | DONE | `cpp/5_Domain_Specific/fluidsGL/fluidsGL_kernels.cu` | 379 | 19 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, CUDA Graphs | DONE |
| 769 | DONE | `cpp/5_Domain_Specific/fluidsGL/fluidsGL_kernels.cuh` | 72 | 1 | no | host/helper source structure | DONE |
| 770 | DONE | `cpp/5_Domain_Specific/fluidsGL/fluidsGL_kernels.h` | 70 | 1 | no | host/helper source structure | DONE |
| 771 | DONE | `cpp/5_Domain_Specific/marchingCubes/CMakeLists.txt` | 93 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 772 | DONE | `cpp/5_Domain_Specific/marchingCubes/defines.h` | 50 | 1 | no | host/helper source structure | DONE |
| 773 | DONE | `cpp/5_Domain_Specific/marchingCubes/marchingCubes.cpp` | 1056 | 17 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, graphics/interop | DONE |
| 774 | DONE | `cpp/5_Domain_Specific/marchingCubes/marchingCubes_kernel.cu` | 769 | 22 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 775 | DONE | `cpp/5_Domain_Specific/marchingCubes/tables.h` | 199 | 1 | no | shader source | DONE |
| 776 | DONE | `cpp/5_Domain_Specific/nbody/CMakeLists.txt` | 92 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 777 | DONE | `cpp/5_Domain_Specific/nbody/bodysystem.h` | 292 | 1 | no | host/helper source structure | DONE |
| 778 | DONE | `cpp/5_Domain_Specific/nbody/bodysystemcpu.h` | 78 | 1 | no | host/helper source structure | DONE |
| 779 | DONE | `cpp/5_Domain_Specific/nbody/bodysystemcpu_impl.h` | 272 | 2 | no | host/helper source structure | DONE |
| 780 | DONE | `cpp/5_Domain_Specific/nbody/bodysystemcuda.cu` | 299 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 781 | DONE | `cpp/5_Domain_Specific/nbody/bodysystemcuda.h` | 107 | 3 | no | streams/events, CUDA Graphs, graphics/interop, multi-GPU/P2P/IPC | DONE |
| 782 | DONE | `cpp/5_Domain_Specific/nbody/bodysystemcuda_impl.h` | 435 | 17 | no | CUDA Runtime memory/transfer, streams/events, CUDA Graphs, Driver API | DONE |
| 783 | DONE | `cpp/5_Domain_Specific/nbody/nbody.cpp` | 1389 | 28 | no | synchronization, streams/events, Driver API, graphics/interop | DONE |
| 784 | DONE | `cpp/5_Domain_Specific/nbody/render_particles.cpp` | 377 | 1 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop, shader source | DONE |
| 785 | DONE | `cpp/5_Domain_Specific/nbody/render_particles.h` | 80 | 1 | no | host/helper source structure | DONE |
| 786 | DONE | `cpp/5_Domain_Specific/nbody/tipsy.h` | 168 | 1 | no | host/helper source structure | DONE |
| 787 | DONE | `cpp/5_Domain_Specific/p2pBandwidthLatencyTest/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 788 | DONE | `cpp/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu` | 885 | 49 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 789 | DONE | `cpp/5_Domain_Specific/postProcessGL/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 790 | DONE | `cpp/5_Domain_Specific/postProcessGL/main.cpp` | 1070 | 15 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 791 | DONE | `cpp/5_Domain_Specific/postProcessGL/postProcessGL.cu` | 279 | 8 | no | CUDA kernel launch, thread/block indexing, shared memory, synchronization | DONE |
| 792 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 793 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator.cpp` | 187 | 9 | no | CUDA Runtime memory/transfer, synchronization, Driver API, multi-GPU/P2P/IPC | DONE |
| 794 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_common.h` | 41 | 1 | no | host/helper source structure | DONE |
| 795 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_gold.cpp` | 331 | 1 | no | Driver API | DONE |
| 796 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_kernel.cu` | 181 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 797 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/CMakeLists.txt` | 49 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 798 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/quasirandomGenerator.cpp` | 187 | 7 | no | Driver API, multi-GPU/P2P/IPC, Python CUDA | DONE |
| 799 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/quasirandomGenerator_common.h` | 42 | 1 | no | host/helper source structure | DONE |
| 800 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/quasirandomGenerator_gold.cpp` | 340 | 1 | no | Driver API | DONE |
| 801 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/quasirandomGenerator_gpu.cuh` | 114 | 6 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 802 | DONE | `cpp/5_Domain_Specific/quasirandomGenerator_nvrtc/quasirandomGenerator_kernel.cu` | 161 | 4 | no | thread/block indexing, Driver API | DONE |
| 803 | DONE | `cpp/5_Domain_Specific/recursiveGaussian/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 804 | DONE | `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian.cpp` | 546 | 16 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 805 | DONE | `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_cuda.cu` | 159 | 4 | no | CUDA kernel launch | DONE |
| 806 | DONE | `cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_kernel.cuh` | 241 | 6 | no | thread/block indexing, shared memory | DONE |
| 807 | DONE | `cpp/5_Domain_Specific/simpleD3D11/CMakeLists.txt` | 62 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 808 | DONE | `cpp/5_Domain_Specific/simpleD3D11/ShaderStructs.h` | 50 | 2 | no | streams/events, shader source | DONE |
| 809 | DONE | `cpp/5_Domain_Specific/simpleD3D11/simpleD3D11.cpp` | 660 | 6 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 810 | DONE | `cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.cu` | 146 | 6 | no | CUDA kernel launch, thread/block indexing, streams/events, Driver API | DONE |
| 811 | DONE | `cpp/5_Domain_Specific/simpleD3D11/sinewave_cuda.h` | 47 | 2 | no | streams/events, Driver API, shader source | DONE |
| 812 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/CMakeLists.txt` | 64 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 813 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/d3dx11effect/d3dx11effect.h` | 1728 | 0 | no | graphics/interop | DONE |
| 814 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/simpleD3D11Texture.cpp` | 1293 | 25 | no | CUDA Runtime memory/transfer, streams/events, CUDA Graphs, Driver API | DONE |
| 815 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/texture_2d.cu` | 81 | 3 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 816 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/texture_3d.cu` | 83 | 3 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 817 | DONE | `cpp/5_Domain_Specific/simpleD3D11Texture/texture_cube.cu` | 95 | 3 | no | CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 818 | DONE | `cpp/5_Domain_Specific/simpleD3D12/CMakeLists.txt` | 66 | 2 | no | CMake/build wiring, graphics/interop, shader source | DONE |
| 819 | DONE | `cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.cpp` | 152 | 2 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 820 | DONE | `cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.h` | 104 | 2 | no | graphics/interop | DONE |
| 821 | DONE | `cpp/5_Domain_Specific/simpleD3D12/DXSampleHelper.h` | 214 | 2 | no | graphics/interop | DONE |
| 822 | DONE | `cpp/5_Domain_Specific/simpleD3D12/Main.cpp` | 36 | 1 | no | graphics/interop | DONE |
| 823 | DONE | `cpp/5_Domain_Specific/simpleD3D12/ShaderStructs.h` | 53 | 2 | no | streams/events, graphics/interop, shader source | DONE |
| 824 | DONE | `cpp/5_Domain_Specific/simpleD3D12/Win32Application.cpp` | 155 | 2 | no | host/helper source structure | DONE |
| 825 | DONE | `cpp/5_Domain_Specific/simpleD3D12/Win32Application.h` | 72 | 2 | no | host/helper source structure | DONE |
| 826 | DONE | `cpp/5_Domain_Specific/simpleD3D12/d3dx12.h` | 1472 | 2 | no | graphics/interop | DONE |
| 827 | DONE | `cpp/5_Domain_Specific/simpleD3D12/shaders.hlsl` | 51 | 0 | no | shader source | DONE |
| 828 | DONE | `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp` | 546 | 8 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 829 | DONE | `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.h` | 151 | 2 | no | streams/events, graphics/interop, shader source | DONE |
| 830 | DONE | `cpp/5_Domain_Specific/simpleD3D12/sinewave_cuda.cu` | 75 | 4 | no | CUDA kernel launch, thread/block indexing, streams/events, shader source | DONE |
| 831 | DONE | `cpp/5_Domain_Specific/simpleD3D12/stdafx.cpp` | 29 | 1 | no | host/helper source structure | DONE |
| 832 | DONE | `cpp/5_Domain_Specific/simpleD3D12/stdafx.h` | 48 | 1 | no | graphics/interop | DONE |
| 833 | DONE | `cpp/5_Domain_Specific/simpleGL/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 834 | DONE | `cpp/5_Domain_Specific/simpleGL/simpleGL.cu` | 583 | 18 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 835 | DONE | `cpp/5_Domain_Specific/simpleVulkan/CMakeLists.txt` | 117 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 836 | DONE | `cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.cu` | 145 | 5 | no | CUDA kernel launch, thread/block indexing, streams/events, Driver API | DONE |
| 837 | DONE | `cpp/5_Domain_Specific/simpleVulkan/SineWaveSimulation.h` | 58 | 2 | no | streams/events | DONE |
| 838 | DONE | `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp` | 1909 | 3 | no | graphics/interop | DONE |
| 839 | DONE | `cpp/5_Domain_Specific/simpleVulkan/VulkanBaseApp.h` | 169 | 1 | no | graphics/interop | DONE |
| 840 | DONE | `cpp/5_Domain_Specific/simpleVulkan/linmath.h` | 580 | 1 | no | graphics/interop | DONE |
| 841 | DONE | `cpp/5_Domain_Specific/simpleVulkan/main.cpp` | 532 | 6 | no | synchronization, streams/events, Driver API, graphics/interop | DONE |
| 842 | DONE | `cpp/5_Domain_Specific/simpleVulkan/sinewave.frag` | 38 | 0 | no | shader source | DONE |
| 843 | DONE | `cpp/5_Domain_Specific/simpleVulkan/sinewave.vert` | 43 | 0 | no | shader source | DONE |
| 844 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/CMakeLists.txt` | 117 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 845 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.cu` | 305 | 15 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, streams/events | DONE |
| 846 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/MonteCarloPi.h` | 96 | 2 | no | streams/events, CUDA libraries, graphics/interop, Python CUDA | DONE |
| 847 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp` | 1797 | 3 | no | graphics/interop | DONE |
| 848 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.h` | 164 | 1 | no | graphics/interop | DONE |
| 849 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/VulkanCudaInterop.h` | 80 | 2 | no | NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 850 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/main.cpp` | 355 | 7 | no | synchronization, streams/events, Driver API, graphics/interop | DONE |
| 851 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.frag` | 37 | 0 | no | shader source | DONE |
| 852 | DONE | `cpp/5_Domain_Specific/simpleVulkanMMAP/montecarlo.vert` | 55 | 0 | no | shader source | DONE |
| 853 | DONE | `cpp/5_Domain_Specific/smokeParticles/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 854 | DONE | `cpp/5_Domain_Specific/smokeParticles/GLSLProgram.cpp` | 257 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 855 | DONE | `cpp/5_Domain_Specific/smokeParticles/GLSLProgram.h` | 72 | 1 | no | NVRTC/JIT/libNVVM/PTX, shader source | DONE |
| 856 | DONE | `cpp/5_Domain_Specific/smokeParticles/GpuArray.h` | 293 | 7 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, graphics/interop | DONE |
| 857 | DONE | `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cpp` | 404 | 2 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 858 | DONE | `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.cuh` | 47 | 1 | no | host/helper source structure | DONE |
| 859 | DONE | `cpp/5_Domain_Specific/smokeParticles/ParticleSystem.h` | 131 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 860 | DONE | `cpp/5_Domain_Specific/smokeParticles/ParticleSystem_cuda.cu` | 163 | 3 | no | CUDA Runtime memory/transfer, Driver API, graphics/interop | DONE |
| 861 | DONE | `cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.cpp` | 586 | 2 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop, shader source | DONE |
| 862 | DONE | `cpp/5_Domain_Specific/smokeParticles/SmokeRenderer.h` | 169 | 1 | no | host/helper source structure | DONE |
| 863 | DONE | `cpp/5_Domain_Specific/smokeParticles/SmokeShaders.cpp` | 303 | 1 | no | NVRTC/JIT/libNVVM/PTX, shader source | DONE |
| 864 | DONE | `cpp/5_Domain_Specific/smokeParticles/SmokeShaders.h` | 33 | 1 | no | host/helper source structure | DONE |
| 865 | DONE | `cpp/5_Domain_Specific/smokeParticles/framebufferObject.cpp` | 368 | 2 | no | host/helper source structure | DONE |
| 866 | DONE | `cpp/5_Domain_Specific/smokeParticles/framebufferObject.h` | 232 | 2 | no | graphics/interop | DONE |
| 867 | DONE | `cpp/5_Domain_Specific/smokeParticles/nvMath.h` | 91 | 2 | no | graphics/interop | DONE |
| 868 | DONE | `cpp/5_Domain_Specific/smokeParticles/nvMatrix.h` | 484 | 2 | no | graphics/interop | DONE |
| 869 | DONE | `cpp/5_Domain_Specific/smokeParticles/nvQuaternion.h` | 457 | 2 | no | graphics/interop | DONE |
| 870 | DONE | `cpp/5_Domain_Specific/smokeParticles/nvVector.h` | 942 | 2 | no | graphics/interop | DONE |
| 871 | DONE | `cpp/5_Domain_Specific/smokeParticles/particleDemo.cpp` | 950 | 4 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 872 | DONE | `cpp/5_Domain_Specific/smokeParticles/particles_kernel.cuh` | 52 | 1 | no | host/helper source structure | DONE |
| 873 | DONE | `cpp/5_Domain_Specific/smokeParticles/particles_kernel_device.cuh` | 125 | 1 | no | host/helper source structure | DONE |
| 874 | DONE | `cpp/5_Domain_Specific/smokeParticles/renderbuffer.cpp` | 119 | 2 | no | host/helper source structure | DONE |
| 875 | DONE | `cpp/5_Domain_Specific/smokeParticles/renderbuffer.h` | 108 | 2 | no | NVRTC/JIT/libNVVM/PTX, graphics/interop | DONE |
| 876 | DONE | `cpp/5_Domain_Specific/stereoDisparity/CMakeLists.txt` | 43 | 2 | no | CMake/build wiring | DONE |
| 877 | DONE | `cpp/5_Domain_Specific/stereoDisparity/stereoDisparity.cu` | 295 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 878 | DONE | `cpp/5_Domain_Specific/stereoDisparity/stereoDisparity_kernel.cuh` | 271 | 7 | no | thread/block indexing, shared memory, NVRTC/JIT/libNVVM/PTX | DONE |
| 879 | DONE | `cpp/5_Domain_Specific/volumeFiltering/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 880 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volume.cpp` | 94 | 4 | no | CUDA Runtime memory/transfer, Driver API | DONE |
| 881 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volume.h` | 89 | 1 | no | host/helper source structure | DONE |
| 882 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volumeFilter.h` | 49 | 1 | no | host/helper source structure | DONE |
| 883 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volumeFilter_kernel.cu` | 120 | 3 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 884 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volumeFiltering.cpp` | 787 | 15 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 885 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volumeRender.h` | 56 | 1 | no | host/helper source structure | DONE |
| 886 | DONE | `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` | 743 | 12 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 887 | DONE | `cpp/5_Domain_Specific/volumeRender/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 888 | DONE | `cpp/5_Domain_Specific/volumeRender/volumeRender.cpp` | 662 | 14 | no | CUDA Runtime memory/transfer, synchronization, CUDA Graphs, Driver API | DONE |
| 889 | DONE | `cpp/5_Domain_Specific/volumeRender/volumeRender_kernel.cu` | 376 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, Driver API | DONE |
| 890 | DONE | `cpp/5_Domain_Specific/vulkanImageCUDA/CMakeLists.txt` | 117 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 891 | DONE | `cpp/5_Domain_Specific/vulkanImageCUDA/linmath.h` | 579 | 1 | no | graphics/interop | DONE |
| 892 | DONE | `cpp/5_Domain_Specific/vulkanImageCUDA/shader.frag` | 13 | 0 | no | shader source | DONE |
| 893 | DONE | `cpp/5_Domain_Specific/vulkanImageCUDA/shader.vert` | 26 | 0 | no | shader source | DONE |
| 894 | DONE | `cpp/5_Domain_Specific/vulkanImageCUDA/vulkanImageCUDA.cu` | 2593 | 14 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, streams/events | DONE |
| 895 | DONE | `cpp/6_Performance/CMakeLists.txt` | 8 | 2 | no | CMake/build wiring, CUDA Graphs | DONE |
| 896 | DONE | `cpp/6_Performance/LargeKernelParameter/CMakeLists.txt` | 38 | 2 | no | CMake/build wiring | DONE |
| 897 | DONE | `cpp/6_Performance/LargeKernelParameter/LargeKernelParameter.cu` | 196 | 16 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 898 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/CMakeLists.txt` | 41 | 2 | no | CMake/build wiring | DONE |
| 899 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/commonDefs.hpp` | 82 | 1 | no | Python CUDA | DONE |
| 900 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/commonKernels.cu` | 35 | 1 | no | host/helper source structure | DONE |
| 901 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/commonKernels.hpp` | 29 | 1 | no | host/helper source structure | DONE |
| 902 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/helperFunctions.cpp` | 296 | 3 | no | host/helper source structure | DONE |
| 903 | DONE | `cpp/6_Performance/UnifiedMemoryPerf/matrixMultiplyPerf.cu` | 716 | 32 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 904 | DONE | `cpp/6_Performance/alignedTypes/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 905 | DONE | `cpp/6_Performance/alignedTypes/alignedTypes.cu` | 323 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 906 | DONE | `cpp/6_Performance/cudaGraphsPerfScaling/CMakeLists.txt` | 39 | 4 | no | CMake/build wiring, CUDA Graphs | DONE |
| 907 | DONE | `cpp/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu` | 477 | 38 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, streams/events | DONE |
| 908 | DONE | `cpp/6_Performance/cudaGraphsPerfScaling/dataCollection.bash` | 17 | 0 | no | CUDA Graphs | DONE |
| 909 | DONE | `cpp/6_Performance/transpose/CMakeLists.txt` | 37 | 2 | no | CMake/build wiring | DONE |
| 910 | DONE | `cpp/6_Performance/transpose/transpose.cu` | 685 | 43 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 911 | DONE | `cpp/7_libNVVM/CMakeLists.txt` | 151 | 6 | no | CMake/build wiring, shared memory, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 912 | DONE | `cpp/7_libNVVM/common/include/DDSWriter.h` | 124 | 1 | no | host/helper source structure | DONE |
| 913 | DONE | `cpp/7_libNVVM/cuda-c-linking/CMakeLists.txt` | 101 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 914 | DONE | `cpp/7_libNVVM/cuda-c-linking/cuda-c-linking.cpp` | 337 | 11 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 915 | DONE | `cpp/7_libNVVM/cuda-c-linking/math-funcs.cu` | 87 | 2 | no | thread/block indexing | DONE |
| 916 | DONE | `cpp/7_libNVVM/cuda-shared-memory/CMakeLists.txt` | 60 | 2 | no | CMake/build wiring, shared memory, NVRTC/JIT/libNVVM/PTX | DONE |
| 917 | DONE | `cpp/7_libNVVM/cuda-shared-memory/extern_shared_memory.ll` | 58 | 0 | no | shared memory, NVRTC/JIT/libNVVM/PTX | DONE |
| 918 | DONE | `cpp/7_libNVVM/cuda-shared-memory/shared_memory.ll` | 58 | 0 | no | shared memory, NVRTC/JIT/libNVVM/PTX | DONE |
| 919 | DONE | `cpp/7_libNVVM/device-side-launch/CMakeLists.txt` | 72 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 920 | DONE | `cpp/7_libNVVM/device-side-launch/dsl-gpu64.ll` | 103 | 0 | no | CUDA kernel launch, thread/block indexing, synchronization, Driver API | DONE |
| 921 | DONE | `cpp/7_libNVVM/device-side-launch/dsl.c` | 272 | 11 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 922 | DONE | `cpp/7_libNVVM/ptxgen/CMakeLists.txt` | 78 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 923 | DONE | `cpp/7_libNVVM/ptxgen/ptxgen.c` | 281 | 3 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 924 | DONE | `cpp/7_libNVVM/ptxgen/test.ll` | 39 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 925 | DONE | `cpp/7_libNVVM/simple/CMakeLists.txt` | 71 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 926 | DONE | `cpp/7_libNVVM/simple/simple-gpu64.ll` | 61 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 927 | DONE | `cpp/7_libNVVM/simple/simple.c` | 264 | 12 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 928 | DONE | `cpp/7_libNVVM/syscalls/CMakeLists.txt` | 56 | 1 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 929 | DONE | `cpp/7_libNVVM/syscalls/malloc-free.ll` | 60 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 930 | DONE | `cpp/7_libNVVM/syscalls/vprintf.ll` | 92 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 931 | DONE | `cpp/7_libNVVM/utils/build.bat` | 6 | 2 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 932 | DONE | `cpp/7_libNVVM/utils/build.sh` | 8 | 2 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 933 | DONE | `cpp/7_libNVVM/uvmlite/CMakeLists.txt` | 75 | 2 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 934 | DONE | `cpp/7_libNVVM/uvmlite/uvmlite.c` | 334 | 13 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 935 | DONE | `cpp/7_libNVVM/uvmlite/uvmlite64.ll` | 61 | 0 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 936 | DONE | `cpp/8_Platform_Specific/Tegra/CMakeLists.txt` | 18 | 1 | no | CMake/build wiring | DONE |
| 937 | DONE | `cpp/8_Platform_Specific/Tegra/EGLSync_CUDAEvent_Interop/CMakeLists.txt` | 73 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 938 | DONE | `cpp/8_Platform_Specific/Tegra/EGLSync_CUDAEvent_Interop/EGLSync_CUDAEvent_Interop.cu` | 807 | 21 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 939 | DONE | `cpp/8_Platform_Specific/Tegra/EGLSync_CUDAEvent_Interop/egl_common.h` | 73 | 1 | no | graphics/interop, Python CUDA | DONE |
| 940 | DONE | `cpp/8_Platform_Specific/Tegra/EGLSync_CUDAEvent_Interop/graphics_interface.h` | 224 | 1 | no | graphics/interop | DONE |
| 941 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAErrorReporting/CMakeLists.txt` | 58 | 4 | no | CMake/build wiring | DONE |
| 942 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAErrorReporting/main.cu` | 438 | 15 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 943 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAHybridMode/CMakeLists.txt` | 58 | 4 | no | CMake/build wiring | DONE |
| 944 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAHybridMode/main.cu` | 504 | 17 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 945 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/CMakeLists.txt` | 58 | 4 | no | CMake/build wiring | DONE |
| 946 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsHybrid/main.cu` | 923 | 24 | no | CUDA Runtime memory/transfer, synchronization, streams/events, Driver API | DONE |
| 947 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsStandalone/CMakeLists.txt` | 65 | 4 | no | CMake/build wiring, graphics/interop | DONE |
| 948 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLALayerwiseStatsStandalone/main.cpp` | 1310 | 4 | no | Driver API | DONE |
| 949 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/CMakeLists.txt` | 65 | 4 | no | CMake/build wiring, graphics/interop | DONE |
| 950 | DONE | `cpp/8_Platform_Specific/Tegra/cuDLAStandaloneMode/main.cpp` | 1019 | 4 | no | Driver API | DONE |
| 951 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/CMakeLists.txt` | 67 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 952 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.cpp` | 463 | 9 | no | CUDA Runtime memory/transfer, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 953 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/cudaNvSciBufMultiplanar.h` | 136 | 1 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop, Python CUDA | DONE |
| 954 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/imageKernels.cu` | 68 | 3 | no | CUDA kernel launch, thread/block indexing, Python CUDA | DONE |
| 955 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciBufMultiplanar/main.cpp` | 74 | 1 | no | Driver API, Python CUDA | DONE |
| 956 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/CMakeLists.txt` | 98 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 957 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.cu` | 428 | 22 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 958 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/cuda_consumer.h` | 82 | 3 | no | streams/events | DONE |
| 959 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/main.cpp` | 212 | 1 | no | graphics/interop | DONE |
| 960 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.cpp` | 490 | 2 | no | graphics/interop | DONE |
| 961 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_producer.h` | 55 | 1 | no | host/helper source structure | DONE |
| 962 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.cpp` | 209 | 1 | no | graphics/interop | DONE |
| 963 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/cmdline.h` | 93 | 1 | no | host/helper source structure | DONE |
| 964 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.cpp` | 618 | 2 | no | host/helper source structure | DONE |
| 965 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/config_parser.h` | 111 | 1 | no | host/helper source structure | DONE |
| 966 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.cpp` | 787 | 2 | no | host/helper source structure | DONE |
| 967 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/image_utils.h` | 128 | 1 | no | host/helper source structure | DONE |
| 968 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.cpp` | 155 | 1 | no | graphics/interop | DONE |
| 969 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/log_utils.h` | 108 | 1 | no | host/helper source structure | DONE |
| 970 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.cpp` | 61 | 1 | no | host/helper source structure | DONE |
| 971 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvmedia_utils/misc_utils.h` | 74 | 1 | no | host/helper source structure | DONE |
| 972 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.cpp` | 156 | 2 | no | Driver API, NVRTC/JIT/libNVVM/PTX, graphics/interop, multi-GPU/P2P/IPC | DONE |
| 973 | DONE | `cpp/8_Platform_Specific/Tegra/cudaNvSciNvMedia/nvsci_setup.h` | 41 | 1 | no | host/helper source structure | DONE |
| 974 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/CMakeLists.txt` | 80 | 3 | no | CMake/build wiring, CUDA libraries, graphics/interop, shader source | DONE |
| 975 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/defines.h` | 48 | 1 | no | host/helper source structure | DONE |
| 976 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES.cpp` | 707 | 14 | no | CUDA Runtime memory/transfer, CUDA Graphs, Driver API, CUDA libraries | DONE |
| 977 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cu` | 375 | 19 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, CUDA Graphs | DONE |
| 978 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.cuh` | 72 | 1 | no | host/helper source structure | DONE |
| 979 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/fluidsGLES_kernels.h` | 54 | 1 | no | host/helper source structure | DONE |
| 980 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/graphics_interface.h` | 214 | 1 | no | graphics/interop | DONE |
| 981 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.frag.glsl` | 34 | 0 | no | shader source | DONE |
| 982 | DONE | `cpp/8_Platform_Specific/Tegra/fluidsGLES/mesh.vert.glsl` | 37 | 0 | no | shader source | DONE |
| 983 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/CMakeLists.txt` | 70 | 2 | no | CMake/build wiring, graphics/interop | DONE |
| 984 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystem.h` | 292 | 1 | no | host/helper source structure | DONE |
| 985 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystemcpu.h` | 78 | 1 | no | host/helper source structure | DONE |
| 986 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystemcpu_impl.h` | 272 | 2 | no | host/helper source structure | DONE |
| 987 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystemcuda.cu` | 289 | 16 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
| 988 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystemcuda.h` | 103 | 3 | no | streams/events, CUDA Graphs, graphics/interop | DONE |
| 989 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/bodysystemcuda_impl.h` | 392 | 16 | no | CUDA Runtime memory/transfer, streams/events, CUDA Graphs, Driver API | DONE |
| 990 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/nbody_opengles.cpp` | 1191 | 25 | no | synchronization, streams/events, Driver API, graphics/interop | DONE |
| 991 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/render_particles.cpp` | 379 | 2 | no | NVRTC/JIT/libNVVM/PTX, shader source | DONE |
| 992 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/render_particles.h` | 107 | 1 | no | graphics/interop | DONE |
| 993 | DONE | `cpp/8_Platform_Specific/Tegra/nbody_opengles/tipsy.h` | 168 | 1 | no | host/helper source structure | DONE |
| 994 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES/CMakeLists.txt` | 79 | 2 | no | CMake/build wiring, graphics/interop, shader source | DONE |
| 995 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES/graphics_interface.c` | 250 | 1 | no | graphics/interop | DONE |
| 996 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES/mesh.frag.glsl` | 31 | 0 | no | shader source | DONE |
| 997 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES/mesh.vert.glsl` | 33 | 0 | no | shader source | DONE |
| 998 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES/simpleGLES.cu` | 648 | 17 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 999 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES_EGLOutput/CMakeLists.txt` | 85 | 2 | no | CMake/build wiring, graphics/interop, shader source | DONE |
| 1000 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES_EGLOutput/graphics_interface_egloutput_via_egl.c` | 610 | 2 | no | Driver API, graphics/interop | DONE |
| 1001 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES_EGLOutput/mesh.frag.glsl` | 31 | 0 | no | shader source | DONE |
| 1002 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES_EGLOutput/mesh.vert.glsl` | 33 | 0 | no | shader source | DONE |
| 1003 | DONE | `cpp/8_Platform_Specific/Tegra/simpleGLES_EGLOutput/simpleGLES_EGLOutput.cu` | 587 | 16 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 1004 | DONE | `cpp/9_CUDA_Tile/Benchmark_Common/benchmark.h` | 212 | 6 | no | synchronization, streams/events, Driver API | DONE |
| 1005 | DONE | `cpp/9_CUDA_Tile/Benchmark_Common/matmul_benchmark.h` | 98 | 2 | no | host/helper source structure | DONE |
| 1006 | DONE | `cpp/9_CUDA_Tile/CMakeLists.txt` | 19 | 1 | no | CMake/build wiring | DONE |
| 1007 | DONE | `cpp/9_CUDA_Tile/helloTile/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1008 | DONE | `cpp/9_CUDA_Tile/helloTile/helloTile.cu` | 83 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 1009 | DONE | `cpp/9_CUDA_Tile/tileBmm/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1010 | DONE | `cpp/9_CUDA_Tile/tileBmm/tileBmm.cu` | 275 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 1011 | DONE | `cpp/9_CUDA_Tile/tileLayerNorm/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1012 | DONE | `cpp/9_CUDA_Tile/tileLayerNorm/tileLayerNorm.cu` | 277 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 1013 | DONE | `cpp/9_CUDA_Tile/tileMatmul/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1014 | DONE | `cpp/9_CUDA_Tile/tileMatmul/tileMatmul.cu` | 289 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, Driver API, multi-GPU/P2P/IPC | DONE |
| 1015 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/CMakeLists.txt` | 71 | 3 | no | CMake/build wiring, NVRTC/JIT/libNVVM/PTX | DONE |
| 1016 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_common.h` | 316 | 2 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 1017 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvcc.h` | 112 | 1 | no | host/helper source structure | DONE |
| 1018 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/backend_nvrtc.h` | 191 | 5 | no | Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 1019 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul.cu` | 102 | 1 | no | host/helper source structure | DONE |
| 1020 | DONE | `cpp/9_CUDA_Tile/tileMatmulAutotuner/matmul_autotuner.cpp` | 343 | 12 | no | CUDA kernel launch, Driver API, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 1021 | DONE | `cpp/9_CUDA_Tile/tileRope/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1022 | DONE | `cpp/9_CUDA_Tile/tileRope/tileRope.cu` | 283 | 9 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 1023 | DONE | `cpp/9_CUDA_Tile/tileSpMV/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1024 | DONE | `cpp/9_CUDA_Tile/tileSpMV/tileSpMV.cu` | 504 | 10 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 1025 | DONE | `cpp/9_CUDA_Tile/tileTranspose/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1026 | DONE | `cpp/9_CUDA_Tile/tileTranspose/tileTranspose.cu` | 132 | 6 | no | CUDA Runtime memory/transfer, CUDA kernel launch, synchronization, Driver API | DONE |
| 1027 | DONE | `cpp/9_CUDA_Tile/tileVectorAdd/CMakeLists.txt` | 32 | 2 | no | CMake/build wiring | DONE |
| 1028 | DONE | `cpp/9_CUDA_Tile/tileVectorAdd/tileVectorAdd.cu` | 143 | 7 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, synchronization | DONE |
| 1029 | DONE | `cpp/CMakeLists.txt` | 38 | 1 | no | CMake/build wiring | DONE |
| 1030 | DONE | `python/1_GettingStarted/blurImageUnifiedMemory/blurImageUnifiedMemory.py` | 284 | 7 | no | thread/block indexing, synchronization, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1031 | DONE | `python/1_GettingStarted/copyImageArraytoGPU/copyImageArraytoGPU.py` | 252 | 13 | no | Driver API, Python CUDA | DONE |
| 1032 | DONE | `python/1_GettingStarted/deviceQuery/deviceQuery.py` | 393 | 4 | yes | shared memory, multi-GPU/P2P/IPC, Python CUDA | DONE |
| 1033 | DONE | `python/1_GettingStarted/kernelNsysProfile/kernelNsysProfile.py` | 336 | 9 | no | thread/block indexing, synchronization, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 1034 | DONE | `python/1_GettingStarted/numpyVsCupy/numpyVsCupy.py` | 145 | 4 | no | Python CUDA | DONE |
| 1035 | DONE | `python/1_GettingStarted/simplePrint/simplePrint.py` | 292 | 5 | no | thread/block indexing, synchronization, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1036 | DONE | `python/1_GettingStarted/systemInfo/systemInfo.py` | 212 | 3 | no | NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC, Python CUDA | DONE |
| 1037 | DONE | `python/1_GettingStarted/vectorAdd/vectorAdd.py` | 203 | 7 | no | thread/block indexing, synchronization, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1038 | DONE | `python/2_CoreConcepts/binarySearch/binarySearch.py` | 152 | 5 | no | CUDA libraries, Python CUDA | DONE |
| 1039 | DONE | `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py` | 272 | 13 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1040 | DONE | `python/2_CoreConcepts/cudaComputeLambdas/cudaComputeLambdas.py` | 186 | 7 | no | Driver API, CUDA libraries, Python CUDA | DONE |
| 1041 | DONE | `python/2_CoreConcepts/cudaGraphs/cudaGraphs.py` | 279 | 13 | no | CUDA kernel launch, thread/block indexing, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1042 | DONE | `python/2_CoreConcepts/fftSignalAnalysis/fftSignalAnalysis.py` | 330 | 12 | no | CUDA libraries, Python CUDA | DONE |
| 1043 | DONE | `python/2_CoreConcepts/greenContext/greenContext.py` | 761 | 9 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1044 | DONE | `python/2_CoreConcepts/jitLtoLinking/jitLtoLinking.py` | 231 | 8 | no | thread/block indexing, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1045 | DONE | `python/2_CoreConcepts/launchConfigTuning/launchConfigTuning.py` | 406 | 10 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1046 | DONE | `python/2_CoreConcepts/matrixMulSharedMem/matrixMulSharedMem.py` | 258 | 10 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1047 | DONE | `python/2_CoreConcepts/memoryResources/memoryResources.py` | 274 | 18 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1048 | DONE | `python/2_CoreConcepts/pageRank/pageRank.py` | 375 | 7 | no | CUDA Graphs, Python CUDA, shader source | DONE |
| 1049 | DONE | `python/2_CoreConcepts/parallelHistogram/parallelHistogram.py` | 246 | 9 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1050 | DONE | `python/2_CoreConcepts/parallelReduction/parallelReduction.py` | 389 | 14 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1051 | DONE | `python/2_CoreConcepts/prefixSum/prefixSum.py` | 212 | 13 | no | Driver API, Python CUDA | DONE |
| 1052 | DONE | `python/2_CoreConcepts/processCheckpoint/processCheckpoint.py` | 273 | 12 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1053 | DONE | `python/2_CoreConcepts/reduction/reduction.py` | 492 | 7 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1054 | DONE | `python/2_CoreConcepts/reductionMultiBlockCG/reductionMultiBlockCG.py` | 481 | 6 | no | thread/block indexing, shared memory, synchronization, NVRTC/JIT/libNVVM/PTX | DONE |
| 1055 | DONE | `python/2_CoreConcepts/simpleZeroCopy/simpleZeroCopy.py` | 283 | 8 | no | CUDA Runtime memory/transfer, thread/block indexing, Driver API, NVRTC/JIT/libNVVM/PTX | DONE |
| 1056 | DONE | `python/2_CoreConcepts/streamingCopyComputeOverlap/streamingCopyComputeOverlap.py` | 329 | 17 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1057 | DONE | `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py` | 285 | 8 | no | thread/block indexing, shared memory, synchronization, CUDA libraries | DONE |
| 1058 | DONE | `python/3_FrameworkInterop/customPyTorchKernel/customPyTorchKernel.py` | 403 | 13 | no | thread/block indexing, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1059 | DONE | `python/3_FrameworkInterop/customTensorFlowKernel/customTensorFlowKernel.py` | 439 | 9 | no | thread/block indexing, Driver API, NVRTC/JIT/libNVVM/PTX, Python CUDA | DONE |
| 1060 | DONE | `python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py` | 227 | 7 | no | Python CUDA | DONE |
| 1061 | DONE | `python/4_DistributedComputing/multiGPUGradientAverage/multiGPUGradientAverage.py` | 429 | 13 | no | thread/block indexing, synchronization, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC | DONE |
| 1062 | DONE | `python/4_DistributedComputing/simpleP2P/simpleP2P.py` | 388 | 11 | no | thread/block indexing, NVRTC/JIT/libNVVM/PTX, multi-GPU/P2P/IPC, Python CUDA | DONE |
| 1063 | DONE | `python/Utilities/__init__.py` | 48 | 1 | no | host/helper source structure | DONE |
| 1064 | DONE | `python/Utilities/cuda_samples_utils.py` | 148 | 4 | no | Python CUDA | DONE |
| 1065 | DONE | `run_tests.py` | 327 | 1 | no | NVRTC/JIT/libNVVM/PTX | DONE |
| 1066 | DONE | `tools/inventory_ja.py` | 1325 | 22 | yes | CMake/build wiring, CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing | DONE |
| 1067 | DONE | `tools/regenerate_sample_readmes_ja.py` | 650 | 1 | no | CMake/build wiring, CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing | DONE |
| 1068 | DONE | `tools/regenerate_theme_guides_ja.py` | 237 | 2 | no | CUDA Runtime memory/transfer, CUDA kernel launch, thread/block indexing, shared memory | DONE |
