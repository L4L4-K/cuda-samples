# simpleD3D12 - Simple D3D12 CUDA Interop - Japanese Learning Guide

English source: [README.md](README.md)
Build file: [CMakeLists.txt](CMakeLists.txt)

## English Overview

A program which demonstrates Direct3D12 interoperability with CUDA.  The program creates a sinewave in DX12 vertex buffer which is created using CUDA kernels. DX12 and CUDA synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA GPU is required on Windows10 or higher OS.

Graphics Interop, CUDA DX12 Interop, Image Processing

Original README headings: `simpleD3D12 - Simple D3D12 CUDA Interop`, `Description`, `Key Concepts`, `Supported SM Architectures`, `Supported OSes`, `Supported CPU Architecture`, `CUDA APIs involved`, `[CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)`, `Dependencies needed to build/run`, `Prerequisites`, `References (for more details)`

> **日本語**
> `cpp/5_Domain_Specific/simpleD3D12` は `C++/CUDA` の CUDA sample です。英語 README を正本として残し、この guide では実装を読む順番と CUDA の学習ポイントを日本語で補足します。
>
> **学習メモ**
> API 名、target 名、command、出力文字列は翻訳しません。英語 README と source の文字列をそのまま参照し、意味だけを日本語で補います。

## Purpose

English anchor: read `simpleD3D12` as a focused example of the CUDA concepts used in `cpp/5_Domain_Specific/simpleD3D12`.
> **日本語**
> この sample の目的は、`simpleD3D12` の小さな実装を通して CUDA Graphs, Streams And Events, Synchronization And Atomics, Memory, Kernel Launch And Indexing を具体的に追うことです。
>
> **学習メモ**
> 最初に `DX12CudaSample.cpp, DX12CudaSample.h, DXSampleHelper.h, Main.cpp, ShaderStructs.h` を読み、setup、GPU work、sync、validation、cleanup の境界を source 内で対応させます。

## Prerequisites

- CUDA Toolkit matching this repository branch
- NVIDIA driver and a CUDA-capable GPU supported by the original README
- CMake 3.20 or newer and a host compiler supported by the toolkit
- The graphics, display, or platform stack named by the English README

> **日本語**
> 必要条件は英語 README と CMake/requirements を優先します。この guide は条件を置き換えず、読むべき確認点を追加します。
>
> **学習メモ**
> 実行できない場合は、source を変える前に driver、toolkit、GPU feature、library、platform guard、Python package version を確認します。

## Files

- `CMakeLists.txt`: CMake target, CUDA architecture, and library dependency wiring.
- `DX12CudaSample.cpp`: Host-side setup, API calls, validation, and cleanup.
- `DX12CudaSample.h`: Host/device declarations, helper types, constants, or library wrappers.
- `DXSampleHelper.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Main.cpp`: Host-side setup, API calls, validation, and cleanup.
- `README.md`: Original English purpose, prerequisites, run notes, and expected behavior.
- `ShaderStructs.h`: Host/device declarations, helper types, constants, or library wrappers.
- `Win32Application.cpp`: Host-side setup, API calls, validation, and cleanup.
- `Win32Application.h`: Host/device declarations, helper types, constants, or library wrappers.
- `d3dx12.h`: Host/device declarations, helper types, constants, or library wrappers.
- `shaders.hlsl`: Supporting file used by `shaders.hlsl`.
- `simpleD3D12.cpp`: Host-side setup, API calls, validation, and cleanup.
- `simpleD3D12.h`: Host/device declarations, helper types, constants, or library wrappers.
- `sinewave_cuda.cu`: CUDA source containing kernels, Runtime API calls, or device-side helper code.
- `stdafx.cpp`: Host-side setup, API calls, validation, and cleanup.
- `stdafx.h`: Host/device declarations, helper types, constants, or library wrappers.

> **日本語**
> 各 file は役割を分けて読みます。README は目的、build file は依存関係、source は CUDA resource と実行順序を示します。
>
> **学習メモ**
> data/reference file がある場合は、GPU result の比較対象または入力条件として扱います。出力名や test string は英語のまま確認します。

## Execution Flow

- Read the original README and identify the supported device, OS, and library assumptions.
- Open `DX12CudaSample.cpp` first and locate the host-side setup or Python entry point.
- Select the CUDA device and allocate host/device resources used by the sample.
- Capture or build CUDA Graph nodes, instantiate the graph, then launch the executable graph.
- Move, map, or expose input data so GPU work can read the intended values.
- Launch the kernel, graph, library call, or Python CUDA operation with the documented configuration.
- Synchronize only at the required correctness or timing boundary.
- Validate results against the CPU/reference path, generated artifact, or expected status message.
- Release CUDA, library, framework, graphics, or external resources in the reverse ownership order.

> **日本語**
> 実行の流れは setup、visibility、GPU work、sync、validation、cleanup の順に読みます。非同期 API がある場合は、host がいつ待つかを別に記録します。
>
> **学習メモ**
> CUDA の bug は kernel 本体だけでなく、copy direction、descriptor、stream dependency、cleanup order にも出ます。

## Concrete Reading Path

- `DX12CudaSample.cpp`: focus on `CUDA`.
- `DX12CudaSample.h`: focus on `CUDA`.
- `DXSampleHelper.h`: focus on `CUDA`.
- `Main.cpp`: focus on `CUDA`.
- `ShaderStructs.h`: focus on `cudaStream_t`, `cudaDevVertptr`.
- `Win32Application.cpp`: focus on `CUDA`.
- `Win32Application.h`: focus on `CUDA`.
- `d3dx12.h`: focus on `CUDA`, `atomic`.
- `simpleD3D12.cpp`: focus on `CUDA`, `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaFree`, `atomic`.
- `simpleD3D12.h`: focus on `cudaStream_t`, `atomic`, `CUDA`, `cudaExternalMemoryHandleType`, `cudaExternalMemory_t`.
- Additional source files: 3 more support files. Use the same setup/work/sync/cleanup lens.

> **日本語**
> 読む順番を file ごとに固定すると、CUDA API と helper code の境界を見失いにくくなります。
>
> **学習メモ**
> まず entry point で resource lifetime を追い、次に kernel/device helper で indexing、shared memory、atomic、library boundary を確認します。

## Code Walkthrough

この節のコードは現在のリポジトリから直接抜き出しています。`Source: path:start-end` は検証スクリプトが照合する契約です。

### `CMakeLists.txt`

Source: cpp/5_Domain_Specific/simpleD3D12/CMakeLists.txt:1-66
```cmake
# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(simpleD3D12 LANGUAGES C CXX CUDA)

# JP: `find_package`: この CMake 行で CUDA target、architecture、library dependency を配線します。target 名と link 設定は挙動に直結します。
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

if(WIN32)
    # Source file
    # Add target for simpleD3D12
    add_executable(simpleD3D12 WIN32
        DX12CudaSample.cpp
        Main.cpp
        Win32Application.cpp
        simpleD3D12.cpp
        stdafx.cpp
        sinewave_cuda.cu
    )

    target_compile_options(simpleD3D12 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    target_compile_features(simpleD3D12 PRIVATE cxx_std_17 cuda_std_17)

    set_target_properties(simpleD3D12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(simpleD3D12 PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(simpleD3D12 PRIVATE
        d3d12
        dxgi
        dxguid
        d3dcompiler
    )

    add_custom_command(TARGET simpleD3D12 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders.hlsl
        ${CMAKE_CURRENT_BINARY_DIR}/shaders.hlsl
    )

else()
    message(STATUS "Sample 'simpleD3D12' is Windows-only - skipping")
endif()

# Include installation configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/InstallSamples.cmake)
setup_samples_install()
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/CMakeLists.txt` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `DX12CudaSample.cpp`

Source: cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.cpp:30-48
```cpp
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.cpp:74-93
```cpp
std::wstring DX12CudaSample::string2wstring(const std::string &s)
{
    int len;
    int slength  = (int)s.length() + 1;
    len          = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t *buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}
// Helper function for resolving the full path of assets.
std::wstring DX12CudaSample::GetAssetFullPath(const char *assetName)
{
    LPTSTR lpBuffer = new char[4096];
    GetCurrentDirectory(FILENAME_MAX, lpBuffer);
    char *tmp = sdkFindFilePath((const char *)assetName, "simpleD3D12");
    if (tmp == NULL) {
        throw std::exception("File not found");
    }
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `DX12CudaSample.h`

Source: cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.h:30-48
```cpp
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/DX12CudaSample.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `DXSampleHelper.h`

Source: cpp/5_Domain_Specific/simpleD3D12/DXSampleHelper.h:30-48
```cpp
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/DXSampleHelper.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/DXSampleHelper.h:123-142
```cpp

    if (fileInfo.EndOfFile.HighPart != 0) {
        throw std::exception();
    }

    *data = reinterpret_cast<byte *>(malloc(fileInfo.EndOfFile.LowPart));
    *size = fileInfo.EndOfFile.LowPart;

    if (!ReadFile(file.Get(), *data, fileInfo.EndOfFile.LowPart, nullptr, nullptr)) {
        throw std::exception();
    }

    return S_OK;
}

// Assign a name to the object to aid with debugging.
#if defined(_DEBUG) || defined(DBG)
inline void SetName(ID3D12Object *pObject, LPCWSTR name) { pObject->SetName(name); }
inline void SetNameIndexed(ID3D12Object *pObject, LPCWSTR name, UINT index)
{
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/DXSampleHelper.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Main.cpp`

Source: cpp/5_Domain_Specific/simpleD3D12/Main.cpp:29-36
```cpp
#include "simpleD3D12.h"
#include "stdafx.h"

_Use_decl_annotations_ int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    DX12CudaInterop sample(1280, 720, "D3D12 CUDA Interop");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/Main.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `ShaderStructs.h`

Source: cpp/5_Domain_Specific/simpleD3D12/ShaderStructs.h:29-53
```cpp
#pragma once

#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <cuda_runtime.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include "helper_cuda.h"

using namespace DirectX;

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT4 color;
};


void RunSineWaveKernel(size_t       mesh_width,
                       size_t       mesh_height,
                       Vertex      *cudaDevVertptr,
                       // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                       cudaStream_t streamToRun,
                       float        AnimTime);
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/ShaderStructs.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Win32Application.cpp`

Source: cpp/5_Domain_Specific/simpleD3D12/Win32Application.cpp:30-48
```cpp
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/Win32Application.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `Win32Application.h`

Source: cpp/5_Domain_Specific/simpleD3D12/Win32Application.h:30-72
```cpp
  which is licensed as follows:

The MIT License (MIT)
    Copyright (c) 2015 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // JP: streams_events: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#pragma once

#include "DX12CudaSample.h"

class DX12CudaSample;

class Win32Application
{
public:
    static int  Run(DX12CudaSample *pSample, HINSTANCE hInstance, int nCmdShow);
    static HWND GetHwnd() { return m_hwnd; }

protected:
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

private:
    static HWND m_hwnd;
};
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/Win32Application.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `d3dx12.h`

Source: cpp/5_Domain_Specific/simpleD3D12/d3dx12.h:12-30
```cpp
#ifndef __D3DX12_H__
#define __D3DX12_H__

#include "d3d12.h"

#if defined(__cplusplus)

struct CD3DX12_DEFAULT
{
};
extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;

//------------------------------------------------------------------------------------------------
inline bool operator==(const D3D12_VIEWPORT &l, const D3D12_VIEWPORT &r)
{
    return l.TopLeftX == r.TopLeftX && l.TopLeftY == r.TopLeftY && l.Width == r.Width && l.Height == r.Height
        && l.MinDepth == r.MinDepth && l.MaxDepth == r.MaxDepth;
}

```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/d3dx12.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/d3dx12.h:778-797
```cpp
                                                UINT                        registerSpace = 0,
                                                D3D12_SHADER_VISIBILITY     visibility    = D3D12_SHADER_VISIBILITY_ALL)
    {
        rootParam.ParameterType    = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParam.ShaderVisibility = visibility;
        // JP: library_resources: CUDA library の handle/descriptor/workspace は外部 resource です。作成、設定、利用、破棄の順序を対応させます。
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsShaderResourceView(_Out_ D3D12_ROOT_PARAMETER &rootParam,
                                                UINT                        shaderRegister,
                                                UINT                        registerSpace = 0,
                                                D3D12_SHADER_VISIBILITY     visibility    = D3D12_SHADER_VISIBILITY_ALL)
    {
        rootParam.ParameterType    = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsUnorderedAccessView(_Out_ D3D12_ROOT_PARAMETER &rootParam,
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/d3dx12.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `shaders.hlsl`

Source: cpp/5_Domain_Specific/simpleD3D12/shaders.hlsl:30-51
```hlsl
struct PSInput
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};

PSInput VSMain(float3 position : POSITION, float4 color : COLOR)
{
	PSInput result;

	result.position = float4(position, 1.0f);

	// Pass the color through without modification.
	result.color = color;

	return result;
}

float4 PSMain(PSInput input) : SV_TARGET
{
	return input.color;
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/shaders.hlsl` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `simpleD3D12.cpp`

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp:29-47
```cpp
#include "simpleD3D12.h"

#include <aclapi.h>
#include <cuda_runtime.h>
#include <shellapi.h>
#include <string>
#include <windows.h>
#include <wrl.h>

#include "ShaderStructs.h"
#include "d3dx12.h"

//////////////////////////////////////////////
// WindowsSecurityAttributes implementation //
//////////////////////////////////////////////

class WindowsSecurityAttributes
{
protected:
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp:56-75
```cpp

WindowsSecurityAttributes::WindowsSecurityAttributes()
{
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    // JP: validation: GPU result を CPU/reference と比較する検証地点です。失敗時は transfer、indexing、sync の順に疑います。
    assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode        = SET_ACCESS;
    explicitAccess.grfInheritance       = INHERIT_ONLY;
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp:96-115
```cpp
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    // JP: cleanup: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
    free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() { return &m_winSecurityAttributes; }

DX12CudaInterop::DX12CudaInterop(UINT width, UINT height, std::string name)
    : DX12CudaSample(width, height, name)
    , m_frameIndex(0)
    , m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height))
    , m_fenceValues{}
    , m_rtvDescriptorSize(0)
{
    m_viewport = {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)};
    m_AnimTime = 1.0f;
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp:235-254
```cpp
        if ((memcmp(&m_dx12deviceluid.LowPart, devProp.luid, sizeof(m_dx12deviceluid.LowPart)) == 0)
            && (memcmp(&m_dx12deviceluid.HighPart,
                       devProp.luid + sizeof(m_dx12deviceluid.LowPart),
                       sizeof(m_dx12deviceluid.HighPart))
                == 0)) {
            checkCudaErrors(cudaSetDevice(devId));
            m_cudaDeviceID = devId;
            m_nodeMask     = devProp.luidDeviceNodeMask;
            // JP: `cudaStreamCreate`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
            checkCudaErrors(cudaStreamCreate(&m_streamToRun));
            printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
            break;
        }
    }
}
// Load the sample assets.
void DX12CudaInterop::LoadAssets()
{
    // Create a root signature.
    {
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `simpleD3D12.h`

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.h:29-47
```cpp
#pragma once

#include <DirectXMath.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include "DX12CudaSample.h"
#include "ShaderStructs.h"
#include "d3dx12.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the
// CPU, it has no understanding of the lifetime of resources on the GPU. Apps
// must account for the GPU lifetime of resources to avoid destroying objects
// that may still be referenced by the GPU. An example of this can be found in
// the class method: OnDestroy().
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

Source: cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.h:132-151
```cpp

    // CUDA objects
    cudaExternalMemoryHandleType m_externalMemoryHandleType;
    cudaExternalMemory_t         m_externalMemory;
    cudaExternalSemaphore_t      m_externalSemaphore;
    // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
    cudaStream_t                 m_streamToRun;
    LUID                         m_dx12deviceluid;
    UINT                         m_cudaDeviceID;
    UINT                         m_nodeMask;
    float                        m_AnimTime;
    void                        *m_cudaDevVertptr = NULL;

    void LoadPipeline();
    void InitCuda();
    void LoadAssets();
    void PopulateCommandList();
    void MoveToNextFrame();
    void WaitForGpu();
};
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/simpleD3D12.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `sinewave_cuda.cu`

Source: cpp/5_Domain_Specific/simpleD3D12/sinewave_cuda.cu:29-75
```cuda
#include "ShaderStructs.h"

__global__ void sinewave_gen_kernel(Vertex *vertices, unsigned int width, unsigned int height, float time)
{
    // JP: `blockIdx`, `blockDim`, `threadIdx`: block/thread index から担当要素を計算します。境界チェックは problem size と同じ単位で合わせます。
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u       = u * 2.0f - 1.0f;
    v       = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w    = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    if (y < height && x < width) {
        // write output vertex
        vertices[y * width + x].position.x = u;
        vertices[y * width + x].position.y = w;
        vertices[y * width + x].position.z = v;
        // vertices[y*width+x].position[3] = 1.0f;
        vertices[y * width + x].color.x = 1.0f;
        vertices[y * width + x].color.y = 0.0f;
        vertices[y * width + x].color.z = 0.0f;
        vertices[y * width + x].color.w = 0.0f;
    }
}

// The host CPU Sinewave thread spawner
void RunSineWaveKernel(size_t       mesh_width,
                       size_t       mesh_height,
                       Vertex      *cudaDevVertptr,
                       // JP: `cudaStream_t`: stream/event は非同期 work の順序、overlap、計測範囲を表します。同じ stream 内では投入順が保たれます。
                       cudaStream_t streamToRun,
                       float        AnimTime)
{
    dim3    block(16, 16, 1);
    dim3    grid(mesh_width / 16, mesh_height / 16, 1);
    Vertex *vertices = (Vertex *)cudaDevVertptr;
    // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
    sinewave_gen_kernel<<<grid, block, 0, streamToRun>>>(vertices, mesh_width, mesh_height, AnimTime);

    getLastCudaError("sinewave_gen_kernel execution failed.\n");
}
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/sinewave_cuda.cu` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `stdafx.cpp`

Source: cpp/5_Domain_Specific/simpleD3D12/stdafx.cpp:29-29
```cpp
#include "stdafx.h"
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/stdafx.cpp` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。

### `stdafx.h`

Source: cpp/5_Domain_Specific/simpleD3D12/stdafx.h:33-48
```cpp
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <shellapi.h>
#include <string>
#include <windows.h>
#include <wrl.h>

#include "d3dx12.h"
```

> JP: この抜粋は `cpp/5_Domain_Specific/simpleD3D12/stdafx.h` の実コードです。setup、allocation、transfer、GPU work、sync、validation、cleanup のどの境界を示すかを、行番号と一緒に確認します。


## Key APIs And Concepts

| API or concept | Why it matters |
| - | - |
| `cudaStream_t` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaStreamCreate` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaStreamSynchronize` | 非同期 work の順序、overlap、計測範囲を表す API です。 |
| `cudaFree` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaDevVertptr` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `atomic` | 複数 thread が同じ address を更新する箇所です。競合と順序の意味を確認します。 |
| `cudaGetDeviceCount` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaGetDeviceProperties` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaSetDevice` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportExternalMemory` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaExternalMemoryGetMappedBuffer` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaImportExternalSemaphore` | この sample の中心 API/概念です。入力、所有権、同期、検証との関係を確認します。 |
| `cudaDestroyExternalSemaphore` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |
| `cudaDestroyExternalMemory` | resource lifetime を閉じる API です。未完了 work が残っていないかを確認します。 |

> **日本語**
> API 名は英語のまま、何を所有するか、何を開始するか、何を待つか、何を検証するかで分類します。
>
> **学習メモ**
> helper macro や wrapper の内側にも CUDA Runtime/Driver/library call があるため、error handling の境界も確認します。

## Memory, Synchronization, And Performance Notes

- CUDA Graph は一連の work を node と依存関係として再利用します。capture 対象と buffer lifetime を確認します。
- kernel launch では grid/block/thread の形と `i < N` のような境界チェックを一緒に確認します。
- allocation、copy/mapping、cleanup を同じ単位で追い、element count と byte count を混同しないようにします。
- stream/event がある場合、host から見た完了点と device 内の順序は別物として読みます。
- 同期や atomic は correctness のための境界です。性能測定では待ちすぎによる overlap 消失も確認します。

> **日本語**
> memory の所有者、転送方向、同期点、計測範囲を分けて読むと、この sample の意図が見えます。
>
> **学習メモ**
> correctness のための同期と、performance 測定のための同期は目的が違います。待ちすぎると overlap が消えることがあります。

## Build And Run

English commands remain authoritative. Typical repository-root CMake flow:

```bash
cmake -S . -B build
cmake --build build --target simpleD3D12
ctest --test-dir build -R simpleD3D12
```

> **日本語**
> 実際の option、target 名、実行 directory は英語 README と build file を優先します。この guide の command は読み方の補助です。
>
> **学習メモ**
> build directory と source directory を分けると、生成物を消しても source や翻訳 companion を壊しにくくなります。

## Expected Behavior

The sample may display a window or produce/validate image-like output; exact visuals depend on platform support.
> **日本語**
> 期待結果は英語の出力文字列、README の validation、生成 file、または reference result と照合します。
>
> **学習メモ**
> `PASS`、`Test passed`、error code、timing label などの出力文字列は翻訳せず、source と同じ表記で確認します。

## Common Mistakes

- API 名や target 名を翻訳してしまい、README や build command と対応できなくなる。
- allocation size を byte で渡す API と element count で考える loop を混同する。
- kernel launch が非同期であることを忘れ、同期前の結果を host 側で読んでしまう。
- different stream 間に依存があるのに event や explicit sync を置かない。
- graph capture 後に buffer lifetime や node dependency が変わったことを見落とす。

> **日本語**
> 失敗時は API の戻り値、現在の device、memory size、同期位置、reference validation の条件を順に確認します。
>
> **学習メモ**
> source を変更する練習では、まず期待出力と validation を壊していないかを小さく確認します。

## Exercises

- `cudaStream_t` の直前と直後で、どの memory/resource が有効になったかをメモする。
- source file を上から読み、setup、GPU work、sync、validation、cleanup の行番号を抜き出す。
- problem size や input size を変更した場合に、境界チェックや allocation size が破綻しないか説明する。
- stream timeline を描き、copy、kernel、event、host wait の位置を分ける。
- graph node の依存関係を箇条書きにし、どの buffer lifetime が graph 実行全体をまたぐか確認する。
- 関連 theme guide を 1 つ読み、同じ API pattern が別 sample でどう変わるか比較する。

> **日本語**
> 演習では code を動かす前に、読み取った仮説を memo として整理します。
>
> **学習メモ**
> 変更する場合は、元の出力、validation、resource cleanup が変わっていないかを確認します。

## Related Themes

- [CUDA Graphs](../../../docs_ja/themes/graphs.md): capture、node dependency、replay、graph update を読むための基礎です。
- [Streams And Events](../../../docs_ja/themes/streams_events.md): 非同期 work、overlap、timing event の順序を読むための基礎です。
- [Synchronization And Atomics](../../../docs_ja/themes/sync_atomics.md): barrier、fence、atomic update の必要性を判断するための基礎です。
- [Memory](../../../docs_ja/themes/memory.md): host/device/managed/external memory の所有権と lifetime を追うための基礎です。
- [Kernel Launch And Indexing](../../../docs_ja/themes/kernel_indexing.md): thread index から data index への対応と境界チェックを読むための基礎です。
- [Execution Model](../../../docs_ja/themes/execution_model.md): kernel launch、grid/block/thread の実行階層を読むための基礎です。
- [Debugging, Profiling, And Testing](../../../docs_ja/themes/debugging_profiling_testing.md): error check、reference validation、profiling/timing の範囲を読むための基礎です。

> **日本語**
> 関連 theme を先に読むと、この sample が CUDA 全体のどの概念を切り出しているか分かります。
>
> **学習メモ**
> 同じ theme を持つ別 sample と比較すると、API の使い分けや性能上の tradeoff が見えます。
