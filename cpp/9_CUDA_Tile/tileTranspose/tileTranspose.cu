/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// JP: この file では memory ownership と host/device transfer、kernel launch と thread indexing、stream/event による非同期実行と同期 を確認します。英語の識別子/API/出力文字列は保持します。

/**
 * This sample demonstrates how to transpose a 2D matrix using CUDA
 * Tile C++. Each block handles an n x m sized chunk of the source
 * matrix. The block loads a chunk, transposes it locally, and stores
 * it to the correct position in the result matrix. A
 * cuda::tiles::partition_view is used to model the chunking of the
 * source and result matrices.
 */

#include "helper_cuda.h"
#include "cuda_tile.h"
#include <cstdio>

constexpr int CHUNK_N = 128;
constexpr int CHUNK_M = 256;

/* Declares a tile kernel with '__restrict__' pointers (important for performance) */
__tile_global__ void transpose(float* __restrict__ a,
                               float* __restrict__ b,
                               std::size_t n,
                               std::size_t m) {
  /* set up the namespace */
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  /* indicate to the compiler that the pointers are aligned (important for optimizations) */
  a = ct::assume_aligned(a, 16_ic);
  b = ct::assume_aligned(b, 16_ic);

  /* get the block index for the x and y dimension */
  auto [idx, idy, idz] = ct::bid();

  /* create tensor spans representing n x m and m x n row major matrices */
  ct::tensor_span a_span{a, ct::extents{n, m}};
  ct::tensor_span b_span{b, ct::extents{m, n}};

  /* create partition views over the arrays */
  auto view_a = ct::partition_view{a_span, ct::shape<CHUNK_N, CHUNK_M>{}};
  auto view_b = ct::partition_view{b_span, ct::shape<CHUNK_M, CHUNK_N>{}};

  /* load the tile from the input partition */
  auto tile_a = view_a.load_masked(idx, idy);

  /* transpose the tile locally */
  auto tile_transposed = ct::transpose(tile_a);

  /* store the tile to the correct output partition */
  view_b.store_masked(tile_transposed, idy, idx);
}

int main() {
  int n = 800;
  int m = 400;

  float* h_a = new float[n * m];
  for (int idx = 0; idx != n * m; ++idx) {
    h_a[idx] = idx;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;

  int num_blocks_n = 1 + (n - 1) / CHUNK_N;
  int num_blocks_m = 1 + (m - 1) / CHUNK_M;

  // JP: `cudaMalloc`: device 側 storage の所有をここで作ります。確保した pointer は後段の cleanup で対応する API により解放します。
  checkCudaErrors(cudaMalloc(&d_a, n * m * sizeof(float)));
  // JP: `cudaMemcpy`, `cudaMemcpyHostToDevice`: host/device 間の転送方向と async ordering を確認します。Async 版は同じ stream 内の順序と後続同期に依存します。
  checkCudaErrors(cudaMemcpy(d_a, h_a, n * m * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&d_b, n * m * sizeof(float)));

  // JP: kernel_launch: launch shape は grid/block/shared-memory/stream をここで決めます。kernel は非同期に開始し、後続の同期や検証で完了を確認します。
  transpose<<<dim3(num_blocks_n, num_blocks_m)>>>(d_a, d_b, n, m);
  checkCudaErrors(cudaGetLastError());

  // JP: `cudaDeviceSynchronize`: ここが同期境界です。これ以降の host 処理や検証は、ここまでの GPU work が完了した前提になります。
  checkCudaErrors(cudaDeviceSynchronize());

  float* h_b = new float[n * m];
  checkCudaErrors(cudaMemcpy(h_b, d_b, n * m * sizeof(float), cudaMemcpyDeviceToHost));

  for (int idx = 0; idx != n; ++idx) {
    for (int jdx = 0; jdx != m; ++jdx) {
      float expected = h_a[idx * m + jdx];
      float actual = h_b[jdx * n + idx];
      if (expected != actual) {
        printf("Expected: h_b[%i][%i] == %f\n", jdx, idx, expected);
        printf("Actual:   h_b[%i][%i] == %f\n", jdx, idx, actual);

        return 1;
      }
    }
  }

  printf("Success! Matrix transpose matches expected results.\n");

  // JP: `cudaFree`: ここで resource lifetime を閉じます。async work が残っていないことを確認してから、確保時と対応する API で解放します。
  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));

  delete[] h_a;
  delete[] h_b;
}
