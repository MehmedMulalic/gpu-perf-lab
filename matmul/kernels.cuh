#pragma once

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
MxK * KxN = MxN
*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * acc + beta * C[x * N + y];
    }
}

template <const uint BLOCKSIZE>
__global__ void sgemm_coalesced_gmem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += A[cRow * K + i] * B[i * N + cCol];
        }

        C[cRow * N + cCol] = alpha * acc + beta * C[cRow * N + cCol];
    }
}

template <const uint BLOCKSIZE>
__global__ void sgemm_smem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float acc = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            acc += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * acc + beta * C[threadRow * N + threadCol];
}

template <const uint BM, const uint BN, const uint BK, const int TM>
__global__ void sgemm1D_blocktiling(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    
}

template <const uint BLOCKSIZE>
__global__ void sgemm2D_blocktiling(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {

}

template <const uint BLOCKSIZE>
__global__ void sgemm_vectorize(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {

}
