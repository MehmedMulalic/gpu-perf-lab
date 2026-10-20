#include <cuda_runtime.h>
#include <fstream>
#include <cstdio>
#include <vector>

#include "kernels.cuh"

inline void cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA Error at file %s:%d:\n%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void run_sgemm_naive(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

void run_sgemm_coalesced_gmem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
                                
    sgemm_coalesced_gmem<32><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

void run_sgemm_smem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    sgemm_smem<32><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

void run_sgemm1D_blocktiling(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);

    sgemm1D_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

void run_sgemm2D_blocktiling(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);

    sgemm1D_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

int main() {
    int M = 128;
    int N = 128;
    int K = 128;

    float alpha = 1.0f;
    float beta = 0.5f;

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);
    
    run_sgemm_naive(M, N, K, alpha, beta, A.data(), B.data(), C.data());
}
