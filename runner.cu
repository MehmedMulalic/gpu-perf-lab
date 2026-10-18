#include "kernels.cuh"
#include <cuda_runtime.h>

void run_sgemm_naive(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);

    sgemm_naive<<<gridDim, blockDim>>>>(M, N, K, alpha, beta, A, B, C)
}

void run_sgemm_coalesced_gmem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
                                
    sgemm_coalesced_gemm<32><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}

void run_sgemm_smem(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    sgemm_smem<32><<<gridDim, blockDim>>>(M, N, K, alpha, beta, A, B, C);
}
