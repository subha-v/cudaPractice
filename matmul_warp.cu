#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Required for 'half' precision
#include <mma.h>       // <--- REQUIRED: The Warp Matrix Multiply API

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Use the NVIDIA CUDA WMMA namespace to save typing
using namespace nvcuda;

// =================================================================================
// YOUR TASK: Implement Warp-Level Matmul using Tensor Cores
// =================================================================================
// Dimensions for the Tensor Core Operation
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmul_wmma(const half *A, const half *B, float *C, int N) {
    // -------------------------------------------------------------------------
    // STEP 1: CALCULATE COORDINATES
    // -------------------------------------------------------------------------
    // Unlike standard CUDA where 1 Thread = 1 Output, here 1 WARP = 1 Output Tile (16x16).
    // You need to calculate which 16x16 tile of C this specific Warp is responsible for.
    
    // TODO: Calculate global Warp ID
    // Hint: (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int globalWarpId = ...

    // TODO: Calculate the Row and Column of the *Tile* (not the thread)
    // Hint: You are working on a grid of 16x16 tiles.
    int numTilesRow = N / 16;
    int tileRow = ...
    int tileCol = ...

    // Safety check: if we are outside the matrix, return.
    if (tileRow >= numTilesRow || tileCol >= numTilesRow) return;

    // -------------------------------------------------------------------------
    // STEP 2: DECLARE FRAGMENTS
    // -------------------------------------------------------------------------
    // A "Fragment" is a register variable that holds a piece of the matrix.
    // Declare three fragments: a_frag, b_frag, and c_frag.
    
    // TODO: Declare a_frag (Matrix A, 16x16x16, half precision, Row Major)
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    
    // TODO: Declare b_frag (Matrix B, 16x16x16, half precision, Col Major)
    // Note: We will assume B is stored Column-Major in memory for simplicity here.
    
    // TODO: Declare c_frag (Accumulator, 16x16x16, float precision)


    // TODO: Initialize c_frag to 0.0f
    // Hint: wmma::fill_fragment(...);


    // -------------------------------------------------------------------------
    // STEP 3: THE MAIN LOOP
    // -------------------------------------------------------------------------
    // Loop across the K dimension in chunks of 16 (WMMA_K)
    for (int k = 0; k < N; k += WMMA_K) {
        
        // TODO: Calculate the pointers to the current 16x16 chunks in A and B
        // Remember: A is Row Major, B is Col Major (transposed).
        // const half* A_tile_ptr = A + ...
        // const half* B_tile_ptr = B + ...

        // TODO: Load data from Memory into Fragments
        // Hint: wmma::load_matrix_sync(frag, ptr, stride);
        // The "stride" for a standard matrix is just N (width of the matrix).
        

        // TODO: Perform the Math!
        // Hint: wmma::mma_sync(dest, src_a, src_b, src_c);
        
    }

    // -------------------------------------------------------------------------
    // STEP 4: STORE RESULT
    // -------------------------------------------------------------------------
    // TODO: Calculate pointer to output tile in C
    // float* C_tile_ptr = ...

    // TODO: Store c_frag back to Global Memory
    // Hint: wmma::store_matrix_sync(...);
}

// =================================================================================
// MAIN (Verification Harness)
// =================================================================================
int main() {
    int N = 256; // Matrix size (Must be multiple of 16)
    int size = N * N;
    size_t bytes_half = size * sizeof(half);
    size_t bytes_float = size * sizeof(float);

    std::cout << "WMMA Challenge | N = " << N << "\n";

    // Alloc Host
    std::vector<half> h_A(size);
    std::vector<half> h_B(size);
    std::vector<float> h_C(size);

    // Initialize A=1.0, B=1.0. Result should be N.
    for (int i = 0; i < size; i++) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    // Alloc Device
    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_float));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_half, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // EXECUTION CONFIGURATION
    // -------------------------------------------------------------------------
    // 1 Warp = 32 Threads.
    // We want 4 Warps per Block (128 threads total).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;

    // Calculate total tiles needed
    int totalTiles = (N / 16) * (N / 16);
    int numBlocks = (totalTiles + warpsPerBlock - 1) / warpsPerBlock;

    std::cout << "Launching " << numBlocks << " blocks (" << totalTiles << " tiles)...\n";
    matmul_wmma<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_float, cudaMemcpyDeviceToHost));
    
    // Check first and last elements
    std::cout << "Verification (Expected: " << N << ".0):\n";
    std::cout << "C[0] = " << h_C[0] << "\n";
    std::cout << "C[end] = " << h_C[size-1] << "\n";

    if (abs(h_C[0] - N) < 0.1) std::cout << "PASSED!\n";
    else std::cout << "FAILED.\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}