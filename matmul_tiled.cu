#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =================================================================================
// YOUR TASK: Implement Tiled Matrix Multiplication
// =================================================================================
// TILE_WIDTH controls the size of the shared memory block.
// 16 is a safe start (16x16 floats = 256 floats = 1KB per array).
#define TILE_WIDTH 16

__global__ void matmulTiled(const float *A, const float *B, float *C, int N) {
    // TODO 1: Allocate Shared Memory
    // __shared__ float As[TILE_WIDTH][TILE_WIDTH];

    __shared__ float 
    // __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // TODO 2: Calculate Row and Column indices
    // int bx = blockIdx.x;  int by = blockIdx.y;
    // int tx = threadIdx.x; int ty = threadIdx.y;
    // int row = ...
    // int col = ...

    // Accumulator for C[row][col]
    float value = 0.0f;

    // TODO 3: Loop over the matrix in "tiles"
    // Loop 'm' from 0 to N/TILE_WIDTH
    for (int m = 0; m < N / TILE_WIDTH; ++m) {
        
        // TODO 4: Collaborative Loading
        // Thread (ty, tx) loads one element of A and one element of B into Shared Memory
        // Hint: You need to calculate the global index for A and B based on 'm'
        // As[ty][tx] = A[...];
        // Bs[ty][tx] = B[...];

        // TODO 5: Barrier Synchronization
        // Wait for all threads to finish loading
        // __syncthreads();

        // TODO 6: Compute Partial Product
        // Multiply the small 16x16 matrices As and Bs
        // for (int k = 0; k < TILE_WIDTH; ++k) {
        //     value += ...
        // }

        // TODO 7: Barrier Synchronization
        // Wait for all threads to finish math before overwriting Shared Memory in next loop
        // __syncthreads();
    }

    // TODO 8: Write result to Global Memory
    // if (row < N && col < N) ...
}

// =================================================================================
// CPU Verification (Same as before)
// =================================================================================
void verify_result(const std::vector<float>& h_C, const std::vector<float>& h_Ref, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = std::abs(h_C[i] - h_Ref[i]);
        if (diff > 1e-2) { 
            std::cerr << "FAILED! Mismatch at " << i << " GPU=" << h_C[i] << " CPU=" << h_Ref[i] << "\n";
            return;
        }
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "PASSED! Max Error: " << max_diff << "\n";
}

int main() {
    int N = 1024; // Standard size
    size_t bytes = N * N * sizeof(float);
    std::cout << "Tiled Matrix Multiplication | N = " << N << "\n";

    std::vector<float> h_A(N*N, 1.0f);
    std::vector<float> h_B(N*N, 0.01f);
    std::vector<float> h_C(N*N);
    std::vector<float> h_Ref(N*N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Execution Configuration
    // NOTE: Block size MUST match TILE_WIDTH for this simplified implementation
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(N / TILE_WIDTH, N / TILE_WIDTH);

    std::cout << "Launching Kernel with Tile Width " << TILE_WIDTH << "...\n";
    matmulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Verifying on CPU...\n";
    // Using OpenMP if available could speed this up, but simple loop is fine for N=1024
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_Ref[i * N + j] = sum;
        }
    }

    verify_result(h_C, h_Ref, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}