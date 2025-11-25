#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =================================================================================
// YOUR TASK: Implement Naive Matrix Multiply
// =================================================================================
// global indexing:
// The grid is 2D. 
// blockIdx.x / threadIdx.x corresponds to the COLUMN (x-axis)
// blockIdx.y / threadIdx.y corresponds to the ROW (y-axis)
__global__ void matmulNaive(const float *A, const float *B, float *C, int N) {
    // TODO 1: Calculate global Row and Column indices
    // Hint: int row = ... (uses .y)
    // Hint: int col = ... (uses .x)

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // this gives us the specific thread we need to update 
    // (x_glothis would give us a value in A, B

    // TODO 2: Boundary check
    // if (row < N && col < N) {

    float sum = 0.0f;

    if (row < N && col <  N ) { 
        for (int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];

        }

        C[row * N + col] = sum;


    }
        
}

// =================================================================================
// CPU Verification (Single-threaded, slow)
// =================================================================================
void verify_result(const std::vector<float>& h_C, const std::vector<float>& h_Ref, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = std::abs(h_C[i] - h_Ref[i]);
        if (diff > 1e-2) { // Higher tolerance for accumulations
            std::cerr << "FAILED! Mismatch at " << i << " GPU=" << h_C[i] << " CPU=" << h_Ref[i] << "\n";
            return;
        }
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "PASSED! Max Error: " << max_diff << "\n";
}

int main() {
    int N = 1024; // 1024 x 1024 matrix
    size_t bytes = N * N * sizeof(float);
    std::cout << "Matrix Multiplication (Naive) | N = " << N << "\n";

    // Alloc Host
    std::vector<float> h_A(N*N);
    std::vector<float> h_B(N*N);
    std::vector<float> h_C(N*N);
    std::vector<float> h_Ref(N*N);

    // Initialize A and B (simple values to avoid overflow)
    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f; // All 1s
        h_B[i] = 0.01f; // Small value
    }

    // Alloc Device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // SETUP EXECUTION CONFIGURATION (The Important Part)
    // -------------------------------------------------------------------------
    // We use a 2D block (e.g., 16x16 threads)
    dim3 threadsPerBlock(16, 16);
    
    // TODO 5: Calculate grid dimensions
    // We need enough blocks in X to cover N columns
    // We need enough blocks in Y to cover N rows
    // Hint: dim3 blocksPerGrid( ... , ... );

    // Comment: Essentially we want N/16 x N/16 blocks per grid so this is an easier way
    int xCoord = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int yCoord = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerGrid(xCoord, yCoord); 

    std::cout << "Launching Kernel...\n";
    matmulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU Reference (Ground Truth)
    std::cout << "Verifying on CPU (this might take a few seconds)...\n";
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