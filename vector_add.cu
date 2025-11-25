#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// =================================================================================
// 1. BOILERPLATE: Error Checking Macro (Keep this!)
// =================================================================================
// Professional CUDA code always checks error codes. This macro wraps CUDA calls
// and crashes nicely with a line number if something goes wrong.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =================================================================================
// 2. YOUR TASK: Implement the Kernel
// =================================================================================
// Goal: Implement C[i] = A[i] + B[i] for N elements.
// Remember: The GPU launches massive grids. You need to calculate a global index
// using blockIdx, blockDim, and threadIdx to know which 'i' this thread should process.
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // TODO 1: Calculate the global thread index (tid)

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO 2: Ensure the index is within bounds (tid < N)

    if (tid < N){
        C[tid] = A[tid] + B[tid];
    }

    // TODO 3: Perform the addition
    
}

// =================================================================================
// 3. BOILERPLATE: CPU Verification (Ground Truth)
// =================================================================================
void verify_result(const std::vector<float>& h_A, const std::vector<float>& h_B, const std::vector<float>& h_C, int N) {
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > 1e-5) {
            std::cerr << "FAILED! Mismatch at index " << i 
                      << ": GPU=" << h_C[i] << ", CPU=" << expected << "\n";
            return;
        }
    }
    std::cout << "PASSED! System Verification Successful.\n";
}

int main() {
    // -------------------------------------------------------------------------
    // SETUP: Define problem size
    // -------------------------------------------------------------------------
    int N = 1 << 20; // 1 million elements (2^20)
    size_t bytes = N * sizeof(float);
    std::cout << "Vector Addition | N = " << N << " elements\n";

    // -------------------------------------------------------------------------
    // STEP 1: Allocate Host Memory (CPU)
    // -------------------------------------------------------------------------
    // h_A is not a pointer, it's a C++ object containing a pointer to its internal array, size, capacity, allocator info, etc.

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    // Initialize with dummy data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // -------------------------------------------------------------------------
    // STEP 2: Allocate Device Memory (GPU)
    // -------------------------------------------------------------------------
    float *d_A, *d_B, *d_C;
    // TODO 4: Allocate memory on the GPU using cudaMalloc
    // Hint: CUDA_CHECK(cudaMalloc(...));
    cudaError_t err1 =  = cudaMalloc(&d_A, N * sizeof(float))
    cudaError_t err2 = cudaMalloc(&d_B, N * sizeof(float))
    cudaError_t err3 = cudaMalloc(&d_C, N * sizeof(float))


    // -------------------------------------------------------------------------
    // STEP 3: Copy Data Host -> Device
    // -------------------------------------------------------------------------
    // TODO 5: Copy h_A and h_B to d_A and d_B using cudaMemcpy

    // Cuda requires a raw pointer to pointer memory shift e.g. the first and second things must be raw pointers 
    // Thankfully, h_A.data() gives us the same thing as &h_A[0]
    cudaMemcpy(d_A, h_A.data(), N* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N* sizeof(float), cudaMemcpyHostToDevice);

    // Hint: Use cudaMemcpyHostToDevice


    // -------------------------------------------------------------------------
    // STEP 4: Launch Kernel
    // -------------------------------------------------------------------------
    int threadsPerBlock = 256;
    // TODO 6: Calculate the number of blocks needed to cover N elements.
    // Hint: It's roughly N / threadsPerBlock, but you need to round UP (ceil).
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    // correctly calculating the amount of blocks we need for the whole grid

    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and " 
              << threadsPerBlock << " threads per block...\n";

    // TODO 7: Launch the kernel!
    // vectorAdd<<<...>>>(...);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    
    // Check for launch errors (Async errors are caught by synchronization)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for GPU to finish

    // -------------------------------------------------------------------------
    // STEP 5: Copy Result Device -> Host
    // -------------------------------------------------------------------------
    // TODO 8: Copy d_C back to h_C
    cudaMemcpy(h_C.data(), d_C,  N * sizeof(float), cudaMemcpyDeviceToHost)
    // Hint: Use cudaMemcpyDeviceToHost


    // -------------------------------------------------------------------------
    // VERIFICATION & CLEANUP
    // -------------------------------------------------------------------------
    verify_result(h_A, h_B, h_C, N);

    // TODO 9: Free GPU memory using cudaFree
    error1 = cudaFree(d_A)
    error2 = cudaFree(d_B)
    error3 = cudaFree(d_C)

    

    return 0;
}