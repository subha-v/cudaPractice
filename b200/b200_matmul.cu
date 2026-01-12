#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda/barrier>

// =============================================================================
// CONSTANTS
// =============================================================================

// We want to launch 148 blocks and have those persistently running on the B200
// A kernel is executed as a grid of blocks of threads
// Each CUDA block is executed by one streaming multiprocessor (SM)

// Tile sizes for our matmul
constexpr int TILE_M = 128;  // Tile size for M dimension (rows of A, rows of C)
constexpr int TILE_N = 256;  // Tile size for N dimension (cols of B, cols of C)
constexpr int TILE_K = 64;   // Tile size for K dimension (cols of A, rows of B)

constexpr int NUM_CONSUMERS = 1;
constexpr int NUM_PRODUCERS = 1;
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;  // 4 warps per warpgroup
constexpr int NUM_THREADS = NUM_WORKERS * 32;  // 32 threads per warp = 256 threads

// =============================================================================
// KERNEL
// =============================================================================

__global__ void my_matmul_kernel(
    const __grid_constant__ CUtensorMap tensor_map_A,  // TMA descriptor for A
    const __grid_constant__ CUtensorMap tensor_map_B,  // TMA descriptor for B
    __nv_bfloat16* __restrict__ C,                     // [M x N] matrix in global memory (output)
    int M, int N, int K                                // Matrix dimensions
) {
    // Figure out which warpgroup this thread belongs to
    // threadIdx.x will be a number from 0 to 255 in this case
    int warp_id = threadIdx.x / 32;        // Which warp (0-7 for 256 threads) because 256/32 = 8

    // This tells us whether the current THREAD is part of a consumer or producer warpgroup
    int wg_id = warp_id / 4;               // Which warpgroup (0 or 1 for 256 threads)

    // specific thread numbering since 1 warp = 32 threads
    int lane_id = threadIdx.x % 32;        // Which thread within the warp (0-31)

    // Allocate shared memory for tiles
    __shared__ __nv_bfloat16 a_smem[TILE_M * TILE_K];  // 128 * 64 = 8192 elements
    __shared__ __nv_bfloat16 b_smem[TILE_K * TILE_N];  // 64 * 256 = 16384 elements

    // Barrier for TMA synchronization
    // TMA will signal this barrier when the load is complete
    __shared__ cuda::barrier<cuda::thread_scope_block> tma_barrier;

    // Initialize the barrier (only one thread does this)
    if (threadIdx.x == 0) {
        init(&tma_barrier, 1); // This sets up a barrier/semaphore for later
        // Which tells us that 1 thread is expected to arrive
        // Per block
    }
    __syncthreads();  // Make sure barrier is initialized before anyone uses it

    // Calculate how many output tiles we have in C
    int num_tiles_m = M / TILE_M;
    int num_tiles_n = N / TILE_N;
    int num_k_tiles = K / TILE_K;
    int total_tiles = num_tiles_m * num_tiles_n;

    // Persistent kernel loop - each block processes multiple output tiles
    // Grid is the number of parallel processes we want to eventually run on our GPU
    // We stride by gridDim.x (148) so all blocks divide up the work
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {

        // Convert tile_id to 2D tile coordinates of where in C It is 
        int tile_row = tile_id / num_tiles_n;  
        int tile_col = tile_id % num_tiles_n; 

        // Essentially now we're looping through all the partitions of A0B0 + A1B1 + ... To make
        // The current tile at C that we want to calculate and accumulating the partial sum. 
        float accumulator = 0.0f
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            if (wg_id == 0) {
                // --- PRODUCER ROLE ---
                // 1. Calculate next A and B tile coordinates
                //    - For A: we want tile at (tile_row, k_tile) in tile-space
                //    - For B: we want tile at (k_tile, tile_col) in tile-space

                // 2. Launch TMA Load (Hardware moves data to Smem)
                if (threadIdx.x == 0) {
                    // Set up the barrier to expect TMA completions
                    // TMA will deliver (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16) bytes
                    uint64_t bytes_A = TILE_M * TILE_K * sizeof(__nv_bfloat16);
                    uint64_t bytes_B = TILE_K * TILE_N * sizeof(__nv_bfloat16);

                    // Tell barrier how many bytes to expect from TMA
                    cuda::barrier_arrive_tx(tma_barrier, 1, bytes_A + bytes_B);

                    // TMA coordinates: which tile to load (in tile-space, not element-space)
                    // For A[M,K] loading tile [TILE_M, TILE_K]: coords are (k_tile, tile_row)
                    // For B[K,N] loading tile [TILE_K, TILE_N]: coords are (tile_col, k_tile)

                    // Launch TMA for matrix A
                    // cp.async.bulk.tensor.2d loads a 2D tile from global to shared memory
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(a_smem)),  // Destination: shared memory
                          "l"(&tensor_map_A),                               // Source: TMA descriptor
                          "r"(k_tile),                                      // Coordinate 0 (column in tile-space)
                          "r"(tile_row),                                    // Coordinate 1 (row in tile-space)
                          "r"((uint32_t)__cvta_generic_to_shared(&tma_barrier))  // Barrier to signal
                        : "memory"
                    );

                    // Launch TMA for matrix B
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                        " [%0], [%1, {%2, %3}], [%4];"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(b_smem)),  // Destination: shared memory
                          "l"(&tensor_map_B),                               // Source: TMA descriptor
                          "r"(tile_col),                                    // Coordinate 0 (column in tile-space)
                          "r"(k_tile),                                      // Coordinate 1 (row in tile-space)
                          "r"((uint32_t)__cvta_generic_to_shared(&tma_barrier))  // Barrier to signal
                        : "memory"
                    );
                }

                // Once this code executes, it signals the barrier automatically 

            }
            else {
                // --- CONSUMER ROLE ---
             
            }

            // All threads wait for TMA loads to complete
            // The barrier will be released when TMA has delivered all expected bytes
            tma_barrier.arrive_and_wait();

            for (int k = 0; k < TILE_K; k++){ 
                accumulator += __bfloat162float(a_smem[local_row * TILE_K + k]) * 
                __bfloat162float(b_smem[k * TILE_N + local_col]);
            }

            // Sync before next iteration (so we don't overwrite smem while someone reads)
            __syncthreads();
        }

        int global_row = tile_row * TILE_M + local_row;
        int global_col = tile_col * TILE_N + local_col; 
        C[global_row * N + global_col] = __float2bfloat16(accumulator); //  Store accumulated result to C
    }
}

/*
Questions:
- Ignoring the 2 consumer / 1 producer thing rn
-

*/

// =============================================================================
// HELPER: Create TMA descriptor
// =============================================================================

CUtensorMap create_tensor_map(
    void* global_ptr,           // Pointer to matrix in global memory
    int rows,                   // Number of rows in full matrix
    int cols,                   // Number of columns in full matrix
    int tile_rows,              // Tile height
    int tile_cols               // Tile width
) {
    CUtensorMap tensor_map;

    // Dimensions of the full matrix (in elements)
    // Note: TMA uses column-major-ish ordering: {inner_dim, outer_dim}
    cuuint64_t globalDim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};

    // Strides in bytes
    // For row-major: stride to next column = sizeof(element), stride to next row = cols * sizeof(element)
    cuuint64_t globalStrides[2] = {
        sizeof(__nv_bfloat16),                    // Stride for moving one column (inner)
        (cuuint64_t)cols * sizeof(__nv_bfloat16)  // Stride for moving one row (outer)
    };

    // Tile dimensions
    cuuint32_t boxDim[2] = {(cuuint32_t)tile_cols, (cuuint32_t)tile_rows};

    // Element strides (typically 1)
    cuuint32_t elementStrides[2] = {1, 1};

    // Create the TMA descriptor
    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,       // Data type
        2,                                       // Rank (2D tensor)
        global_ptr,                              // Global memory pointer
        globalDim,                               // Dimensions of full tensor
        globalStrides,                           // Strides in bytes
        boxDim,                                  // Tile dimensions
        elementStrides,                          // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,          // No interleaving
        CU_TENSOR_MAP_SWIZZLE_NONE,             // No swizzling (simpler, but less efficient)
        CU_TENSOR_MAP_L2_PROMOTION_NONE,        // L2 promotion setting
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE       // Out-of-bounds fill
    );

    if (result != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled failed with error: " << result << std::endl;
    }

    return tensor_map;
}

// =============================================================================
// MAIN - Setup and launch kernel
// =============================================================================

int main() {
    // Initialize CUDA driver API (required for TMA)
    cuInit(0);

    // Matrix dimensions (must be divisible by tile sizes for now)
    int M = 1024;  // Rows of A and C
    int N = 1024;  // Cols of B and C
    int K = 1024;  // Cols of A and Rows of B

    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Tile sizes: TILE_M=" << TILE_M << ", TILE_N=" << TILE_N << ", TILE_K=" << TILE_K << std::endl;

    // =========================================================================
    // Step 1: Allocate memory on CPU (Host)
    // =========================================================================
    // "h_" prefix means "host" (CPU)

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];  // Output - will be filled by GPU

    std::cout << "Allocated host memory" << std::endl;

    // =========================================================================
    // Step 2: Initialize matrices with random values
    // =========================================================================

    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (int i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) h_B[i] = dis(gen);

    std::cout << "Initialized matrices with random values" << std::endl;

    // =========================================================================
    // Step 3: Allocate memory on GPU (Device)
    // =========================================================================
    // "d_" prefix means "device" (GPU)
    // cudaMalloc allocates memory in GPU's global memory (HBM)

    __nv_bfloat16 *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // =========================================================================
    // Step 4: Convert to bfloat16 and copy CPU → GPU
    // =========================================================================
    // We need to convert float32 to bfloat16 before copying

    __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];

    for (int i = 0; i < M * K; i++) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    // cudaMemcpy copies data between host and device
    // cudaMemcpyHostToDevice = CPU → GPU
    cudaMemcpy(d_A, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    // =========================================================================
    // Step 5: Create TMA descriptors
    // =========================================================================
    // TMA descriptors tell the hardware how to interpret the memory layout

    // For matrix A [M x K]: tiles are [TILE_M x TILE_K] = [128 x 64]
    CUtensorMap tensor_map_A = create_tensor_map(d_A, M, K, TILE_M, TILE_K);
    std::cout << "Created TMA descriptor for A" << std::endl;

    // For matrix B [K x N]: tiles are [TILE_K x TILE_N] = [64 x 256]
    CUtensorMap tensor_map_B = create_tensor_map(d_B, K, N, TILE_K, TILE_N);
    std::cout << "Created TMA descriptor for B" << std::endl;

    // =========================================================================
    // Step 6: Launch the kernel
    // =========================================================================

    dim3 grid(148, 1);       // 148 blocks (one per SM on B200)
    dim3 block(NUM_THREADS); // 256 threads per block (2 warpgroups)

    std::cout << "Launching kernel with grid(" << grid.x << "), block(" << block.x << ")" << std::endl;
    std::cout << "  - " << grid.x << " blocks" << std::endl;
    std::cout << "  - " << block.x << " threads per block" << std::endl;
    std::cout << "  - " << block.x / 32 << " warps per block" << std::endl;
    std::cout << "  - " << block.x / 128 << " warpgroups per block" << std::endl;

    // Launch! The <<<grid, block>>> syntax is CUDA-specific
    // Now passing TMA descriptors instead of raw pointers for A and B
    my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Kernel finished!" << std::endl;

    // =========================================================================
    // Step 7: Copy result GPU → CPU
    // =========================================================================

    __nv_bfloat16* h_C_bf16 = new __nv_bfloat16[M * N];

    // cudaMemcpyDeviceToHost = GPU → CPU
    cudaMemcpy(h_C_bf16, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Convert back to float for verification
    for (int i = 0; i < M * N; i++) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Copied result back to host" << std::endl;

    // =========================================================================
    // Step 8: Clean up
    // =========================================================================

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;

    std::cout << "Cleaned up memory" << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}
