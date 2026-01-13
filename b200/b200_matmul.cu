#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda/barrier>
#include <chrono>
#include <cmath>
#include <omp.h>

// We want to launch 148 blocks and have those persistently running on the B200
// A kernel is executed as a grid of blocks of threads
// Each CUDA block is executed by one streaming multiprocessor (SM)

// Tile sizes for our matmul
constexpr int TILE_M = 128;  
constexpr int TILE_N = 256;  
constexpr int TILE_K = 64;   

constexpr int NUM_CONSUMERS = 1;
constexpr int NUM_PRODUCERS = 1;
constexpr int PIPE_DEPTH = 4;  // Ring buffer depth for pipelining
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;  // 4 warps per warpgroup
constexpr int NUM_THREADS = NUM_WORKERS * 32;  // 32 threads per warp = 256 threads

// TMEM allocation size: 128x256 output tile with float32 accumulators
// Organized in 512-byte columns: 128 * 256 * 4 bytes = 131072 bytes = 256 columns
constexpr int TMEM_COLS = 256;

// =============================================================================
// DEVICE HELPER: Launch TMA loads for A and B tiles into a pipeline slot
// =============================================================================
__device__ __forceinline__ void launch_tma_load(
    int slot,                                              // Which pipeline slot to load into
    int k_tile,                                            // Which k-tile to load
    int tile_row,                                          // Row coordinate for A
    int tile_col,                                          // Column coordinate for B
    __nv_bfloat16 a_smem[][TILE_M * TILE_K],              // Shared memory for A tiles
    __nv_bfloat16 b_smem[][TILE_K * TILE_N],              // Shared memory for B tiles
    cuda::barrier<cuda::thread_scope_block>* barrier,      // Barrier to signal on completion
    const CUtensorMap* tensor_map_A,                       // TMA descriptor for A
    const CUtensorMap* tensor_map_B                        // TMA descriptor for B
) {
    // Calculate expected bytes for barrier
    uint64_t bytes_A = TILE_M * TILE_K * sizeof(__nv_bfloat16);
    uint64_t bytes_B = TILE_K * TILE_N * sizeof(__nv_bfloat16);

    // Signal barrier with expected transaction bytes
    cuda::barrier_arrive_tx(*barrier, 1, bytes_A + bytes_B);

    // Launch TMA for matrix A: loads tile [TILE_M x TILE_K] at position (tile_row, k_tile)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(a_smem[slot])),
          "l"(tensor_map_A),
          "r"(k_tile),
          "r"(tile_row),
          "r"((uint32_t)__cvta_generic_to_shared(barrier))
        : "memory"
    );

    // Launch TMA for matrix B: loads tile [TILE_K x TILE_N] at position (k_tile, tile_col)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(b_smem[slot])),
          "l"(tensor_map_B),
          "r"(tile_col),
          "r"(k_tile),
          "r"((uint32_t)__cvta_generic_to_shared(barrier))
        : "memory"
    );
}

__global__ void my_matmul_kernel(
    const __grid_constant__ CUtensorMap tensor_map_A,  // TMA descriptor for A
    const __grid_constant__ CUtensorMap tensor_map_B,  // TMA descriptor for B
    __nv_bfloat16* __restrict__ C,                     // [M x N] matrix in global memory (output)
    int M, int N, int K                                // Matrix dimensions
) {
    // what thread am i? producer or consumer and which warp am i in 
    int warp_id = threadIdx.x / 32;        // Which warp (0-7 for 256 threads)
    int wg_id = warp_id / 4;               // Which warpgroup (0=consumer, 1=producer)

    // smem allocation
    // Ring buffer for A and B tiles
    __shared__ __nv_bfloat16 a_smem[PIPE_DEPTH][TILE_M * TILE_K];  // 4 x 128 x 64
    __shared__ __nv_bfloat16 b_smem[PIPE_DEPTH][TILE_K * TILE_N];  // 4 x 64 x 256

    // TMEM base address for accumulator (allocated per-block)
    __shared__ uint32_t tmem_base;

    // Barriers for producer/consumer synchronization
    __shared__ cuda::barrier<cuda::thread_scope_block> inputs_arrived[PIPE_DEPTH];   // Producer -> Consumer: data ready
    __shared__ cuda::barrier<cuda::thread_scope_block> inputs_finished[PIPE_DEPTH];  // Consumer -> Producer: slot free

    // initialization
    if (threadIdx.x == 0) {
        // Initialize barriers
        // inputs_arrived: expects 1 arrival from barrier_arrive_tx + 1 from consumer's arrive_and_wait
        // inputs_finished: expects 1 arrival from consumer when done with slot
        for (int i = 0; i < PIPE_DEPTH; i++) {
            init(&inputs_arrived[i], 2);
            init(&inputs_finished[i], 2);  // Producer waits, consumer signals
        }

        // allocates TMEM for 128x256 float accumulator (256 columns of 512 bytes each)
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : "=r"(tmem_base)
            : "r"(TMEM_COLS)
        );
    }
    __syncthreads();

    int num_tiles_m = M / TILE_M;
    int num_tiles_n = N / TILE_N;
    int num_k_tiles = K / TILE_K;
    int total_tiles = num_tiles_m * num_tiles_n;

    // each block processes multiple output tiles
    // note that blockIdx.x  goes from 1 to 148
    // block 0 would process tiles 0, 148, 296, etc. and then block 1 would process 1, 149, etc. since we're striding by gridDim = 148
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {

        int tile_row = tile_id / num_tiles_n;
        int tile_col = tile_id % num_tiles_n;

        // Note the following 2 if statements run in parallel :D

        // Consumer warpgroup zeros the TMEM at start of each output tile
        if (wg_id == 0) {
            // Each thread in consumer warpgroup zeros a portion of TMEM
            // 128x256 floats = 32768 floats, 128 threads, so 256 floats per thread
            // TMEM is column-major, 512 bytes per column = 128 floats per column
            for (int i = threadIdx.x; i < TILE_M * TILE_N; i += 128) {
                uint32_t offset = i * sizeof(float);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x1b.x1.b32 [%0], {%1};" 
                    :: "r"(tmem_base + offset), "r"(0)
                );
            }
        }
        // No __syncthreads__ here! Consumer and Producer work in parallel

        // 1 thread of the producer loads the first tiles
        if (wg_id == 1 && threadIdx.x == 128) {  // First thread of producer warpgroup
            int prefetch_count = min(PIPE_DEPTH, num_k_tiles); // min (4, tiles) = 4 usually
            for (int k_tile = 0; k_tile < prefetch_count; k_tile++) {
                int slot = k_tile % PIPE_DEPTH;
                launch_tma_load( // loads in tiles A and B to SMEM
                    slot, k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived[slot],
                    &tensor_map_A, &tensor_map_B
                );
            }
        }
        // No __syncthreads__ here! Barriers handle synchronization


        // we split up the tiles of  A and B into further tiles
        // note that each tile of A is 128 x 64 and each tile of B is 64 x 256, where K = 64 (we just set that lol)
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) { // i think we have like 32 k tiles for 1024 x 1024
            int compute_slot = k_tile % PIPE_DEPTH; // which slot is the tile were computing on is in
            int load_k_tile = k_tile + PIPE_DEPTH; // which one to load for the producer
            int load_slot = load_k_tile % PIPE_DEPTH; // which slot to load it into

            // consumer thread executes wgmma
            if (wg_id == 0) {
                // Consumer waits for the producer (only thread 0 does the barrier wait)
                if (threadIdx.x == 0) {
                    inputs_arrived[compute_slot].arrive_and_wait();
                }
                // at this point, only thread 0 does the barrier wait
                // threads 1-127 in the consumer warpgroup dont know yet
                // Warpgroup fence ensures all consumer threads see the data

                // this is basically like syncthreads but only syncs the 128 consumer threads and not all the 256 threads in the block lol
                asm volatile("wgmma.fence.sync.aligned;"); // this code synchronizes the warpgroup. 
                // fence creates a memory barrier and then .aligned is a memory alignment qualifier
                // wgmma is defined to operate at a warpgroup granularity so it will take the current warpgroup. 
                
                // Get shared memory addresses for current slot
                uint32_t a_addr = __cvta_generic_to_shared(a_smem[compute_slot]);
                uint32_t b_addr = __cvta_generic_to_shared(b_smem[compute_slot]);

                // WGMMA for 128x256 output tile with K=64
                // Using Blackwell m128n256k16 WGMMA: 4 k-steps (TILE_K=64, step=16)
                // Each wgmma handles 128 rows, 256 cols, 16 k-elements in ONE call

                #pragma unroll // idk chatgpt said to do this ... but apparently it unrolls this for loop and speeds up the math :D
                for (int k_step = 0; k_step < TILE_K; k_step += 16) { // K = 64
                    // A offset: k_step columns into the 128x64 A tile
                    // B offset: k_step rows into the 64x256 B tile
                    uint32_t a_offset = k_step * sizeof(__nv_bfloat16);
                    uint32_t b_offset = k_step * TILE_N * sizeof(__nv_bfloat16);

                    // Single WGMMA computes: C[128x256] += A[128x16] × B[16x256]
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m128n256k16.f32.bf16.bf16 "
                        // this command means wgmma mma (matrix multiply accumulate) with 128 x 256 x 16 dimensions
                        // output is f32 and inputs are bf16
                        "[%0], [%1], [%2], 1, 1, 1, 0, 1;"
                        :
                        : "r"(tmem_base),           // Output: full 128x256 tile in TMEM
                          "r"(a_addr + a_offset),   // Input A: 128 rows × 16 cols
                          "r"(b_addr + b_offset)    // Input B: 16 rows × 256 cols
                        : "memory"
                    );
                }

                // Wait for all WGMMA operations to complete
                // like syncthreads
                asm volatile("wgmma.wait_group.sync.aligned 0;");

                // consumer tells producer that it's done with this slot
                if (threadIdx.x == 0) {
                    inputs_finished[compute_slot].arrive();
                }
            }

            // Producer loads the next tile (runs in parallel with consumer's WGMMA!)
            if (wg_id == 1 && threadIdx.x == 128 && load_k_tile < num_k_tiles) { // note that because we didn't use syncthreads and instead use wgmma.fence we can actually move onto this part and not get stuck at syncthreads
                // Wait for consumer to finish with the slot we're about to reuse
                if (k_tile >= PIPE_DEPTH) {
                    inputs_finished[load_slot].arrive_and_wait();
                }

                launch_tma_load(
                    load_slot, load_k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived[load_slot],
                    &tensor_map_A, &tensor_map_B
                );
            }

        }

        // store results from TMEM to global memory
        // Consumer warpgroup reads from TMEM and writes to global memory
        if (wg_id == 0) {
            // Ensure all WGMMA operations are complete before reading TMEM
            asm volatile("wgmma.commit_group.sync.aligned;");
            asm volatile("wgmma.wait_group.sync.aligned 0;");

            // Each thread reads and stores multiple elements
            // 128x256 = 32768 elements, 128 threads = 256 elements per thread
            for (int i = threadIdx.x; i < TILE_M * TILE_N; i += 128) {
                int local_row = i / TILE_N;
                int local_col = i % TILE_N;
                int global_row = tile_row * TILE_M + local_row;
                int global_col = tile_col * TILE_N + local_col;

                // Read from TMEM
                float val;
                uint32_t tmem_offset = (local_row * TILE_N + local_col) * sizeof(float);
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x1b.x1.b32 {%0}, [%1];"
                    : "=r"(*(uint32_t*)&val)
                    : "r"(tmem_base + tmem_offset)
                );

                // Store to global memory
                C[global_row * N + global_col] = __float2bfloat16(val);
            }
        }
        // Sync before next output tile to ensure all stores complete
        __syncthreads();
    }

    // deallocate tmem
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(TMEM_COLS));
    }
}


// create TMA tensor map
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
// CPU reference GEMM for validation
// =============================================================================
void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2)  // Parallelize outer two loops across CPU cores
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// =============================================================================
// Benchmark function - runs kernel multiple times and measures TFLOPS
// =============================================================================
int run_benchmark(size_t M, size_t N, size_t K) {
    // Initialize CUDA driver API (required for TMA)
    cuInit(0);

    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Tile sizes: TILE_M=" << TILE_M << ", TILE_N=" << TILE_N << ", TILE_K=" << TILE_K << std::endl;

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize matrices with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (size_t i = 0; i < M * K; i++) h_A[i] = dis(gen);
    for (size_t i = 0; i < K * N; i++) h_B[i] = dis(gen);

    std::cout << "Initialized matrices with random values" << std::endl;

    // Perform CPU matrix multiplication for reference (only for small sizes)
    bool do_validation = (M <= 2048);
    if (do_validation) {
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to bfloat16 and copy to device
    __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];

    for (size_t i = 0; i < M * K; i++) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (size_t i = 0; i < K * N; i++) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    // Create TMA descriptors
    CUtensorMap tensor_map_A = create_tensor_map(d_A, M, K, TILE_M, TILE_K);
    CUtensorMap tensor_map_B = create_tensor_map(d_B, K, N, TILE_K, TILE_N);

    std::cout << "Created TMA descriptors" << std::endl;

    // Launch configuration
    dim3 grid(148, 1);       // 148 blocks (one per SM on B200)
    dim3 block(NUM_THREADS); // 256 threads per block (2 warpgroups)

    std::cout << "Launching kernel with grid(" << grid.x << "), block(" << block.x << ")" << std::endl;

    // Warmup run
    std::cout << "Warmup..." << std::endl;
    my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Benchmark runs
    constexpr int ITERS = 5;
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERS; i++) {
        my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16* h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Convert result back to float for comparison
    for (size_t i = 0; i < M * N; i++) h_C[i] = __bfloat162float(h_C_bf16[i]);

    // Validate results (only for small sizes)
    if (do_validation) {
        float max_error = 0.0f;
        float average_error = 0.0f;
        int error_count = 0;
        for (size_t i = 0; i < M * N; i++) {
            float error = std::abs(h_C[i] - h_C_ref[i]);
            if (error > 1.0f) { // large threshold because of bf16 vs fp32 numerics
                if (error_count < 20) {
                    std::cout << "Error at row " << i / N << " col " << i % N
                              << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
                } else if (error_count == 20) {
                    std::cout << "Too many errors to show them all.\n";
                }
                error_count++;
            }
            max_error = std::max(max_error, error);
            average_error += error;
        }
        average_error /= M * N;

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Average error: " << average_error << std::endl;
        std::cout << "Error count: " << error_count << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Done!\n" << std::endl;

    return 0;
}

int main() {
    // Run benchmarks at different sizes
    run_benchmark(1024, 1024, 1024);
    run_benchmark(2048, 2048, 2048);
    run_benchmark(4096, 4096, 4096);
    run_benchmark(8192, 8192, 8192);

    return 0;
}
