#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda/barrier>
#include <chrono>
#include <cmath>
#include <omp.h>

// =============================================================================
// DEBUGGING UTILITIES
// =============================================================================

// Device-side assert - triggers trap on failure (use with compute-sanitizer)
__device__ __forceinline__ void dassert(bool condition) {
    if (!condition) {
        asm volatile("trap;");
    }
}

// Device-side checkpoint marker - writes to global memory for post-mortem analysis
// checkpoint_buffer should be allocated as: int[gridDim.x * NUM_CHECKPOINTS]
#define NUM_CHECKPOINTS 16
__device__ __forceinline__ void checkpoint(int* buffer, int checkpoint_id) {
    if (threadIdx.x == 0) {
        buffer[blockIdx.x * NUM_CHECKPOINTS + checkpoint_id] = 1;
    }
}

// Host-side CUDA error checking macro
#define CUDA_CHECK(x) do {                                              \
    cudaError_t err = (x);                                              \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CU_CHECK(x) do {                                                \
    CUresult err = (x);                                                 \
    if (err != CUDA_SUCCESS) {                                          \
        const char* errStr;                                             \
        cuGetErrorString(err, &errStr);                                 \
        fprintf(stderr, "CUDA driver error %s:%d: %s\n",                \
                __FILE__, __LINE__, errStr);                            \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// =============================================================================
// KERNEL CONSTANTS
// =============================================================================

// We want to launch 148 blocks and have those persistently running on the B200
// A kernel is executed as a grid of blocks of threads
// Each CUDA block is executed by one streaming multiprocessor (SM)

// Tile sizes for our matmul
constexpr int TILE_M = 128;  
constexpr int TILE_N = 256;  
constexpr int TILE_K = 64;   

constexpr int NUM_CONSUMERS = 1;
constexpr int NUM_PRODUCERS = 1;
constexpr int PIPE_DEPTH = 4;   // 4 slots for the pipeline
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;  // 4 warps per warpgroup
constexpr int NUM_THREADS = NUM_WORKERS * 32;  // 32 threads per warp = 256 threads


// Organized in 512-byte columns: 128 * 256 * 4 bytes = 131072 bytes = 256 columns
constexpr int TMEM_COLS = 256;

// helper function to make tma load from gmem to smem
__device__ __forceinline__ void launch_tma_load(
    int slot,
    int k_tile,
    int tile_row,
    int tile_col,
    __nv_bfloat16 a_smem[][TILE_M * TILE_K],
    __nv_bfloat16 b_smem[][TILE_K * TILE_N],
    uint64_t* barrier,      // raw mbarrier storage pointer
    const CUtensorMap* tensor_map_A,
    const CUtensorMap* tensor_map_B
) {
    // Calculate expected bytes for barrier
    uint32_t bytes_A = TILE_M * TILE_K * sizeof(__nv_bfloat16);
    uint32_t bytes_B = TILE_K * TILE_N * sizeof(__nv_bfloat16);
    uint32_t expected_bytes = bytes_A + bytes_B;

    // Signal barrier with expected transaction bytes using PTX
    uint64_t barrier_addr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :
        : "l"(barrier_addr), "r"(expected_bytes)
        : "memory"
    );

    // TMA expects ELEMENT coordinates, not tile indices!
    // Convert tile indices to element coordinates
    int a_coord_k = k_tile * TILE_K;      // K dimension element coordinate
    int a_coord_m = tile_row * TILE_M;    // M dimension element coordinate
    int b_coord_k = k_tile * TILE_K;      // K dimension element coordinate
    int b_coord_n = tile_col * TILE_N;    // N dimension element coordinate

    // Launch TMA for A - explicitly specify .tile mode for sm_100a compatibility
    // A tensor map: globalDim={K, M}, so coords are {k_coord, m_coord}
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(a_smem[slot])),
          "l"(tensor_map_A),
          "r"(a_coord_k),     // coordinate 0: K dimension (element coord)
          "r"(a_coord_m),     // coordinate 1: M dimension (element coord)
          "r"((uint32_t)barrier_addr)
        : "memory"
    );

    // Launch TMA for B - explicitly specify .tile mode for sm_100a compatibility
    // B tensor map: globalDim={K, N}, so coords are {k_coord, n_coord}
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(b_smem[slot])),
          "l"(tensor_map_B),
          "r"(b_coord_k),     // coordinate 0: K dimension (element coord)
          "r"(b_coord_n),     // coordinate 1: N dimension (element coord)
          "r"((uint32_t)barrier_addr)
        : "memory"
    );
}

__global__ void my_matmul_kernel(
    const __grid_constant__ CUtensorMap tensor_map_A,
    const __grid_constant__ CUtensorMap tensor_map_B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int* checkpoint_buffer  // Debug: tracks execution progress per block
) {
    // Checkpoint IDs:
    // 0 = kernel entry
    // 1 = after barrier init
    // 2 = before TMEM alloc
    // 3 = after TMEM alloc
    // 4 = entering tile loop
    // 5 = after tile processing
    // 6 = before TMEM dealloc
    // 7 = kernel exit

    checkpoint(checkpoint_buffer, 0);  // CHECKPOINT 0: Kernel entry

    // what thread am i? producer or consumer and which warp am i in
    int warp_id = threadIdx.x / 32;        // Which warp (0-7 for 256 threads)
    int wg_id = warp_id / 4;               // Which warpgroup (0=consumer, 1=producer)

    // smem allocation
    // Ring buffer for A and B tiles
    // TMA requires 128-byte alignment for shared memory destinations
    __shared__ __align__(128) __nv_bfloat16 a_smem[PIPE_DEPTH][TILE_M * TILE_K];  // 4 x 128 x 64 x 2 = 65,536 bytes
    __shared__ __align__(128) __nv_bfloat16 b_smem[PIPE_DEPTH][TILE_K * TILE_N];  // 4 x 64 x 256 x 2 = 131,072 bytes

    __shared__ uint32_t tmem_base;

    // barriers - raw PTX mbarrier storage (NOT cuda::barrier - they're incompatible!)
    __shared__ alignas(8) uint64_t inputs_arrived_storage[PIPE_DEPTH];
    __shared__ alignas(8) uint64_t inputs_finished_storage[PIPE_DEPTH];

    // initialization - barrier init only needs thread 0
    if (threadIdx.x == 0) {
        // Initialize barriers using PTX mbarrier.init
        for (int i = 0; i < PIPE_DEPTH; i++) {
            uint64_t* arrived_ptr = &inputs_arrived_storage[i];
            uint64_t* finished_ptr = &inputs_finished_storage[i];
            asm volatile(
                "mbarrier.init.shared.b64 [%0], %1;"
                :: "l"(__cvta_generic_to_shared(arrived_ptr)), "r"(2)
            );
            asm volatile(
                "mbarrier.init.shared.b64 [%0], %1;"
                :: "l"(__cvta_generic_to_shared(finished_ptr)), "r"(2)
            );
        }
    }
    __syncthreads();  // Ensure barriers are initialized before TMEM alloc

    checkpoint(checkpoint_buffer, 1);  // CHECKPOINT 1: After barrier init

    // TMEM allocation - tcgen05.alloc with cta_group::1 is a WARP-collective operation
    // Only ONE warp (32 threads) should execute it, not a warpgroup!

    checkpoint(checkpoint_buffer, 2);  // CHECKPOINT 2: Before TMEM alloc
    __syncthreads();

    uint32_t num_cols = TMEM_COLS;

    // Only warp 0 (threads 0-31) executes tcgen05.alloc
    // Per PTX docs: "When .cta_group::1 is specified, one warp from the CTA must perform the allocation"
    // Using register-based output (like CUTLASS) instead of shared memory output
    if (threadIdx.x < 32) {
        uint32_t tmem_addr_result;
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : "=r"(tmem_addr_result)
            : "r"(num_cols)
            : "memory"
        );
        // Only one thread writes to shared memory to avoid race
        if (threadIdx.x == 0) {
            tmem_base = tmem_addr_result;
        }
    }
    __syncthreads();  // Ensure all threads see the allocated tmem_base

    checkpoint(checkpoint_buffer, 3);  // CHECKPOINT 3: After TMEM alloc

    // Assert that TMEM allocation succeeded (tmem_base should be non-zero or valid)
    // Note: tmem_base == 0 might actually be valid, so this is just for debugging
    // dassert(tmem_base != 0xFFFFFFFF);  // Uncomment to check for allocation failure

    int num_tiles_m = M / TILE_M;
    int num_tiles_n = N / TILE_N;
    int num_k_tiles = K / TILE_K;
    int total_tiles = num_tiles_m * num_tiles_n;

    checkpoint(checkpoint_buffer, 4);  // CHECKPOINT 4: Entering tile loop

    // each block processes multiple output tiles
    // note that blockIdx.x  goes from 1 to 148
    // block 0 would process tiles 0, 148, 296, etc. and then block 1 would process 1, 149, etc. since we're striding by gridDim = 148
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {

        int tile_row = tile_id / num_tiles_n;
        int tile_col = tile_id % num_tiles_n;

        // Note the following 2 if statements run in parallel :D
        // TMEM zeroing is handled by using scale_d=0 on first WGMMA iteration

        // 1 thread of the producer loads the first tiles
        if (wg_id == 1 && threadIdx.x == 128) {  // First thread of producer warpgroup
            int prefetch_count = min(PIPE_DEPTH, num_k_tiles); // min (4, tiles) = 4 usually
            for (int k_tile = 0; k_tile < prefetch_count; k_tile++) {
                int slot = k_tile % PIPE_DEPTH;
                launch_tma_load( // loads in tiles A and B to SMEM
                    slot, k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived_storage[slot],  // raw mbarrier storage
                    &tensor_map_A, &tensor_map_B
                );
            }
        }

        // we split up the tiles of  A and B into further tiles
        // note that each tile of A is 128 x 64 and each tile of B is 64 x 256, where K = 64 (we just set that lol)
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) { // i think we have like 32 k tiles for 1024 x 1024
            int compute_slot = k_tile % PIPE_DEPTH; // which slot is the tile were computing on is in
            int load_k_tile = k_tile + PIPE_DEPTH; // which one to load for the producer
            int load_slot = load_k_tile % PIPE_DEPTH; // which slot to load it into

            // Consumer warpgroup waits for data, but ONLY ONE THREAD issues tcgen05.mma
            // tcgen05.mma is NOT warpgroup-collective like WGMMA - only 1 thread executes it!
            if (wg_id == 0) {
                // All threads in consumer warpgroup wait for data to arrive
                // Use PTX mbarrier operations (not C++ cuda::barrier) for consistency
                if (threadIdx.x == 0) {
                    uint64_t* barrier_ptr = &inputs_arrived_storage[compute_slot];
                    uint64_t barrier_addr = __cvta_generic_to_shared(barrier_ptr);

                    // Arrive and wait using PTX
                    // Phase bit alternates 0,1,0,1... for each use of the barrier
                    uint32_t phase = (k_tile / PIPE_DEPTH) & 1;

                    // First arrive
                    asm volatile(
                        "mbarrier.arrive.shared.b64 _, [%0];"
                        :: "l"(barrier_addr) : "memory"
                    );

                    // Then wait for phase completion
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "WAIT_LOOP:\n\t"
                        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
                        "@!p bra WAIT_LOOP;\n\t"
                        "}"
                        :: "l"(barrier_addr), "r"(phase) : "memory"
                    );
                }
                __syncwarp();  // Ensure all threads in warp 0 know data is ready

                // Only thread 0 of the entire CTA issues the tcgen05.mma instruction
                // Per NVIDIA docs: "Unlike WGMMA, only one thread is used to launch UMMA"
                if (threadIdx.x == 0) {
                    // Fence before tcgen05 operations to ensure memory ordering
                    asm volatile("tcgen05.fence::before_thread_sync;");

                    // Get shared memory addresses for current slot
                    uint32_t a_addr = __cvta_generic_to_shared(a_smem[compute_slot]);
                    uint32_t b_addr = __cvta_generic_to_shared(b_smem[compute_slot]);

                    // Build instruction descriptor for tcgen05.mma
                    // Table 42: .kind::f16 format for bf16 inputs, f32 output
                    // M=128, N=256, K=16
                    uint32_t idesc = 0;
                    idesc |= (1 << 4);    // bits 4-5: dtype = F32
                    idesc |= (1 << 7);    // bits 7-9: atype = BF16
                    idesc |= (1 << 10);   // bits 10-12: btype = BF16
                    idesc |= (32 << 17);  // bits 17-22: N >> 3 = 256 >> 3 = 32
                    idesc |= (8 << 24);   // bits 24-28: M >> 4 = 128 >> 4 = 8

                    // tcgen05.mma for 128x256 output tile with K=64
                    // Each tcgen05.mma handles 128 rows, 256 cols, 16 k-elements
                    // Need 4 k-steps (TILE_K=64, step=16)

                    // SMEM descriptor format (64-bit):
                    // bits 0-13:  base address >> 4
                    // bits 16-29: LBO (leading byte offset) - stride in K dimension
                    // bits 32-45: SBO (stride byte offset) - stride in M/N dimension
                    // bits 61-63: swizzle mode (0 = none)

                    // For A (128 x 64 bf16, row-major):
                    //   - LBO = stride to next K element = 2 bytes (adjacent in memory)
                    //   - SBO = stride to next M row = TILE_K * 2 = 128 bytes
                    // For B (64 x 256 bf16, stored column-major as 256 x 64):
                    //   - LBO = stride to next K element = 2 bytes (adjacent)
                    //   - SBO = stride to next N column = TILE_K * 2 = 128 bytes

                    #pragma unroll
                    for (int k_step = 0; k_step < TILE_K; k_step += 16) {
                        // A offset: k_step columns into the 128x64 A tile
                        // B offset: k_step rows into the 64x256 B tile
                        uint32_t a_offset = k_step * sizeof(__nv_bfloat16);
                        uint32_t b_offset = k_step * sizeof(__nv_bfloat16);  // B is K-major

                        uint32_t a_smem_addr = a_addr + a_offset;
                        uint32_t b_smem_addr = b_addr + b_offset;

                        // A descriptor: 128 rows x K bf16
                        // LBO = 16 bytes (8 bf16 elements, core matrix width)
                        // SBO = TILE_K * 2 = 128 bytes (stride to next row of 8 rows)
                        uint64_t a_lbo = 16;   // Core matrix is 8 elements wide = 16 bytes
                        uint64_t a_sbo = TILE_K * sizeof(__nv_bfloat16);  // 128 bytes

                        uint64_t a_desc = ((uint64_t)(a_smem_addr) & 0x3FFFF) |
                                          ((a_lbo & 0x3FFF) << 16) |
                                          ((a_sbo & 0x3FFF) << 32);

                        // B descriptor: K x 256 bf16 (K-major)
                        // LBO = 16 bytes (core matrix width)
                        // SBO = TILE_K * 2 = 128 bytes (stride to next column group)
                        uint64_t b_lbo = 16;
                        uint64_t b_sbo = TILE_K * sizeof(__nv_bfloat16);  // 128 bytes

                        uint64_t b_desc = ((uint64_t)(b_smem_addr) & 0x3FFFF) |
                                          ((b_lbo & 0x3FFF) << 16) |
                                          ((b_sbo & 0x3FFF) << 32);

                        // tcgen05.mma format (from PTX docs):
                        // tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc,
                        //                                    {disable-output-lane x4}, enable-input-d;
                        // - disable-output-lane: 4 x 32-bit masks (0 = don't disable)
                        // - enable-input-d: predicate (0 = D=A*B, 1 = D=A*B+D)

                        // First iteration zeros accumulator, the rest accumulate
                        if (k_tile == 0 && k_step == 0) {
                            // D = A*B (enable_d = false, no accumulation)
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.eq.u32 p, 0, 1;\n\t"  // p = false
                                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                                "}"
                                :
                                : "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                                : "memory"
                            );
                        } else {
                            // D = A*B + D (enable_d = true, accumulate)
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.eq.u32 p, 1, 1;\n\t"  // p = true
                                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                                "}"
                                :
                                : "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                                : "memory"
                            );
                        }
                    }

                }

                // All consumer threads sync before signaling completion
                __syncwarp();

                // Consumer tells producer that it's done with this slot (PTX arrive)
                if (threadIdx.x == 0) {
                    uint64_t* barrier_ptr = &inputs_finished_storage[compute_slot];
                    uint64_t barrier_addr = __cvta_generic_to_shared(barrier_ptr);
                    asm volatile(
                        "mbarrier.arrive.shared.b64 _, [%0];"
                        :: "l"(barrier_addr) : "memory"
                    );
                }
            }

            // Producer loads the next tile (runs in parallel with consumer's WGMMA!)
            if (wg_id == 1 && threadIdx.x == 128 && load_k_tile < num_k_tiles) {
                // Wait for consumer to finish with the slot we're about to reuse
                if (k_tile >= PIPE_DEPTH) {
                    uint64_t* barrier_ptr = &inputs_finished_storage[load_slot];
                    uint64_t barrier_addr = __cvta_generic_to_shared(barrier_ptr);

                    // Phase bit for this barrier
                    // First wait on slot 0 happens at k_tile=4, phase should be 0
                    // Second wait on slot 0 happens at k_tile=8, phase should be 1
                    uint32_t phase = ((k_tile - PIPE_DEPTH) / PIPE_DEPTH) & 1;

                    // First arrive
                    asm volatile(
                        "mbarrier.arrive.shared.b64 _, [%0];"
                        :: "l"(barrier_addr) : "memory"
                    );

                    // Then wait for phase completion
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "WAIT_LOOP2:\n\t"
                        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
                        "@!p bra WAIT_LOOP2;\n\t"
                        "}"
                        :: "l"(barrier_addr), "r"(phase) : "memory"
                    );
                }

                launch_tma_load(
                    load_slot, load_k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived_storage[load_slot],  // raw mbarrier storage
                    &tensor_map_A, &tensor_map_B
                );
            }

        }

        // Thread 0 must wait for MMA to complete BEFORE syncing with other threads
        // tcgen05.fence only waits for ops "issued by the executing thread"
        // So only thread 0 (which issued the MMA) can wait for its completion
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::before_thread_sync;");
        }

        // Now sync all threads - after this, all threads know MMA is complete
        __syncthreads();

        // store results from TMEM to global memory
        // Consumer warpgroup reads from TMEM and writes directly to global memory
        if (wg_id == 0) {

            // tcgen05.ld is warp-collective: all 32 threads in a warp must use same taddr
            // using shape .16x256b.x1: loads 512 bytes (1 TMEM column = 128 floats)
            // each thread gets 4 registers (4 floats), 32 threads × 4 = 128 floats
            // consumer warpgroup has 4 warps, each handles 64 columns (256/4)

            int warp_in_consumer = threadIdx.x / 32;  // 0-3 within consumer warpgroup
            int lane_id = threadIdx.x % 32;           // 0-31 within warp
            int cols_per_warp = TMEM_COLS / 4;        // 64 columns per warp

            for (int col_idx = 0; col_idx < cols_per_warp; col_idx++) {
                int col = warp_in_consumer * cols_per_warp + col_idx;

                // all threads in warp use the SAME taddr (base of this column)
                uint32_t taddr = tmem_base + col * 512;  // 512 bytes per column

                // Collective load: each thread receives 4 floats from the column
                float r0, r1, r2, r3;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
                    : "r"(taddr)
                );

                // Wait for tcgen05.ld to complete before using data
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                // Write directly to global memory (no SMEM staging)
                // Assuming lane t gets rows [t*4, t*4+3] of this column
                int base_row = lane_id * 4;
                int global_col = tile_col * TILE_N + col;

                if (base_row + 3 < TILE_M) {
                    int global_row_base = tile_row * TILE_M + base_row;
                    C[(global_row_base + 0) * N + global_col] = __float2bfloat16(r0);
                    C[(global_row_base + 1) * N + global_col] = __float2bfloat16(r1);
                    C[(global_row_base + 2) * N + global_col] = __float2bfloat16(r2);
                    C[(global_row_base + 3) * N + global_col] = __float2bfloat16(r3);
                }
            }
        }

        // Sync before next output tile
        __syncthreads();
    }

    checkpoint(checkpoint_buffer, 5);  // CHECKPOINT 5: After tile processing

    // deallocate tmem - tcgen05.dealloc with cta_group::1 needs only ONE warp
    __syncthreads();  // Ensure all threads are done before dealloc

    checkpoint(checkpoint_buffer, 6);  // CHECKPOINT 6: Before TMEM dealloc

    uint32_t dealloc_num_cols = TMEM_COLS;
    // Only warp 0 (threads 0-31) executes tcgen05.dealloc
    // Per PTX docs: "When .cta_group::1 is specified, one warp from the CTA must perform the allocation and de-allocation"
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_base), "r"(dealloc_num_cols)
            : "memory"
        );
    }

    checkpoint(checkpoint_buffer, 7);  // CHECKPOINT 7: Kernel exit
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
    cuuint64_t globalDim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};

    // strides
    cuuint64_t globalStrides[1] = {
        (cuuint64_t)cols * sizeof(__nv_bfloat16)  // Stride to next row in bytes
    };
    cuuint32_t boxDim[2] = {(cuuint32_t)tile_cols, (cuuint32_t)tile_rows};
    cuuint32_t elementStrides[2] = {1, 1};

    //debug
    std::cout << "TMA params: globalDim={" << globalDim[0] << "," << globalDim[1] << "}"
              << " boxDim={" << boxDim[0] << "," << boxDim[1] << "}"
              << " stride=" << globalStrides[0]
              << " boxDim[0]*sizeof=" << (boxDim[0] * sizeof(__nv_bfloat16)) << " bytes" << std::endl;


    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,      
        2,                                      
        global_ptr,                          
        globalDim,                             
        globalStrides,                         
        boxDim,                                 
        elementStrides,                        
        CU_TENSOR_MAP_INTERLEAVE_NONE,         
        CU_TENSOR_MAP_SWIZZLE_NONE,           
        CU_TENSOR_MAP_L2_PROMOTION_NONE,       
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE     
    );

    if (result != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled FAILED with error: " << result << std::endl;
        std::cerr << "  Possible issue: boxDim[0]*sizeof(element) = "
                  << (boxDim[0] * sizeof(__nv_bfloat16))
                  << " bytes (max 128 bytes for SWIZZLE_NONE)" << std::endl;
    }

    return tensor_map;
}


// cpu
void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2)  
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

// Helper function to print checkpoint results
void print_checkpoint_results(int* h_checkpoints, int num_blocks, const char* checkpoint_names[]) {
    std::cout << "\n=== CHECKPOINT RESULTS ===" << std::endl;
    for (int cp = 0; cp < NUM_CHECKPOINTS; cp++) {
        int reached_count = 0;
        for (int b = 0; b < num_blocks; b++) {
            if (h_checkpoints[b * NUM_CHECKPOINTS + cp]) {
                reached_count++;
            }
        }
        if (checkpoint_names[cp]) {
            std::cout << "  Checkpoint " << cp << " (" << checkpoint_names[cp] << "): "
                      << reached_count << "/" << num_blocks << " blocks" << std::endl;
        }
    }
    std::cout << "==========================\n" << std::endl;
}

// benchmark
int run_benchmark(size_t M, size_t N, size_t K) {
    // Initialize CUDA driver API (required for TMA)
    CU_CHECK(cuInit(0));

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Tile sizes: TILE_M=" << TILE_M << ", TILE_N=" << TILE_N << ", TILE_K=" << TILE_K << std::endl;

    // Launch configuration
    // DEBUG: Start with 1 block to verify tcgen05 instructions work
    // Then increase gradually: 1 -> 4 -> 16 -> 148
    const int NUM_BLOCKS = 1;
    dim3 grid(NUM_BLOCKS, 1);       // Reduced for debugging
    dim3 block(NUM_THREADS);        // 256 threads per block (2 warpgroups)

    // Checkpoint names for debugging
    const char* checkpoint_names[NUM_CHECKPOINTS] = {
        "kernel entry",           // 0
        "after barrier init",     // 1
        "before TMEM alloc",      // 2
        "after TMEM alloc",       // 3
        "entering tile loop",     // 4
        "after tile processing",  // 5
        "before TMEM dealloc",    // 6
        "kernel exit",            // 7
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
    };

    // Allocate checkpoint buffer (device + host)
    int* d_checkpoints;
    int* h_checkpoints = new int[NUM_BLOCKS * NUM_CHECKPOINTS]();
    CUDA_CHECK(cudaMalloc(&d_checkpoints, NUM_BLOCKS * NUM_CHECKPOINTS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_checkpoints, 0, NUM_BLOCKS * NUM_CHECKPOINTS * sizeof(int)));

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

    // cpu
    bool do_validation = (M <= 2048);
    if (do_validation) {
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));

    std::cout << "Allocated device memory" << std::endl;

    // Convert to bfloat16 and copy to device
    __nv_bfloat16* h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B_bf16 = new __nv_bfloat16[K * N];

    // A is row-major (M x K)
    for (size_t i = 0; i < M * K; i++) h_A_bf16[i] = __float2bfloat16(h_A[i]);

    // B needs to be column-major for TMA (K contiguous, not N)
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            h_B_bf16[n * K + k] = __float2bfloat16(h_B[k * N + n]);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    std::cout << "Copied matrices to device (B in column-major)" << std::endl;

    // Create TMA descriptors
    // A: M x K row-major, inner dim = K, boxDim[0] = TILE_K = 64
    CUtensorMap tensor_map_A = create_tensor_map(d_A, M, K, TILE_M, TILE_K);
    // B: now N x K column-major (K contiguous), inner dim = K, boxDim[0] = TILE_K = 64
    CUtensorMap tensor_map_B = create_tensor_map(d_B, N, K, TILE_N, TILE_K);

    std::cout << "Created TMA descriptors" << std::endl;
    std::cout << "Launching kernel with grid(" << grid.x << "), block(" << block.x << ")" << std::endl;

    // Reset checkpoint buffer before kernel launch
    CUDA_CHECK(cudaMemset(d_checkpoints, 0, NUM_BLOCKS * NUM_CHECKPOINTS * sizeof(int)));

    // Warmup/debug run
    std::cout << "Running kernel..." << std::endl;
    my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K, d_checkpoints);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait and check for execution errors

    // Copy checkpoint results back to host
    CUDA_CHECK(cudaMemcpy(h_checkpoints, d_checkpoints, NUM_BLOCKS * NUM_CHECKPOINTS * sizeof(int), cudaMemcpyDeviceToHost));

    // Print checkpoint results
    print_checkpoint_results(h_checkpoints, NUM_BLOCKS, checkpoint_names);

    // Benchmark runs (only if warmup succeeded)
    constexpr int ITERS = 5;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERS; i++) {
        my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K, d_checkpoints);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Copy result back to host
    __nv_bfloat16* h_C_bf16 = new __nv_bfloat16[M * N];
    CUDA_CHECK(cudaMemcpy(h_C_bf16, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

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
    delete[] h_checkpoints;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_checkpoints));

    std::cout << "Done!\n" << std::endl;

    return 0;
}

int main() {
    // Run single benchmark for debugging
    // Uncomment additional sizes once the kernel works correctly
    run_benchmark(1024, 1024, 1024);
    // run_benchmark(2048, 2048, 2048);
    // run_benchmark(4096, 4096, 4096);
    // run_benchmark(8192, 8192, 8192);

    return 0;
}
