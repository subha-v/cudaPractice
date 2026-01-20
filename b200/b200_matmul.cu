#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda/barrier>
#include <chrono>
#include <cmath>
#include <omp.h>

// debug
__device__ __forceinline__ void dassert(bool condition) {
    if (!condition) {
        asm volatile("trap;");
    }
}

//debug
#define NUM_CHECKPOINTS 16
__device__ __forceinline__ void checkpoint(int* buffer, int checkpoint_id) {
    if (threadIdx.x == 0) {
        buffer[blockIdx.x * NUM_CHECKPOINTS + checkpoint_id] = 1;
    }
}


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

// We want to launch 148 blocks and have those persistently running on the B200
// A kernel is executed as a grid of blocks of threads
// Each CUDA block is executed by one streaming multiprocessor (SM)

constexpr int TILE_M = 128;  
constexpr int TILE_N = 256;  
constexpr int TILE_K = 64;   

constexpr int NUM_CONSUMERS = 1;
constexpr int NUM_PRODUCERS = 1;
constexpr int PIPE_DEPTH = 4;   // 4 slots for the pipeline
constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;  // 4 warps per warpgroup
constexpr int NUM_THREADS = NUM_WORKERS * 32;  // 32 threads per warp = 256 threads


constexpr int TMEM_COLS = 256;  

// 3d tensor map
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
    uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(barrier_addr), "r"(expected_bytes)
        : "memory"
    );

    int a_coord_0 = 0;                        
    int a_coord_1 = tile_row * TILE_M;       
    int a_coord_2 = k_tile;                
    int b_coord_0 = 0;                      
    int b_coord_1 = tile_col * TILE_N;       
    int b_coord_2 = k_tile;                  

    // tma A
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(a_smem[slot])),
          "l"(tensor_map_A),
          "r"(a_coord_0),     // coordinate 0: always 0
          "r"(a_coord_1),     // coordinate 1: M offset
          "r"(a_coord_2),     // coordinate 2: K/64 block
          "r"((uint32_t)barrier_addr)
        : "memory"
    );

    // tma B
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(b_smem[slot])),
          "l"(tensor_map_B),
          "r"(b_coord_0),     // coordinate 0: always 0
          "r"(b_coord_1),     // coordinate 1: N offset
          "r"(b_coord_2),     // coordinate 2: K/64 block
          "r"((uint32_t)barrier_addr)
        : "memory"
    );
}

__global__ __cluster_dims__(1, 1, 1) void my_matmul_kernel(
    const __grid_constant__ CUtensorMap tensor_map_A,
    const __grid_constant__ CUtensorMap tensor_map_B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int* checkpoint_buffer   
) {
    

    checkpoint(checkpoint_buffer, 0);  

    // what thread am i? producer or consumer and which warp am i in
    int warp_id = threadIdx.x / 32;        // Which warp (0-7 for 256 threads)
    int wg_id = warp_id / 4;               // Which warpgroup (0=consumer, 1=producer)

    // smem allocation
    // Ring buffer for A and B tiles
    // TMA requires 128-byte alignment for shared memory destinations
    __shared__ __align__(128) __nv_bfloat16 a_smem[PIPE_DEPTH][TILE_M * TILE_K];  // 4 x 128 x 64 x 2 = 65,536 bytes
    __shared__ __align__(128) __nv_bfloat16 b_smem[PIPE_DEPTH][TILE_K * TILE_N];  // 4 x 64 x 256 x 2 = 131,072 bytes

    __shared__ int tmem_base[1];  // tmem address is 32-bit

    // barriers - raw PTX mbarrier storage (NOT cuda::barrier - they're incompatible!)
    __shared__ alignas(8) uint64_t inputs_arrived_storage[PIPE_DEPTH];
    __shared__ alignas(8) uint64_t inputs_finished_storage[PIPE_DEPTH];
    __shared__ alignas(8) uint64_t mma_mbar[1];  // For async MMA completion tracking

    // initialization - barrier init only needs thread 0
    if (threadIdx.x == 0) {
        // Initialize barriers using PTX mbarrier.init
       
        for (int i = 0; i < PIPE_DEPTH; i++) {
            uint64_t* arrived_ptr = &inputs_arrived_storage[i];
            uint64_t* finished_ptr = &inputs_finished_storage[i];
            asm volatile(
                "mbarrier.init.shared.b64 [%0], %1;"
                :: "l"(__cvta_generic_to_shared(arrived_ptr)), "r"(1)
            );
            asm volatile(
                "mbarrier.init.shared.b64 [%0], %1;"
                :: "l"(__cvta_generic_to_shared(finished_ptr)), "r"(1)
            );
        }
      // mma completion barrier
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;"
            :: "l"(__cvta_generic_to_shared(mma_mbar)), "r"(1)
        );
        // MAKE BARRIER VISIBLE YAY
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();  // Ensure barriers are initialized before TMEM alloc

    checkpoint(checkpoint_buffer, 1);  

   
    // Only ONE warp (32 threads) should execute it, not a warpgroup!

    checkpoint(checkpoint_buffer, 2);  
    __syncthreads();

    uint32_t num_cols = TMEM_COLS;

    // use warp 1 like the example ig
    if (threadIdx.x >= 32 && threadIdx.x < 64) {  // warp 1
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_base));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(addr), "r"(num_cols)
            : "memory"
        );
    }
    __syncthreads();  // ensure everyone got tmem base

    checkpoint(checkpoint_buffer, 3);   

    int num_tiles_m = M / TILE_M;
    int num_tiles_n = N / TILE_N;
    int num_k_tiles = K / TILE_K;
    int total_tiles = num_tiles_m * num_tiles_n;

    
    const uint32_t mma_mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mma_mbar));
    int mma_phase = 0;  // alternates 0/1

    checkpoint(checkpoint_buffer, 4);   

    // each block processes multiple output tiles
    // note that blockIdx.x  goes from 1 to 148
    // block 0 would process tiles 0, 148, 296, etc. and then block 1 would process 1, 149, etc. since we're striding by gridDim = 148
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {

        int tile_row = tile_id / num_tiles_n;
        int tile_col = tile_id % num_tiles_n;

        // Note the following 2 if statements run in parallel :D
        // TMEM zeroing is handled by using scale_d=0 on first WGMMA iteration

        // 1 thread of the producer loads the first tiles
        // Prefetch PIPE_DEPTH - 1 tiles, leaving one slot free for the main loop

        if (wg_id == 1 && threadIdx.x == 128) {  
            int prefetch_count = min(PIPE_DEPTH - 1, num_k_tiles); 
            for (int k_tile = 0; k_tile < prefetch_count; k_tile++) {
                int slot = k_tile % PIPE_DEPTH;
                launch_tma_load(  // loads A and B
                    slot, k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived_storage[slot],  
                    &tensor_map_A, &tensor_map_B
                );
            }
        }

        __syncthreads();

        // we split up the tiles of  A and B into further tiles
        // note that each tile of A is 128 x 64 and each tile of B is 64 x 256, where K = 64 (we just set that lol)
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) { // i think we have like 32 k tiles for 1024 x 1024
            int compute_slot = k_tile % PIPE_DEPTH; // which slot is the tile were computing on is in
            // With PIPE_DEPTH-1 prefetch, we've loaded tiles 0..(PIPE_DEPTH-2)
            // At k_tile=0, load tile (PIPE_DEPTH-1) which fills the last empty slot
            int load_k_tile = k_tile + (PIPE_DEPTH - 1); // which one to load for the producer
            int load_slot = load_k_tile % PIPE_DEPTH; // which slot to load it into

            // Consumer warpgroup waits for data, but ONLY ONE THREAD issues tcgen05.mma
            // tcgen05.mma is NOT warpgroup-collective like WGMMA - only 1 thread executes it!
            if (wg_id == 0) {
                // Thread 0 waits for TMA data to arrive (producer signals via expect_tx)
                // Consumer does NOT arrive - only waits! The barrier is signaled by:
                //   1. Producer's mbarrier.arrive.expect_tx
                //   2. TMA hardware completion
                if (threadIdx.x == 0) {
                    uint64_t* barrier_ptr = &inputs_arrived_storage[compute_slot];
                    uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));

                    //  phase bit alternates
                    uint32_t phase = (k_tile / PIPE_DEPTH) & 1;

                    // wait for TMA completion
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "WAIT_LOOP:\n\t"
                        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                        "@!p bra WAIT_LOOP;\n\t"
                        "}"
                        :: "r"(barrier_addr), "r"(phase) : "memory"
                    );
                }
                __syncwarp();  // Ensure all threads in warp 0 know data is ready

                // only one thread is needed
                if (threadIdx.x == 0) {
                    // Fence before tcgen05 operations to ensure memory ordering
                    asm volatile("tcgen05.fence::before_thread_sync;");

                    // Get shared memory addresses for current slot
                    uint32_t a_addr = __cvta_generic_to_shared(a_smem[compute_slot]);
                    uint32_t b_addr = __cvta_generic_to_shared(b_smem[compute_slot]);

                    // build instruction descriptor
                    uint32_t idesc = 0;
                    idesc |= (1 << 4);    // bits 4-5: dtype = F32
                    idesc |= (1 << 7);    // bits 7-9: atype = BF16
                    idesc |= (1 << 10);   // bits 10-12: btype = BF16
                    idesc |= (32 << 17);  // bits 17-22: N >> 3 = 256 >> 3 = 32
                    idesc |= (8 << 24);   // bits 24-28: M >> 4 = 128 >> 4 = 8

                   

                    // Helper lambda for descriptor encoding
                    auto desc_encode = [](uint32_t x) -> uint64_t {
                        return (x & 0x3FFFF) >> 4;
                    };

                    // SBO = 8 * 128 = 1024 bytes (stride for 128B swizzle pattern)
                    // This is the stride to the next swizzle block
                    constexpr uint32_t SBO = 8 * 128;  // 1024 bytes

                    // Make SMEM descriptor with 128B swizzle  
                    auto make_desc = [&](uint32_t addr) -> uint64_t {
                        return desc_encode(addr)              // bits 0-13: encoded address
                             | (desc_encode(SBO) << 32)       // bits 32-45: encoded SBO
                             | (1ULL << 46)                   // bits 46-48: fixed 0b001
                             | (2ULL << 61);                  // bits 61-63: 128B swizzle
                    };

                    // tcgen05.mma 
                    constexpr int MMA_K = 16;

                    #pragma unroll
                    for (int k2 = 0; k2 < TILE_K / MMA_K; k2++) {
                        // Each k2 iteration processes 16 K elements (32 bytes)
                        uint64_t a_desc = make_desc(a_addr + k2 * 32);
                        uint64_t b_desc = make_desc(b_addr + k2 * 32);

                        // First iteration zeros accumulator, the rest accumulate
                        if (k_tile == 0 && k2 == 0) {
                            // D = A*B (enable_d = false, no accumulation)
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.eq.u32 p, 0, 1;\n\t"  // p = false
                                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                                "}"
                                :
                                : "r"(tmem_base[0]), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
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
                                : "r"(tmem_base[0]), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                                : "memory"
                            );
                        }
                    }

                    // ASYNC MMA COMPLETION: tcgen05.commit signals mbarrier when MMA finishes
                    // This is NON-BLOCKING - thread continues immediately, hardware signals later
                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mma_mbar_addr) : "memory"
                    );
                }

                // All consumer threads sync before signaling completion
                __syncwarp();

                // Consumer tells producer that it's done with this slot (PTX arrive)
                if (threadIdx.x == 0) {
                    uint64_t* barrier_ptr = &inputs_finished_storage[compute_slot];
                    uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                        :: "r"(barrier_addr) : "memory"
                    );
                }
            }

            // Producer loads the next tile (runs in parallel with consumer's WGMMA!)
            if (wg_id == 1 && threadIdx.x == 128 && load_k_tile < num_k_tiles) {
               
                if (load_k_tile >= PIPE_DEPTH) {
                    uint64_t* barrier_ptr = &inputs_finished_storage[load_slot];
                    uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));

                  
                    uint32_t phase = ((k_tile - 1) / PIPE_DEPTH) & 1;

                    // Producer only waits - consumer signals via arrive
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "WAIT_LOOP2:\n\t"
                        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                        "@!p bra WAIT_LOOP2;\n\t"
                        "}"
                        :: "r"(barrier_addr), "r"(phase) : "memory"
                    );
                }

                launch_tma_load(
                    load_slot, load_k_tile, tile_row, tile_col,
                    a_smem, b_smem,
                    &inputs_arrived_storage[load_slot],  // raw mbarrier storage
                    &tensor_map_A, &tensor_map_B
                );
            }

            if (threadIdx.x == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "MMA_WAIT_LOOP:\n\t"
                    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                    "@!p bra MMA_WAIT_LOOP;\n\t"
                    "}"
                    :: "r"(mma_mbar_addr), "r"(mma_phase) : "memory"
                );
            }
            mma_phase ^= 1;  // flip phase
        }
        asm volatile("tcgen05.fence::after_thread_sync;");

      
        __syncthreads();

        // epilogue

        float* output_smem = reinterpret_cast<float*>(a_smem[0]);  // 64KB = 16K floats = 128×128

        // Process columns in two halves (128 cols each)
        for (int col_half = 0; col_half < 2; col_half++) {
            int col_base = col_half * 128;  // 0 or 128

            if (wg_id == 0) {
                int warp_id = threadIdx.x / 32;
                int lane_id = threadIdx.x % 32;

                for (int chunk = 0; chunk < 16; chunk++) {
                    int n = col_base / 8 + chunk;  // TMEM column chunk index
                    uint32_t addr = tmem_base[0] + ((warp_id * 32) << 16) + (n * 8);

                    float tmp[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                        : "r"(addr)
                    );
                    asm volatile("tcgen05.wait::ld.sync.aligned;");

                    // Store to SMEM: [row][col] layout, 128 cols stride
                    int my_row = warp_id * 32 + lane_id;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        output_smem[my_row * 128 + chunk * 8 + c] = tmp[c];
                    }
                }
            }
            __syncthreads();

            if (wg_id == 0) {
                int warp_id = threadIdx.x / 32;
                int lane_id = threadIdx.x % 32;

                // 4 warps write 4 rows per iteration, 32 iterations for 128 rows
                for (int r = 0; r < TILE_M; r += 4) {
                    int row = r + warp_id;  // Which row this warp handles
                    int col_start = lane_id * 4;  // Each lane writes 4 cols (8 bytes)
                    // Note: 128 cols / 32 lanes = 4 cols per lane

                    // Load 4 floats from SMEM
                    float vals[4];
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        vals[c] = output_smem[row * 128 + col_start + c];
                    }

                    // Convert to bf16 pairs and write as int2 (8 bytes)
                    __nv_bfloat162 bvals[2];
                    bvals[0] = __floats2bfloat162_rn(vals[0], vals[1]);
                    bvals[1] = __floats2bfloat162_rn(vals[2], vals[3]);

                    int global_row = tile_row * TILE_M + row;
                    int global_col = tile_col * TILE_N + col_base + col_start;
                    __nv_bfloat16* out_ptr = C + global_row * N + global_col;
                    reinterpret_cast<int2*>(out_ptr)[0] = reinterpret_cast<int2*>(bvals)[0];
                }
            }
            __syncthreads();
        }

    
        __syncthreads();
    }

    checkpoint(checkpoint_buffer, 5);  

    // deallocate tmem - tcgen05.dealloc with cta_group::1 needs only ONE warp
    __syncthreads();  // Ensure all threads are done before dealloc

    checkpoint(checkpoint_buffer, 6);  // CHECKPOINT 6: Before TMEM dealloc

    uint32_t dealloc_num_cols = TMEM_COLS;
    // Use warp 1 (threads 32-63) for dealloc - same warp that did alloc
    // Per PTX docs: "When .cta_group::1 is specified, one warp from the CTA must perform the allocation and de-allocation"
    if (threadIdx.x >= 32 && threadIdx.x < 64) {  // warp 1
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_base[0]), "r"(dealloc_num_cols)
            : "memory"
        );
    }

    checkpoint(checkpoint_buffer, 7);  
}


// create TMA tensor map - 3D with 128B swizzle for tcgen05.mma
// Layout: (64, HEIGHT, K/64) with strides (K*sizeof, 128)
// This creates the swizzled layout expected by tcgen05.mma
CUtensorMap create_tensor_map_3d(
    void* global_ptr,           // Pointer to matrix in global memory
    int height,                 // Number of rows (M for A, N for B)
    int K,                      // K dimension size
    int tile_height,            // Tile height (TILE_M for A, TILE_N for B)
    int tile_k                  // Tile K dimension (TILE_K)
) {
    CUtensorMap tensor_map;

    
    constexpr uint32_t rank = 3;
    cuuint64_t globalDim[rank] = {64, (cuuint64_t)height, (cuuint64_t)K / 64};

   
    cuuint64_t globalStrides[rank-1] = {
        (cuuint64_t)K * sizeof(__nv_bfloat16),  // stride to next row
        128                                      // stride between 64-element K blocks
    };

    // Box dimensions for the tile
    cuuint32_t boxDim[rank] = {
        64,                                      // always 64 (the swizzle unit)
        (cuuint32_t)tile_height,                 // TILE_M or TILE_N
        (cuuint32_t)tile_k / 64                  // TILE_K / 64
    };
    cuuint32_t elementStrides[rank] = {1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,                                   // 3D tensor
        global_ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,            // 128-byte swizzling!
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        std::cerr << "cuTensorMapEncodeTiled FAILED with error: " << result << std::endl;
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
    // B200 has 148 SMs - use all of them for peak performance
    int num_tiles = (M / TILE_M) * (N / TILE_N);
    int NUM_BLOCKS = min(num_tiles, 148);  // Use up to 148 blocks (one per SM)
    dim3 grid(NUM_BLOCKS, 1);
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

    // Create TMA descriptors - 3D with 128B swizzle
    // A: M x K row-major, loaded as 3D (64, M, K/64)
    CUtensorMap tensor_map_A = create_tensor_map_3d(d_A, M, K, TILE_M, TILE_K);
    // B: N x K (transposed), loaded as 3D (64, N, K/64)
    CUtensorMap tensor_map_B = create_tensor_map_3d(d_B, N, K, TILE_N, TILE_K);

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
    // Run benchmarks with same sizes as TK kernel for comparison
    // TK only tests 8192 and 16384 since small sizes don't saturate GPU
    run_benchmark(8192, 8192, 8192);
    run_benchmark(16384, 16384, 16384);

    return 0;
}
