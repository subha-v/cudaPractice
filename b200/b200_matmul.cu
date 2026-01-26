#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda/barrier>
#include <chrono>
#include <cmath>
#include <omp.h>

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

__global__ __cluster_dims__(1, 1, 1) void my_matmul_kernel(
    const __grid_constant__ CUtensorMap tensor_map_A,
    const __grid_constant__ CUtensorMap tensor_map_B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
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
    // Per-stage barriers for pipelined execution:
    //   - tma_mbar[PIPE_DEPTH]: TMA completion per stage (producer signals, consumer waits)
    //   - mma_mbar[PIPE_DEPTH]: MMA completion per stage (consumer signals, producer waits for reuse)
    //   - mainloop_mbar[1]: Final MMA completion for epilogue
    __shared__ alignas(8) uint64_t tma_mbar[PIPE_DEPTH];      // TMA arrival barriers
    __shared__ alignas(8) uint64_t mma_mbar[PIPE_DEPTH];      // MMA completion barriers (per-stage!)
    __shared__ alignas(8) uint64_t mainloop_mbar[1];          // Final completion for epilogue

    // initialization - barrier init only needs thread 0
    if (threadIdx.x == 0) {
        // Initialize all barriers using PTX mbarrier.init
        for (int i = 0; i < PIPE_DEPTH; i++) {
            uint32_t tma_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tma_mbar[i]));
            uint32_t mma_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_mbar[i]));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(tma_addr), "r"(1));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mma_addr), "r"(1));
        }
        // mainloop completion barrier
        uint32_t mainloop_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mainloop_mbar));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mainloop_addr), "r"(1));
        // MAKE BARRIER VISIBLE
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();  // Ensure barriers are initialized before TMEM alloc

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

    int num_tiles_m = M / TILE_M;
    int num_tiles_n = N / TILE_N;
    int num_k_tiles = K / TILE_K;
    int total_tiles = num_tiles_m * num_tiles_n;

    // Compute barrier base addresses for efficient addressing
    const uint32_t tma_mbar_base = static_cast<uint32_t>(__cvta_generic_to_shared(tma_mbar));
    const uint32_t mma_mbar_base = static_cast<uint32_t>(__cvta_generic_to_shared(mma_mbar));
    const uint32_t mainloop_mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mainloop_mbar));

    // Phase tracking - alternates 0/1 as we cycle through stages
    int phase = 0;
    int mainloop_phase = 0;  // Tracks mainloop barrier phase across tiles

    // Build instruction descriptor once (constant across all tiles)
    constexpr uint32_t idesc = (1U << 4U)    // bits 4-5: dtype = F32
                             | (1U << 7U)    // bits 7-9: atype = BF16
                             | (1U << 10U)   // bits 10-12: btype = BF16
                             | ((uint32_t)TILE_N >> 3U << 17U)   // bits 17-22: N >> 3
                             | ((uint32_t)TILE_M >> 4U << 24U);  // bits 24-28: M >> 4

    // Helper for SMEM descriptor encoding
    auto desc_encode = [](uint32_t x) -> uint64_t {
        return (x & 0x3FFFF) >> 4;
    };
    constexpr uint32_t SBO = 8 * 128;  // 1024 bytes (stride for 128B swizzle pattern)
    auto make_desc = [&](uint32_t addr) -> uint64_t {
        return desc_encode(addr)              // bits 0-13: encoded address
             | (desc_encode(SBO) << 32)       // bits 32-45: encoded SBO
             | (1ULL << 46)                   // bits 46-48: fixed 0b001
             | (2ULL << 61);                  // bits 61-63: 128B swizzle
    };

    constexpr int MMA_K = 16;  // Process 16 K elements per MMA

    // Helper function to wait on an mbarrier with given phase
    // Uses .acquire semantics for proper memory ordering (from example kernel)
    auto mbarrier_wait = [](uint32_t mbar_addr, uint32_t wait_phase) {
        uint32_t ticks = 0x989680;  // timeout ticks (optional)
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_%=:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
            "@P1 bra.uni DONE_%=;\n\t"
            "bra.uni LAB_WAIT_%=;\n\t"
            "DONE_%=:\n\t"
            "}"
            :: "r"(mbar_addr), "r"(wait_phase), "r"(ticks)
        );
    };

    // each block processes multiple output tiles
    for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {

        int tile_row = tile_id / num_tiles_n;
        int tile_col = tile_id % num_tiles_n;

        // Reset phase for each new tile
        phase = 0;

        // ============================================================================
        // PIPELINED MAINLOOP: TMA and MMA run as separate warps with independent loops
        //
        // Key insight from example kernel:
        // - Warp 0 (TMA): loads data, waits on mma_mbar[stage] before reusing slot
        // - Warp 1 (MMA): computes, waits on tma_mbar[stage] before using data
        // - MMA warp NEVER waits on its own completion - just issues and moves on
        // - Only final mainloop_mbar wait before epilogue
        // ============================================================================

        if (warp_id == 0 && threadIdx.x == 0) {
            // ========== TMA WARP ==========
            // Runs independent loop issuing TMA loads
            // Waits on mma_mbar[stage] before reusing slot (producer-side wait)

            for (int iter_k = 0; iter_k < num_k_tiles; iter_k++) {
                int stage_id = iter_k % PIPE_DEPTH;
                uint32_t mma_stage_addr = mma_mbar_base + stage_id * 8;
                uint32_t tma_stage_addr = tma_mbar_base + stage_id * 8;

                // Wait for MMA to finish with this stage before reusing
                // Initial phase is 0 and available, so we wait for phase^1 on first cycle
                // This trick: on first iteration (iter_k < PIPE_DEPTH), mma_mbar is untouched
                // so we skip wait. After PIPE_DEPTH iterations, we must wait.
                if (iter_k >= PIPE_DEPTH) {
                    mbarrier_wait(mma_stage_addr, phase ^ 1);
                }

                // Flip phase when we cycle through all buffers
                if (stage_id == PIPE_DEPTH - 1) {
                    phase ^= 1;
                }

                // Issue TMA loads with expect_tx
                uint32_t bytes_A = TILE_M * TILE_K * sizeof(__nv_bfloat16);
                uint32_t bytes_B = TILE_K * TILE_N * sizeof(__nv_bfloat16);
                uint32_t expected_bytes = bytes_A + bytes_B;

                // Signal barrier with expected transaction bytes
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(tma_stage_addr), "r"(expected_bytes) : "memory"
                );

                // TMA coordinates
                int a_coord_0 = 0;
                int a_coord_1 = tile_row * TILE_M;
                int a_coord_2 = iter_k;
                int b_coord_0 = 0;
                int b_coord_1 = tile_col * TILE_N;
                int b_coord_2 = iter_k;

                // TMA A
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"((uint32_t)__cvta_generic_to_shared(a_smem[stage_id])),
                       "l"(&tensor_map_A),
                       "r"(a_coord_0), "r"(a_coord_1), "r"(a_coord_2),
                       "r"(tma_stage_addr)
                    : "memory"
                );

                // TMA B
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4}], [%5];"
                    :: "r"((uint32_t)__cvta_generic_to_shared(b_smem[stage_id])),
                       "l"(&tensor_map_B),
                       "r"(b_coord_0), "r"(b_coord_1), "r"(b_coord_2),
                       "r"(tma_stage_addr)
                    : "memory"
                );
            }
        }
        else if (warp_id == 1 && threadIdx.x == 32) {
            // ========== MMA WARP ==========
            // Runs independent loop issuing tcgen05.mma
            // Waits on tma_mbar[stage] before computing (consumer-side wait)
            // NEVER waits on its own mma_mbar - just issues and continues!

            for (int iter_k = 0; iter_k < num_k_tiles; iter_k++) {
                int stage_id = iter_k % PIPE_DEPTH;
                uint32_t tma_stage_addr = tma_mbar_base + stage_id * 8;
                uint32_t mma_stage_addr = mma_mbar_base + stage_id * 8;

                // Wait for TMA to complete loading this stage
                mbarrier_wait(tma_stage_addr, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");

                // Flip phase when we cycle through all buffers
                if (stage_id == PIPE_DEPTH - 1) {
                    phase ^= 1;
                }

                // Get shared memory addresses for current stage
                uint32_t a_addr = __cvta_generic_to_shared(a_smem[stage_id]);
                uint32_t b_addr = __cvta_generic_to_shared(b_smem[stage_id]);

                // Issue MMAs for this k_tile (unroll the inner k2 loop)
                // First MMA of first iter_k: disable accumulation (zero out D)
                {
                    uint64_t a_desc = make_desc(a_addr);
                    uint64_t b_desc = make_desc(b_addr);
                    if (iter_k == 0) {
                        // D = A*B (enable_d = false)
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.eq.u32 p, 0, 1;\n\t"
                            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                            "}"
                            :: "r"(tmem_base[0]), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                            : "memory"
                        );
                    } else {
                        // D = A*B + D (enable_d = true)
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.eq.u32 p, 1, 1;\n\t"
                            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                            "}"
                            :: "r"(tmem_base[0]), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                            : "memory"
                        );
                    }
                }

                // Remaining k2 iterations (always accumulate)
                #pragma unroll
                for (int k2 = 1; k2 < TILE_K / MMA_K; k2++) {
                    uint64_t a_desc = make_desc(a_addr + k2 * 32);
                    uint64_t b_desc = make_desc(b_addr + k2 * 32);
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.eq.u32 p, 1, 1;\n\t"
                        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%4, %4, %4, %4}, p;\n\t"
                        "}"
                        :: "r"(tmem_base[0]), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0)
                        : "memory"
                    );
                }

                // Commit MMA completion to this stage's mbarrier (NON-BLOCKING!)
                // This allows TMA warp to know when it's safe to reuse this stage
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(mma_stage_addr) : "memory"
                );
            }

            // Signal mainloop completion for epilogue
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory"
            );
        }

        // All warps sync here, then wait for mainloop to complete before epilogue
        __syncthreads();

        // Wait for all MMA operations to complete before reading TMEM
        if (threadIdx.x == 0) {
            mbarrier_wait(mainloop_mbar_addr, mainloop_phase);
        }
        __syncthreads();
        mainloop_phase ^= 1;  // Flip phase for next tile

        // Fence before tcgen05.ld as per PTX docs
        asm volatile("tcgen05.fence::after_thread_sync;");

        // Epilogue: Direct TMEM → global, no SMEM staging (like example kernel)
        // Each thread in consumer warpgroup handles one row, iterates over columns
        if (wg_id == 0) {
            int local_warp_id = warp_id;  // 0-3 within consumer warpgroup
            int lane_id = threadIdx.x % 32;
            int my_row = local_warp_id * 32 + lane_id;  // 0-127

            // Load 8 columns at a time, write directly to global
            for (int n = 0; n < TILE_N / 8; n++) {  // 32 iterations for TILE_N=256
                uint32_t addr = tmem_base[0] + ((local_warp_id * 32) << 16) + (n * 8);

                float tmp[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                      "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                    : "r"(addr)
                );
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                // Convert to bf16 pairs (4 x bf16x2 = 8 bf16 = 16 bytes)
                __nv_bfloat162 out[4];
                out[0] = __floats2bfloat162_rn(tmp[0], tmp[1]);
                out[1] = __floats2bfloat162_rn(tmp[2], tmp[3]);
                out[2] = __floats2bfloat162_rn(tmp[4], tmp[5]);
                out[3] = __floats2bfloat162_rn(tmp[6], tmp[7]);

                // Direct write to global (16 bytes = int4)
                int global_row = tile_row * TILE_M + my_row;
                int global_col = tile_col * TILE_N + n * 8;
                __nv_bfloat16* out_ptr = C + global_row * N + global_col;
                reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out)[0];
            }
        }
    }

    // Single sync before dealloc - all threads must finish reading TMEM
    __syncthreads();

    // Deallocate TMEM (warp 1 does dealloc, same warp that did alloc)
    if (threadIdx.x >= 32 && threadIdx.x < 64) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_base[0]), "r"(TMEM_COLS)
            : "memory"
        );
    }
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

    // Warmup run
    std::cout << "Running kernel..." << std::endl;
    my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark runs
    constexpr int ITERS = 5;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERS; i++) {
        my_matmul_kernel<<<grid, block>>>(tensor_map_A, tensor_map_B, d_C, M, N, K);
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
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

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
