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
    cuda::barrier<cuda::thread_scope_block>* barrier,      // which barrier to signal
    const CUtensorMap* tensor_map_A,                     
    const CUtensorMap* tensor_map_B                  
) {
    // Calculate expected bytes for barrier
    uint32_t bytes_A = TILE_M * TILE_K * sizeof(__nv_bfloat16);
    uint32_t bytes_B = TILE_K * TILE_N * sizeof(__nv_bfloat16);
    uint32_t expected_bytes = bytes_A + bytes_B;

    // Signal barrier with expected transaction bytes using PTX

    uint64_t barrier_ptr = __cvta_generic_to_shared(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :
        : "l"(barrier_ptr), "r"(expected_bytes)
        : "memory"
    );

    //launch tma for a
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

    // launch tma for b
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(b_smem[slot])),
          "l"(tensor_map_B),
          "r"(k_tile),       // coordinate 0: K dimension (inner)
          "r"(tile_col),     // coordinate 1: N dimension (outer)
          "r"((uint32_t)__cvta_generic_to_shared(barrier))
        : "memory"
    );
}

__global__ void my_matmul_kernel(
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
    __shared__ __nv_bfloat16 a_smem[PIPE_DEPTH][TILE_M * TILE_K];  // 4 x 128 x 64
    __shared__ __nv_bfloat16 b_smem[PIPE_DEPTH][TILE_K * TILE_N];  // 4 x 64 x 256

    __shared__ uint32_t tmem_base;

    // barriers
    __shared__ alignas(8) uint64_t inputs_arrived_storage[PIPE_DEPTH];
    __shared__ alignas(8) uint64_t inputs_finished_storage[PIPE_DEPTH];

    // cast to barrier pointers
    cuda::barrier<cuda::thread_scope_block>* inputs_arrived =
        reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(inputs_arrived_storage);
    cuda::barrier<cuda::thread_scope_block>* inputs_finished =
        reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(inputs_finished_storage);

    // initialization - barrier init only needs thread 0
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) printf("[DEBUG] Block 0: Starting barrier init\n");

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

    // TMEM allocation - tcgen05.alloc with cta_group::1 is a COLLECTIVE operation
    // for exactly 1 warpgroup (128 threads). Only threads 0-127 should execute it.
    // somehow this is printing but then it hangs after this :((()))
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: Warpgroup 0 allocating TMEM\n");
    __syncthreads();  // IMPORTANT: Reconverge after printf ???

    uint32_t num_cols = TMEM_COLS;
    uint32_t tmem_base_smem_addr = __cvta_generic_to_shared(&tmem_base);

    // Only first warpgroup (threads 0-127) executes tcgen05.alloc - i dont know if this is correct....
    if (threadIdx.x < 128) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :
            : "r"(tmem_base_smem_addr), "r"(num_cols)
            : "memory"
        );
    }
    __syncthreads();  // Ensure all threads see the allocated tmem_base

    // This is currently not printing
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: TMEM allocated, tmem_base=%u\n", tmem_base);

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
        // TMEM zeroing is handled by using scale_d=0 on first WGMMA iteration

        // 1 thread of the producer loads the first tiles
        if (wg_id == 1 && threadIdx.x == 128) {  // First thread of producer warpgroup
            if (blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Producer starting prefetch\n");
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
            if (blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Producer prefetch done\n");
        }
        if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Entering k_tile loop, num_k_tiles=%d\n", num_k_tiles);

        // we split up the tiles of  A and B into further tiles
        // note that each tile of A is 128 x 64 and each tile of B is 64 x 256, where K = 64 (we just set that lol)
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) { // i think we have like 32 k tiles for 1024 x 1024
            int compute_slot = k_tile % PIPE_DEPTH; // which slot is the tile were computing on is in
            int load_k_tile = k_tile + PIPE_DEPTH; // which one to load for the producer
            int load_slot = load_k_tile % PIPE_DEPTH; // which slot to load it into

            if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0 && k_tile < 3) printf("[DEBUG] Block 0: k_tile=%d, compute_slot=%d\n", k_tile, compute_slot);

            // consumer thread executes wgmma
            if (wg_id == 0) {
                // Consumer waits for the producer (only thread 0 does the barrier wait)
                if (threadIdx.x == 0) {
                    if (blockIdx.x == 0 && k_tile == 0) printf("[DEBUG] Block 0: Consumer waiting on barrier (k_tile=%d)\n", k_tile);
                    inputs_arrived[compute_slot].arrive_and_wait();
                    if (blockIdx.x == 0 && k_tile == 0) printf("[DEBUG] Block 0: Consumer past barrier (k_tile=%d)\n", k_tile);
                }
                // at this point, only thread 0 does the barrier wait
                // threads 1-127 in the consumer warpgroup dont know yet
                // Warpgroup fence ensures all consumer threads see the data

                // Fence before tcgen05 operations to ensure memory ordering
                asm volatile("tcgen05.fence::before_thread_sync;");

                if (threadIdx.x == 0 && blockIdx.x == 0 && k_tile == 0) printf("[DEBUG] Block 0: Past tcgen05.fence (k_tile=%d)\n", k_tile); 
                
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
                // idesc = 0x8400490

                // tcgen05.mma for 128x256 output tile with K=64
                // Each tcgen05.mma handles 128 rows, 256 cols, 16 k-elements
                // 4 k-steps needed (TILE_K=64, step=16)

                if (threadIdx.x == 0 && blockIdx.x == 0 && k_tile == 0) printf("[DEBUG] Block 0: Starting tcgen05.mma loop\n");

                 // paragma unroll could be used.. 
                for (int k_step = 0; k_step < TILE_K; k_step += 16) { // K = 64
                    // A offset: k_step columns into the 128x64 A tile
                    // B offset: k_step rows into the 64x256 B tile
                    uint32_t a_offset = k_step * sizeof(__nv_bfloat16);
                    uint32_t b_offset = k_step * TILE_N * sizeof(__nv_bfloat16);

                    // Build matrix descriptors (64-bit)
                    // Format: bits 0-13 = addr>>4, bits 16-29 = ldim>>4, bits 32-45 = stride>>4
                    uint32_t a_smem_addr = a_addr + a_offset;
                    uint32_t b_smem_addr = b_addr + b_offset;

                    // a desc
                    uint64_t a_desc = ((uint64_t)((a_smem_addr >> 4) & 0x3FFF)) |
                                      ((uint64_t)(((TILE_K * 2) >> 4) & 0x3FFF) << 16);

                    // b desc
                    uint64_t b_desc = ((uint64_t)((b_smem_addr >> 4) & 0x3FFF)) |
                                      ((uint64_t)(((TILE_K * 2) >> 4) & 0x3FFF) << 16);

                    if (threadIdx.x == 0 && blockIdx.x == 0 && k_tile == 0 && k_step == 0) {
                        printf("[DEBUG] Block 0: tcgen05.mma params: tmem_base=%u, a_desc=0x%llx, b_desc=0x%llx, idesc=0x%x\n",
                               tmem_base, (unsigned long long)a_desc, (unsigned long long)b_desc, idesc);
                    }

                    // first iteration zeros accumulator lol the rest are mm
                    if (k_tile == 0 && k_step == 0) {
                        if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: Calling tcgen05.mma (first, no accum)\n");
                        // D = A*B (enable_d = 0, no accumulation)
                        asm volatile(
                            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;"
                            :
                            : "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(idesc)
                            : "memory"
                        );
                        if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: tcgen05.mma returned (first)\n");
                    } else {
                        // D = A*B + D (enable_d = 1, accumulate)
                        asm volatile(
                            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;"
                            :
                            : "r"(tmem_base), "l"(a_desc), "l"(b_desc), "r"(idesc)
                            : "memory"
                        );
                    }
                }

                if (threadIdx.x == 0 && blockIdx.x == 0 && k_tile == 0) printf("[DEBUG] Block 0: Finished tcgen05.mma loop for k_tile=0\n");

                // Wait for all MMA operations to complete
                // For now, using fence as synchronization point
                asm volatile("tcgen05.fence::before_thread_sync;");

                // consumer tells producer that it's done with this slot
                if (threadIdx.x == 0) {
                    (void)inputs_finished[compute_slot].arrive();
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
        // Consumer warpgroup reads from TMEM and writes directly to global memory
        if (wg_id == 0) {
            if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Starting store phase\n");

            // Fence to ensure MMA completion before reading
            asm volatile("tcgen05.fence::before_thread_sync;");

            // tcgen05.ld is warp-collective: all 32 threads in a warp must use same taddr
            // using shape .16x256b.x1: loads 512 bytes (1 TMEM column = 128 floats)
            // each thread gets 4 registers (4 floats), 32 threads × 4 = 128 floats
            // consumer warpgroup has 4 warps, each handles 64 columns (256/4)

            int warp_in_consumer = threadIdx.x / 32;  // 0-3 within consumer warpgroup
            int lane_id = threadIdx.x % 32;           // 0-31 within warp
            int cols_per_warp = TMEM_COLS / 4;        // 64 columns per warp

            if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Starting tcgen05.ld loop\n");

            for (int col_idx = 0; col_idx < cols_per_warp; col_idx++) {
                int col = warp_in_consumer * cols_per_warp + col_idx;

                // all threads in warp use the SAME taddr (base of this column)
                uint32_t taddr = tmem_base + col * 512;  // 512 bytes per column

                if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0 && col_idx == 0) {
                    printf("[DEBUG] Block 0: tcgen05.ld taddr=%u (col=%d)\n", taddr, col);
                }

                // Collective load: each thread receives 4 floats from the column
                float r0, r1, r2, r3;
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
                    : "r"(taddr)
                );

                if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0 && col_idx == 0) {
                    printf("[DEBUG] Block 0: tcgen05.ld returned (col=%d)\n", col);
                }

                // Fence after async load before using the data
                asm volatile("tcgen05.fence::after_thread_sync;");

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

            if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Store phase complete\n");
        }

        // Sync before next output tile
        __syncthreads();

        if (threadIdx.x == 0 && blockIdx.x == 0 && tile_id == 0) printf("[DEBUG] Block 0: Finished tile_id=0\n");
    }

    // deallocate tmem - tcgen05.dealloc with cta_group::1 needs only warpgroup 0
    __syncthreads();  // Ensure all threads are done before dealloc
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: Warpgroup 0 deallocating TMEM\n");
    __syncthreads();  // Reconverge after printf

    uint32_t dealloc_num_cols = TMEM_COLS;
    // idk if this is correct. only the first 128 threads do this? 
    if (threadIdx.x < 128) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_base), "r"(dealloc_num_cols)
            : "memory"
        );
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) printf("[DEBUG] Block 0: TMEM deallocated, kernel done!\n");
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

// benchmark
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

    // cpu
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

    // A is row-major (M x K)
    for (size_t i = 0; i < M * K; i++) h_A_bf16[i] = __float2bfloat16(h_A[i]);

    // B needs to be column-major for TMA (K contiguous, not N)
  
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            h_B_bf16[n * K + k] = __float2bfloat16(h_B[k * N + n]);
        }
    }

    cudaMemcpy(d_A, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device (B in column-major)" << std::endl;

    // Create TMA descriptors
    // A: M x K row-major, inner dim = K, boxDim[0] = TILE_K = 64 
    CUtensorMap tensor_map_A = create_tensor_map(d_A, M, K, TILE_M, TILE_K);
    // B: now N x K column-major (K contiguous), inner dim = K, boxDim[0] = TILE_K = 64 
    CUtensorMap tensor_map_B = create_tensor_map(d_B, N, K, TILE_N, TILE_K);

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
