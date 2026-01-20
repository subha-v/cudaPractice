# TCGEN05.ALLOC Issue on B200 (Blackwell) - RESOLVED

## Summary

We were trying to use Blackwell's 5th generation tensor core instructions (`tcgen05`) for a matrix multiplication kernel. The `tcgen05.alloc` instruction appeared to be failing, but **it was actually working correctly**.

## Resolution

**The issue was a misunderstanding**: `tmem_base = 0` is a **valid** tensor memory address. Tensor memory starts at address 0 in its own address space. The allocation was working the whole time!

This was confirmed when reducing TMEM_COLS to 32 produced the error:
```
Tensor Memory column 32 being accessed by instruction tcgen05.mma is not allocated.
Columns allocated are: 0-31.
```

This proves that `tcgen05.alloc` successfully allocated columns 0-31.

## Environment

- **GPU**: NVIDIA B200 (Blackwell architecture)
- **Compute Capability**: 10.0 (sm_100)
- **CUDA Toolkit**: 13.0 (release 13.0, V13.0.88)
- **Compilation Command**:
  ```bash
  nvcc -gencode arch=compute_100a,code=sm_100a -O3 -Xcompiler -fopenmp b200_matmul.cu -o b200_matmul -lcuda -lgomp
  ```

## The Problem

The `tcgen05.alloc` instruction is supposed to allocate Tensor Memory and write the allocated base address to a shared memory location. Instead, it either:
1. Returns 0 (allocation appears to fail silently), OR
2. Causes an "illegal instruction" error at runtime

## What tcgen05.alloc Should Do

According to the PTX documentation:
```
tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32  [dst], nCols;
```

- Allocates `nCols` columns of Tensor Memory (each column is 512 bytes)
- Writes the allocated TMEM base address to shared memory at `[dst]`
- Must be executed by exactly one warp (warp-collective operation)
- Returns 0 or 0xFFFFFFFF on failure

## Current Code (Relevant Section)

```cpp
__shared__ int tmem_base[1];  // Where alloc writes the TMEM address

// Warp 1 performs the allocation
if (threadIdx.x >= 32 && threadIdx.x < 64) {  // warp 1
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_base));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(addr), "r"(num_cols)
        : "memory"
    );
}
```

## Debug Output (Issue #1: Returns 0)

When the code runs, we see:
```
DEBUG: smem_addr for tmem_base = 197632 (0x30400)
DEBUG: num_cols = 256
DEBUG: After manual write, tmem_base = 0xdeadbeef    <-- Manual write works!
DEBUG: Warp 1 executing tcgen05.alloc with addr=0x30400, num_cols=8
DEBUG: tmem_base = 0 (0x0)                           <-- But alloc writes nothing!
WARNING: tmem_base is 0 - allocation may have failed!
```

Key observation: Manual write to shared memory works perfectly (0xDEADBEEF), but `tcgen05.alloc` leaves the value as 0.

## Debug Output (Issue #2: Illegal Instruction)

With compute-sanitizer:
```
DEBUG: Warp 1 executing tcgen05.alloc with addr=0x30400, num_cols=8
========= Illegal instruction
=========     at my_matmul_kernel(...)+0x860
=========     by thread (32,0,0) in block (0,0,0)
CUDA error: an illegal instruction was encountered
```

This suggests the `tcgen05.alloc` PTX instruction itself is not recognized at runtime.

## What We've Tried

### 1. Different Warp for Allocation
- **Original**: Warp 0 (threads 0-31)
- **Changed to**: Warp 1 (threads 32-63)
- **Result**: No change

### 2. Matching Example Code Syntax
Changed from:
```cpp
uint32_t smem_addr = __cvta_generic_to_shared(&tmem_base);
```
To (matching online example):
```cpp
const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_base));
```
And changed `tmem_base` from single variable to array:
```cpp
__shared__ int tmem_base[1];
```
- **Result**: No change

### 3. Removed `.shared::cta` Qualifier
The qualifier is optional per docs. Tried:
```
tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;
```
- **Result**: No change

### 4. Reduced num_cols (Resource Exhaustion Test)
Changed from 256 columns to 8 columns to rule out TMEM size limits.
- **Result**: Still fails (now with "illegal instruction" error)

### 5. Added `__cluster_dims__(1, 1, 1)`
Added cluster launch attribute to kernel:
```cpp
__global__ __cluster_dims__(1, 1, 1) void my_matmul_kernel(...)
```
- **Result**: Not yet tested

### 6. Different Compilation Flags
Tried `-arch=sm_100a`:
```bash
nvcc -arch=sm_100a b200_matmul.cu -o b200_matmul
```
- **Result**: Compilation fails with:
  ```
  ptxas error: Instruction 'tcgen05.alloc' not supported on .target 'sm_100'
  ```
  Note: The error says `sm_100` even though we specified `sm_100a`. The PTX file is named `compute_100.ptx` instead of `compute_100a.ptx`.

## Observations & Theories

1. **Compilation Works, Runtime Fails**: With `-gencode arch=compute_100a,code=sm_100a`, the code compiles successfully. But at runtime, the GPU encounters an "illegal instruction". This suggests a mismatch between what the compiler thinks is supported and what the hardware actually supports.

2. **PTX Target Mismatch**: When using `-arch=sm_100a`, the ptxas error shows target `sm_100` (without 'a'), suggesting the architecture suffix might not be propagating correctly.

3. **Possible Missing Prerequisites**: The `tcgen05` instructions might require:
   - Cluster launch mode (`__cluster_dims__`)
   - Specific kernel launch configuration
   - Some initialization before allocation
   - A specific CUDA driver version

4. **Hardware/Driver Compatibility**: B200 reports compute capability 10.0. There might be a distinction between sm_100 (base) and sm_100a (with tensor memory features) that requires driver updates.

## Next Steps to Try

1. Test with `__cluster_dims__(1, 1, 1)` added (just done)
2. Check CUDA driver version: `nvidia-smi`
3. Look for NVIDIA official samples for Blackwell tensor memory
4. Try different cluster sizes
5. Check if there's required mbarrier initialization before tcgen05.alloc
6. Contact NVIDIA developer support

## Reference: Working Example Code (from online)

```cpp
#pragma nv_diag_suppress static_var_with_dynamic_init
__shared__ uint64_t mbars[1];
__shared__ int tmem_addr[1];

if (tid == 0) {
    // lane0 of warp0 initializes mbarriers
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;");
} else if (warp_id == 1) {
    // warp 1 allocates tmem
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(addr), "r"(BLOCK_N));
}
```

Note: This example has mbarrier initialization before tcgen05.alloc. This might be a required prerequisite.
