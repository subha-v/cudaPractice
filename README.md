

# CUDA practice


## B200 Matmul Kernel

Make sure to run it with this command, otherwise it will be so slow.
`nvcc -O3 -gencode arch=compute_100a,code=sm_100a b200_matmul.cu -o b200_matmul -lcuda -lcudart`

Currently achieves 842.371 TFLOPs on the B200, could be further optimized

## Matmul Naive
The slow part is the loads for A and B (e.g. A[row * N + k]) because that accesses the HBM to load that specific element in A
- This loop runs N times
- We do 2 loads per 2 math ops
- Spends 400 cycles waiting for A's LOAD and then 1 cycle for doing the math and then 400 cycles waiting for B's LOAD

Actually, matrix B is slower than matrix A!
- Matrix A needs everything in the same row, which means we simply just need A[0], A[1],... A[32]. The memory controller can give us bytes 0 to 128 so all threads can get their data instantly
- However, matrix B needs to access by column which means we must access B[0], B[1024], B[2048] etc which can be very far apart in physical memory
- Our memory controller has to issue 32 separate memory requests

## Matmul Tiled
Often, we can't even fit a full row or full column of our large matrix onto the shared memory. That's why we need to **tile** our data
- Even if we could, e.g. a width of 2048 is 8 KB because we have 2048 * 4 (assuming fp4) bytes, we want to load 32 rows and 32 columns onto shared memory so that each thread can compute its own output, but that's way too big for our shared memory
- Shared memory only has a 48 KB limit