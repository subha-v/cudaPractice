# CUDA practice

## Matmul Naive
The slow part is the loads for A and B (e.g. A[row * N + k]) because that accesses the HBM to load that specific element in A
- This loop runs N times
- We do 2 loads per 2 math ops
- Spends 400 cycles waiting for A's LOAD and then 1 cycle for doing the math and then 400 cycles waiting for B's LOAD

Actually, matrix B is slower than matrix A!
- Matrix A needs everything in the same row, which means we simply just need A[0], A[1],... A[32]. The memory controller can give us bytes 0 to 128 so all threads can get their data instantly
- However, matrix B needs to access by column which means we must access B[0], B[1024], B[2048] etc which can be very far apart in physical memory
- Our memory controller has to issue 32 separate memory requests