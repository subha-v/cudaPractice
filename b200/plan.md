
1. Tile matrix C into $128 \times 256$ tiles (as many as necessary)
	1. ~~We can get the index of each tile with `blockIdx.x` and `blockIdx.y`~~
	2. We want to use persistent threads/blocks(?) 
		1. Instead of launching thousands of blocks we launch a small, fixed number of blocks which are exactly enough to fill the B200's hardware (148 blocks, one for each SM)
2. In each SM, we do the following:
	1. Reserve memory in the L1 cache for 
		1. $128 \times 64$ piece of matrix $A$ 
		2. $256 \times 64$ piece of matrix $B$ 
	2. Note that we make 2-4 sets of these spots (depending on what the `PIPE_DEPTH` is) which lets us have the producer and consumer be at different places
3.  ~~At every clock cycle (?)~~ Whenever there is an empty slot in the Ring Buffer
	1. Producer Warp (32 threads) takes in the `(blockIdx.x, blockIdx.y)` tuple and then asks the TMA hardware to go to global memory and load it into the L1/Shared memory on each SM
	2. Consumer Warp (32 threads)
		1.  Whenever the producer sends the semaphore signal that there's something available to compute, then
		2. It pulls the $128 \times 64$ slice from `a_smem` and the $256 \times 64$ slice from `b_smem`
		3. **The "Magic" part:** On the B200, the **B matrix does not even go into the threads' registers.** The WGMMA hardware instruction reads $B$ directly from Shared Memory on its way into the Tensor Core. This saves a massive amount of register space, which is why this kernel can be so fast.
		4. Instead of registers, the B200 has something called **TMEM** which just stores the matrix results
4. Move from TMEM to SM
	1. warpgroup::store(d_smem, d_reg[0]);

