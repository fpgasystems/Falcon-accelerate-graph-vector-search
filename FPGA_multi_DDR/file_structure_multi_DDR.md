# Inter-query Parallel + Multi DRAM Channel

PEs working on different queries.

## V1 : FPGA_inter_query_v1

Based on single channel V1.2 FPGA_single_DDR_single_layer_v1.2_support_multi_batch

We support breaking down all queries to batches, so we prepare for measuring performance with smaller batches, as well as preparing for the networked version.

## V1.3 : FPGA_inter_query_v1.3_longer_FIFO_alt_PR : updating FIFO size to 512 and use alternative P&R strategies

I found that FIFO depth plays a important role in placemet and routing. When setting as a small number, it will be treated as LUTs rather than FIFO. The P&R may then over-consume LUTs in a single SLR. I also evaluated alternative P&R strategies, which eventually can achieve 180 MHz for 4 channels. Specifically, Performance_WLBlockPlacement and Performance_Explore can achieve 180 MHz for 4 channels.

Note: optimizing queues will affect P&R! So I ROLLED Back the following changes!
```
# Rolled back
* Update queue insertion: only if sucessfully inserted would we trigger the two-cycle compare-swap
  * update `results_collection` in DRAM_utils.hpp; also only sort if there is any insertion
  * update `insert_only` in priority_queue.hpp; also move wait to after sort
* No filter after compute (this requires one more pipeline stage, harmful for performance given a single channel)
```

## eval_trace_FPGA_inter_query_v1.3

This one is used to track the distance over steps, to compare BFS, MCS, and DST.

Commands for testing: 

```
cd /mnt/scratch/wenqi/tmp_bitstreams/eval_trace_FPGA_inter_query_v1.3_D128

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 HNSW SIFT1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 HNSW SIFT1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 HNSW SIFT1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 HNSW SIFT1M 64 10000

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 NSG SIFT1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 NSG SIFT1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 NSG SIFT1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 NSG SIFT1M 64 10000

cp per_query_* /mnt/scratch/wenqi/graph-vector-search-on-FPGA/plots/saved_distances_over_steps/
```

```
cd /mnt/scratch/wenqi/tmp_bitstreams/eval_trace_FPGA_inter_query_v1.3_D96

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 HNSW Deep1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 HNSW Deep1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 HNSW Deep1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 HNSW Deep1M 64 10000

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 NSG Deep1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 NSG Deep1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 NSG Deep1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 NSG Deep1M 64 10000

cp per_query_* /mnt/scratch/wenqi/graph-vector-search-on-FPGA/plots/saved_distances_over_steps/
```

```
cd /mnt/scratch/wenqi/tmp_bitstreams/eval_trace_FPGA_inter_query_v1.3_D100

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 HNSW SPACEV1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 HNSW SPACEV1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 HNSW SPACEV1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 HNSW SPACEV1M 64 10000

# BFS: mc=1, mg=1
./host xclbin/vadd.hw.xclbin 1 1 64 NSG SPACEV1M 64 10000

# MCS: mc=4, mg=1
./host xclbin/vadd.hw.xclbin 4 1 64 NSG SPACEV1M 64 10000

# DST: mc=2, mg=2
./host xclbin/vadd.hw.xclbin 2 2 64 NSG SPACEV1M 64 10000

# DST: mc=1, mg=4
./host xclbin/vadd.hw.xclbin 1 4 64 NSG SPACEV1M 64 10000

cp per_query_* /mnt/scratch/wenqi/graph-vector-search-on-FPGA/plots/saved_distances_over_steps/
```

## Unused V1.1 FPGA_inter_query_v1.1_multi_kernel : multi-kernel

Based on FPGA_inter_query_v1, detached one kernel to the scheduler and multiple per-channel kernels.

However:
(1) I tried several P&R directives and frequencies, even the 1-kernel version does not work...
(2) I have not implemented the host for multi-kernel yet.

## Unused V1.2 FPGA_inter_query_v1.2_reduce_mem_interfaces : reduce input / output interfaces

Based on FPGA_inter_query_v1, merging the inputs (queries, entries) into one interface; and outputs (dist, ids, debug) into one interface, in order to achieving better P&R.

However:
(1) This does not help P&R at all actually
(2) The correctness is not tuned -> I have not set the offset between multiple inputs/outputs, just wanted to test whether P&R is better


# Intra-query Parallel + Multi DRAM Channel

Multiple PEs working on the same query.


## V1 : FPGA_intra_query_v1_async_bloom

Based on single-channel V1 : FPGA_single_DDR_single_layer_v1_async

## V1.1 : FPGA_intra_query_v1.1_async_opt_mem : update memory access + reduce bloom filter number to 3

Based on single-channel V1.1 : FPGA_single_DDR_single_layer_v1.1_async_opt_mem

Instead of manual burst with variable length, here we use fixed length (D), and also reduce bloom filter number to 3. This is because the burst memory access of this version is very fast (close to 40 ns per D=128 vector given 200 MHz, i.e., 8 cycles), we would like reduce blooom II to (3 * 2 = 6 cycles), otherwise it can become the bottleneck.

## V1.2 : FPGA_intra_query_v1.2_opt_queue : optimize queue insertion

Compared to V1.1, this version: 
* Reduces the number of insertions by filtering out the computed distance that is larger than the bound.
  * change in utils.hpp by adding function `filter_computed_distances`
  * change in vadd.cpp by adding the function, changing the FIFO connections, etc.
* Update queue insertion: only if sucessfully inserted would we trigger the two-cycle compare-swap
  * update `results_collection` in DRAM_utils.hpp; also only sort if there is any insertion
  * update `insert_only` in priority_queue.hpp; also move wait to after sort


## Unused: V1.3 FPGA_intra_query_v1.3_alt_shutdown_protocol: alternative protocol for sending tasks

Based on V1.2, I updated the task scheduler --- if in one iteration, the received valid results is zero, then do not send out any task. 

Wenqi comment: this actually may not make sense if we want to send out stuffs as early as possible --- does not have valid results several stages ago does not mean right now we should not send out tasks. 

However: 
* It failed P&R using several directives.
* In simulation it seems to be even slower...

## V1.4 FPGA_intra_query_v1.4_fast_task_split

Based on V1.2, I split the neighbor ID readers to 2 functions, detaching memory accessing (512-bit) with parsing. In 4-channel version, this improves the performance by 10~15%; in 2 channel, it improves only around 5%.