# Compute PEs

We use various compute PEs and evaluate their performance.

## Floating point performance Microbenchmark

Microbenchmark the latency of floating point operations. An official doc can be found in: https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/pragma-HLS-bind_op (Table: Supported Combinations of Floating Point Operations, Implementation, and Latency), but it only specifies maximum latency (which fluctuates according to frequency), e.g., fadd is up to 12 cycles (fulldsp) and fmul is up to 7 cycles (maxdsp).

```
	// 140MHz: Pipelining result : Target II = 1, Final II = 1, Depth = 5, loop 'ADD_LOOP'
	// 200MHz: Pipelining result : Target II = 1, Final II = 1, Depth = 8, loop 'ADD_LOOP'
	ADD_LOOP:	
	for (int i = 0; i < query_num; i++) {
#pragma HLS pipeline II=1
// #pragma HLS BIND_OP variable=local_array_C op=fmul latency=1
		local_array_C[i] = local_array_A[i] + local_array_B[i];	
	}

	// 140MHz: Pipelining result : Target II = 1, Final II = 1, Depth = 5, loop 'MULT_LOOP'
	// 200MHz: Pipelining result : Target II = 1, Final II = 1, Depth = 5, loop 'MULT_LOOP'
	MULT_LOOP:
	for (int i = 0; i < query_num; i++) {
#pragma HLS pipeline II=1
// #pragma HLS BIND_OP variable=local_array_C op=fmul latency=1
		local_array_C[i] = local_array_A[i] * local_array_B[i];
	}
```


## V1: compute everything in a single loop

49.1165 ms, 140 MHz

Cycles = 49.1165 / 1000 * 140 * 1e6 = 6,876,310

```
int query_num = 1000;
int num_fetched_vectors_per_query = 64;
int d = 128;
```

Real per vector cycles = 6,876,310 / 1000 / 64 = 107.4

Syn results:  Pipelining result : Target II = 1, Final II = 3, Depth = 83, loop 'VITIS_LOOP_68_7' -> given d=128 (iter=8), latency = 83 + (8 - 1) * 3 = 104 cycles.

In the optimal case, consume 1000 * 64 * (128/16) = 512,000 cycles

There is a 10x difference to the theoretical performance!

## V2: use 2 sub-PEs + 1 helper

PE A: compute the L2 distance for every loaded ap_uint<512> vectors.

PE helper: pack the distances from PE A (float) into ap_uint<512> vectors.

PE B: reduction sum the distances from PE helper. 

### V2.1

Optimize V2 by writing loops in perfect loop format (https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Working-with-Nested-Loops). 


The pipeline depth and latency of the three PEs:

```
INFO: [v++ 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 23, loop 'VITIS_LOOP_84_6'
INFO: [v++ 204-61] Pipelining loop 'VITIS_LOOP_137_4'.
INFO: [v++ 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 4, loop 'VITIS_LOOP_137_4'
INFO: [v++ 204-61] Pipelining loop 'VITIS_LOOP_191_3'.
INFO: [v++ 200-1470] Pipelining result : Target II = 1, Final II = 3, Depth = 24, loop 'VITIS_LOOP_191_3'
```

Syn total latency = 23 + 4 + 24, given 128 dim vectors (8 AXI width), final latency should be 23 + 4 + 24 + (8 - 1) = 58 cycles. But since different PEs can work on different vectors. The actual latency if more than one vector (but can be only one query), is max(23 + 8 - 1, 4 + 8 - 1, 24) = 30 cycles

15.0312 ms, 140 MHz

Cycles = 15.0312 / 1000 * 140 * 1e6 = 2,104,368

```
int query_num = 1000;
int num_fetched_vectors_per_query = 64;
int d = 128;
```

Real per vector cycles = 2,104,368 / 1000 / 64 = 32.88 -> close to estimated, still far from ideal.


### V2.2 

Optimize V2.1 by allowing batched compute, knowing how many number of requests to process. Here, we set batch size as 64


```
int query_num = 10000;
int num_fetched_vectors_per_query = N;
int d = 128;
```

Different batch size, their runtime in ms running at 140MHz -> their cycles per AXI (num_ms / 1000 * 140 * 1e6 / (batch_size * 1000 * (128/16))).

```
batch_size = 1: 6.28856 ms -> 11.00 cycles

batch_size = 2: 6.85911 ms -> 6.00 cycles

batch_size = 4: 8.71696 ms -> 3.81 cycles

batch_size = 8: 12.1445 ms -> 2.65 cycles

batch_size = 16: 19.0025 ms -> 2.07 cycles

batch_size = 32: 32.7169 ms -> 1.78 cycles -> A good point close to optimal

batch_size = 64: 60.1444 ms -> 1.64 cycles

batch_size = 128: 115.002 ms -> 1.57 cycles

batch_size = 1024: 883.002 ms -> 1.50 cycles
```

Not sure why it cannot achieve closer performance to 1 cycle / AXI, but is acceptable given the memory access itself can be even a bigger issue than compute.

### (Use this) V3.1 achieve line rate / support bloom filter + use fixed D

**This version can achieve close-to-optimal performance even for batch size = 1**

Compared to v2.2, here we support bloom filter before compute so all data are valid. Also, we allow the next batch size to be added before current batch finished - so we keep pipeline busy rather than restart the pipeline. 

Besides, we improve the performance of compute to saturate memory bandwidth. The trick is to use fixed D instead of variable, then put pipeline in the outer loop (which automatically unrolls in the inner loops that involved D).

Achieved performance: 

* D = 128

batch size = 1:

50.7916 ms @ 200MHz for 1M inputs -> 10 cycles per vec (size = 8 AXI)

batch size = 32

40.7111 ms @ 200MHz for 1M inputs -> 8 cycles per vec (size = 8 AXI) -> achieved theoretical throughput


* D = 256

batch size = 1:

90.7117 ms @ 200MHz for 1M inputs -> 18.14 cycles per vec (size = 16 AXI)

batch size = 32

80.7913 ms @ 200MHz for 1M inputs -> 16.16 cycles per vec (size = 16 AXI) -> achieved theoretical throughput


* D = 512

batch size = 1:

170.732 ms @ 200MHz for 1M inputs -> 34.14 cycles per vec (size = 32 AXI)

batch size = 32

160.967 ms @ 200MHz for 1M inputs -> 32.2 cycles per vec (size = 32 AXI) -> achieved theoretical throughput

* D = 1024

batch size = 1:

331.093 ms @ 200MHz for 1M inputs -> 66.21 cycles per vec (size = 64 AXI)

batch size = 32

321.327 ms @ 200MHz for 1M inputs -> 64.26 cycles per vec (size = 64 AXI) -> achieved theoretical throughput