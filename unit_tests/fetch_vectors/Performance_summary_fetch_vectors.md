# Performance fetch vectors

Read write performance can be lookedup here in the spatial join project: https://github.com/WenqiJiang/spatial-join-on-FPGA/tree/main/test_multi_kernel#compare-random-read-performance 

## read vector + visited tag + write tag

### V1

1 read + 1 write per query; no prefetching / batched pipeline fetching, because the next read operation may be influenced by the current write operation, especially in the scenario where we use multiple nodes and they have overlap on neighbors

```
	const int AXI_num_per_vector = d % FLOAT_PER_AXI == 0? 
		d / FLOAT_PER_AXI + 1 : d / FLOAT_PER_AXI + 2; // 16 for d = 512 + visited padding

	for (int i = 0; i < AXI_num_per_vector; i++) {
	#pragma HLS pipeline II=1
		ap_uint<512> vector_AXI = db_vectors[start_addr + i];
		if (i < AXI_num_per_vector - 1) {
			s_fetched_vectors.write(vector_AXI);
		} else {
			ap_uint<32> last_visit_qid = vector_AXI.range(31, 0);
			valid = level_id > 0? true : last_visit_qid != qid;
			s_fetch_valid.write(valid);
		}
	}
```

Performance:

```
    int query_num = 1000;
	int read_iter_per_query = 1000;
	int d = 128;
```

780.183 ms @200 MHz

1 group (read + write) = 780.183 ms / 1e3 / 1e3 = 780 ns. Very high -> almost 10x to the best pipelined read (throughput translated to 83 ns / len=8 vector).

Conclusion: the way to go is to use bloom filter -> which can lead to up to 10x performance improvement, and also ~2x number of visits given the duplicates. 


## Only read vectors

### V1: no prefetching, no batching

1 read per query; no prefetching / batched pipeline fetching

```
    int query_num = 1000;
	int read_iter_per_query = 1000;
	int d = 128;
```

433.682 ms @200 MHz -> 433 ns per read


### V1.1: no prefetching, w/ batching

Compared to V1, add a batch size indicating the number of reads, but still no prefetching. 

Literally no improvement over V1.


### V1.2: prefetching + batching

set max_burst_length = 16. I have tried max_burst_length of 32 and 64, and the performance is exactly the same.

Prefetching + batching.

Large read batches (1000): 

```
    int query_num = 1000;
	int read_iter_per_query = 1000;
	int d = 128;
```

125.515 ms @200 MHz -> 125 ns per read (length of 8) == 25 cycles (given lenght = 8, the pure latency is 25 - 8 + 1 = 18 cycles)

Medium read batches (100): 

13.0161 ms @200 MHz -> 130 ns per read (length of 8) 

Small read batches (10):

1.8214 @200 MHz -> 182 ns per read (length of 8) -> 1.46x time compared to maximal batch sizes. 


### V1.3 Using implicit prefetching - by referring to Vitis Tutorial

Refer to Vitis tutorial: 

https://github.com/Xilinx/Vitis-Tutorials/blob/2022.1/Hardware_Acceleration/Feature_Tutorials/07-using-hbm/3_BW_Explorations.md

Two keys in inferring high-performance fetching:

1. Using fixed fetch length
2. Pipeline outside the read loop

For exmaple:

```
	const int AXI_num_per_vector_and_padding = D % FLOAT_PER_AXI == 0? 
		D / FLOAT_PER_AXI + 1 : D / FLOAT_PER_AXI + 2; // 16 for D = 512 + visited padding

	const int AXI_num_per_vector_only = D % FLOAT_PER_AXI == 0? 
		D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; // 16 for D = 512
		
	for (int bid = 0; bid < fetch_batch_size; bid++) {
	#pragma HLS pipeline // put the pipeline here so hopefully Vitis can handle prefetching automatically
		// receive task & read vectors
		cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
		int start_addr = reg_cand.node_id * AXI_num_per_vector_and_padding;
		for (int i = 0; i < AXI_num_per_vector_only; i++) {
			ap_uint<512> vector_AXI = db_vectors[start_addr + i];
			s_fetched_vectors.write(vector_AXI);
		}
	}
```


#### latency=64, num_read_outstanding=64

Large read batches (1000): 

```
    int query_num = 1000;
	int read_iter_per_query = 1000;
	int read_batch_size = 1000;
	int d = 128;
```

42.5765 ms @200 MHz -> 42.6 ns per read (length of 8) == 8.5 cycles (given lenght = 8, this almost achieved the theoretical bandwidth)

Achieved bandwidth: 12.151 GB/s

Medium read batches (100): 

46.3133 ms @200 MHz -> 46.3 ns per read (length of 8) -> ~10 cycles

Small read batches (10), latency=64:

82.5177 ms @200 MHz -> 82 ns per read -> 16 cycles/vec

theoretically, 64 + 10 * 8 = 144 cycles to fetch 10 vectors; it could be that the entire flow overhead (e.g., finish signal), it too noisy to evaluate small read numbers accurately.


#### latency=32, num_read_outstanding=32/64 (same perf)

Large read batches (1000): 

57.5261 ms @200 MHz 

Medium read batches (100): 

59.8016 ms @200 MHz 

Small read batches (10):

82.4586 ms @200 MHz -> similar to latency=64, num_read_outstanding=64


#### latency=16, num_read_outstanding=16

Large read batches (1000): 

57.5274 ms @200 MHz -> slower than latency = 64

Medium read batches (100): 

62.4019 ms @200 MHz -> slower than latency = 64

Small read batches (10)

115.557 ms @200 MHz -> slightly better than latency = 64

#### latency=1, num_read_outstanding=64/32 (same perf)

Large read batches (1000): 

278.852 ms @200 MHz -> slower than latency = 64

Medium read batches (100): 

281.479 ms @200 MHz -> slower than latency = 64

Small read batches (10)

319.092 ms @200 MHz -> slower than latency = 64

### V1.4 Using implicit prefetching + allow continuous batching

Compared to v1.3, we allow the next batch size to be added before current batch finished - so we keep pipeline busy rather than restart the pipeline. 

## Unused

### failed_test_simple_fetch

I wanted to test bandwidth using a simple function, but it seems that the compiler automatically optimized away most of the code for data reading. 

Conclusion: use latency=64, num_read_outstanding=64, batch size = 16

#### (Use this) latency=64, num_read_outstanding=64

Large read batches (1000): 

42.6013 ms @200 MHz 

Medium read batches (100): 

42.6009 ms @200 MHz 

Medium small read batches (32): 

42.6013ms @200 MHz 

Medium small read batches (16): 

42.6021 ms @200 MHz 

Small read batches (10):

52.6506 ms @200 MHz 

#### latency=32, num_read_outstanding=32

Large read batches (1000): 

57.5281 ms @200 MHz 

Small read batches (10):

57.5271 ms @200 MHz 