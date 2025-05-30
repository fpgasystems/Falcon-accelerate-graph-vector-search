# Bloom-Fetch-Compute

## V1: multi-layer 

**Important:** the Bloom-Fetch-Compute unit will return the total number of valid elements as the output, thus cannot support input size of more than 512 elements (which should not appear in the search if one node has up to 64 neighbors), otherwise it will run into deadlocks. 

`bloom_fetch_compute_v1_multi_layer`

```
    int query_num = 100;
    int read_iter_per_query = 10000;        
	int run_upper_levels = 1;
    int run_base_level = 1;
```

Achieved: 253.231 ms @ 200 MHz -> close to 2 x 125 ns / group

## V1.1: single-layer 

`bloom_fetch_compute_v1.1_base_layer`

Split the upper and lower layer workload, upper just forward while base layer goes through bloom-fetch-compute

```
    int query_num = 100;
    int read_iter_per_query = 10000;
```

Achieved: 128.118 ms @ 200 MHz -> close to memory access performance of 125 ns / access