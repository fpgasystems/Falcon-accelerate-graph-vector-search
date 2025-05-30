# Bloom Filter and Hashing

## Hash 

### V1

An implementation of murmur2, can lead to a performance of II=1 and Latency=6, so pretty good performance. 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 1000;
```

Real performance: 50.9625 ms @ 200MHz ~= 10M cycles => achieved II=1.

## Bloom Filter

### V1: bool array + mod operation

Using a bool array to store bits, and the address handling is through the mod operation (hash % this->runtime_num_buckets;), thus leading to high latency (45 cycles). 

II = 2 * hash_num (2 * 4 = 8)

INFO: [v++ 200-1470] Pipelining result : Target II = NA, Final II = 8, Depth = 45, loop 'VITIS_LOOP_127_3'

```
	for (int i = 0; i < num_keys; i++) {
		// check each bucket, if false, write true
		int bit_match_cnt = 0;
		ap_uint<32> key = s_keys.read();
		for (int j = 0; j < num_hash_funs; j++) {
			ap_uint<32> hash = s_hash_values_per_pe[j].read();
			ap_uint<32> bucket_id = hash % this->runtime_num_buckets;
			if (!this->buckets[bucket_id]) {
				this->buckets[bucket_id] = true;
			} else {
				bit_match_cnt++;
			}
		}
		if (bit_match_cnt < num_hash_funs) { // does not contain
			s_valid_keys.write(key);
			num_valid++;
		}
		// if already some data in data fifo, write num acount
		if (num_valid == match_burst_size) {
			s_num_valid_keys.write(num_valid);
			num_valid = 0;
		}
	}
```

Small reads per iter (10 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 10;
```

58.1516 ms @ 200MHz -> real cycles / num_elements = II_real -> 58.1516 / 1000 * 200 * 1e6 / (10 * 1000 * 10) = 58 -> far larger than 8, because of the high latency.


Medium reads per iter (100 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 100;
```

94.1513 ms @ 200MHz -> real cycles / num_elements = II_real -> 94.1513 / 1000 * 200 * 1e6 / (100 * 1000 * 10) = 18.8 -> okay, but still much larger than 8, because of the high latency.


Many reads per iter (1000 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 1000;
```

454.151 ms @ 200MHz -> real cycles / num_elements = II_real -> 454.151 / 1000 * 200 * 1e6 / (1000 * 1000 * 10) = 9.1 -> finally close to the ideal II=8


Resource (For 64 * 1024 bits -> 48 BRAM):
```
    +--------------------------------+-----------------------------+---------+----+-------+-------+-----+
    |            Instance            |            Module           | BRAM_18K| DSP|   FF  |  LUT  | URAM|
    +--------------------------------+-----------------------------+---------+----+-------+-------+-----+
    |run_continuous_insert_check_U0  |run_continuous_insert_check  |        9|  48|  13667|  13969|    0|
```

### V1.1: bool array + bit shift operations

Lower the processing latency by always using 2^N size of bloom filter (both max capacity and runtime capacity), which allows us to use bit shift and range operations to do mod. 

Latency reduction: from 45 to 10 cycles.

INFO: [v++ 200-1470] Pipelining result : Target II = NA, Final II = 8, Depth = 10, loop 'VITIS_LOOP_138_3'

II = 2 * hash_num (2 * 4 = 8)

Hardware performance skipped.


### V1.2: 512-bit-wide array + bit shift operations

Saves hardware resources compared to V1.1.

Hash latency = 6.

Final II = 8, Depth = 10, loop 'VITIS_LOOP_138_3' -> loop up itself

II = 2 * hash_num (2 * 4 = 8)

Small reads per iter (10 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 10;
```

13.2025 ms @ 200MHz -> real cycles / num_elements = II_real -> 13.2025 / 1000 * 200 * 1e6 / (10 * 1000 * 10) = 26.4 -> still larger than 8, but much better than V1 (58 cycles).

Medium reads per iter (100 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 100;
```

50.4713 @ 200MHz -> real cycles / num_elements = II_real -> 50.4713 / 1000 * 200 * 1e6 / (100 * 1000 * 10) = 10.1 -> already close to 8

Many reads per iter (1000 reads per query): 

```
    int query_num = 10 * 1000;
	int read_iter_per_query = 1000;
```

409.501 ms @ 200MHz -> real cycles / num_elements = II_real -> 409.501 / 1000 * 200 * 1e6 / (1000 * 1000 * 10) = 8.2 -> very close to the ideal II=8

Resource (For 64 * 1024 bits -> only 9 BRAM, compared to the bool version using 48 BRAM):

```
    +--------------------------------+-----------------------------+---------+----+------+------+-----+
    |            Instance            |            Module           | BRAM_18K| DSP|  FF  |  LUT | URAM|
    +--------------------------------+-----------------------------+---------+----+------+------+-----+
    |run_continuous_insert_check_U0  |run_continuous_insert_check  |        9|  48|  4294|  7878|    0|
```

### V1.3 (failed) tried to unroll bloom bit array to further improve performance

The unrolling operation takes super long... And not sure the performance is improved. 


### (Use this for base-layer search) V1.4: support higher-level interface

Developped on V1.2, support cand_t and interface to the base-level-only accelerator.  

### (Use this for multi-layer search) V2: +support for upper/lower layer split based on V1.2

Based on V1.2, add higher-level interface to support the split of upper/base layers. Essentially bypassing the bloom filter for the upper layers.

Debug log: the task spliter and results collector before and after the bloom filter has to be 2 different PEs. Otherwise, if we merge them into one, there are feedback signals between the two, and thus the finish sequence cannot be correctly implemented, leading to deadlock in some cases. 

Performance: 

```
    int query_num = 100;
    int read_iter_per_query = 10000;
```

Theoretical performance: 8 cycle per base layer + 1 cycle per upper layer = 9 cycle

Achieved 46.5705 ms @ 200 MHz -> 46.5705 / 1000 * 200 * 1e6 / (100 * 10000) = 9.3 cycle -> very close to optimal.