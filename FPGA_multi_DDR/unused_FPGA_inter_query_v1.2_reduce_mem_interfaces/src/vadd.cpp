#include "bloom_filter.hpp"
#include "bloom_fetch_compute.hpp"
#include "compute.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "scheduler.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "per_channel_processing_wrapper.hpp"

extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int query_batch_size,
	const int ef, // size of the result priority queue
	const int candidate_queue_runtime_size, 
	const int max_cand_batch_size, 
	const int max_async_stage_num,
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int max_bloom_out_burst_size,
	const int max_link_num_base,

    // in runtime (from DRAM)
	const int entry_point_id,
	// const int* entry_point_ids,
	const ap_uint<512>* query_vectors,

    ap_uint<512>* db_vectors_chan_0, 
#if N_CHANNEL >= 2
	ap_uint<512>* db_vectors_chan_1,
#endif
#if N_CHANNEL >= 3
	ap_uint<512>* db_vectors_chan_2,
#if N_CHANNEL >= 4
#endif
	ap_uint<512>* db_vectors_chan_3,
#endif
#if N_CHANNEL >= 8
	ap_uint<512>* db_vectors_chan_4,
	ap_uint<512>* db_vectors_chan_5,
	ap_uint<512>* db_vectors_chan_6,
	ap_uint<512>* db_vectors_chan_7,
#endif
#if N_CHANNEL >= 16
	ap_uint<512>* db_vectors_chan_8,
	ap_uint<512>* db_vectors_chan_9,
	ap_uint<512>* db_vectors_chan_10,
	ap_uint<512>* db_vectors_chan_11,
	ap_uint<512>* db_vectors_chan_12,
	ap_uint<512>* db_vectors_chan_13,
	ap_uint<512>* db_vectors_chan_14,
	ap_uint<512>* db_vectors_chan_15,
#endif

    const ap_uint<512>* links_base_chan_0,
#if N_CHANNEL >= 2
	const ap_uint<512>* links_base_chan_1,
#endif
#if N_CHANNEL >= 3
	const ap_uint<512>* links_base_chan_2,
#endif
#if N_CHANNEL >= 4
	const ap_uint<512>* links_base_chan_3,
#endif
#if N_CHANNEL >= 8
	const ap_uint<512>* links_base_chan_4,
	const ap_uint<512>* links_base_chan_5,
	const ap_uint<512>* links_base_chan_6,
	const ap_uint<512>* links_base_chan_7,
#endif
#if N_CHANNEL >= 16
	const ap_uint<512>* links_base_chan_8,
	const ap_uint<512>* links_base_chan_9,
	const ap_uint<512>* links_base_chan_10,
	const ap_uint<512>* links_base_chan_11,
	const ap_uint<512>* links_base_chan_12,
	const ap_uint<512>* links_base_chan_13,
	const ap_uint<512>* links_base_chan_14,
	const ap_uint<512>* links_base_chan_15,
#endif
	   
    // out
    // int* out_id,
	// float* out_dist,
    ap_uint<64>* out_id_dist

	// debug signals (each 4 byte): 
	//   0: bottom layer entry node id, 
	//   1: number of hops in upper layers 
	//   2: number of read vectors in upper layers
	//   3: number of hops in base layer (number of pop operations)
	//   4: number of valid read vectors in base layer
	// int* mem_debug
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
// #pragma HLS INTERFACE m_axi port=entry_point_ids latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=query_vectors latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
// #pragma HLS INTERFACE m_axi port=mem_debug latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem10 // cannot share gmem with out as they are different PEs

#pragma HLS INTERFACE m_axi port=db_vectors_chan_0 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors0
#if N_CHANNEL >= 2
#pragma HLS INTERFACE m_axi port=db_vectors_chan_1 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors1
#endif
#if N_CHANNEL >= 3
#pragma HLS INTERFACE m_axi port=db_vectors_chan_2 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors2
#endif
#if N_CHANNEL >= 4
#pragma HLS INTERFACE m_axi port=db_vectors_chan_3 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors3
#endif
#if N_CHANNEL >= 8
#pragma HLS INTERFACE m_axi port=db_vectors_chan_4 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors4
#pragma HLS INTERFACE m_axi port=db_vectors_chan_5 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors5
#pragma HLS INTERFACE m_axi port=db_vectors_chan_6 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors6
#pragma HLS INTERFACE m_axi port=db_vectors_chan_7 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors7
#endif
#if N_CHANNEL >= 16
#pragma HLS INTERFACE m_axi port=db_vectors_chan_8 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors8
#pragma HLS INTERFACE m_axi port=db_vectors_chan_9 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors9
#pragma HLS INTERFACE m_axi port=db_vectors_chan_10 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors10
#pragma HLS INTERFACE m_axi port=db_vectors_chan_11 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors11
#pragma HLS INTERFACE m_axi port=db_vectors_chan_12 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors12
#pragma HLS INTERFACE m_axi port=db_vectors_chan_13 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors13
#pragma HLS INTERFACE m_axi port=db_vectors_chan_14 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors14
#pragma HLS INTERFACE m_axi port=db_vectors_chan_15 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors15
#endif

#pragma HLS INTERFACE m_axi port=links_base_chan_0 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base0
#if N_CHANNEL >= 2
#pragma HLS INTERFACE m_axi port=links_base_chan_1 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base1
#endif
#if N_CHANNEL >= 4
#pragma HLS INTERFACE m_axi port=links_base_chan_2 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base2
#pragma HLS INTERFACE m_axi port=links_base_chan_3 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base3
#endif
#if N_CHANNEL >= 8
#pragma HLS INTERFACE m_axi port=links_base_chan_4 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base4
#pragma HLS INTERFACE m_axi port=links_base_chan_5 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base5
#pragma HLS INTERFACE m_axi port=links_base_chan_6 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base6
#pragma HLS INTERFACE m_axi port=links_base_chan_7 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base7
#endif
#if N_CHANNEL >= 16
#pragma HLS INTERFACE m_axi port=links_base_chan_8 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base8
#pragma HLS INTERFACE m_axi port=links_base_chan_9 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base9
#pragma HLS INTERFACE m_axi port=links_base_chan_10 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base10
#pragma HLS INTERFACE m_axi port=links_base_chan_11 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base11
#pragma HLS INTERFACE m_axi port=links_base_chan_12 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base12
#pragma HLS INTERFACE m_axi port=links_base_chan_13 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base13
#pragma HLS INTERFACE m_axi port=links_base_chan_14 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base14
#pragma HLS INTERFACE m_axi port=links_base_chan_15 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base15
#endif

// out
// #pragma HLS INTERFACE m_axi port=out_id latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9
// #pragma HLS INTERFACE m_axi port=out_dist latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=out_id_dist latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9

#pragma HLS dataflow

	hls::stream<int> s_query_batch_size;
#pragma HLS stream variable=s_query_batch_size depth=16

	hls::stream<ap_uint<512>> s_query_vectors_in;
#pragma HLS stream variable=s_query_vectors_in depth=16

	hls::stream<int> s_entry_point_ids;
#pragma HLS stream variable=s_entry_point_ids depth=16

	hls::stream<int> s_finish_batch;
#pragma HLS stream variable=s_finish_batch depth=16

	read_queries(
		// in initialization
		query_num,
		query_batch_size,
		// in DRAM
		entry_point_id,
		// entry_point_ids,
		query_vectors,
		// in stream
		s_finish_batch,

		// out streams
		s_query_batch_size,
		s_query_vectors_in,
		s_entry_point_ids
	);

	// replicate s_query_batch_size to multiple streams
	const int replicate_factor_s_query_batch_size = 2;
	hls::stream<int> s_query_batch_size_replicated[replicate_factor_s_query_batch_size];
#pragma HLS stream variable=s_query_batch_size_replicated depth=16

	replicate_s_query_batch_size<replicate_factor_s_query_batch_size>(
		s_query_batch_size,
		s_query_batch_size_replicated
	);

	hls::stream<int> s_query_batch_size_in_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_query_batch_size_in_per_channel depth=16


	hls::stream<ap_uint<512>> s_query_vectors_in_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_query_vectors_in_per_channel depth=16

	hls::stream<int> s_entry_point_ids_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_entry_point_ids_per_channel depth=16

	split_queries(
		// in streams
		s_query_batch_size_replicated[0],
		s_query_vectors_in,
		s_entry_point_ids,

		// out streams
		s_query_batch_size_in_per_channel,
		s_query_vectors_in_per_channel,
		s_entry_point_ids_per_channel
	);


	hls::stream<int> s_out_ids_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_out_ids_per_channel depth=512

	hls::stream<float> s_out_dists_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_out_dists_per_channel depth=512

	hls::stream<int> s_debug_signals_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_debug_signals_per_channel depth=512

	// using an array to represent different DRAM channels would lead to fail to find the per_channel_processing_wrapper
	// thus manually unrolling
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_0,
		links_base_chan_0,	

		// in streams
		s_query_batch_size_in_per_channel[0], // -1: stop
		s_query_vectors_in_per_channel[0],	
		s_entry_point_ids_per_channel[0], 

		// out streams
		s_out_ids_per_channel[0],
		s_out_dists_per_channel[0],
		s_debug_signals_per_channel[0]
	);
#if N_CHANNEL >= 2
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_1,
		links_base_chan_1,	

		// in streams
		s_query_batch_size_in_per_channel[1], // -1: stop
		s_query_vectors_in_per_channel[1],	
		s_entry_point_ids_per_channel[1], 

		// out streams
		s_out_ids_per_channel[1],
		s_out_dists_per_channel[1],
		s_debug_signals_per_channel[1]
	);
#endif
#if N_CHANNEL >= 3
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_2,
		links_base_chan_2,	

		// in streams
		s_query_batch_size_in_per_channel[2], // -1: stop
		s_query_vectors_in_per_channel[2],	
		s_entry_point_ids_per_channel[2], 

		// out streams
		s_out_ids_per_channel[2],
		s_out_dists_per_channel[2],
		s_debug_signals_per_channel[2]
	);
#endif
#if N_CHANNEL >= 4
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_3,
		links_base_chan_3,	

		// in streams
		s_query_batch_size_in_per_channel[3], // -1: stop
		s_query_vectors_in_per_channel[3],	
		s_entry_point_ids_per_channel[3], 

		// out streams
		s_out_ids_per_channel[3],
		s_out_dists_per_channel[3],
		s_debug_signals_per_channel[3]
	);
#endif
#if N_CHANNEL >=8 
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_4,
		links_base_chan_4,	

		// in streams
		s_query_batch_size_in_per_channel[4], // -1: stop
		s_query_vectors_in_per_channel[4],	
		s_entry_point_ids_per_channel[4], 

		// out streams
		s_out_ids_per_channel[4],
		s_out_dists_per_channel[4],
		s_debug_signals_per_channel[4]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_5,
		links_base_chan_5,	

		// in streams
		s_query_batch_size_in_per_channel[5], // -1: stop
		s_query_vectors_in_per_channel[5],	
		s_entry_point_ids_per_channel[5], 

		// out streams
		s_out_ids_per_channel[5],
		s_out_dists_per_channel[5],
		s_debug_signals_per_channel[5]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_6,
		links_base_chan_6,	

		// in streams
		s_query_batch_size_in_per_channel[6], // -1: stop
		s_query_vectors_in_per_channel[6],
		s_entry_point_ids_per_channel[6],

		// out streams
		s_out_ids_per_channel[6],
		s_out_dists_per_channel[6],
		s_debug_signals_per_channel[6]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_7,
		links_base_chan_7,	

		// in streams
		s_query_batch_size_in_per_channel[7], // -1: stop
		s_query_vectors_in_per_channel[7],
		s_entry_point_ids_per_channel[7],

		// out streams
		s_out_ids_per_channel[7],
		s_out_dists_per_channel[7],
		s_debug_signals_per_channel[7]
	);
#endif
#if N_CHANNEL >= 16
	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_8,
		links_base_chan_8,	

		// in streams
		s_query_batch_size_in_per_channel[8], // -1: stop
		s_query_vectors_in_per_channel[8],
		s_entry_point_ids_per_channel[8],

		// out streams
		s_out_ids_per_channel[8],
		s_out_dists_per_channel[8],
		s_debug_signals_per_channel[8]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_9,
		links_base_chan_9,	

		// in streams
		s_query_batch_size_in_per_channel[9], // -1: stop
		s_query_vectors_in_per_channel[9],
		s_entry_point_ids_per_channel[9],

		// out streams
		s_out_ids_per_channel[9],
		s_out_dists_per_channel[9],
		s_debug_signals_per_channel[9]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_10,
		links_base_chan_10,	

		// in streams
		s_query_batch_size_in_per_channel[10], // -1: stop
		s_query_vectors_in_per_channel[10],
		s_entry_point_ids_per_channel[10],

		// out streams
		s_out_ids_per_channel[10],
		s_out_dists_per_channel[10],
		s_debug_signals_per_channel[10]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_11,
		links_base_chan_11,	

		// in streams
		s_query_batch_size_in_per_channel[11], // -1: stop
		s_query_vectors_in_per_channel[11],
		s_entry_point_ids_per_channel[11],

		// out streams
		s_out_ids_per_channel[11],
		s_out_dists_per_channel[11],
		s_debug_signals_per_channel[11]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_12,
		links_base_chan_12,	

		// in streams
		s_query_batch_size_in_per_channel[12], // -1: stop
		s_query_vectors_in_per_channel[12],
		s_entry_point_ids_per_channel[12],

		// out streams
		s_out_ids_per_channel[12],
		s_out_dists_per_channel[12],
		s_debug_signals_per_channel[12]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_13,
		links_base_chan_13,	

		// in streams
		s_query_batch_size_in_per_channel[13], // -1: stop
		s_query_vectors_in_per_channel[13],
		s_entry_point_ids_per_channel[13],

		// out streams
		s_out_ids_per_channel[13],
		s_out_dists_per_channel[13],
		s_debug_signals_per_channel[13]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_14,
		links_base_chan_14,	

		// in streams
		s_query_batch_size_in_per_channel[14], // -1: stop
		s_query_vectors_in_per_channel[14],
		s_entry_point_ids_per_channel[14],

		// out streams
		s_out_ids_per_channel[14],
		s_out_dists_per_channel[14],
		s_debug_signals_per_channel[14]
	);

	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors_chan_15,
		links_base_chan_15,	

		// in streams
		s_query_batch_size_in_per_channel[15], // -1: stop
		s_query_vectors_in_per_channel[15],
		s_entry_point_ids_per_channel[15],

		// out streams
		s_out_ids_per_channel[15],
		s_out_dists_per_channel[15],
		s_debug_signals_per_channel[15]
	);
#endif

	// write in round robine same as split_queries
	write_results(
		ef,

		// in streams
		s_query_batch_size_replicated[1], // -1: stop
		s_out_ids_per_channel,
		s_out_dists_per_channel,
		s_debug_signals_per_channel,

		// out streams
		s_finish_batch,

		// out
		// out_id,
		// out_dist
		out_id_dist
		// mem_debug
	);

}

}
