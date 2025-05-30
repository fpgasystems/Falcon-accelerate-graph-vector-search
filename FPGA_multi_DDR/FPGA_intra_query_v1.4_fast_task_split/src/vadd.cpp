#include "bloom_filter.hpp"
#include "bloom_fetch_compute.hpp"
#include "compute.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "scheduler.hpp"
#include "types.hpp"
#include "utils.hpp"

extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int ef, // size of the result priority queue
	const int candidate_queue_runtime_size, 
	const int max_cand_batch_size, 
	const int max_async_stage_num,
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int max_bloom_out_burst_size,
	const int max_link_num_base,

    // in runtime (from DRAM)
	const int* entry_point_ids,
	const ap_uint<512>* query_vectors,
    ap_uint<512>* db_vectors_chan_0, 
#if N_CHANNEL >= 2
	ap_uint<512>* db_vectors_chan_1,
#endif
#if N_CHANNEL >= 4
	ap_uint<512>* db_vectors_chan_2,
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
#if N_CHANNEL >= 4
	const ap_uint<512>* links_base_chan_2,
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
    int* out_id,
	float* out_dist,

	// debug signals (each 4 byte): 
	//   0: bottom layer entry node id, 
	//   1: number of hops in upper layers 
	//   2: number of read vectors in upper layers
	//   3: number of hops in base layer (number of pop operations)
	//   4: number of valid read vectors in base layer
	int* mem_debug
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=entry_point_ids latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=query_vectors latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=mem_debug latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem10 // cannot share gmem with out as they are different PEs

// If the port is a read-only port, then set the num_write_outstanding=1 and max_write_burst_length=2 to conserve memory resources. For write-only ports, set the num_read_outstanding=1 and max_read_burst_length=2.
// https://docs.xilinx.com/r/2022.1-English/ug1399-vitis-hls/pragma-HLS-interface

#pragma HLS INTERFACE m_axi port=db_vectors_chan_0 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors0
#if N_CHANNEL >= 2
#pragma HLS INTERFACE m_axi port=db_vectors_chan_1 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors1
#endif
#if N_CHANNEL >= 4
#pragma HLS INTERFACE m_axi port=db_vectors_chan_2 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors2
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
#pragma HLS INTERFACE m_axi port=out_id latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=out_dist latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9

#pragma HLS dataflow

    hls::stream<int> s_finish_query_task_scheduler; // finish the current query
#pragma HLS stream variable=s_finish_query_task_scheduler depth=16

    hls::stream<int> s_finish_query_results_collection; // finish all queries
#pragma HLS stream variable=s_finish_query_results_collection depth=16
	
	hls::stream<ap_uint<512>> s_query_vectors_replicated[N_CHANNEL];
#pragma HLS stream variable=s_query_vectors_replicated depth=16

// 	hls::stream<result_t> s_entry_point_base_level;
// #pragma HLS stream variable=s_entry_point_base_level depth=16

	hls::stream<int> s_cand_batch_size;
#pragma HLS stream variable=s_cand_batch_size depth=16

    hls::stream<cand_t> s_top_candidates; // current top candidates
#pragma HLS stream variable=s_top_candidates depth=512

	hls::stream<int> s_num_inserted_candidates;
#pragma HLS stream variable=s_num_inserted_candidates depth=512

	hls::stream<result_t> s_inserted_candidates;
#pragma HLS stream variable=s_inserted_candidates depth=512

	hls::stream<int> s_num_valid_candidates_base_level_total;
#pragma HLS stream variable=s_num_valid_candidates_base_level_total depth=16

	hls::stream<result_t> s_distances_base_level;
#pragma HLS stream variable=s_distances_base_level depth=512

	hls::stream<float> s_largest_result_queue_elements;
#pragma HLS stream variable=s_largest_result_queue_elements depth=512	

	const int rep_factor_s_largest_result_queue_elements = 1 + N_CHANNEL;
	hls::stream<float> s_largest_result_queue_elements_replicated[rep_factor_s_largest_result_queue_elements];
#pragma HLS stream variable=s_largest_result_queue_elements_replicated depth=512		

// 	hls::stream<int> s_debug_num_vec_base_layer;
// #pragma HLS stream variable=s_debug_num_vec_base_layer depth=16

	hls::stream<int> s_finish_query_replicate_s_largest_result_queue_elements;
#pragma HLS stream variable=s_finish_query_replicate_s_largest_result_queue_elements depth=16

	// controls the traversal and maintains the candidate queue
	task_scheduler(
		query_num, 
		candidate_queue_runtime_size,
		max_cand_batch_size,
		max_async_stage_num,

    	// in runtime (should from DRAM)
		entry_point_ids,
		query_vectors,

		// in streams
		s_num_inserted_candidates,
		s_inserted_candidates,
		s_largest_result_queue_elements_replicated[0],
		// s_debug_num_vec_base_layer,
		s_finish_query_replicate_s_largest_result_queue_elements,
		
		// out streams
		s_query_vectors_replicated,
		// s_entry_point_base_level,
		s_cand_batch_size,
		s_top_candidates,
		s_finish_query_task_scheduler,

		mem_debug
	);


	hls::stream<int> s_fetch_batch_size;
#pragma HLS stream variable=s_fetch_batch_size depth=16

    hls::stream<ap_uint<512>> s_neighbor_ids_raw;
#pragma HLS stream variable=s_neighbor_ids_raw depth=128

    hls::stream<int> s_finish_query_fetch_neighbor_ids; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_neighbor_ids depth=16

	fetch_neighbor_ids(
		// in initialization
		query_num,
		max_link_num_base,
		// in runtime (should from DRAM)
    	links_base_chan_0,
#if N_CHANNEL >= 2
		links_base_chan_1,
#endif
#if N_CHANNEL >= 4
		links_base_chan_2,
		links_base_chan_3,
#endif
#if N_CHANNEL >= 8
		links_base_chan_4,
		links_base_chan_5,
		links_base_chan_6,
		links_base_chan_7,
#endif
#if N_CHANNEL >= 16
		links_base_chan_8,
		links_base_chan_9,
		links_base_chan_10,
		links_base_chan_11,
		links_base_chan_12,
		links_base_chan_13,
		links_base_chan_14,
		links_base_chan_15,
#endif
		// in runtime (stream)
		s_top_candidates,
		s_finish_query_task_scheduler,

		// out (stream)
		s_neighbor_ids_raw,
		s_finish_query_fetch_neighbor_ids
	);

    hls::stream<int> s_num_neighbors_base_level_per_channel[N_CHANNEL]; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_base_level_per_channel depth=512

	hls::stream<cand_t> s_fetched_neighbor_ids_per_channel[N_CHANNEL]; 
#pragma HLS stream variable=s_fetched_neighbor_ids_per_channel depth=512

    hls::stream<int> s_finish_query_split_tasks_to_channels; // finish all queries
#pragma HLS stream variable=s_finish_query_split_tasks_to_channels depth=16

	split_tasks_to_channels(
		query_num,
		max_link_num_base,

		// in streams
		s_cand_batch_size,
		s_neighbor_ids_raw,
		s_finish_query_fetch_neighbor_ids,

		// out streams
		s_num_neighbors_base_level_per_channel,
		s_fetched_neighbor_ids_per_channel,
		s_finish_query_split_tasks_to_channels
	);

    hls::stream<int> s_finish_query_split_tasks_to_channels_replicated[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_query_split_tasks_to_channels_replicated depth=16

	replicate_s_finish<N_CHANNEL>(
		query_num,
		s_finish_query_split_tasks_to_channels,
		s_finish_query_split_tasks_to_channels_replicated
	);

	hls::stream<int> s_num_valid_candidates_base_level_total_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_num_valid_candidates_base_level_total_per_channel depth=16

	hls::stream<result_t> s_distances_base_level_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_distances_base_level_per_channel depth=512

    hls::stream<int> s_finish_query_bloom_fetch_compute_per_channel[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_query_bloom_fetch_compute_per_channel depth=16


	// using loop unrolling for bloom_fetch_compute can lead to compilation error
	//   due to failed dataflow checking in HLS, when using inline pragma for bloom_fetch_compute

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_0,

		// in streams
		s_query_vectors_replicated[0],
		s_num_neighbors_base_level_per_channel[0],
		s_fetched_neighbor_ids_per_channel[0],
		s_finish_query_split_tasks_to_channels_replicated[0],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[0],
		s_distances_base_level_per_channel[0],
		s_finish_query_bloom_fetch_compute_per_channel[0]
	);
	#if N_CHANNEL >= 2
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_1,

		// in streams
		s_query_vectors_replicated[1],
		s_num_neighbors_base_level_per_channel[1],
		s_fetched_neighbor_ids_per_channel[1],
		s_finish_query_split_tasks_to_channels_replicated[1],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[1],
		s_distances_base_level_per_channel[1],
		s_finish_query_bloom_fetch_compute_per_channel[1]
	);
	#endif
	#if N_CHANNEL >= 4
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_2,

		// in streams
		s_query_vectors_replicated[2],
		s_num_neighbors_base_level_per_channel[2],
		s_fetched_neighbor_ids_per_channel[2],
		s_finish_query_split_tasks_to_channels_replicated[2],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[2],
		s_distances_base_level_per_channel[2],
		s_finish_query_bloom_fetch_compute_per_channel[2]
	);
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_3,

		// in streams
		s_query_vectors_replicated[3],
		s_num_neighbors_base_level_per_channel[3],
		s_fetched_neighbor_ids_per_channel[3],
		s_finish_query_split_tasks_to_channels_replicated[3],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[3],
		s_distances_base_level_per_channel[3],
		s_finish_query_bloom_fetch_compute_per_channel[3]
	);
	#endif
	#if N_CHANNEL >= 8
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		
		// in runtime (from DRAM)
		db_vectors_chan_4,

		// in streams
		s_query_vectors_replicated[4],
		s_num_neighbors_base_level_per_channel[4],
		s_fetched_neighbor_ids_per_channel[4],
		s_finish_query_split_tasks_to_channels_replicated[4],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[4],
		s_distances_base_level_per_channel[4],
		s_finish_query_bloom_fetch_compute_per_channel[4]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_5,

		// in streams
		s_query_vectors_replicated[5],
		s_num_neighbors_base_level_per_channel[5],
		s_fetched_neighbor_ids_per_channel[5],
		s_finish_query_split_tasks_to_channels_replicated[5],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[5],
		s_distances_base_level_per_channel[5],
		s_finish_query_bloom_fetch_compute_per_channel[5]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_6,

		// in streams
		s_query_vectors_replicated[6],
		s_num_neighbors_base_level_per_channel[6],
		s_fetched_neighbor_ids_per_channel[6],
		s_finish_query_split_tasks_to_channels_replicated[6],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[6],
		s_distances_base_level_per_channel[6],
		s_finish_query_bloom_fetch_compute_per_channel[6]
	);
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_7,

		// in streams
		s_query_vectors_replicated[7],
		s_num_neighbors_base_level_per_channel[7],
		s_fetched_neighbor_ids_per_channel[7],
		s_finish_query_split_tasks_to_channels_replicated[7],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[7],
		s_distances_base_level_per_channel[7],
		s_finish_query_bloom_fetch_compute_per_channel[7]
	);
	#endif
	#if N_CHANNEL >= 16
	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_8,

		// in streams
		s_query_vectors_replicated[8],
		s_num_neighbors_base_level_per_channel[8],
		s_fetched_neighbor_ids_per_channel[8],
		s_finish_query_split_tasks_to_channels_replicated[8],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[8],
		s_distances_base_level_per_channel[8],
		s_finish_query_bloom_fetch_compute_per_channel[8]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_9,

		// in streams
		s_query_vectors_replicated[9],
		s_num_neighbors_base_level_per_channel[9],
		s_fetched_neighbor_ids_per_channel[9],
		s_finish_query_split_tasks_to_channels_replicated[9],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[9],
		s_distances_base_level_per_channel[9],
		s_finish_query_bloom_fetch_compute_per_channel[9]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		
		// in runtime (from DRAM)
		db_vectors_chan_10,

		// in streams
		s_query_vectors_replicated[10],
		s_num_neighbors_base_level_per_channel[10],
		s_fetched_neighbor_ids_per_channel[10],
		s_finish_query_split_tasks_to_channels_replicated[10],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[10],
		s_distances_base_level_per_channel[10],
		s_finish_query_bloom_fetch_compute_per_channel[10]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_11,

		// in streams
		s_query_vectors_replicated[11],
		s_num_neighbors_base_level_per_channel[11],
		s_fetched_neighbor_ids_per_channel[11],
		s_finish_query_split_tasks_to_channels_replicated[11],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[11],
		s_distances_base_level_per_channel[11],
		s_finish_query_bloom_fetch_compute_per_channel[11]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_12,

		// in streams
		s_query_vectors_replicated[12],
		s_num_neighbors_base_level_per_channel[12],
		s_fetched_neighbor_ids_per_channel[12],
		s_finish_query_split_tasks_to_channels_replicated[12],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[12],
		s_distances_base_level_per_channel[12],
		s_finish_query_bloom_fetch_compute_per_channel[12]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_13,

		// in streams
		s_query_vectors_replicated[13],
		s_num_neighbors_base_level_per_channel[13],
		s_fetched_neighbor_ids_per_channel[13],
		s_finish_query_split_tasks_to_channels_replicated[13],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[13],
		s_distances_base_level_per_channel[13],
		s_finish_query_bloom_fetch_compute_per_channel[13]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_14,

		// in streams
		s_query_vectors_replicated[14],
		s_num_neighbors_base_level_per_channel[14],
		s_fetched_neighbor_ids_per_channel[14],
		s_finish_query_split_tasks_to_channels_replicated[14],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[14],
		s_distances_base_level_per_channel[14],
		s_finish_query_bloom_fetch_compute_per_channel[14]
	);

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_15,

		// in streams
		s_query_vectors_replicated[15],
		s_num_neighbors_base_level_per_channel[15],
		s_fetched_neighbor_ids_per_channel[15],
		s_finish_query_split_tasks_to_channels_replicated[15],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[15],
		s_distances_base_level_per_channel[15],
		s_finish_query_bloom_fetch_compute_per_channel[15]
	);
	#endif

	hls::stream<int> s_num_valid_candidates_base_level_filtered_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_num_valid_candidates_base_level_filtered_per_channel depth=16

	hls::stream<result_t> s_distances_base_filtered_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_distances_base_filtered_per_channel depth=512

    hls::stream<int> s_finish_filter_computed_distances_per_channel[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_filter_computed_distances_per_channel depth=16	

filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[0],
    s_largest_result_queue_elements_replicated[0 + 1],
    s_distances_base_level_per_channel[0],
    s_finish_query_bloom_fetch_compute_per_channel[0],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[0],
    s_distances_base_filtered_per_channel[0],
    s_finish_filter_computed_distances_per_channel[0]
);
#if N_CHANNEL >= 2
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[1],
    s_largest_result_queue_elements_replicated[1 + 1],
    s_distances_base_level_per_channel[1],
    s_finish_query_bloom_fetch_compute_per_channel[1],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[1],
    s_distances_base_filtered_per_channel[1],
    s_finish_filter_computed_distances_per_channel[1]
);
#endif
#if N_CHANNEL >= 4
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[2],
    s_largest_result_queue_elements_replicated[2 + 1],
    s_distances_base_level_per_channel[2],
    s_finish_query_bloom_fetch_compute_per_channel[2],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[2],
    s_distances_base_filtered_per_channel[2],
    s_finish_filter_computed_distances_per_channel[2]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[3],
    s_largest_result_queue_elements_replicated[3 + 1],
    s_distances_base_level_per_channel[3],
    s_finish_query_bloom_fetch_compute_per_channel[3],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[3],
    s_distances_base_filtered_per_channel[3],
    s_finish_filter_computed_distances_per_channel[3]
);
#endif
#if N_CHANNEL >= 8
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[4],
    s_largest_result_queue_elements_replicated[4 + 1],
    s_distances_base_level_per_channel[4],
    s_finish_query_bloom_fetch_compute_per_channel[4],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[4],
    s_distances_base_filtered_per_channel[4],
    s_finish_filter_computed_distances_per_channel[4]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[5],
    s_largest_result_queue_elements_replicated[5 + 1],
    s_distances_base_level_per_channel[5],
    s_finish_query_bloom_fetch_compute_per_channel[5],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[5],
    s_distances_base_filtered_per_channel[5],
    s_finish_filter_computed_distances_per_channel[5]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[6],
    s_largest_result_queue_elements_replicated[6 + 1],
    s_distances_base_level_per_channel[6],
    s_finish_query_bloom_fetch_compute_per_channel[6],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[6],
    s_distances_base_filtered_per_channel[6],
    s_finish_filter_computed_distances_per_channel[6]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[7],
    s_largest_result_queue_elements_replicated[7 + 1],
    s_distances_base_level_per_channel[7],
    s_finish_query_bloom_fetch_compute_per_channel[7],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[7],
    s_distances_base_filtered_per_channel[7],
    s_finish_filter_computed_distances_per_channel[7]
);
#endif
#if N_CHANNEL >= 16
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[8],
    s_largest_result_queue_elements_replicated[8 + 1],
    s_distances_base_level_per_channel[8],
    s_finish_query_bloom_fetch_compute_per_channel[8],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[8],
    s_distances_base_filtered_per_channel[8],
    s_finish_filter_computed_distances_per_channel[8]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[9],
    s_largest_result_queue_elements_replicated[9 + 1],
    s_distances_base_level_per_channel[9],
    s_finish_query_bloom_fetch_compute_per_channel[9],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[9],
    s_distances_base_filtered_per_channel[9],
    s_finish_filter_computed_distances_per_channel[9]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[10],
    s_largest_result_queue_elements_replicated[10 + 1],
    s_distances_base_level_per_channel[10],
    s_finish_query_bloom_fetch_compute_per_channel[10],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[10],
    s_distances_base_filtered_per_channel[10],
    s_finish_filter_computed_distances_per_channel[10]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[11],
    s_largest_result_queue_elements_replicated[11 + 1],
    s_distances_base_level_per_channel[11],
    s_finish_query_bloom_fetch_compute_per_channel[11],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[11],
    s_distances_base_filtered_per_channel[11],
    s_finish_filter_computed_distances_per_channel[11]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[12],
    s_largest_result_queue_elements_replicated[12 + 1],
    s_distances_base_level_per_channel[12],
    s_finish_query_bloom_fetch_compute_per_channel[12],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[12],
    s_distances_base_filtered_per_channel[12],
    s_finish_filter_computed_distances_per_channel[12]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[13],
    s_largest_result_queue_elements_replicated[13 + 1],
    s_distances_base_level_per_channel[13],
    s_finish_query_bloom_fetch_compute_per_channel[13],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[13],
    s_distances_base_filtered_per_channel[13],
    s_finish_filter_computed_distances_per_channel[13]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[14],
    s_largest_result_queue_elements_replicated[14 + 1],
    s_distances_base_level_per_channel[14],
    s_finish_query_bloom_fetch_compute_per_channel[14],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[14],
    s_distances_base_filtered_per_channel[14],
    s_finish_filter_computed_distances_per_channel[14]
);
filter_computed_distances(
    query_num,
    // in stream
    s_num_valid_candidates_base_level_total_per_channel[15],
    s_largest_result_queue_elements_replicated[15 + 1],
    s_distances_base_level_per_channel[15],
    s_finish_query_bloom_fetch_compute_per_channel[15],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[15],
    s_distances_base_filtered_per_channel[15],
    s_finish_filter_computed_distances_per_channel[15]
);
#endif

    hls::stream<int> s_finish_query_filter_computed_distances; // finish all queries
#pragma HLS stream variable=s_finish_query_filter_computed_distances depth=16

	gather_s_finish<N_CHANNEL>(
		query_num,
		s_finish_filter_computed_distances_per_channel,
		s_finish_query_filter_computed_distances
	);

    hls::stream<int> s_finish_gather_distances_from_channels; // finish all queries
#pragma HLS stream variable=s_finish_gather_distances_from_channels depth=16

	gather_distances_from_channels(
		query_num,
		s_num_valid_candidates_base_level_filtered_per_channel,
		s_distances_base_filtered_per_channel,
		s_finish_query_filter_computed_distances,

		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_gather_distances_from_channels
	);

	results_collection(
		// in (initialization)
		query_num,
		ef,
		// in runtime (stream)
		// s_entry_point_base_level,
		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_gather_distances_from_channels,

		// out (stream)
		s_inserted_candidates,
		s_num_inserted_candidates,
		s_largest_result_queue_elements,
		// s_debug_num_vec_base_layer,
		s_finish_query_results_collection,

		// out (DRAM)
		out_id,
		out_dist
	);

	replicate_s_control<rep_factor_s_largest_result_queue_elements, float>(
		query_num,
		// in (stream)
		s_largest_result_queue_elements,
		s_finish_query_results_collection,
		
		// out (stream)
		s_largest_result_queue_elements_replicated,
		s_finish_query_replicate_s_largest_result_queue_elements
	);
}

}
