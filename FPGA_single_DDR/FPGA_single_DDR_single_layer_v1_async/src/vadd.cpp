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
	const int d,
	const int max_link_num_base,

    // in runtime (from DRAM)
	const int* entry_point_ids,
	const ap_uint<512>* query_vectors,
    hls::burst_maxi<ap_uint<512>> db_vectors, // need to write visited tag

    const ap_uint<512>* links_base,
	   
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
#pragma HLS INTERFACE m_axi port=entry_point_ids num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=query_vectors num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=mem_debug num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem10 // cannot share gmem with out as they are different PEs

#pragma HLS INTERFACE m_axi port=db_vectors latency=1 num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem4 
#pragma HLS INTERFACE m_axi port=links_base num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem2

// out
#pragma HLS INTERFACE m_axi port=out_id num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=out_dist num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9

#pragma HLS dataflow

    hls::stream<int> s_finish_query_task_scheduler; // finish the current query
#pragma HLS stream variable=s_finish_query_task_scheduler depth=16

    hls::stream<int> s_finish_query_results_collection; // finish all queries
#pragma HLS stream variable=s_finish_query_results_collection depth=16
	
	hls::stream<ap_uint<512>> s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=128

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
	
// 	hls::stream<int> s_debug_num_vec_base_layer;
// #pragma HLS stream variable=s_debug_num_vec_base_layer depth=16

	// controls the traversal and maintains the candidate queue
	task_scheduler(
		query_num, 
		candidate_queue_runtime_size,
		max_cand_batch_size,
		max_async_stage_num,
		d, 

    	// in runtime (should from DRAM)
		entry_point_ids,
		query_vectors,

		// in streams
		s_num_inserted_candidates,
		s_inserted_candidates,
		s_largest_result_queue_elements,
		// s_debug_num_vec_base_layer,
		s_finish_query_results_collection,
		
		// out streams
		s_query_vectors,
		// s_entry_point_base_level,
		s_cand_batch_size,
		s_top_candidates,
		s_finish_query_task_scheduler,

		mem_debug
	);

	hls::stream<int> s_fetch_batch_size;
#pragma HLS stream variable=s_fetch_batch_size depth=16
	
    hls::stream<int> s_num_neighbors_base_level; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_base_level depth=512

    hls::stream<cand_t> s_fetched_neighbor_ids; 
#pragma HLS stream variable=s_fetched_neighbor_ids depth=512

    hls::stream<int> s_finish_query_fetch_neighbor_ids; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_neighbor_ids depth=16

	fetch_neighbor_ids(
		// in initialization
		query_num,
		max_link_num_base,
		// in runtime (should from DRAM)
    	links_base,
		// in runtime (stream)
		s_top_candidates,
		s_finish_query_task_scheduler,

		// out (stream)
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids
	);


    hls::stream<int> s_finish_query_bloom_fetch_compute; // finish all queries
#pragma HLS stream variable=s_finish_query_bloom_fetch_compute depth=16

	bloom_fetch_compute(
		// in initialization
		query_num, 
		runtime_n_bucket_addr_bits,
		hash_seed,
		d,

		// in runtime (from DRAM)
		db_vectors, // need to write visited tag

		// in streams
		s_query_vectors,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids,

		// out streams
		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_query_bloom_fetch_compute
	);

	results_collection(
		// in (initialization)
		query_num,
		ef,
		// in runtime (stream)
		// s_entry_point_base_level,
		s_cand_batch_size,
		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_query_bloom_fetch_compute,

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

}

}
