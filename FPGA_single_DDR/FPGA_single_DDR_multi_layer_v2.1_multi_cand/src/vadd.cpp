#include "bloom_filter.hpp"
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
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int d,
	const int max_level,
    const int max_link_num_upper, 
	const int max_link_num_base,
	const int entry_point_id,
	// in initialization (from DRAM)
	const ap_uint<512>* entry_vector, 

    // in runtime (from DRAM)
	const ap_uint<512>* query_vectors,
    hls::burst_maxi<ap_uint<512>> db_vectors, // need to write visited tag

	const ap_uint<64>* ptr_to_upper_links, // start addr to upper link address per node
    const ap_uint<512>* links_upper,
    const ap_uint<512>* links_base,
	   
    // out
    int* out_id,
	float* out_dist,

	// debug signals (each 4 byte):
	//   0: bottom layer entry node id,
	//   1: number of hops in base layer (number of pop operations)
	int* mem_debug
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// If the port is a read-only port, then set the num_write_outstanding=1 and max_write_burst_length=2 to conserve memory resources. For write-only ports, set the num_read_outstanding=1 and max_read_burst_length=2.
// https://docs.xilinx.com/r/2022.1-English/ug1399-vitis-hls/pragma-HLS-interface

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=query_vectors num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=entry_vector num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem0 // share the same AXI interface with query_vectors
#pragma HLS INTERFACE m_axi port=db_vectors latency=1 num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem4 
#pragma HLS INTERFACE m_axi port=ptr_to_upper_links num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=links_upper num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=links_base num_read_outstanding=16 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem2

// out
#pragma HLS INTERFACE m_axi port=out_id  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=out_dist  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=mem_debug  offset=slave bundle=gmem10 // cannot share gmem with out as they are different PEs


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

    hls::stream<int> s_num_neighbors_upper_levels; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_upper_levels depth=512

	hls::stream<int> s_num_inserted_candidates;
#pragma HLS stream variable=s_num_inserted_candidates depth=512

	hls::stream<result_t> s_inserted_candidates;
#pragma HLS stream variable=s_inserted_candidates depth=512

	hls::stream<int> s_num_valid_candidates_upper_levels_total;
#pragma HLS stream variable=s_num_valid_candidates_upper_levels_total depth=16

	hls::stream<result_t> s_distances_upper_levels;
#pragma HLS stream variable=s_distances_upper_levels depth=512

	hls::stream<float> s_largest_result_queue_elements;
#pragma HLS stream variable=s_largest_result_queue_elements depth=512	
	
	// controls the traversal and maintains the candidate queue
	task_scheduler(
		query_num, 
		candidate_queue_runtime_size,
		max_cand_batch_size,
		d, 
		max_level,
    	max_link_num_upper, 
		max_link_num_base,
		entry_point_id,
    	// in runtime (should from DRAM)
		entry_vector,
		query_vectors,

		// in streams
		s_num_valid_candidates_upper_levels_total,
		s_distances_upper_levels,
		s_num_inserted_candidates,
		s_inserted_candidates,
		s_largest_result_queue_elements,
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
    	max_link_num_upper, 
		max_link_num_base,
		// in runtime (should from DRAM)
		ptr_to_upper_links,
    	links_upper,
    	links_base,
		// in runtime (stream)
		s_top_candidates,
		s_finish_query_task_scheduler,

		// out (stream)
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids
	);


	hls::stream<int> s_num_valid_candidates_burst;
#pragma HLS stream variable=s_num_valid_candidates_burst depth=16

	hls::stream<int> s_num_valid_candidates_base_level_total;
#pragma HLS stream variable=s_num_valid_candidates_base_level_total depth=16

	hls::stream<cand_t> s_valid_candidates;
#pragma HLS stream variable=s_valid_candidates depth=512

    hls::stream<int> s_finish_bloom; // finish all queries
#pragma HLS stream variable=s_finish_bloom depth=16

	BloomFilter<bloom_num_hash_funs, bloom_num_bucket_addr_bits> 
		bloom_filter(runtime_n_bucket_addr_bits);

	bloom_filter.bloom_filter_top_level(
		query_num, 
		hash_seed,
		// in streams
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids,

		// out streams
		s_num_valid_candidates_burst, // one round (s_num_neighbors) can contain multiple bursts
		s_num_valid_candidates_upper_levels_total, // one round can contain multiple bursts
		s_num_valid_candidates_base_level_total, // one round can contain multiple bursts
		s_valid_candidates,
		s_finish_bloom);
	
    hls::stream<ap_uint<512>> s_fetched_vectors; 
#pragma HLS stream variable=s_fetched_vectors depth=512
	
    hls::stream<int> s_finish_query_fetch_vectors; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_vectors depth=16

	const int rep_factor_s_num_valid_candidates_burst = 2;

	hls::stream<int> s_num_valid_candidates_burst_replicated[rep_factor_s_num_valid_candidates_burst];
#pragma HLS stream variable=s_num_valid_candidates_burst_replicated depth=16

	hls::stream<cand_t> s_valid_candidates_replicated[rep_factor_s_num_valid_candidates_burst];
#pragma HLS stream variable=s_valid_candidates_replicated depth=512

	hls::stream<int> s_finish_query_replicate_candidates; // finish all queries
#pragma HLS stream variable=s_finish_query_replicate_candidates depth=16

	replicate_s_read_iter_and_s_data<rep_factor_s_num_valid_candidates_burst, cand_t>(
		// in (initialization)
		query_num,
		// in (stream)
		s_num_valid_candidates_burst,
		s_valid_candidates,
		s_finish_bloom,
		
		// out (stream)
		s_num_valid_candidates_burst_replicated,
		s_valid_candidates_replicated,
		s_finish_query_replicate_candidates
	);

	fetch_vectors(
		// in initialization
		query_num,
		d,
		// in runtime (should from DRAM)
    	db_vectors,
		// in runtime (stream)
		s_num_valid_candidates_burst_replicated[0], 
		s_valid_candidates_replicated[0], 
		s_finish_query_replicate_candidates,
		
		// out (stream)
		s_fetched_vectors,
		s_finish_query_fetch_vectors
	);

    hls::stream<result_t> s_distances; 
#pragma HLS stream variable=s_distances depth=512

    hls::stream<int> s_finish_query_compute_distances; // finish all queries
#pragma HLS stream variable=s_finish_query_compute_distances depth=16

	compute_distances(
		// in initialization
		query_num,
		d,
		// in runtime (stream)
		s_query_vectors,
		s_num_valid_candidates_burst_replicated[1], 
		s_fetched_vectors,
		s_valid_candidates_replicated[1],
		s_finish_query_fetch_vectors,
		
		// out (stream)
		s_distances,
		s_finish_query_compute_distances
	);

	hls::stream<result_t> s_distances_base_level;
#pragma HLS stream variable=s_distances_base_level depth=512

	hls::stream<int> s_finish_query_replicate_distances; // finish all queries
#pragma HLS stream variable=s_finish_query_replicate_distances depth=16

	split_s_distances(
		// in (initialization)
		query_num,
		// in runtime (stream)
		s_distances,
		s_finish_query_compute_distances,

		// out (stream)
		s_distances_upper_levels,
		s_distances_base_level,
		s_finish_query_replicate_distances
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
		s_finish_query_replicate_distances,

		// out (stream)
		s_inserted_candidates,
		s_num_inserted_candidates,
		s_largest_result_queue_elements,
		s_finish_query_results_collection,

		// out (DRAM)
		out_id,
		out_dist
	);

}

}
