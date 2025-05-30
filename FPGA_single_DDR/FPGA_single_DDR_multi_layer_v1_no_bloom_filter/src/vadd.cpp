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
	const int d,
	const int max_level,
    const int max_link_num_upper, 
	const int max_link_num_base,
	const int entry_point_id,
	// in initialization (from DRAM)
	const ap_uint<512>* entry_vector, 

    // in runtime (from DRAM)
	const ap_uint<512>* query_vectors,
    ap_uint<512>* db_vectors, // need to write visited tag

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

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=query_vectors offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=entry_vector offset=slave bundle=gmem0 // share the same AXI interface with query_vectors
#pragma HLS INTERFACE m_axi port=db_vectors offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=ptr_to_upper_links offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=links_upper offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=links_base offset=slave bundle=gmem2
// // for debugging use, seperate the bundles
// #pragma HLS INTERFACE m_axi port=ptr_to_upper_links offset=slave bundle=gmem1
// #pragma HLS INTERFACE m_axi port=links_upper offset=slave bundle=gmem2
// #pragma HLS INTERFACE m_axi port=links_base offset=slave bundle=gmem3

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

    hls::stream<cand_t> s_top_candidates; // current top candidates
#pragma HLS stream variable=s_top_candidates depth=512

    hls::stream<int> s_num_neighbors_upper_levels; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_upper_levels depth=512

	hls::stream<int> s_num_inserted_candidates;
#pragma HLS stream variable=s_num_inserted_candidates depth=512

	hls::stream<result_t> s_inserted_candidates;
#pragma HLS stream variable=s_inserted_candidates depth=512

	hls::stream<result_t> s_distances_upper_levels;
#pragma HLS stream variable=s_distances_upper_levels depth=512

	hls::stream<float> s_largest_result_queue_elements;
#pragma HLS stream variable=s_largest_result_queue_elements depth=512	
	
	// controls the traversal and maintains the candidate queue
	task_scheduler(
		query_num, 
		candidate_queue_runtime_size,
		d, 
		max_level,
    	max_link_num_upper, 
		max_link_num_base,
		entry_point_id,
    	// in runtime (should from DRAM)
		entry_vector,
		query_vectors,

		// in streams
		s_num_neighbors_upper_levels,
		s_distances_upper_levels,
		s_num_inserted_candidates,
		s_inserted_candidates,
		s_largest_result_queue_elements,
		s_finish_query_results_collection,
		
		// out streams
		s_query_vectors,
		// s_entry_point_base_level,
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
		s_fetch_batch_size,
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids
	);

	const int replication_factor_s_fetched_neighbor_ids = 2;

    hls::stream<cand_t> s_fetched_neighbor_ids_replicated[replication_factor_s_fetched_neighbor_ids]; 
#pragma HLS stream variable=s_fetched_neighbor_ids_replicated depth=512

    hls::stream<int> s_finish_query_replicate_s_fetched_neighbor_ids; // finish all queries
#pragma HLS stream variable=s_finish_query_replicate_s_fetched_neighbor_ids depth=16

	replicate_s_control<replication_factor_s_fetched_neighbor_ids, cand_t>(
		// in (initialization)
		query_num,
		// in (stream)
		s_fetched_neighbor_ids,
		s_finish_query_fetch_neighbor_ids,
		
		// out (stream)
		s_fetched_neighbor_ids_replicated, 
		s_finish_query_replicate_s_fetched_neighbor_ids
	);


    hls::stream<ap_uint<512>> s_fetched_vectors; 
#pragma HLS stream variable=s_fetched_vectors depth=512

    hls::stream<bool> s_fetch_valid; // if visited, not valid
#pragma HLS stream variable=s_fetch_valid depth=512

    hls::stream<int> s_finish_query_fetch_vectors; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_vectors depth=16

	fetch_vectors(
		// in initialization
		query_num,
		d,
		// in runtime (should from DRAM)
    	db_vectors,
		// in runtime (stream)
		s_fetched_neighbor_ids_replicated[0],
		s_finish_query_replicate_s_fetched_neighbor_ids,
		
		// out (stream)
		s_fetched_vectors,
		s_fetch_valid,
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
		s_fetch_batch_size,
		s_fetched_vectors,
		s_fetch_valid,
		s_fetched_neighbor_ids_replicated[1],
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
		s_num_neighbors_base_level,
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
