#pragma once

#include "bloom_filter.hpp"
#include "compute.hpp"
#include "DRAM_utils.hpp"
#include "types.hpp"
#include "utils.hpp"

void bloom_fetch_compute(
	// in initialization
	const int query_num, 
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int max_bloom_out_burst_size,

    // in runtime (from DRAM)
    ap_uint<512>* db_vectors,

	// in streams
	hls::stream<ap_uint<512>>& s_query_vectors,
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<cand_t>& s_fetched_neighbor_ids,
	hls::stream<int>& s_finish_query_in,

	// out streams
	hls::stream<int>& s_num_valid_candidates_base_level_total,
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_out
) {

#pragma HLS inline

	hls::stream<int> s_num_valid_candidates_burst;
#pragma HLS stream variable=s_num_valid_candidates_burst depth=16

	hls::stream<cand_t> s_valid_candidates;
#pragma HLS stream variable=s_valid_candidates depth=512

    hls::stream<int> s_finish_bloom; // finish all queries
#pragma HLS stream variable=s_finish_bloom depth=16

	BloomFilter<bloom_num_hash_funs, bloom_num_bucket_addr_bits> 
		bloom_filter(runtime_n_bucket_addr_bits);

	bloom_filter.run_bloom_filter(
		query_num, 
		hash_seed,
		max_bloom_out_burst_size,
		// in streams
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_in,

		// out streams
		s_num_valid_candidates_burst, // one round (s_num_neighbors) can contain multiple bursts
		s_num_valid_candidates_base_level_total, // one round can contain multiple bursts
		s_valid_candidates,
		s_finish_bloom);

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

	hls::stream<ap_uint<512>> s_fetched_vectors; 
#pragma HLS stream variable=s_fetched_vectors depth=512
	
    hls::stream<int> s_finish_query_fetch_vectors; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_vectors depth=16

	fetch_vectors(
		// in initialization
		query_num,
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
		// in runtime (stream)
		s_query_vectors,
		s_num_valid_candidates_burst_replicated[1], 
		s_fetched_vectors,
		s_valid_candidates_replicated[1],
		s_finish_query_fetch_vectors,
		
		// out (stream)
		s_distances_base_level,
		s_finish_query_out
	);
}