#pragma once

#include "bloom_filter.hpp"
#include "compute.hpp"
#include "DRAM_utils.hpp"
#include "types.hpp"
#include "utils.hpp"

void split_upper_base_inputs(
	const int query_num,

	// in streams
	hls::stream<int>& s_num_neighbors_upper_levels,
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<cand_t>& s_fetched_neighbor_ids,
	hls::stream<int>& s_finish_query_in,

	// out streams
	hls::stream<int>& s_num_valid_candidates_burst,
	hls::stream<int>& s_num_valid_candidates_upper_levels_total,
	hls::stream<int>& s_num_valid_candidates_base_level_total,
	hls::stream<cand_t>& s_valid_candidates,
	hls::stream<int>& s_finish_split
) {
	bool first_iter_s_fetched_neighbor_ids = true;
	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			if (!s_finish_query_in.empty() && s_num_neighbors_upper_levels.empty()
				&& s_num_neighbors_base_level.empty() && s_fetched_neighbor_ids.empty()) {
				int finish = s_finish_query_in.read();
				s_finish_split.write(finish);
				break;
			} else if (!s_num_neighbors_upper_levels.empty()) {
				int num_neighbors_upper_levels = s_num_neighbors_upper_levels.read();
				wait_data_fifo_first_iter<cand_t>(
					num_neighbors_upper_levels, s_fetched_neighbor_ids, first_iter_s_fetched_neighbor_ids);
				s_num_valid_candidates_burst.write(num_neighbors_upper_levels);
				s_num_valid_candidates_upper_levels_total.write(num_neighbors_upper_levels);
				for (int i = 0; i < num_neighbors_upper_levels; i++) {
					cand_t reg_cand = s_fetched_neighbor_ids.read();
					s_valid_candidates.write(reg_cand);
				}
			} else if (!s_num_neighbors_base_level.empty()) {
				int num_neighbors_base_level = s_num_neighbors_base_level.read();
				wait_data_fifo_first_iter<cand_t>(
					num_neighbors_base_level, s_fetched_neighbor_ids, first_iter_s_fetched_neighbor_ids);
				s_num_valid_candidates_burst.write(num_neighbors_base_level);
				s_num_valid_candidates_base_level_total.write(num_neighbors_base_level);
				for (int i = 0; i < num_neighbors_base_level; i++) {
					cand_t reg_cand = s_fetched_neighbor_ids.read();
					s_valid_candidates.write(reg_cand);
				}
			}
		}
	}
}


void dummy_compute_distances(
    // in initialization
    const int query_num,
    const int d,
    // in runtime (stream)
    hls::stream<ap_uint<512>>& s_query_vectors, 
    hls::stream<int>& s_fetch_batch_size_replicated,
    hls::stream<ap_uint<512>>& s_fetched_vectors,
    hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
    hls::stream<int>& s_finish_query_in,
    
    // out (stream)
    hls::stream<result_t>& s_distances,
    hls::stream<int>& s_finish_query_out
) {
	const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 

    bool first_iter_s_query_vectors = true;
	bool first_iter_s_fetched_vectors = true;
	bool first_iter_s_fetched_neighbor_ids_replicated = true;

    for (int qid = 0; qid < query_num; qid++) {

        // read query vector
		wait_data_fifo_first_iter<ap_uint<512>>(
			vec_AXI_num, s_query_vectors, first_iter_s_query_vectors);
        for (int i = 0; i < vec_AXI_num; i++) {
        #pragma HLS pipeline II=1
            ap_uint<512> query_reg = s_query_vectors.read();
        }

        while (true) {
            
            if (!s_finish_query_in.empty() && s_fetch_batch_size_replicated.empty() 
				&& s_fetched_vectors.empty() && s_fetched_neighbor_ids_replicated.empty()) {
                s_finish_query_out.write(s_finish_query_in.read());
                break;
            } else if (!s_fetch_batch_size_replicated.empty()) {

                int fetch_batch_size = s_fetch_batch_size_replicated.read();
				wait_data_fifo_first_iter<ap_uint<512>>(
					fetch_batch_size, s_fetched_vectors, first_iter_s_fetched_vectors);
				wait_data_fifo_first_iter<cand_t>(
					fetch_batch_size, s_fetched_neighbor_ids_replicated, first_iter_s_fetched_neighbor_ids_replicated);

                for (int b = 0; b < fetch_batch_size; b++) {
					ap_uint<512> db_vec_reg;
					cand_t cand = s_fetched_neighbor_ids_replicated.read();
                    for (int i = 0; i < vec_AXI_num; i++) {
                    #pragma HLS pipeline II=1
                        db_vec_reg = s_fetched_vectors.read();
                    }
					float out = cand.node_id + db_vec_reg.range(31, 0);
					result_t out_reg;
					out_reg.dist = out;
					out_reg.level_id = cand.level_id;
					s_distances.write(out_reg);
                }
            }
        }
    }
}

void fetch_compute(
	// in initialization
	const int query_num, 
	const int d,

    // in runtime (from DRAM)
    hls::burst_maxi<ap_uint<512>>& db_vectors, // need to write visited tag

	// in streams
	hls::stream<ap_uint<512>>& s_query_vectors,
	hls::stream<int>& s_num_neighbors_upper_levels,
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<cand_t>& s_fetched_neighbor_ids,
	hls::stream<int>& s_finish_query_in,

	// out streams
	hls::stream<int>& s_num_valid_candidates_upper_levels_total,
	hls::stream<int>& s_num_valid_candidates_base_level_total,
	hls::stream<result_t>& s_distances_upper_levels,
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_out
) {

#pragma HLS dataflow

	hls::stream<int> s_num_valid_candidates_burst;
#pragma HLS stream variable=s_num_valid_candidates_burst depth=16


	hls::stream<cand_t> s_valid_candidates;
#pragma HLS stream variable=s_valid_candidates depth=512

    hls::stream<int> s_finish_split; // finish all queries
#pragma HLS stream variable=s_finish_split depth=16

	split_upper_base_inputs(
		// in (initialization)
		query_num,
		// in (stream)
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_in,
		
		// out (stream)
		s_num_valid_candidates_burst,
		s_num_valid_candidates_upper_levels_total,
		s_num_valid_candidates_base_level_total,
		s_valid_candidates,
		s_finish_split
	);

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
		s_finish_split,
		
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

	dummy_compute_distances(
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
		s_finish_query_out
	);
}