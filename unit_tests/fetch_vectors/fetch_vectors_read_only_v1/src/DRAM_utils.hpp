#pragma once

#include "types.hpp"

void fetch_vectors(
	// in initialization
	const int query_num,
	const int d,
	// in runtime (should from DRAM)
	ap_uint<512>* db_vectors,
	// in runtime (stream)
	hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<ap_uint<512>>& s_fetched_vectors,
	hls::stream<int>& s_finish_query_out
) {

	const int AXI_num_per_vector_and_padding = d % FLOAT_PER_AXI == 0? 
		d / FLOAT_PER_AXI + 1 : d / FLOAT_PER_AXI + 2; // 16 for d = 512 + visited padding
	const int AXI_num_per_vector_only = d % FLOAT_PER_AXI == 0? 
		d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; // 16 for d = 512

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_fetched_neighbor_ids_replicated.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_fetched_neighbor_ids_replicated.empty()) {
				// receive task
				cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
				int node_id = reg_cand.node_id;
				int level_id = reg_cand.level_id;

				// read vectors
				int start_addr = node_id * AXI_num_per_vector_and_padding;
				for (int i = 0; i < AXI_num_per_vector_only; i++) {
				#pragma HLS pipeline II=1
					ap_uint<512> vector_AXI = db_vectors[start_addr + i];
					s_fetched_vectors.write(vector_AXI);
				}
			}
		}
	}
}