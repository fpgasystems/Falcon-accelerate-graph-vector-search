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
	hls::stream<bool>& s_fetch_valid,
	hls::stream<int>& s_finish_query_out
) {

	const int AXI_num_per_vector = d % FLOAT_PER_AXI == 0? 
		d / FLOAT_PER_AXI + 1 : d / FLOAT_PER_AXI + 2; // 16 for d = 512 + visited padding

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
				int start_addr = node_id * AXI_num_per_vector;
				bool valid;
				for (int i = 0; i < AXI_num_per_vector; i++) {
				#pragma HLS pipeline II=1
					ap_uint<512> vector_AXI = db_vectors[start_addr + i];
					if (i < AXI_num_per_vector - 1) {
						s_fetched_vectors.write(vector_AXI);
					} else {
						ap_uint<32> last_visit_qid = vector_AXI.range(31, 0);
						valid = level_id > 0? true : last_visit_qid != qid;
						s_fetch_valid.write(valid);
					}
				}
				// update the visited qid if needed, only mark visited on the ground layer
				if (valid && level_id == 0) {
					ap_uint<32> new_last_visit_qid = qid;
					ap_uint<512> vector_AXI_updated;
					vector_AXI_updated.range(31, 0) = new_last_visit_qid;	
					db_vectors[start_addr + AXI_num_per_vector - 1] = vector_AXI_updated;
				}
			}
		}
	}
}