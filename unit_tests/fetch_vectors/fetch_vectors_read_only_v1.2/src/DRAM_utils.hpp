#pragma once

#include "types.hpp"
#include "utils.hpp"

void fetch_vectors(
	// in initialization
	const int query_num,
	const int d,
	// in runtime (should from DRAM)
	hls::burst_maxi<ap_uint<512>>& db_vectors,
	// in runtime (stream)
	hls::stream<int>& s_fetch_batch_size,
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


	const int max_prefetch_request = 16;

	bool first_iter_s_fetched_neighbor_ids_replicated = true;

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_fetch_batch_size.empty() && s_fetched_neighbor_ids_replicated.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_fetch_batch_size.empty()) {
				int fetch_batch_size = s_fetch_batch_size.read();
				wait_data_fifo_first_iter<cand_t>(
					fetch_batch_size, s_fetched_neighbor_ids_replicated, first_iter_s_fetched_neighbor_ids_replicated);
				
				int current_request_count = 0; // ongoing unfinished prefetch requests
				int total_request_count = 0; // total sent prefetch requests
				int read_finish_count = 0;
				// prefetch-solution ref: https://github.com/WenqiJiang/spatial-join-on-FPGA/tree/main/test_multi_kernel#compare-random-read-performance
				//   and: https://github.com/WenqiJiang/spatial-join-on-FPGA/blob/main/test_multi_kernel/multi_kernel_read_from_mem_3_latency_1/src/vadd.cpp

				while (read_finish_count < fetch_batch_size) {
					
					// send prefetch request if requests are not enough
					while (current_request_count < max_prefetch_request && total_request_count < fetch_batch_size) {
						cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
						int start_addr = reg_cand.node_id * AXI_num_per_vector_and_padding;
						db_vectors.read_request(start_addr, AXI_num_per_vector_only);
						current_request_count++;
						total_request_count++;
					}

					// read one vector, so keep time delta between prefetch request and data consumption
					for (int i = 0; i < AXI_num_per_vector_only; i++) {
					#pragma HLS pipeline II=1
						s_fetched_vectors.write(db_vectors.read());
					}
					current_request_count--;
					read_finish_count++;
				}
			}
		}
	}
}