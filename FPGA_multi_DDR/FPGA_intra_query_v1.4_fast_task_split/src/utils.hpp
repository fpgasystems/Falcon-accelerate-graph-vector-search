#pragma once

#include "types.hpp"

// wait until data ready and read, please be careful
//   about the using of this function as it can slow down the performance
template<typename T>
inline T block_read(hls::stream<T>& s) {
#pragma HLS inline

    while (s.empty()) {}
    return s.read();
}

template <const int rep_factor, typename T>
bool all_streams_empty(
	// in (stream)
	hls::stream<T> (&s_input)[rep_factor]
) {
#pragma HLS inline

	bool all_empty = true;
	for (int i = 0; i < rep_factor; i++) {
		if (!s_input[i].empty()) {
			all_empty = false;
		}
	}
	return all_empty;
}

template <const int rep_factor, typename T>
bool all_streams_ready(
	// in (stream)
	hls::stream<T> (&s_input)[rep_factor]
) {
#pragma HLS inline

	bool all_ready = true;
	for (int i = 0; i < rep_factor; i++) {
		if (s_input[i].empty()) {
			all_ready = false;
		}
	}
	return all_ready;
}

// in the first iteration, wait for the first data to arrive
template<typename s_data_t>
void wait_data_fifo_first_iter(
	int read_iter,
	hls::stream<s_data_t>& s_data,
	bool& first_iter_s_data
) {
#pragma HLS inline

	if (first_iter_s_data && read_iter > 0) {
		while (s_data.empty()) {}
		first_iter_s_data = false;
	}
}

// in the first iteration, wait for the first data to arrive (a group of FIFOs)
template<const int rep_factor, typename s_data_t>
void wait_data_fifo_group_first_iter(
	int read_iter,
	hls::stream<s_data_t> (&s_data)[rep_factor],
	bool& first_iter_s_data
) {
#pragma HLS inline

	if (first_iter_s_data && read_iter > 0) {
		while (!all_streams_ready<rep_factor, s_data_t>(s_data)) {}
		first_iter_s_data = false;
	}
}

template <const int rep_factor, typename s_control_t>
void replicate_s_control(
	// in (initialization)
	const int query_num,
	// in (stream)
	hls::stream<s_control_t>& s_control,
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<s_control_t> (&s_control_replicated)[rep_factor], 
	hls::stream<int>& s_finish_query_out
) {
	
	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_control.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_control.empty()) {
				// receive task
				s_control_t reg_control = s_control.read();
				for (int i = 0; i < rep_factor; i++) {
				#pragma HLS UNROLL
					s_control_replicated[i].write(reg_control);
				}
			}
		}
	}
}

template <const int rep_factor, typename s_data_t>
void replicate_s_read_iter_and_s_data(
	// in (initialization)
	const int query_num,
	// in (stream)
	hls::stream<int>& s_read_iter,
	hls::stream<s_data_t>& s_data,
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<int> (&s_read_iter_replicated)[rep_factor], 
	hls::stream<s_data_t> (&s_data_replicated)[rep_factor],
	hls::stream<int>& s_finish_query_out
) {
	
	bool first_iter_s_data = true;

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_read_iter.empty() && s_data.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_read_iter.empty()) {
				// receive task
				int read_iter = s_read_iter.read();
				wait_data_fifo_first_iter<s_data_t>(read_iter, s_data, first_iter_s_data);
				for (int i = 0; i < rep_factor; i++) {
				#pragma HLS UNROLL
					s_read_iter_replicated[i].write(read_iter);
				}
				for (int j = 0; j < read_iter; j++) {
				#pragma HLS pipeline II=1
					s_data_t reg_data = s_data.read();
					for (int i = 0; i < rep_factor; i++) {
					#pragma HLS UNROLL
						s_data_replicated[i].write(reg_data);
					}
				}
			}
		}
	}
}

template <const int rep_factor>
void replicate_s_finish(
	// in (initialization)
	const int query_num,
	// in (stream)
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<int> (&s_finish_query_out_replicated)[rep_factor]
) {
	
	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty()) {
				int finish = s_finish_query_in.read();
				for (int i = 0; i < rep_factor; i++) {
				#pragma HLS UNROLL
					s_finish_query_out_replicated[i].write(finish);
				}
				break;
			} 
		}
	}
}

template <const int rep_factor>
void gather_s_finish(
	// in (initialization)
	const int query_num,
	// in (stream)
	hls::stream<int> (&s_finish_query_in_replicated)[rep_factor],
	
	// out (stream)
	hls::stream<int>& s_finish_query_out
) {
	
	for (int qid = 0; qid < query_num; qid++) {
		// wait for all finish signals arrive, read them
		int finish[rep_factor];
#pragma HLS bind_storage variable=finish type=RAM_2P impl=BRAM

		for (int i = 0; i < rep_factor; i++) {
			while (true) {
				if (!s_finish_query_in_replicated[i].empty()) {
					finish[i] = s_finish_query_in_replicated[i].read();
					break;
				}
			}
		}
		// send out
		s_finish_query_out.write(finish[0]);
	}
}

void filter_computed_distances(
	// in (initialization)
	const int query_num,
	// in stream
	hls::stream<int>& s_num_valid_candidates_base_level_total, 
	hls::stream<float>& s_largest_result_queue_elements, 
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_in,

	// out stream
	hls::stream<int>& s_num_valid_candidates_base_level_filtered, 
	hls::stream<result_t> &s_distances_base_filtered,
	hls::stream<int>& s_finish_query_out
) {

	bool first_iter_s_largest_result_queue_elements = true;
	bool first_iter_s_distances_base_level = true;

	for (int qid = 0; qid < query_num; qid++) {
		
		bool query_first_iter = true; // first iteration has no s_largest_result_queue_elements input

		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_num_valid_candidates_base_level_total.empty()
				&& s_distances_base_level.empty()) {
				// Note: here the last iteration of s_largest_result_queue_elements must be consumed
				s_largest_result_queue_elements.read();
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_num_valid_candidates_base_level_total.empty()) {
				// receive task
				int num_valid_candidates_base_level_total = s_num_valid_candidates_base_level_total.read();
				float largest_result_queue_elements;
				if (query_first_iter) { // first iteration has no s_largest_result_queue_elements input
					largest_result_queue_elements = large_float;
					query_first_iter = false;
				} else {
					wait_data_fifo_first_iter<float>(
						1, s_largest_result_queue_elements, first_iter_s_largest_result_queue_elements);
					largest_result_queue_elements = s_largest_result_queue_elements.read();
				}

				wait_data_fifo_first_iter<result_t>(
					num_valid_candidates_base_level_total, s_distances_base_level, first_iter_s_distances_base_level);

				int valid_cnt = 0;
				for (int i = 0; i < num_valid_candidates_base_level_total; i++) {
					result_t reg_out = s_distances_base_level.read();
					if (reg_out.dist < largest_result_queue_elements) {
						s_distances_base_filtered.write(reg_out);
						valid_cnt++;
					}
				}
				s_num_valid_candidates_base_level_filtered.write(valid_cnt);
			}
		}
	}

}


// Send only upper-level results to results collection; replicate to other levels
void split_s_distances(
	// in (initialization)
	const int query_num,
	// in runtime (stream)
	hls::stream<result_t>& s_distances_filtered,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<result_t> &s_distances_upper_levels,
	hls::stream<result_t> &s_distances_base_level,
	hls::stream<int>& s_finish_query_out
) {

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_distances_filtered.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_distances_filtered.empty()) {
				// receive task
				result_t reg_out = s_distances_filtered.read();
				if (reg_out.level_id == 0) {
					s_distances_base_level.write(reg_out);
				} else {
					s_distances_upper_levels.write(reg_out);
				}
			}
		}
	}
}


// split bloom-fetch-compute tasks to different channels based on node ids
//   Merging multiple candidate batches into ONE!
//   Note: cannot handle case where output per channel of multi-batches > FIFO size
void split_tasks_to_channels(
		const int query_num,
		const int max_link_num_base,

		// in streams
		hls::stream<int>& s_cand_batch_size,
		hls::stream<ap_uint<512>>& s_neighbor_ids_raw,
		hls::stream<int>& s_finish_query_in,

		// out streams
		hls::stream<int> (&s_num_neighbors_base_level_per_channel)[N_CHANNEL],
		hls::stream<cand_t> (&s_fetched_neighbor_ids_per_channel)[N_CHANNEL],
		hls::stream<int>& s_finish_query_out
	) {

	bool first_iter_s_num_neighbors_base_level = true;
	bool first_iter_s_fetched_neighbor_ids = true;

	int node_count_per_channel[N_CHANNEL];
#pragma HLS array_partition variable=node_count_per_channel complete
	
	const int AXI_num_per_base_link = max_link_num_base % INT_PER_AXI == 0? 
		1 + max_link_num_base / INT_PER_AXI : 2 + max_link_num_base / INT_PER_AXI; // 4 = int size, 64 = 512 bit

	const int max_buffer_size = 32 + 1; // supporting max of 32 * INT_PER_AXI (16) = 512 edges per node
			ap_uint<512> local_links_buffer[max_buffer_size]; 
#pragma HLS bind_storage variable=local_links_buffer type=RAM_2P impl=BRAM

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_cand_batch_size.empty()
				&& s_neighbor_ids_raw.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_cand_batch_size.empty()) {

				// reset counters
				for (int i = 0; i < N_CHANNEL; i++) {
				#pragma HLS UNROLL
					node_count_per_channel[i] = 0;
				}

				// receive task
				int cand_batch_size = s_cand_batch_size.read();
				wait_data_fifo_first_iter<ap_uint<512>>(
					cand_batch_size, s_neighbor_ids_raw, first_iter_s_num_neighbors_base_level);

				// each round can contain multiple batches
				for (int bid = 0; bid < cand_batch_size; bid++) {
					for (int i = 0; i < AXI_num_per_base_link; i++) {
					#pragma HLS pipeline II=1
						ap_uint<512> reg = s_neighbor_ids_raw.read();
						local_links_buffer[i] = reg;
					}

					ap_uint<32> links_num_ap = local_links_buffer[0].range(31, 0);
					int num_links = links_num_ap;

					// process each node's laber
					for (int i = 0; i < AXI_num_per_base_link - 1; i++) { // first one is the num_links
						for (int j = 0; j < INT_PER_AXI && i * INT_PER_AXI + j < num_links; j++) {
						#pragma HLS pipeline II=1
							ap_uint<32> link_ap = local_links_buffer[i + 1].range(32 * (j + 1) - 1, 32 * j);
							int link = link_ap;
							cand_t fetched_neighbor_ids;
							fetched_neighbor_ids.node_id = link;
							fetched_neighbor_ids.level_id = 0; // hard-code to 0, support only base layer
							ap_uint<32> node_id_ap = link;
#if N_CHANNEL == 1
							ap_uint<8> channel_id = 0;
#else
							ap_uint<8> channel_id = node_id_ap.range(CHANNEL_ADDR_BITS - 1, 0);
#endif
							s_fetched_neighbor_ids_per_channel[channel_id].write(fetched_neighbor_ids);
							node_count_per_channel[channel_id]++;
						}
					}
				}

				// merge multiple cand batches into one
				for (int channel_id = 0; channel_id < N_CHANNEL; channel_id++) {
				#pragma HLS UNROLL
					s_num_neighbors_base_level_per_channel[channel_id].write(node_count_per_channel[channel_id]);
				}
			}
		}
	}
}

void gather_distances_from_channels(
		const int query_num,
		hls::stream<int> (&s_num_valid_candidates_base_level_total_per_channel)[N_CHANNEL],
		hls::stream<result_t> (&s_distances_base_level_per_channel)[N_CHANNEL],
		hls::stream<int>& s_finish_query_in,

		hls::stream<int>& s_num_valid_candidates_base_level_total, // write once per channel
		hls::stream<result_t>& s_distances_base_level,
		hls::stream<int>& s_finish_query_out
	) {

	bool first_iter_s_distances_base_level_per_channel[N_CHANNEL];
	for (int i = 0; i < N_CHANNEL; i++) {
		first_iter_s_distances_base_level_per_channel[i] = true;
	}
	bool visited_channel[N_CHANNEL];

	for (int qid = 0; qid < query_num; qid++) {
		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && all_streams_empty<N_CHANNEL, int>(s_num_valid_candidates_base_level_total_per_channel)
				&& all_streams_empty<N_CHANNEL, result_t>(s_distances_base_level_per_channel)) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!all_streams_empty<N_CHANNEL, int>(s_num_valid_candidates_base_level_total_per_channel)) {

				// start a round of pulling data of each channel
				for (int i = 0; i < N_CHANNEL; i++) {
					visited_channel[i] = false;
				}
				bool stop = false;

				// loop over each channel once and forward the data
				while (!stop) {
					for (int channel_id = 0; channel_id < N_CHANNEL; channel_id++) {
						if (!visited_channel[channel_id] && !s_num_valid_candidates_base_level_total_per_channel[channel_id].empty()) {
							int num_valid_candidates_base_level_total_per_channel = 
								s_num_valid_candidates_base_level_total_per_channel[channel_id].read();
							visited_channel[channel_id] = true;
							wait_data_fifo_first_iter<result_t>(
								num_valid_candidates_base_level_total_per_channel, s_distances_base_level_per_channel[channel_id], 
								first_iter_s_distances_base_level_per_channel[channel_id]);
							s_num_valid_candidates_base_level_total.write(num_valid_candidates_base_level_total_per_channel);
							for (int i = 0; i < num_valid_candidates_base_level_total_per_channel; i++) {
							#pragma HLS pipeline II=1
								result_t reg = s_distances_base_level_per_channel[channel_id].read();
								s_distances_base_level.write(reg);
							}
						}
					}

					// stop if all channels are visited
					stop = true;
					for (int i = 0; i < N_CHANNEL; i++) {
						if (!visited_channel[i]) {
							stop = false;
						}
					}
				}
			}
		}
	}
}