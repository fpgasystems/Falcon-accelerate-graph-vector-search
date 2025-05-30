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

template <const int rep_factor>
void replicate_s_query_batch_size(
	// in (stream)
	hls::stream<int>& s_query_batch_size_in, // -1: stop
	// out stream
	hls::stream<int> (&s_query_batch_size_out)[rep_factor]
) {

	bool first_s_query_batch_size = true;

	bool stop_all = false;
	while (!stop_all) {
		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size_in, first_s_query_batch_size);
		int query_num = s_query_batch_size_in.read();

		// even if -1, send out to stop all
		for (int i = 0; i < rep_factor; i++) {
		#pragma HLS UNROLL
			s_query_batch_size_out[i].write(query_num);
		}

		if (query_num == -1) {
			stop_all = true;
		}
	}
}

template <const int rep_factor, typename s_control_t>
void replicate_s_control(
	// in (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<s_control_t>& s_control, // must not be s_query_batch_size
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<s_control_t> (&s_control_replicated)[rep_factor], 
	hls::stream<int>& s_finish_query_out
) {

	bool first_s_query_batch_size = true;
	
	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

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
}

template <const int rep_factor, typename s_data_t>
void replicate_s_read_iter_and_s_data(
	// in (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int>& s_read_iter,
	hls::stream<s_data_t>& s_data,
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<int> (&s_read_iter_replicated)[rep_factor], 
	hls::stream<s_data_t> (&s_data_replicated)[rep_factor],
	hls::stream<int>& s_finish_query_out
) {
	
	bool first_s_query_batch_size = true;
	bool first_iter_s_data = true;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

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
}

template <const int rep_factor>
void replicate_s_finish(
	// in (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<int> (&s_finish_query_out_replicated)[rep_factor]
) {
	
	bool first_s_query_batch_size = true;
	
	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

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
}

template <const int rep_factor>
void gather_s_finish(
	// in (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int> (&s_finish_query_in_replicated)[rep_factor],
	
	// out (stream)
	hls::stream<int>& s_finish_query_out
) {
	bool first_s_query_batch_size = true;	
	
	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

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
}

void filter_computed_distances(
	// in stream
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int>& s_cand_batch_size,
	hls::stream<int>& s_num_valid_candidates_base_level_total, 
	hls::stream<float>& s_largest_result_queue_elements, 
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_in,

	// out stream
	hls::stream<int>& s_num_valid_candidates_base_level_filtered, 
	hls::stream<result_t> &s_distances_base_filtered,
	hls::stream<int>& s_finish_query_out
) {
	
	bool first_s_query_batch_size = true;
	bool first_s_largest_result_queue_elements = true;
	bool first_s_num_valid_candidates_base_level_total = true;
	bool first_s_distances_base_level = true;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

		for (int qid = 0; qid < query_num; qid++) {

			bool query_first_iter = true; // first iteration has no s_largest_result_queue_elements input

			while (true) {
				if (!s_finish_query_in.empty() && s_cand_batch_size.empty() 
					&& s_num_valid_candidates_base_level_total.empty() && s_distances_base_level.empty()) {
					// Note: here the last iteration of s_largest_result_queue_elements must be consumed
					s_largest_result_queue_elements.read();
					s_finish_query_out.write(s_finish_query_in.read());
					break;
				} else if (!s_cand_batch_size.empty()) {

					int cand_batch_size = s_cand_batch_size.read();

					// get largest_result_queue_elements every candidate batch
					float largest_result_queue_elements;
					if (query_first_iter) { // first iteration has no s_largest_result_queue_elements input
						largest_result_queue_elements = large_float;
						query_first_iter = false;
					} else {
						wait_data_fifo_first_iter<float>(
							1, s_largest_result_queue_elements, first_s_largest_result_queue_elements);
						largest_result_queue_elements = s_largest_result_queue_elements.read();
					}

					for (int bid = 0; bid < cand_batch_size; bid++) {
						wait_data_fifo_first_iter<int>(
							cand_batch_size, s_num_valid_candidates_base_level_total, first_s_num_valid_candidates_base_level_total);
						int num_valid_candidates_base_level_total = s_num_valid_candidates_base_level_total.read();

						wait_data_fifo_first_iter<result_t>(
							num_valid_candidates_base_level_total, s_distances_base_level, first_s_distances_base_level);

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
	}
}


// Send only upper-level results to results collection; replicate to other levels
void split_s_distances(
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<result_t>& s_distances_filtered,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<result_t> &s_distances_upper_levels,
	hls::stream<result_t> &s_distances_base_level,
	hls::stream<int>& s_finish_query_out
) {

	bool first_s_query_batch_size = true;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}
		
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
}

