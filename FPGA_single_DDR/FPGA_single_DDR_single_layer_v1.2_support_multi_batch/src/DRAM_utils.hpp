#pragma once

#include "types.hpp"
#include "priority_queue.hpp"


void read_queries(
	// in initialization
	const int query_num, 
	const int query_batch_size,

    // in runtime (from DRAM)
	const int* entry_point_ids,
	const ap_uint<512>* query_vectors,

	// in streams
	hls::stream<int>& s_finish_batch,

	// out streams
	hls::stream<int>& s_query_batch_size,
	hls::stream<ap_uint<512>>& s_query_vectors_in,
	hls::stream<int>& s_entry_point_ids
) {

	const int vec_AXI_num = D % FLOAT_PER_AXI == 0? D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; 

	int remained_query_num = query_num;
	int processed_query_num = 0;

	while (remained_query_num > 0) {
		int current_query_batch_size = remained_query_num > query_batch_size? query_batch_size : remained_query_num;
		s_query_batch_size.write(current_query_batch_size);
		for (int i = 0; i < current_query_batch_size; i++) {
			int qid = processed_query_num + i;
			for (int j = 0; j < vec_AXI_num; j++) {
			#pragma HLS pipeline II=1
				ap_uint<512> query_vector_AXI = query_vectors[qid * vec_AXI_num + j];
				s_query_vectors_in.write(query_vector_AXI);
			}
			s_entry_point_ids.write(entry_point_ids[qid]);
		}
		remained_query_num -= query_batch_size;
		processed_query_num += query_batch_size;

		while (s_finish_batch.empty()) {}
		int finish_batch = s_finish_batch.read();
	}

	// write finish all 
	s_query_batch_size.write(-1);
}

void write_results(
	// in initialization
	const int ef,
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int>& s_out_ids,
	hls::stream<float>& s_out_dists,
	hls::stream<int>& s_debug_signals,

	// out streams
	hls::stream<int>& s_finish_batch,

	// out (DRAM)
    int* out_id,
	float* out_dist,
	int* mem_debug
) {

	bool first_s_query_batch_size = true;
	bool first_iter_s_out_ids = true;
	bool first_iter_s_out_dists = true;
	bool first_iter_s_debug_signals = true;

	int processed_query_num = 0;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

		for (int qid = 0; qid < query_num; qid++) {

			wait_data_fifo_first_iter<int>(
				ef, s_out_ids, first_iter_s_out_ids);
			wait_data_fifo_first_iter<float>(
				ef, s_out_dists, first_iter_s_out_dists);

			// use two loops to infer burst per loop
			for (int i = 0; i < ef; i++) {
			#pragma HLS pipeline II=1
				int start_addr = (processed_query_num + qid) * ef + i;
				out_id[start_addr] = s_out_ids.read();
			}

			for (int i = 0; i < ef; i++) {
			#pragma HLS pipeline II=1
				int start_addr = (processed_query_num + qid) * ef + i;
				out_dist[start_addr] = s_out_dists.read();
			}

			wait_data_fifo_first_iter<int>(
				debug_size, s_debug_signals, first_iter_s_debug_signals);

			for (int i = 0; i < debug_size; i++) {
			#pragma HLS pipeline II=1
				int start_addr = (processed_query_num + qid) * debug_size + i;
				mem_debug[start_addr] = s_debug_signals.read();
			}
		}
		processed_query_num += query_num;
		
		// finish processing this entire batch
		s_finish_batch.write(1);
	}
}


void fetch_neighbor_ids(
	// in initialization
	const int max_link_num_base,
	// in runtime (should from DRAM)
	const ap_uint<512>* links_base,
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<cand_t>& s_top_candidates,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<cand_t>& s_fetched_neighbor_ids,
	hls::stream<int>& s_finish_query_out
) {

	const int AXI_num_per_base_link = max_link_num_base % INT_PER_AXI == 0? 
		1 + max_link_num_base / INT_PER_AXI : 2 + max_link_num_base / INT_PER_AXI; // 4 = int size, 64 = 512 bit
	bool first_s_query_batch_size = true;

	const int max_buffer_size = 32 + 1; // supporting max of 32 * INT_PER_AXI (16) = 512 edges per node
			ap_uint<512> local_links_buffer[max_buffer_size]; 
#pragma HLS bind_storage variable=local_links_buffer type=RAM_2P impl=BRAM
	
	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

		for (int qid = 0; qid < query_num; qid++) {

			// bool is_entry_point = true; // first base layer task

			while (true) {

				// check query finish
				if (!s_finish_query_in.empty() && s_top_candidates.empty()) {
					s_finish_query_out.write(s_finish_query_in.read());
					break;
				} else if (!s_top_candidates.empty()) {
					// receive task
					cand_t reg_cand = s_top_candidates.read();
					int node_id = reg_cand.node_id;
					int level_id = reg_cand.level_id;
					bool send_node_itself = false;

					ap_uint<64> start_addr = start_addr = node_id * AXI_num_per_base_link;
					// first 64-byte = header (4 byte num links + 60 byte padding)
					// then we have the links (4 byte each, total number = max_link_num)
					for (int i = 0; i < AXI_num_per_base_link; i++) {
					#pragma HLS pipeline II=1
						local_links_buffer[i] = links_base[start_addr + i];
					}
					// if (is_entry_point) {
					// 	send_node_itself = true;
					// 	is_entry_point = false;
					// }

					// write out links num & links id
					ap_uint<32> links_num_ap = local_links_buffer[0].range(31, 0);
					int num_links = links_num_ap;
					// if (send_node_itself) {
					// 	s_num_neighbors_base_level.write(num_links + 1);
					// } else {
						s_num_neighbors_base_level.write(num_links);
					// }
						
					for (int i = 0; i < AXI_num_per_base_link - 1; i++) { // first one is the num_links
						for (int j = 0; j < INT_PER_AXI && i * INT_PER_AXI + j < num_links; j++) {
						#pragma HLS pipeline II=1
							ap_uint<32> link_ap = local_links_buffer[i + 1].range(32 * (j + 1) - 1, 32 * j);
							int link = link_ap;
							cand_t reg_neighbor;
							reg_neighbor.node_id = link;
							reg_neighbor.level_id = level_id;
							s_fetched_neighbor_ids.write(reg_neighbor);
						}
					}
					// if (send_node_itself) {
					// 	cand_t reg_node_itself;
					// 	reg_node_itself.node_id = node_id;
					// 	reg_node_itself.level_id = level_id;
					// 	s_fetched_neighbor_ids.write(reg_node_itself);
					// }
				}
			}
		}
	}
}

void fetch_vectors(
	// in runtime (should from DRAM)
	ap_uint<512>* db_vectors,
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	hls::stream<int>& s_fetch_batch_size,
	hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
	hls::stream<int>& s_finish_query_in,
	
	// out (stream)
	hls::stream<ap_uint<512>>& s_fetched_vectors,
	hls::stream<int>& s_finish_query_out
) {

	const int AXI_num_per_vector_and_padding = D % FLOAT_PER_AXI == 0? 
		D / FLOAT_PER_AXI + 1 : D / FLOAT_PER_AXI + 2; // 16 for D = 512 + visited padding

	const int AXI_num_per_vector_only = D % FLOAT_PER_AXI == 0? 
		D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; // 16 for D = 512

	bool first_s_query_batch_size = true;
	bool first_iter_s_fetched_neighbor_ids_replicated = true;

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
				if (!s_finish_query_in.empty() && s_fetch_batch_size.empty() && s_fetched_neighbor_ids_replicated.empty()) {
					s_finish_query_out.write(s_finish_query_in.read());
					break;
				} else if (!s_fetch_batch_size.empty()) {
					int fetch_batch_size = s_fetch_batch_size.read();
					wait_data_fifo_first_iter<cand_t>(
						fetch_batch_size, s_fetched_neighbor_ids_replicated, first_iter_s_fetched_neighbor_ids_replicated);
					
					for (int bid = 0; bid < fetch_batch_size; bid++) {
					#pragma HLS pipeline // put the pipeline here so hopefully Vitis can handle prefetching automatically
						if (s_finish_query_in.empty() && !s_fetch_batch_size.empty()) {
							// no need to check wait_data_fifo_first_iter, because if this loop is executed, then first iter already passed
							fetch_batch_size += s_fetch_batch_size.read();
						}
						// receive task & read vectors
						cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
						int start_addr = reg_cand.node_id * AXI_num_per_vector_and_padding;
						for (int i = 0; i < AXI_num_per_vector_only; i++) {
							ap_uint<512> vector_AXI = db_vectors[start_addr + i];
							s_fetched_vectors.write(vector_AXI);
						}
					}
				}
			}
		}
	}
}

void results_collection(
	// in (initialization)
	const int ef,
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size, // -1: stop
	// hls::stream<result_t>& s_entry_point_base_level,
	hls::stream<int>& s_cand_batch_size, 
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<result_t>& s_inserted_candidates,
	hls::stream<int>& s_num_inserted_candidates,
	hls::stream<float>& s_largest_result_queue_elements,
	hls::stream<int>& s_debug_num_vec_base_layer,
	hls::stream<int>& s_finish_query_out,
	hls::stream<int>& s_out_ids,
	hls::stream<float>& s_out_dists
) {

	Priority_queue<result_t, hardware_result_queue_size, Collect_smallest> result_queue(ef);
	const int sort_swap_round = ef % 2 == 0? ef / 2 : ef / 2 + 1;

	bool first_s_query_batch_size = true;
	bool first_iter_s_distances_base_level = true;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

		for (int qid = 0; qid < query_num; qid++) {

			result_queue.reset_queue(); // reset content to large_float
			// int effect_queue_size = 0; // number of results in queue
			// while (s_entry_point_base_level.empty()) {}
			// result_t entry_point = s_entry_point_base_level.read();
			// result_queue.queue[ef - 1] = entry_point;

			int debug_num_vec_base_layer = 0;

			while (true) {
				// check query finish
				if (!s_finish_query_in.empty() && s_cand_batch_size.empty() 
					&& s_num_neighbors_base_level.empty() && s_distances_base_level.empty()) {
					// volatile int reg_finish = s_finish_query_in.read();
					s_debug_num_vec_base_layer.write(debug_num_vec_base_layer);
					s_finish_query_out.write(s_finish_query_in.read());
					break;
				} else if (!s_cand_batch_size.empty() && !s_num_neighbors_base_level.empty()) {

					int cand_batch_size = s_cand_batch_size.read();

					for (int bid = 0; bid < cand_batch_size; bid++) {
						int num_neighbors = s_num_neighbors_base_level.read();
						wait_data_fifo_first_iter<result_t>(
							num_neighbors, s_distances_base_level, first_iter_s_distances_base_level);
						debug_num_vec_base_layer += num_neighbors;

						int inserted_num_this_iter = 0;

						if (num_neighbors > 0) {
						// insert new values but without sorting
							for (int i = 0; i < num_neighbors; i++) {
							#pragma HLS pipeline II=1
								result_t reg = s_distances_base_level.read();
								// if both input & queue element are large_float, then do not insert
								if (reg.dist < result_queue.queue[0].dist) {
									result_queue.queue[0] = reg;
									s_inserted_candidates.write(reg);
									inserted_num_this_iter++;
								}
								result_queue.compare_swap_array_step_A();
								result_queue.compare_swap_array_step_B();
							}
							s_num_inserted_candidates.write(inserted_num_this_iter);
						} else { // num_neighbors == 0
							s_num_inserted_candidates.write(0);
						}
					}

					// sorting
					for (int i = 0; i < sort_swap_round; i++) {
	#pragma HLS pipeline II=1
						result_queue.compare_swap_array_step_A();
						result_queue.compare_swap_array_step_B();
					}

					// send out largest dist in the queue:
					//   if the queue is not full, always consider the candidate
					//   if the queue is full, only consider the candidate when it is smaller than the largest element in the queue
					s_largest_result_queue_elements.write(result_queue.queue[0].dist);

					// effect_queue_size = effect_queue_size + inserted_num_this_iter < ef? effect_queue_size + inserted_num_this_iter : ef;
					// int largest_element_position = ef - effect_queue_size;
					// s_largest_result_queue_elements.write(result_queue.queue[largest_element_position].dist);
				}
			}

			// write results back 
			for (int i = 0; i < ef; i++) {
	#pragma HLS pipeline II=1
				s_out_ids.write(result_queue.queue[ef - 1 - i].node_id);
				s_out_dists.write(result_queue.queue[ef - 1 - i].dist);
			}
		}
	}
}