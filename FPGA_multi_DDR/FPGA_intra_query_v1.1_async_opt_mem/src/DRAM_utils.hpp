#pragma once

#include "types.hpp"
#include "priority_queue.hpp"

void fetch_neighbor_ids(
	// in initialization
	const int query_num,
	const int max_link_num_base,
	// in runtime (should from DRAM)
		const ap_uint<512>* links_base_chan_0,
#if N_CHANNEL >= 2
		const ap_uint<512>* links_base_chan_1,
#endif
#if N_CHANNEL >= 4
		const ap_uint<512>* links_base_chan_2,
		const ap_uint<512>* links_base_chan_3,
#endif
#if N_CHANNEL >= 8
		const ap_uint<512>* links_base_chan_4,
		const ap_uint<512>* links_base_chan_5,
		const ap_uint<512>* links_base_chan_6,
		const ap_uint<512>* links_base_chan_7,
#endif
#if N_CHANNEL >= 16
		const ap_uint<512>* links_base_chan_8,
		const ap_uint<512>* links_base_chan_9,
		const ap_uint<512>* links_base_chan_10,
		const ap_uint<512>* links_base_chan_11,
		const ap_uint<512>* links_base_chan_12,
		const ap_uint<512>* links_base_chan_13,
		const ap_uint<512>* links_base_chan_14,
		const ap_uint<512>* links_base_chan_15,
#endif
	// in runtime (stream)
	hls::stream<cand_t>& s_top_candidates,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<cand_t>& s_fetched_neighbor_ids,
	hls::stream<int>& s_finish_query_out
) {

	const int AXI_num_per_base_link = max_link_num_base % INT_PER_AXI == 0? 
		1 + max_link_num_base / INT_PER_AXI : 2 + max_link_num_base / INT_PER_AXI; // 4 = int size, 64 = 512 bit

	const int max_buffer_size = 32 + 1; // supporting max of 32 * INT_PER_AXI (16) = 512 edges per node
			ap_uint<512> local_links_buffer[max_buffer_size]; 
#pragma HLS bind_storage variable=local_links_buffer type=RAM_2P impl=BRAM

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
				ap_uint<32> node_id_ap = node_id;
				int level_id = reg_cand.level_id;
				// bool send_node_itself = false;

#if N_CHANNEL == 1
				ap_uint<8> channel_id = 0;
				ap_uint<32> in_channel_node_id = node_id_ap;
#else
				ap_uint<8> channel_id = node_id_ap.range(CHANNEL_ADDR_BITS - 1, 0);
				ap_uint<32> in_channel_node_id = node_id_ap >> CHANNEL_ADDR_BITS;
#endif

				const ap_uint<512>* links_base_selected_channel;
				switch (channel_id) {
					case 0:
						links_base_selected_channel = links_base_chan_0;
						break;
				#if N_CHANNEL >= 2
					case 1:
						links_base_selected_channel = links_base_chan_1;
						break;
				#endif
				#if N_CHANNEL >= 4
					case 2:
						links_base_selected_channel = links_base_chan_2;
						break;
					case 3:
						links_base_selected_channel = links_base_chan_3;
						break;
				#endif
				#if N_CHANNEL >= 8
					case 4:
						links_base_selected_channel = links_base_chan_4;
						break;
					case 5:
						links_base_selected_channel = links_base_chan_5;
						break;
					case 6:
						links_base_selected_channel = links_base_chan_6;
						break;
					case 7:
						links_base_selected_channel = links_base_chan_7;
						break;
				#endif
				#if N_CHANNEL >= 16
					case 8:
						links_base_selected_channel = links_base_chan_8;
						break;
					case 9:
						links_base_selected_channel = links_base_chan_9;
						break;
					case 10:
						links_base_selected_channel = links_base_chan_10;
						break;
					case 11:
						links_base_selected_channel = links_base_chan_11;
						break;
					case 12:
						links_base_selected_channel = links_base_chan_12;
						break;
					case 13:
						links_base_selected_channel = links_base_chan_13;
						break;
					case 14:
						links_base_selected_channel = links_base_chan_14;
						break;
					case 15:
						links_base_selected_channel = links_base_chan_15;
						break;
				#endif
				}

				ap_uint<64> start_addr;
				int read_num;
				if (level_id == 0) { // base layer
					start_addr = in_channel_node_id * AXI_num_per_base_link;
					read_num  = AXI_num_per_base_link;
					// first 64-byte = header (4 byte num links + 60 byte padding)
					// then we have the links (4 byte each, total number = max_link_num)
					for (int i = 0; i < read_num; i++) {
					#pragma HLS pipeline II=1
						local_links_buffer[i] = links_base_selected_channel[start_addr + i];
					}
					// if (is_entry_point) {
					// 	send_node_itself = true;
					// 	is_entry_point = false;
					// }
				} 

				// write out links num & links id
				ap_uint<32> links_num_ap = local_links_buffer[0].range(31, 0);
				int num_links = links_num_ap;
				if (level_id == 0) { // base layer
					// if (send_node_itself) {
					// 	s_num_neighbors_base_level.write(num_links + 1);
					// } else {
						s_num_neighbors_base_level.write(num_links);
					// }
				} 
				for (int i = 0; i < read_num - 1; i++) { // first one is the num_links
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

void fetch_vectors(
	// in initialization
	const int query_num,
	// in runtime (should from DRAM)
	ap_uint<512>* db_vectors,
	// in runtime (stream)
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
				
				for (int bid = 0; bid < fetch_batch_size; bid++) {
				#pragma HLS pipeline // put the pipeline here so hopefully Vitis can handle prefetching automatically
					if (s_finish_query_in.empty() && !s_fetch_batch_size.empty()) {
						// no need to check wait_data_fifo_first_iter, because if this loop is executed, then first iter already passed
						fetch_batch_size += s_fetch_batch_size.read();
					}
					// receive task & read vectors
					cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
					int node_id = reg_cand.node_id;
#if N_CHANNEL == 1
						ap_uint<32> in_channel_node_id = node_id;
#else
						ap_uint<32> in_channel_node_id = node_id >> CHANNEL_ADDR_BITS;
#endif
					int start_addr = in_channel_node_id * AXI_num_per_vector_and_padding;
					for (int i = 0; i < AXI_num_per_vector_only; i++) {
						ap_uint<512> vector_AXI = db_vectors[start_addr + i];
						s_fetched_vectors.write(vector_AXI);
					}
				}
			}
		}
	}
}

void results_collection(
	// in (initialization)
	const int query_num,
	const int ef,
	// in runtime (stream)
	// hls::stream<result_t>& s_entry_point_base_level,
	hls::stream<int>& s_num_neighbors_base_level,
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_in,

	// out (stream)
	hls::stream<result_t>& s_inserted_candidates,
	hls::stream<int>& s_num_inserted_candidates,
	hls::stream<float>& s_largest_result_queue_elements,
	// hls::stream<int>& s_debug_num_vec_base_layer,
	hls::stream<int>& s_finish_query_out,
	
	// out (DRAM)
	int* out_id,
	float* out_dist
) {

	Priority_queue<result_t, hardware_result_queue_size, Collect_smallest> result_queue(ef);
	const int sort_swap_round = ef % 2 == 0? ef / 2 : ef / 2 + 1;

	bool first_iter_s_distances_base_level = true;

	for (int qid = 0; qid < query_num; qid++) {

		result_queue.reset_queue(); // reset content to large_float
		// int effect_queue_size = 0; // number of results in queue
		// while (s_entry_point_base_level.empty()) {}
		// result_t entry_point = s_entry_point_base_level.read();
		// result_queue.queue[ef - 1] = entry_point;

		int debug_num_vec_base_layer = 0;

		while (true) {
			// check query finish
			if (!s_finish_query_in.empty() && s_num_neighbors_base_level.empty() && s_distances_base_level.empty()) {
				// volatile int reg_finish = s_finish_query_in.read();
				// s_debug_num_vec_base_layer.write(debug_num_vec_base_layer);
				s_finish_query_out.write(s_finish_query_in.read());
				break;
			} else if (!s_num_neighbors_base_level.empty()) {

				for (int bid = 0; bid < N_CHANNEL; bid++) {
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

		// write results back to DRAM, write separately to infer burst
		for (int i = 0; i < ef; i++) {
#pragma HLS pipeline II=1
			out_id[qid * ef + i] = result_queue.queue[ef - 1 - i].node_id;
		}
		for (int i = 0; i < ef; i++) {
#pragma HLS pipeline II=1
			out_dist[qid * ef + i] = result_queue.queue[ef - 1 - i].dist;
		}
	}
}