#include "bloom_filter.hpp"
#include "fetch_compute.hpp"
#include "compute.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "scheduler.hpp"
#include "types.hpp"
#include "utils.hpp"

void send_requests(
	const int query_num, 
	const int iter_per_query,
	const int d,
	const int run_upper_levels,
	const int run_base_level,
	
    // in
	const ap_uint<512>* query_vectors,
	ap_uint<32>* mem_keys, 
	hls::stream<int>& s_finish_query_write_memory, // finish the current query

    // out
	hls::stream<ap_uint<512>>& s_query_vectors,
	hls::stream<int>& s_num_neighbors_upper_levels,
	hls::stream<int>& s_num_neighbors_base_level,
    hls::stream<cand_t>& s_fetched_neighbor_ids,
    hls::stream<int>& s_finish_query_send_requests // finish the current query
    ) {

	const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 

	int finish;
	for (int qid = 0; qid < query_num; qid++) {
		if (qid > 0) {
			finish = block_read<int>(s_finish_query_write_memory);
		}
		// write query
		for (int i = 0; i < vec_AXI_num; i++) {
		#pragma HLS pipeline II=1
			ap_uint<512> query_vector_AXI = query_vectors[i];
			s_query_vectors.write(query_vector_AXI);
		}

		if (run_upper_levels) {
		// write upper level
			s_num_neighbors_upper_levels.write(iter_per_query);
			for (int i = 0; i < iter_per_query; i++) {
			#pragma HLS pipeline II=1
				int reg_key = mem_keys[i];
				cand_t reg_cand = {reg_key, 1};
				s_fetched_neighbor_ids.write(reg_cand);
			}
		}
		if (run_base_level) {
			// write base level
			s_num_neighbors_base_level.write(iter_per_query);
			for (int i = 0; i < iter_per_query; i++) {
			#pragma HLS pipeline II=1
				int reg_key = mem_keys[i];
				cand_t reg_cand = {reg_key, 0};
				s_fetched_neighbor_ids.write(reg_cand);
			}
		}
		s_finish_query_send_requests.write(1);
	}
	finish = block_read<int>(s_finish_query_write_memory);
}

void write_memory(
	const int query_num, 

	// in
	hls::stream<int>& s_num_valid_candidates_upper_levels_total, // one round can contain multiple bursts
	hls::stream<int>& s_num_valid_candidates_base_level_total, // one round can contain multiple bursts
	hls::stream<result_t>& s_distances_upper_levels,
	hls::stream<result_t>& s_distances_base_level,
	hls::stream<int>& s_finish_query_bloom_fetch_compute,

	// out 
	ap_uint<64>* out_id_dist,
	hls::stream<int>& s_finish_query_write_memory
) {
	bool first_iter_s_distances_upper_levels = true;
	bool first_iter_s_distances_base_level = true;
	
	// Expected results:
	//    first iter_per_query numbers: all from upper level
	//    second up to iter_per_query: the valid ones from base level
	for (int qid = 0; qid < query_num; qid++) {
		
		int addr_cnt = 0;
		while (true) {
			
			if (!s_finish_query_bloom_fetch_compute.empty() 
				&& s_num_valid_candidates_upper_levels_total.empty() && s_num_valid_candidates_base_level_total.empty() 
				&& s_distances_upper_levels.empty() && s_distances_base_level.empty()) {
				s_finish_query_write_memory.write(s_finish_query_bloom_fetch_compute.read());
				break;
			} else if (!s_num_valid_candidates_upper_levels_total.empty()) {

				int num_keys = s_num_valid_candidates_upper_levels_total.read();
				wait_data_fifo_first_iter<result_t>(
					num_keys, s_distances_upper_levels, first_iter_s_distances_upper_levels);
				for (int j = 0; j < num_keys; j++) {
					#pragma HLS pipeline II=1
					result_t reg_cand = s_distances_upper_levels.read();
					int id = reg_cand.node_id;
					float dist = reg_cand.dist;
					ap_uint<32> id_uint32 = id;
					ap_uint<32> dist_uint32 = *((ap_uint<32>*)(&dist));
					ap_uint<64> id_dist;
					id_dist.range(31, 0) = id_uint32;
					id_dist.range(63, 32) = dist_uint32;
					out_id_dist[addr_cnt + j] = id_dist;
				}
				addr_cnt += num_keys;
			} else if (!s_num_valid_candidates_base_level_total.empty()) {
				
				int num_keys = s_num_valid_candidates_base_level_total.read();
				wait_data_fifo_first_iter<result_t>(
					num_keys, s_distances_base_level, first_iter_s_distances_base_level);
				for (int j = 0; j < num_keys; j++) {
					#pragma HLS pipeline II=1
					result_t reg_cand = s_distances_base_level.read();
					int id = reg_cand.node_id;
					float dist = reg_cand.dist;
					ap_uint<32> id_uint32 = id;
					ap_uint<32> dist_uint32 = *((ap_uint<32>*)(&dist));
					ap_uint<64> id_dist;
					id_dist.range(31, 0) = id_uint32;
					id_dist.range(63, 32) = dist_uint32;
					out_id_dist[addr_cnt + j] = id_dist;
				}
			}
		}
	}
}
extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int iter_per_query,
	const int d,
	const int run_upper_levels,
	const int run_base_level,

    // in runtime (from DRAM)
	const ap_uint<512>* query_vectors,
	ap_uint<32>* mem_keys, 
    hls::burst_maxi<ap_uint<512>> db_vectors, // need to write visited tag
	   
    // out
    ap_uint<64>* out_id_dist
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=query_vectors offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=mem_keys offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=db_vectors latency=1 num_read_outstanding=32 max_read_burst_length=16 offset=slave bundle=gmem4 

// out
#pragma HLS INTERFACE m_axi port=out_id_dist  offset=slave bundle=gmem9

#pragma HLS dataflow
	
	hls::stream<ap_uint<512>> s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=128

    hls::stream<int> s_num_neighbors_upper_levels; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_upper_levels depth=512

    hls::stream<int> s_num_neighbors_base_level; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_base_level depth=512

    hls::stream<cand_t> s_fetched_neighbor_ids; 
#pragma HLS stream variable=s_fetched_neighbor_ids depth=512

	hls::stream<int> s_num_valid_candidates_upper_levels_total;
#pragma HLS stream variable=s_num_valid_candidates_upper_levels_total depth=16

	hls::stream<int> s_num_valid_candidates_base_level_total;
#pragma HLS stream variable=s_num_valid_candidates_base_level_total depth=16

    hls::stream<int> s_finish_query_write_memory; // finish all queries
#pragma HLS stream variable=s_finish_query_write_memory depth=16

    hls::stream<int> s_finish_query_send_requests; // finish all queries
#pragma HLS stream variable=s_finish_query_send_requests depth=16

	send_requests(
		query_num, 
		iter_per_query,
		d,
		run_upper_levels,
		run_base_level,
		
		// in
		query_vectors,
		mem_keys, 
		s_finish_query_write_memory, // finish the current query

		// out
		s_query_vectors,
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_send_requests // finish the current query
		);

	hls::stream<result_t> s_distances_upper_levels;
#pragma HLS stream variable=s_distances_upper_levels depth=512

	hls::stream<result_t> s_distances_base_level;
#pragma HLS stream variable=s_distances_base_level depth=512

    hls::stream<int> s_finish_query_bloom_fetch_compute; // finish all queries
#pragma HLS stream variable=s_finish_query_bloom_fetch_compute depth=16

	fetch_compute(
		// in initialization
		query_num, 
		d,

		// in runtime (from DRAM)
		db_vectors, // need to write visited tag

		// in streams
		s_query_vectors,
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_fetched_neighbor_ids,
		s_finish_query_send_requests,

		// out streams
		s_num_valid_candidates_upper_levels_total,
		s_num_valid_candidates_base_level_total,
		s_distances_upper_levels,
		s_distances_base_level,
		s_finish_query_bloom_fetch_compute
	);

	write_memory(
		query_num, 

		// in
		s_num_valid_candidates_upper_levels_total, // one round can contain multiple bursts
		s_num_valid_candidates_base_level_total, // one round can contain multiple bursts
		s_distances_upper_levels,
		s_distances_base_level,
		s_finish_query_bloom_fetch_compute,

		// out 
		out_id_dist,
		s_finish_query_write_memory
	);

}

}
