#include "bloom_filter.hpp"
#include "constants.hpp"
#include "types.hpp"
#include "utils.hpp"	

const int num_hash_funs = 4; 
const int num_bucket_addr_bits = 6 + 10; // 64 * 1024
const int num_buckets = 1 << num_bucket_addr_bits;

void send_requests(
	const int query_num, 
	const int iter_per_query,
	
    // in
	ap_uint<32>* mem_keys, 
	hls::stream<int>& s_finish_query_write_memory, // finish the current query

    // out
	hls::stream<int>& s_num_neighbors_upper_levels,
	hls::stream<int>& s_num_neighbors_base_level,
    hls::stream<cand_t>& s_all_candidates,
    hls::stream<int>& s_finish_query_send_requests // finish the current query
    ) {

	int finish;
	for (int qid = 0; qid < query_num; qid++) {
		if (qid > 0) {
			finish = block_read<int>(s_finish_query_write_memory);
		}
		// write upper level
		s_num_neighbors_upper_levels.write(iter_per_query);
		for (int i = 0; i < iter_per_query; i++) {
		#pragma HLS pipeline II=1
			int reg_key = mem_keys[i];
			cand_t reg_cand = {reg_key, 1};
			s_all_candidates.write(reg_cand);
		}
		// write base level
		s_num_neighbors_base_level.write(iter_per_query);
		for (int i = 0; i < iter_per_query; i++) {
		#pragma HLS pipeline II=1
			int reg_key = mem_keys[i];
			cand_t reg_cand = {reg_key, 0};
			s_all_candidates.write(reg_cand);
		}
		s_finish_query_send_requests.write(1);
	}
	finish = block_read<int>(s_finish_query_write_memory);
}

void write_memory(
	const int query_num, 

	// in
	hls::stream<int>& s_num_valid_candidates_burst,
	hls::stream<int>& s_num_valid_candidates_upper_levels_total_out, // one round can contain multiple bursts
	hls::stream<int>& s_num_valid_candidates_base_level_total_out, // one round can contain multiple bursts
	hls::stream<cand_t>& s_valid_candidates,
	hls::stream<int>& s_finish_bloom,

	// out 
	ap_uint<32>* mem_out,
	hls::stream<int>& s_finish_query_write_memory
) {
	bool first_iter_s_valid_candidates = true;
	
	// Expected results:
	//    first iter_per_query numbers: all from upper level
	//    second up to iter_per_query: the valid ones from base level
	for (int qid = 0; qid < query_num; qid++) {
		
		int addr_cnt = 0;
		while (true) {
			
			if (!s_finish_bloom.empty() && s_num_valid_candidates_burst.empty() 
				&& s_num_valid_candidates_upper_levels_total_out.empty() && s_num_valid_candidates_base_level_total_out.empty() 
				&& s_valid_candidates.empty()) {
				s_finish_query_write_memory.write(s_finish_bloom.read());
				break;
			} else if (!s_num_valid_candidates_burst.empty()) {

				int num_keys = s_num_valid_candidates_burst.read();
				wait_data_fifo_first_iter<cand_t>(
					num_keys, s_valid_candidates, first_iter_s_valid_candidates);
				for (int j = 0; j < num_keys; j++) {
					#pragma HLS pipeline II=1
					cand_t reg_cand = s_valid_candidates.read();
					mem_out[addr_cnt + j] = reg_cand.node_id;
				}
				addr_cnt += num_keys;
			} else if (!s_num_valid_candidates_upper_levels_total_out.empty()) {
				s_num_valid_candidates_upper_levels_total_out.read();
			} else if (!s_num_valid_candidates_base_level_total_out.empty()) {
				s_num_valid_candidates_base_level_total_out.read();
			}
		}
	}
}


extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int iter_per_query,
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,

    // in runtime (from DRAM)
	ap_uint<32>* mem_keys, 

    // out
	ap_uint<32>* mem_out
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=mem_keys offset=slave bundle=gmem0

// out
#pragma HLS INTERFACE m_axi port=mem_out  offset=slave bundle=gmem2

#pragma HLS dataflow

    hls::stream<int> s_finish_query_send_requests; // finish the current query
#pragma HLS stream variable=s_finish_query_send_requests depth=16
	
    hls::stream<int> s_finish_query_write_memory; // finish the current query
#pragma HLS stream variable=s_finish_query_write_memory depth=16
	
	hls::stream<int> s_num_neighbors_upper_levels;
#pragma HLS stream variable=s_num_neighbors_upper_levels depth=16

	hls::stream<int> s_num_neighbors_base_level;
#pragma HLS stream variable=s_num_neighbors_base_level depth=16

	hls::stream<cand_t> s_all_candidates;
#pragma HLS stream variable=s_all_candidates depth=512

	send_requests(
		query_num, 
		iter_per_query,
		// in
		mem_keys, 
		s_finish_query_write_memory, // finish the current query

		// out
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_all_candidates,
		s_finish_query_send_requests // finish the current query
    );


	hls::stream<int> s_num_valid_candidates_burst;
#pragma HLS stream variable=s_num_valid_candidates_burst depth=16

	hls::stream<int> s_num_valid_candidates_upper_levels_total_out;
#pragma HLS stream variable=s_num_valid_candidates_upper_levels_total_out depth=16

	hls::stream<int> s_num_valid_candidates_base_level_total_out;
#pragma HLS stream variable=s_num_valid_candidates_base_level_total_out depth=16

	hls::stream<cand_t> s_valid_candidates;
#pragma HLS stream variable=s_valid_candidates depth=512

    hls::stream<int> s_finish_bloom; // finish the current query
#pragma HLS stream variable=s_finish_bloom depth=16

	BloomFilter<num_hash_funs, num_bucket_addr_bits> bloom_filter(runtime_n_bucket_addr_bits);

	bloom_filter.bloom_filter_top_level(
		query_num, 
		hash_seed,
		// in streams
		s_num_neighbors_upper_levels,
		s_num_neighbors_base_level,
		s_all_candidates,
		s_finish_query_send_requests,

		// out streams
		s_num_valid_candidates_burst, // one round (s_num_neighbors) can contain multiple bursts
		s_num_valid_candidates_upper_levels_total_out, // one round can contain multiple bursts
		s_num_valid_candidates_base_level_total_out,
		s_valid_candidates,
		s_finish_bloom);
	
	write_memory(
		query_num, 

		// in
		s_num_valid_candidates_burst,
		s_num_valid_candidates_upper_levels_total_out,
		s_num_valid_candidates_base_level_total_out,
		s_valid_candidates,
		s_finish_bloom,

		// out 
		mem_out,
		s_finish_query_write_memory
	);

}

}
