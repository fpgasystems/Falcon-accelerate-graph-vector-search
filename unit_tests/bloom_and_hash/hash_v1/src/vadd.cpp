#include "bloom_filter.hpp"
#include "constants.hpp"
#include "types.hpp"

#define D_MAX 1024

template<typename T>
inline T block_read(hls::stream<T>& s) {
#pragma HLS inline

    while (s.empty()) {}
    return s.read();
}

const int hw_buffer_size = 16 * 1024;

void send_requests(
	const int query_num, 
	const int iter_per_query,
	
    // in
	ap_uint<32>* mem_keys, 
	hls::stream<int>& s_finish_query_write_memory, // finish the current query

    // out
	hls::stream<int>& s_num_keys,
    hls::stream<ap_uint<32>>& s_keys,
    hls::stream<int>& s_finish_query_send_requests // finish the current query
    ) {
	
	ap_uint<32> local_buffer[hw_buffer_size];
	int max_iter_size = hw_buffer_size < iter_per_query? hw_buffer_size : iter_per_query;
	for (int i = 0; i < max_iter_size; i++) {
		#pragma HLS pipeline II=1
		local_buffer[i] = mem_keys[i];
	}

	int finish = 1;
	bool first_iter_s_finish_query_write_memory = true;
	for (int qid = 0; qid < query_num; qid++) {
		if (qid > 0) {
			if (first_iter_s_finish_query_write_memory) {
				while (s_finish_query_write_memory.empty()) {}
				first_iter_s_finish_query_write_memory = false;
			}
			finish = s_finish_query_write_memory.read();
		}
		s_num_keys.write(max_iter_size);
		for (int i = 0; i < max_iter_size; i++) {
	#pragma HLS pipeline II=1
			s_keys.write(local_buffer[i]);
		}
		s_finish_query_send_requests.write(finish);
	}
	if (first_iter_s_finish_query_write_memory) {
		while (s_finish_query_write_memory.empty()) {}
		first_iter_s_finish_query_write_memory = false;
	}
	finish = s_finish_query_write_memory.read();
}

void write_memory(
	const int query_num, 
	const int iter_per_query,

	// in
	hls::stream<ap_uint<32>>& s_hash_values,
	hls::stream<int>& s_finish_hashing,

	// out 
	ap_uint<32>* mem_out,
	hls::stream<int>& s_finish_query_write_memory
) {
	
	int max_iter_size = hw_buffer_size < iter_per_query? hw_buffer_size : iter_per_query;
	ap_uint<32> local_buffer[hw_buffer_size];

	for (int i = 0; i < query_num; i++) {

		while (true) {
			
			if (!s_finish_hashing.empty() && s_hash_values.empty()) {
				s_finish_query_write_memory.write(s_finish_hashing.read());
				break;
			} else if (!s_hash_values.empty()) {
				// read values
				for (int j = 0; j < max_iter_size; j++) {
					#pragma HLS pipeline II=1
					local_buffer[j] = s_hash_values.read();
				}
			}
		}
	}
	for (int j = 0; j < max_iter_size; j++) {
		#pragma HLS pipeline II=1
		mem_out[j] = local_buffer[j];
	}
}


extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int iter_per_query,
	const int seed,

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
	
	hls::stream<int> s_num_keys;
#pragma HLS stream variable=s_num_keys depth=16

	hls::stream<ap_uint<32>> s_keys;
#pragma HLS stream variable=s_keys depth=512

	send_requests(
		query_num, 
		iter_per_query,
		// in
		mem_keys, 
		s_finish_query_write_memory, // finish the current query

		// out
		s_num_keys,
		s_keys,
		s_finish_query_send_requests // finish the current query
    );

	hls::stream<ap_uint<32>> s_hash_values;
#pragma HLS stream variable=s_hash_values depth=512

    hls::stream<int> s_finish_hashing; // finish the current query
#pragma HLS stream variable=s_finish_hashing depth=16

	stream_hash(
		query_num, 
		seed,
		// in streams
		s_num_keys,
		s_keys,
		s_finish_query_send_requests,

		// out streams
		s_hash_values,
		s_finish_hashing);

	write_memory(
		query_num, 
		iter_per_query,

		// in
		s_hash_values,
		s_finish_hashing,

		// out 
		mem_out,
		s_finish_query_write_memory
	);

}

}
