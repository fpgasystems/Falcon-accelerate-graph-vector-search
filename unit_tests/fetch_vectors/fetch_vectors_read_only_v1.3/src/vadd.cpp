#include "DRAM_utils.hpp"
#include "constants.hpp"
#include "types.hpp"

#define D_MAX 1024

void send_requests(
	const int query_num, 
	const int read_iter_per_query,
	const int read_batch_size,
    // in
	int* mem_read_node_id, 
	hls::stream<int>& s_finish_query_write_memory, // finish the current query

    // out
	hls::stream<int>& s_fetch_batch_size,
    hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
    hls::stream<int>& s_finish_query_send_requests // finish the current query
    ) {
	int finish;
	bool first_iter_s_finish_query_write_memory = true;
	for (int qid = 0; qid < query_num; qid++) {
		if (qid > 0) { 
			if (first_iter_s_finish_query_write_memory) {
				while (s_finish_query_write_memory.empty()) {}
				first_iter_s_finish_query_write_memory = false;
			}
			finish = s_finish_query_write_memory.read();
		}
		for (int i = 0; i < read_iter_per_query; i += read_batch_size) {
			#pragma HLS pipeline II=1
			int remaining = read_iter_per_query - i;
			int num_elements = (remaining < read_batch_size) ? remaining : read_batch_size;
			s_fetch_batch_size.write(num_elements);
			for (int j = 0; j < num_elements; j++) {
				int node_id = mem_read_node_id[i + j];
				s_fetched_neighbor_ids_replicated.write({node_id, 0});
			}
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
	const int read_iter_per_query,

	// in
	hls::stream<ap_uint<512>>& s_fetched_vectors,
	hls::stream<int>& s_finish_fetch_vectors,

	// out 
	ap_uint<512>* mem_out,
	hls::stream<int>& s_finish_query_write_memory
) {
	
	const int AXI_num_per_vector_only = D % FLOAT_PER_AXI == 0? 
		D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; // 16 for D = 512

	ap_uint<512> reg;
	for (int i = 0; i < query_num; i++) {
		while (true) {
			if (!s_finish_fetch_vectors.empty() && s_fetched_vectors.empty()) {
				int finish = s_finish_fetch_vectors.read();
				s_finish_query_write_memory.write(finish);
				break;
			}
			else if (!s_fetched_vectors.empty()) {
				// read distance
				for (int j = 0; j < read_iter_per_query; j++) {
				#pragma HLS pipeline
					for (int k = 0; k < AXI_num_per_vector_only; k++) {
						reg = s_fetched_vectors.read();
					}	
				}
			}
		}
	}

	mem_out[0] = reg;
}


extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int read_iter_per_query,
	const int read_batch_size,

    // in runtime (from DRAM)
	int* mem_read_node_id, 
   	ap_uint<512>* db_vectors,

    // out
	ap_uint<512>* mem_out
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=mem_read_node_id offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=db_vectors latency=32 num_read_outstanding=32 num_write_outstanding=1 offset=slave bundle=gmem1 // share the same AXI interface with query_vectors

// out
#pragma HLS INTERFACE m_axi port=mem_out  offset=slave bundle=gmem2

#pragma HLS dataflow

    hls::stream<int> s_finish_query_send_requests; // finish the current query
#pragma HLS stream variable=s_finish_query_send_requests depth=16
	
    hls::stream<int> s_finish_query_write_memory; // finish the current query
#pragma HLS stream variable=s_finish_query_write_memory depth=16
	
	hls::stream<int> s_fetch_batch_size;
#pragma HLS stream variable=s_fetch_batch_size depth=16

	hls::stream<cand_t> s_fetched_neighbor_ids_replicated;
#pragma HLS stream variable=s_fetched_neighbor_ids_replicated depth=512


	send_requests(
		query_num, 
		read_iter_per_query,
		read_batch_size,
		// in
		mem_read_node_id, 
		s_finish_query_write_memory, // finish the current query

		// out
		s_fetch_batch_size,
		s_fetched_neighbor_ids_replicated,
		s_finish_query_send_requests // finish the current query
    );

    hls::stream<ap_uint<512>> s_fetched_vectors; 
#pragma HLS stream variable=s_fetched_vectors depth=512

    hls::stream<int> s_finish_fetch_vectors; // finish the current query
#pragma HLS stream variable=s_finish_fetch_vectors depth=16

	fetch_vectors(
		// in initialization
		query_num,
		// in runtime (should from DRAM)
		db_vectors,
		// in runtime (stream)
		s_fetch_batch_size,
		s_fetched_neighbor_ids_replicated,
		s_finish_query_send_requests,
		
		// out (stream)
		s_fetched_vectors,
		s_finish_fetch_vectors
	);

	write_memory(
		query_num, 
		read_iter_per_query,
		// in
		s_fetched_vectors,
		s_finish_fetch_vectors,

		// out 
		mem_out,
		s_finish_query_write_memory
	);

}

}
