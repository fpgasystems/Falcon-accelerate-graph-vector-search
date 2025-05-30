#include "compute.hpp"
#include "constants.hpp"
#include "types.hpp"

#define D_MAX 1024

void read_memory(
	const int query_num, 
	const int num_fetched_vectors_per_query,
	const int d,
	const bool valid,

    // in runtime (from DRAM)
	const ap_uint<512>* query_vectors,
   	const ap_uint<512>* fetched_vectors,
    hls::stream<int>& s_finish_query_write_memory, // finish the current query

	// out streams
    hls::stream<int>& s_finish_query_read_memory, // finish the current query
	hls::stream<ap_uint<512>>& s_query_vectors,
	hls::stream<int>& s_fetch_batch_size,
    hls::stream<ap_uint<512>>& s_fetched_vectors,
	hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
    hls::stream<bool>& s_fetch_valid // if visited, not valid
) {

	ap_uint<512> query_buffer[D_MAX / FLOAT_PER_AXI];
	ap_uint<512> fetched_buffer[D_MAX / FLOAT_PER_AXI];
	cand_t neighbor_ids_dummy = {0, 0};
	bool fetch_valid = valid;

	const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 

	for (int j = 0; j < vec_AXI_num; j++) {
		#pragma HLS pipeline II=1
		query_buffer[j] = query_vectors[j];
		fetched_buffer[j] = fetched_vectors[j];
	}

	for (int i = 0; i < query_num; i++) {
		// write query
		for (int j = 0; j < vec_AXI_num; j++) {
			#pragma HLS pipeline II=1
			s_query_vectors.write(query_buffer[j]);
		}

		s_fetch_batch_size.write(num_fetched_vectors_per_query);
		// write fetched vector & s_fetched_neighbor_ids_replicated & s_fetch_valid
		for (int k = 0; k < num_fetched_vectors_per_query; k++) {
			for (int j = 0; j < vec_AXI_num; j++) {
				#pragma HLS pipeline II=1
				s_fetched_vectors.write(fetched_buffer[j]);
			}
			s_fetched_neighbor_ids_replicated.write(neighbor_ids_dummy);
			s_fetch_valid.write(fetch_valid);
		}

		// write finish query
		s_finish_query_read_memory.write(1);

		// wait for the next query
		while (s_finish_query_write_memory.empty()) {}
		volatile int finish = s_finish_query_write_memory.read();
	}
}

void write_memory(
	const int query_num, 
	const int num_fetched_vectors_per_query,
	hls::stream<result_t>& s_distances,
	hls::stream<int>& s_finish_query_compute, // finish the current query
	float* out_dist,
	hls::stream<int>& s_finish_query_write_memory
) {
	result_t dist;
	volatile int finish;
	for (int i = 0; i < query_num - 1; i++) {
		// read distance
		for (int j = 0; j < num_fetched_vectors_per_query; j++) {
			#pragma HLS pipeline II=1
			dist = s_distances.read();
		}

		// read finish query
		finish = s_finish_query_compute.read();
		s_finish_query_write_memory.write(1);
	}
	out_dist[0] = dist.dist;
	for (int j = 0; j < num_fetched_vectors_per_query; j++) {
		#pragma HLS pipeline II=1
		dist = s_distances.read();
		out_dist[j] = dist.dist;
	}
	finish = s_finish_query_compute.read();
	s_finish_query_write_memory.write(1);
}



extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int num_fetched_vectors_per_query,
	const int d,
	const bool valid,

    // in runtime (from DRAM)
	const ap_uint<512>* query_vectors,
   	const ap_uint<512>* fetched_vectors,

    // out
	float* out_dist
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=query_vectors offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=fetched_vectors offset=slave bundle=gmem1 // share the same AXI interface with query_vectors

// out
#pragma HLS INTERFACE m_axi port=out_dist  offset=slave bundle=gmem10

#pragma HLS dataflow

    hls::stream<int> s_finish_query_read_memory; // finish the current query
#pragma HLS stream variable=s_finish_query_read_memory depth=16
	
    hls::stream<int> s_finish_query_write_memory; // finish the current query
#pragma HLS stream variable=s_finish_query_write_memory depth=16

	hls::stream<ap_uint<512>> s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=128

	hls::stream<int> s_fetch_batch_size;
#pragma HLS stream variable=s_fetch_batch_size depth=16

    hls::stream<ap_uint<512>> s_fetched_vectors; 
#pragma HLS stream variable=s_fetched_vectors depth=512

	hls::stream<cand_t> s_fetched_neighbor_ids_replicated;
#pragma HLS stream variable=s_fetched_neighbor_ids_replicated depth=512

    hls::stream<bool> s_fetch_valid; // if visited, not valid
#pragma HLS stream variable=s_fetch_valid depth=512


	read_memory(
		query_num, 
		num_fetched_vectors_per_query,
		d,
		valid,
		query_vectors,
		fetched_vectors,
		s_finish_query_write_memory,

		s_finish_query_read_memory,
		s_query_vectors,
		s_fetch_batch_size,
		s_fetched_vectors,
		s_fetched_neighbor_ids_replicated,
		s_fetch_valid
	);

    hls::stream<result_t> s_distances; 
#pragma HLS stream variable=s_distances depth=512

    hls::stream<int> s_finish_query_compute; // finish the current query
#pragma HLS stream variable=s_finish_query_compute depth=16

	compute_distances(
		// in initialization
		query_num,
		d,
		// in runtime (stream)
		s_query_vectors,
		s_fetch_batch_size,
		s_fetched_vectors,
		s_fetch_valid,
		s_fetched_neighbor_ids_replicated,
		s_finish_query_read_memory,
		
		// out (stream)
		s_distances,
		s_finish_query_compute
	);

	write_memory(
		query_num, 
		num_fetched_vectors_per_query,
		s_distances,
		s_finish_query_compute,
		out_dist,
		s_finish_query_write_memory
	);

}

}
