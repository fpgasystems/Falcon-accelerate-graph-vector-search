#include "bloom_filter.hpp"
#include "bloom_fetch_compute.hpp"
#include "compute.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "scheduler.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "per_channel_processing_wrapper.hpp"

extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int query_batch_size,
	const int ef, // size of the result priority queue
	const int candidate_queue_runtime_size, 
	const int max_cand_batch_size, 
	const int max_async_stage_num,
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int max_bloom_out_burst_size,
	const int max_link_num_base,

    // in runtime (from DRAM)
	const int* entry_point_ids,
	const ap_uint<512>* query_vectors,
    ap_uint<512>* db_vectors,
    const ap_uint<512>* links_base,
	   
    // out
    int* out_id,
	float* out_dist,

	// debug signals (each 4 byte): 
	//   0: bottom layer entry node id, 
	//   1: number of hops in upper layers 
	//   2: number of read vectors in upper layers
	//   3: number of hops in base layer (number of pop operations)
	//   4: number of valid read vectors in base layer
	int* mem_debug
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (from DRAM)
#pragma HLS INTERFACE m_axi port=entry_point_ids latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=query_vectors latency=32 num_read_outstanding=4 max_read_burst_length=16  num_write_outstanding=1 max_write_burst_length=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=mem_debug latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem10 // cannot share gmem with out as they are different PEs

#pragma HLS INTERFACE m_axi port=db_vectors latency=64 num_read_outstanding=64 num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem4 
#pragma HLS INTERFACE m_axi port=links_base latency=32 num_read_outstanding=16 num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmem2

// out
#pragma HLS INTERFACE m_axi port=out_id latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=out_dist latency=32 num_read_outstanding=1 max_read_burst_length=2  num_write_outstanding=8 max_write_burst_length=16 offset=slave bundle=gmem9

#pragma HLS dataflow

	hls::stream<int> s_query_batch_size;
#pragma HLS stream variable=s_query_batch_size depth=16

	hls::stream<ap_uint<512>> s_query_vectors_in;
#pragma HLS stream variable=s_query_vectors_in depth=16

	hls::stream<int> s_entry_point_ids;
#pragma HLS stream variable=s_entry_point_ids depth=16

	hls::stream<int> s_finish_batch;
#pragma HLS stream variable=s_finish_batch depth=16

	read_queries(
		// in initialization
		query_num,
		query_batch_size,
		// in DRAM
		entry_point_ids,
		query_vectors,
		// in stream
		s_finish_batch,

		// out streams
		s_query_batch_size,
		s_query_vectors_in,
		s_entry_point_ids
	);

	// replicate s_query_batch_size to multiple streams
	const int replicate_factor_s_query_batch_size = 2;
	hls::stream<int> s_query_batch_size_replicated[replicate_factor_s_query_batch_size];
#pragma HLS stream variable=s_query_batch_size_replicated depth=16

	replicate_s_query_batch_size<replicate_factor_s_query_batch_size>(
		s_query_batch_size,
		s_query_batch_size_replicated
	);

	hls::stream<int> s_out_ids;
#pragma HLS stream variable=s_out_ids depth=512

	hls::stream<float> s_out_dists;
#pragma HLS stream variable=s_out_dists depth=512

	hls::stream<int> s_debug_signals;
#pragma HLS stream variable=s_debug_signals depth=512


	per_channel_processing_wrapper(
		// in initialization
		ef, // size of the result priority queue
		candidate_queue_runtime_size, 
		max_cand_batch_size, 
		max_async_stage_num,
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		max_link_num_base,
	
		// in runtime (from DRAM)
		db_vectors,
		links_base,		

		// in streams
		s_query_batch_size_replicated[0], // -1: stop
		s_query_vectors_in,	
		s_entry_point_ids, 

		// out streams
		s_out_ids,
		s_out_dists,
		s_debug_signals
	);

	write_results(
		ef,

		// in streams
		s_query_batch_size_replicated[1], // -1: stop
		s_out_ids,
		s_out_dists,
		s_debug_signals,

		// out streams
		s_finish_batch,
		
		// out
		out_id,
		out_dist,
		mem_debug
	);

}

}
