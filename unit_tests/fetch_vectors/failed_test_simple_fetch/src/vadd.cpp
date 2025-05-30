#include "DRAM_utils.hpp"
#include "constants.hpp"
#include "types.hpp"

#define D_MAX 1024

extern "C" {

void vadd(  
	// in initialization
	const int query_num, 
	const int read_iter_per_query,

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
#pragma HLS INTERFACE m_axi port=db_vectors latency=300 num_read_outstanding=128 num_write_outstanding=1 offset=slave bundle=gmem1 // share the same AXI interface with query_vectors

// out
#pragma HLS INTERFACE m_axi port=mem_out  offset=slave bundle=gmem2


	ap_uint<512> vector_AXI;

	int cache_size = 16 * 1024;
	int node_id_array[cache_size];

	int actual_read_iter_per_query = read_iter_per_query < cache_size ? read_iter_per_query : cache_size;
	for (int i = 0; i < actual_read_iter_per_query; i++) {
		node_id_array[i] = mem_read_node_id[i];
	}

	for (int qid = 0; qid < query_num; qid++) {

		for (int bid = 0; bid < actual_read_iter_per_query; bid++) {
		#pragma HLS pipeline // put the pipeline here so hopefully Vitis can handle prefetching automatically
			// receive task & read vectors
			int start_addr = node_id_array[bid] * AXI_num_per_vector_and_padding;
			for (int i = 0; i < AXI_num_per_vector_only; i++) {
				vector_AXI = db_vectors[start_addr + i];
			}
		}
	}

	mem_out[0] = vector_AXI;
}

}
