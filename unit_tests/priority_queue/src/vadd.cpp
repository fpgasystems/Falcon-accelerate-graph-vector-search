#include "constants.hpp"
#include "priority_queue.hpp"
#include "types.hpp"

#define INPUT_BUF_SIZE 16384

void read_memory(
	int input_array_size, // size of the input array to sort
	int iter_insert_sort, // number of repetitions of insert & sort
	float* input_array,
	hls::stream<int>& s_num_inserted_candidates,
	hls::stream<result_t>& s_input
) {
	float local_array[INPUT_BUF_SIZE];
	input_array_size = input_array_size > INPUT_BUF_SIZE? INPUT_BUF_SIZE : input_array_size;

	// read from memory
	for (int i = 0; i < input_array_size; i++) {
		#pragma HLS pipeline II=1
		local_array[i] = input_array[i];
	}

	// send out
	for (int i = 0; i < iter_insert_sort; i++) {
		s_num_inserted_candidates.write(input_array_size);
		for (int j = 0; j < input_array_size; j++) {
			#pragma HLS pipeline II=1
			result_t reg;
			reg.node_id = j;
			reg.level_id = 0;
			reg.dist = local_array[j];
			s_input.write(reg);
		}
	}
}

void queue_operations(
	int runtime_queue_size, // size of the candidate priority queue
	int input_array_size, // size of the input array to sort
	int iter_insert_sort, // number of repetitions of insert & sort
    int iter_pop, // number of repetitions of pop

	hls::stream<int>& s_num_inserted_candidates,
	hls::stream<result_t>& s_input,
	
	hls::stream<float>& s_output,
	hls::stream<cand_t>& s_top_candidates
) {

	Priority_queue<result_t, hardware_result_queue_size, Collect_smallest> result_queue(runtime_queue_size);

	for (int i = 0; i < iter_insert_sort; i++) {
		result_queue.reset_queue();
		result_queue.insert_sort(s_num_inserted_candidates, s_input);
	}

	// write results in ascending order
	for (int i = 0; i < runtime_queue_size; i++) {
		#pragma HLS pipeline II=1
		s_output.write(result_queue.queue[runtime_queue_size - i - 1].dist);
	}

	for (int i = 0; i < iter_pop; i++) {
		result_queue.pop_top(s_top_candidates);
	}
}

void write_memory(
	int runtime_queue_size, // size of the candidate priority queue
    int iter_pop, // number of repetitions of pop

	hls::stream<float>& s_output,
	hls::stream<cand_t>& s_top_candidates,

	float* sorted_array
) {
	// first runtime_queue_size elements: ascending by sort
	for (int i = 0; i < runtime_queue_size; i++) {
		#pragma HLS pipeline II=1
		sorted_array[i] = s_output.read();
	}

	// second runtime_queue_size elements: ids by pop
	int min_iter = iter_pop < runtime_queue_size? iter_pop : runtime_queue_size;
	for (int i = 0; i < min_iter; i++) {
		#pragma HLS pipeline II=1
		cand_t reg_cand = s_top_candidates.read();
		sorted_array[runtime_queue_size + i] = reg_cand.node_id;
	}
	cand_t reg_cand;
	for (int i = min_iter; i < iter_pop - min_iter; i++) {
		#pragma HLS pipeline II=1
		reg_cand = s_top_candidates.read();
	}

	// last element: last popped id
	sorted_array[runtime_queue_size * 2] = reg_cand.node_id;
}

extern "C" {

void vadd(  
	// in initialization
	int runtime_queue_size, // size of the candidate priority queue
	int input_array_size, // size of the input array to sort
	int iter_insert_sort, // number of repetitions of insert & sort
    int iter_pop, // number of repetitions of pop
	   
    // out
    float* input_array,
	float* sorted_array
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior
#pragma HLS INTERFACE m_axi port=input_array  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=sorted_array  offset=slave bundle=gmem10

#pragma HLS dataflow

	input_array_size = input_array_size > INPUT_BUF_SIZE? INPUT_BUF_SIZE : input_array_size;

    hls::stream<result_t> s_input; 
#pragma HLS stream variable=s_input depth=512

    hls::stream<int> s_num_inserted_candidates; 
#pragma HLS stream variable=s_num_inserted_candidates depth=512

	hls::stream<float> s_output;
#pragma HLS stream variable=s_output depth=512

	hls::stream<cand_t> s_top_candidates;	
#pragma HLS stream variable=s_top_candidates depth=512

	// controls the traversal and maintains the candidate queue
	read_memory(
		input_array_size, // size of the input array to sort
		iter_insert_sort, // number of repetitions of insert & sort
		input_array,
		s_num_inserted_candidates,
		s_input
	);

	queue_operations(
		runtime_queue_size, // size of the candidate priority queue
		input_array_size, // size of the input array to sort
		iter_insert_sort, // number of repetitions of insert & sort
    	iter_pop, // number of repetitions of pop

		s_num_inserted_candidates,
		s_input,
	
		s_output,
		s_top_candidates
	);
	
	write_memory(
		runtime_queue_size, // size of the candidate priority queue
		iter_pop, // number of repetitions of pop
		s_output,
		s_top_candidates,
		sorted_array
	);

}

}
