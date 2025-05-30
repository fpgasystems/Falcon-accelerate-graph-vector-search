#pragma once

#include "constants.hpp"
#include "types.hpp"
#include "helpers.hpp"
#include "priority_queue_L1.hpp"
#include "priority_queue_L2.hpp"

template<const int stream_num>
void replicate_scanned_entries_per_query_Redirected_sorted_stream(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue, 
    hls::stream<int> (&s_insertion_per_queue_L1)[stream_num]);

void split_single_stream(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<PQ_out_t> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_A, 
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_B, 
    hls::stream<PQ_out_t> &output_stream_A,
    hls::stream<PQ_out_t> &output_stream_B);

template<const int stream_num>
void split_single_PQ_result_wrapper(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<PQ_out_t> (&s_input)[stream_num], 
    hls::stream<int> &s_scanned_entries_per_query_In_Priority_queue,
    hls::stream<int> (&s_scanned_entries_every_cell_Out_priority_queue)[2 * stream_num],
    hls::stream<PQ_out_t> (&s_input_splitted)[2 * stream_num]);

template<const int iter_num_per_query>
void send_iter_num(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_insertion_per_queue);


template<const int priority_queue_len, const int stream_num>
void merge_streams(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<result_t> (&intermediate_result)[stream_num],
    hls::stream<result_t> &output_stream);

template<const int priority_queue_len, const int stream_num>
void convert_vec_ID_offset_to_addr(
    int nlist,
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<PQ_out_t> (&s_intermediate_result_with_offset)[stream_num],
    hls::stream<PQ_lookup_t> (&s_intermediate_result_with_addr)[stream_num]);

template<const int priority_queue_len, const int stream_num>
void convert_addr_to_vec_ID(
    hls::stream<batch_header_t>& s_batch_header,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,
    hls::stream<PQ_lookup_t> (&s_intermediate_result_with_addr)[stream_num],
    hls::stream<result_t> (&s_intermediate_result_with_vec_ID)[stream_num]);

void hierarchical_priority_queue( 
    int nlist,
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue,
    hls::stream<PQ_out_t> (&s_input)[ADC_PE_NUM], 
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,
    hls::stream<result_t> &output_stream);

////////////////////      Definitions      ////////////////////

template<const int stream_num>
void replicate_scanned_entries_per_query_Redirected_sorted_stream(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue, 
    hls::stream<int> (&s_insertion_per_queue_L1)[stream_num]) {
   
    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        } 

		for (int i = 0; i < query_num; i++) {

			int scanned_entries_per_query = s_scanned_entries_per_query_Priority_queue.read();
			
			for (int s = 0; s < stream_num; s++) {
#pragma HLS UNROLL
				s_insertion_per_queue_L1[s].write(scanned_entries_per_query);
			}
		}
	}
}

void split_single_stream(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<PQ_out_t> &input_stream,
    hls::stream<int> &s_scanned_entries_every_cell,
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_A, 
    hls::stream<int> &s_scanned_entries_every_cell_Out_priority_queue_B, 
    hls::stream<PQ_out_t> &output_stream_A,
    hls::stream<PQ_out_t> &output_stream_B) {

    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {

			int scanned_entries_every_cell = s_scanned_entries_every_cell.read();
			int half_scanned_entries_every_cell = scanned_entries_every_cell / 2;

			if ((scanned_entries_every_cell - 2 * half_scanned_entries_every_cell) == 1) {
				s_scanned_entries_every_cell_Out_priority_queue_A.write(half_scanned_entries_every_cell + 1);
				output_stream_A.write(input_stream.read());
			}
			else {
				s_scanned_entries_every_cell_Out_priority_queue_A.write(half_scanned_entries_every_cell); 
			}
			s_scanned_entries_every_cell_Out_priority_queue_B.write(half_scanned_entries_every_cell);

			for (int entry_id = 0; entry_id < half_scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=2
				output_stream_A.write(input_stream.read());
				output_stream_B.write(input_stream.read());
			}
		}
	}
}

template<const int stream_num>
void split_single_PQ_result_wrapper(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<PQ_out_t> (&s_input)[stream_num], 
    hls::stream<int> &s_scanned_entries_per_query_In_Priority_queue,
    hls::stream<int> (&s_scanned_entries_every_cell_Out_priority_queue)[2 * stream_num],
    hls::stream<PQ_out_t> (&s_input_splitted)[2 * stream_num]) {
    
#pragma HLS inline
    // for the top 16 elements, discard the last 6 
    // for the rest 10 elements, split them to 2 streams, since Priority Queue's
    //   insertion takes 2 CC

    hls::stream<int> s_scanned_entries_every_cell_Replicated[stream_num];
#pragma HLS stream variable=s_scanned_entries_every_cell_Replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Replicated core=FIFO_SRL

    hls::stream<batch_header_t> s_batch_header_replicated[stream_num + 1];
#pragma HLS stream variable=s_batch_header_replicated depth=8
#pragma HLS array_partition variable=s_batch_header_replicated complete

	replicate_s_batch_header<stream_num + 1>(
		s_batch_header, 
		s_batch_header_replicated);

    replicate_scanned_entries_per_query_Redirected_sorted_stream<stream_num>(
        s_batch_header_replicated[0],
        s_scanned_entries_per_query_In_Priority_queue, 
        s_scanned_entries_every_cell_Replicated);

    for (int i = 0; i < stream_num; i++) {
#pragma HLS UNROLL
        split_single_stream(
            s_batch_header_replicated[i + 1], 
            s_input[i], 
            s_scanned_entries_every_cell_Replicated[i],
            s_scanned_entries_every_cell_Out_priority_queue[2 * i],
            s_scanned_entries_every_cell_Out_priority_queue[2 * i + 1],
            s_input_splitted[2 * i], 
            s_input_splitted[2 * i + 1]);
    }
}

template<const int iter_num_per_query>
void send_iter_num(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_insertion_per_queue) {

    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {
			s_insertion_per_queue.write(iter_num_per_query);
		}
	}
}


template<const int priority_queue_len, const int stream_num>
void merge_streams(
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<result_t> (&intermediate_result)[stream_num],
    hls::stream<result_t> &output_stream) {
   
    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        } 

		for (int query_id = 0; query_id < query_num; query_id++) {
			for (int d = 0; d < priority_queue_len; d++) {
				for (int s = 0; s < stream_num; s++) {
#pragma HLS pipeline II=1
					output_stream.write(intermediate_result[s].read());
				}
			}
		}
	}
}

template<const int priority_queue_len, const int stream_num>
void convert_vec_ID_offset_to_addr(
    int nlist,
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<PQ_out_t> (&s_intermediate_result_with_offset)[stream_num],
    hls::stream<PQ_lookup_t> (&s_intermediate_result_with_addr)[stream_num]) {

    // init
    int cell_ID_to_addr[NLIST_MAX];
#pragma HLS resource variable=cell_ID_to_addr core=RAM_2P_URAM
    for (int i = 0; i < nlist; i++) {
        cell_ID_to_addr[i] = s_nlist_vec_ID_start_addr.read();
    }
    
    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {
			for (int d = 0; d < priority_queue_len; d++) {
				for (int s = 0; s < stream_num; s++) {
#pragma HLS pipeline II=1
					PQ_out_t reg = s_intermediate_result_with_offset[s].read();
					int start_addr = cell_ID_to_addr[reg.cell_ID];
					PQ_lookup_t out;
					out.dist = reg.dist;
					out.vec_ID_addr = start_addr + reg.offset;
					s_intermediate_result_with_addr[s].write(out);
				}
			}
		}
	}
}

template<const int priority_queue_len, const int stream_num>
void convert_addr_to_vec_ID(
    hls::stream<batch_header_t>& s_batch_header,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,
    hls::stream<PQ_lookup_t> (&s_intermediate_result_with_addr)[stream_num],
    hls::stream<result_t> (&s_intermediate_result_with_vec_ID)[stream_num]) {

    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;
		int query_num = batch_size; 

        // termination detection
        if (terminate) {
            break;
        }

		for (int query_id = 0; query_id < query_num; query_id++) {
			for (int d = 0; d < priority_queue_len; d++) {

				/* first partition of the streams belong to DRAM 0, and so forth */
				for (int s = 0; s < PRIORITY_QUEUE_PER_BANK; s++) {
#pragma HLS pipeline II=1 // style=flp
					int s_id_DRAM_0 = s + 0 * PRIORITY_QUEUE_PER_BANK;
					int s_id_DRAM_1 = s + 1 * PRIORITY_QUEUE_PER_BANK;
					int s_id_DRAM_2 = s + 2 * PRIORITY_QUEUE_PER_BANK;
					int s_id_DRAM_3 = s + 3 * PRIORITY_QUEUE_PER_BANK;

					PQ_lookup_t in_0 = s_intermediate_result_with_addr[s_id_DRAM_0].read();
					PQ_lookup_t in_1 = s_intermediate_result_with_addr[s_id_DRAM_1].read();
					PQ_lookup_t in_2 = s_intermediate_result_with_addr[s_id_DRAM_2].read();
					PQ_lookup_t in_3 = s_intermediate_result_with_addr[s_id_DRAM_3].read();

					result_t out_0;
					result_t out_1;
					result_t out_2;
					result_t out_3;

					out_0.vec_ID = vec_ID_DRAM_0[in_0.vec_ID_addr];
					out_1.vec_ID = vec_ID_DRAM_1[in_1.vec_ID_addr];
					out_2.vec_ID = vec_ID_DRAM_2[in_2.vec_ID_addr];
					out_3.vec_ID = vec_ID_DRAM_3[in_3.vec_ID_addr];

					out_0.dist = in_0.dist;
					out_1.dist = in_1.dist;
					out_2.dist = in_2.dist;
					out_3.dist = in_3.dist;

					s_intermediate_result_with_vec_ID[s_id_DRAM_0].write(out_0);
					s_intermediate_result_with_vec_ID[s_id_DRAM_1].write(out_1);
					s_intermediate_result_with_vec_ID[s_id_DRAM_2].write(out_2);
					s_intermediate_result_with_vec_ID[s_id_DRAM_3].write(out_3);
				}
			}
		}
	}
}

void hierarchical_priority_queue( 
    int nlist,
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int> &s_nlist_vec_ID_start_addr,
    hls::stream<int> &s_scanned_entries_per_query_Priority_queue,
    hls::stream<PQ_out_t> (&s_input)[ADC_PE_NUM], 
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3,
    hls::stream<result_t> &output_stream) {
/*
    Hardcode number of memory channels as 4

    Input: 
        the table mapping cell ID to its start address of the array of vec_IDs (s_nprobe_vec_ID_start_addr)
        the number of outputs per compute PE (s_scanned_entries_per_query_Priority_queue)
        the compute PE's result (s_input)

    Output:
        the topK results of (vec_ID, dist) pairs
    
    Process: 
        per memory channel, gather the compute PE's result by L1 priority queues,
            then lookup the vector ID from memory channels.
        L2 priory queue consumes the per channel results
*/

#pragma HLS inline

    hls::stream<int> s_insertion_per_queue_L1[PRIORITY_QUEUE_NUM_L1];
#pragma HLS stream variable=s_insertion_per_queue_L1 depth=8
#pragma HLS array_partition variable=s_insertion_per_queue_L1 complete
// #pragma HLS RESOURCE variable=s_insertion_per_queue_L1 core=FIFO_SRL

    hls::stream<PQ_out_t> s_input_splitted[PRIORITY_QUEUE_NUM_L1];
#pragma HLS stream variable=s_input_splitted depth=8
#pragma HLS array_partition variable=s_input_splitted complete
// #pragma HLS RESOURCE variable=s_input_splitted core=FIFO_SRL

    hls::stream<PQ_out_t> s_intermediate_result_with_offset[PRIORITY_QUEUE_NUM_L1];
#pragma HLS stream variable=s_intermediate_result_with_offset depth=8
#pragma HLS array_partition variable=s_intermediate_result_with_offset complete
// #pragma HLS RESOURCE variable=s_intermediate_result_with_offset core=FIFO_SRL

    hls::stream<PQ_lookup_t> s_intermediate_result_with_addr[PRIORITY_QUEUE_NUM_L1];
#pragma HLS stream variable=s_intermediate_result_with_addr depth=8
#pragma HLS array_partition variable=s_intermediate_result_with_addr complete
// #pragma HLS RESOURCE variable=s_intermediate_result_with_offset core=FIFO_SRL

    hls::stream<result_t> s_intermediate_result_with_vec_ID[PRIORITY_QUEUE_NUM_L1];
#pragma HLS stream variable=s_intermediate_result_with_vec_ID depth=8
#pragma HLS array_partition variable=s_intermediate_result_with_vec_ID complete
// #pragma HLS RESOURCE variable=s_intermediate_result_with_vec_ID core=FIFO_SRL

    // collecting results from multiple sources need deeper FIFO
    const int intermediate_result_depth = PRIORITY_QUEUE_NUM_L1 * PRIORITY_QUEUE_LEN_L1;
    hls::stream<result_t> s_merged_intermediate_result;
#pragma HLS stream variable=s_merged_intermediate_result depth=intermediate_result_depth

    // WENQI: Here, the number of priority queue must be a constant passed by macro,
    //   I have tried to use the template argument, i.e., PRIORITY_QUEUE_NUM_L1, but HLS has bug on it
    Priority_queue_L1<PQ_out_t, PRIORITY_QUEUE_LEN_L1, Collect_smallest> priority_queue_L1[PRIORITY_QUEUE_NUM_L1];
#pragma HLS array_partition variable=priority_queue_L1 complete


    hls::stream<batch_header_t> s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 6];
#pragma HLS stream variable=s_batch_header_replicated depth=8
#pragma HLS array_partition variable=s_batch_header_replicated complete

	replicate_s_batch_header<PRIORITY_QUEUE_NUM_L1 + 6>(
		s_batch_header, 
		s_batch_header_replicated);

    ////////////////////         Priority Queue Level 1          ////////////////////
    split_single_PQ_result_wrapper<ADC_PE_NUM>(
        s_batch_header_replicated[0],
        s_input, 
        s_scanned_entries_per_query_Priority_queue,
        s_insertion_per_queue_L1,
        s_input_splitted); 

    // 2 CC per insertion
    for (int i = 0; i < PRIORITY_QUEUE_NUM_L1; i++) {
#pragma HLS UNROLL
        // for each individual query, output intermediate_result
        priority_queue_L1[i].insert_wrapper(
            s_batch_header_replicated[1 + i],
            s_insertion_per_queue_L1[i], 
            s_input_splitted[i], 
            s_intermediate_result_with_offset[i]);
    }

    // cell_ID + offset -> vec_ID address
    convert_vec_ID_offset_to_addr<PRIORITY_QUEUE_LEN_L1, PRIORITY_QUEUE_NUM_L1>(
	    nlist,
		s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 1],
        s_nlist_vec_ID_start_addr,
        s_intermediate_result_with_offset,
        s_intermediate_result_with_addr);

    // retrieve vec_ID from DRAM
    convert_addr_to_vec_ID<PRIORITY_QUEUE_LEN_L1, PRIORITY_QUEUE_NUM_L1>(
		s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 2],
        vec_ID_DRAM_0,
        vec_ID_DRAM_1,
        vec_ID_DRAM_2,
        vec_ID_DRAM_3,
        s_intermediate_result_with_addr,
        s_intermediate_result_with_vec_ID); 

    merge_streams<PRIORITY_QUEUE_LEN_L1, PRIORITY_QUEUE_NUM_L1>(
		s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 3],
        s_intermediate_result_with_vec_ID, 
        s_merged_intermediate_result);

    ////////////////////         Priority Queue Level 2          ////////////////////

    hls::stream<int> s_insertion_per_queue_L2;
#pragma HLS stream variable=s_insertion_per_queue_L2 depth=8
// #pragma HLS RESOURCE variable=s_insertion_per_queue_L2 core=FIFO_SRL

    Priority_queue_L2<result_t, TOPK, Collect_smallest> priority_queue_final;

    send_iter_num<PRIORITY_QUEUE_NUM_L1 * PRIORITY_QUEUE_LEN_L1>(
		s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 4],
        s_insertion_per_queue_L2);

    priority_queue_final.insert_wrapper(
		s_batch_header_replicated[PRIORITY_QUEUE_NUM_L1 + 5],
        s_insertion_per_queue_L2,
        s_merged_intermediate_result, 
        output_stream); 
}
