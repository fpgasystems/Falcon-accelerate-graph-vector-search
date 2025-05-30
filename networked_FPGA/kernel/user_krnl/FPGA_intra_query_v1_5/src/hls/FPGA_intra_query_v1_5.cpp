/*
 * Copyright (c) 2020, Systems Group, ETH Zurich
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "../../../../common/include/communication.hpp"
#include "hls_stream.h"

#include "bloom_filter.hpp"
#include "bloom_fetch_compute.hpp"
#include "compute.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "scheduler.hpp"
#include "types.hpp"
#include "utils.hpp"

/*
Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/


void network_input_processing(
	// in init
	const int entry_point_id, // hnsw and nsg always start from a single point

    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
	hls::stream<int>& s_finish_batch,

    // out streams
	hls::stream<int>& s_query_batch_size,
	hls::stream<ap_uint<512>>& s_query_vectors_in,
	hls::stream<int>& s_entry_point_ids
    ) {

    // Format: for each query
    // packet 0: header (batch_size, ) -> batch size as -1 or 0 means terminate
    //   for the following packets, for each query
    // 		packet 1~k: query_vectors

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
	const int AXI_num_vec = D % FLOAT_PER_AXI == 0? D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; 

    while (true) {

        // decode batch search header
        ap_uint<512> input_header = s_kernel_network_in.read();
        ap_uint<32> batch_size_uint = input_header.range(1 * 32 - 1, 0 * 32);

        int batch_size = batch_size_uint;
        
		if (batch_size <= 0) {
			// write finish all 
			s_query_batch_size.write(-1);
			break;
		} else {
			s_query_batch_size.write(batch_size);
		}

        // decode the content of the batch search
        for (int query_id = 0; query_id < batch_size; query_id++) {

			s_entry_point_ids.write(entry_point_id);
            for (int i = 0; i < AXI_num_vec; i++) {
			#pragma HLS pipeline II=1
                s_query_vectors_in.write(s_kernel_network_in.read());
            }
        }

		// wait batch processing finish
		while (s_finish_batch.empty()) {}
		int finish_batch = s_finish_batch.read();
    }
}

void forward_result_streams(
	// in initialization
	const int ef,
	// in runtime (stream)
	hls::stream<int>& s_query_batch_size,
    hls::stream<int>& s_out_ids,
	hls::stream<float>& s_out_dists,
	hls::stream<int>& s_debug_signals,

	// out streams
    hls::stream<int>& s_out_ids_forward,
	hls::stream<float>& s_out_dists_forward,
	hls::stream<int>& s_finish_batch
) {

	bool first_s_query_batch_size = true;
	bool first_iter_s_out_ids = true;
	bool first_iter_s_out_dists = true;
	bool first_iter_s_debug_signals = true;

	int processed_query_num = 0;

	while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

		for (int qid = 0; qid < query_num; qid++) {
			wait_data_fifo_first_iter<int>(ef, s_out_ids, first_iter_s_out_ids);
			wait_data_fifo_first_iter<float>(ef, s_out_dists, first_iter_s_out_dists);

			// use two loops to infer burst per loop
			for (int i = 0; i < ef; i++) {
			#pragma HLS pipeline II=1
				s_out_ids_forward.write(s_out_ids.read());
				s_out_dists_forward.write(s_out_dists.read());
			}
			
			wait_data_fifo_first_iter<int>(debug_size, s_debug_signals, first_iter_s_debug_signals);
			for (int i = 0; i < debug_size; i++) {
			#pragma HLS pipeline II=1
				s_debug_signals.read();
			}
		}


		// finish processing this entire batch
		s_finish_batch.write(1);
	}
}

void network_output_processing(
	// input init
	const int ef,

    // input streams
	hls::stream<int>& s_query_batch_size,
	hls::stream<int>& s_out_ids,
	hls::stream<float>& s_out_dists,

    // output
    hls::stream<ap_uint<512>>& s_kernel_network_out) {

    // Format: for each query
    // packet 0: header (topK == ef)
    // packet 1~k: topK results, including vec_ID (4-byte) array and dist_array (4-byte)
	//    -> size = ceil(topK * 4 / 64) + ceil(topK * 4 / 64)

    // in 512-bit packets
    const int AXI_num_results_vec_ID = ef % INT_PER_AXI == 0? ef / INT_PER_AXI : ef / INT_PER_AXI + 1;
    const int AXI_num_results_dist = ef % FLOAT_PER_AXI == 0? ef / FLOAT_PER_AXI : ef / FLOAT_PER_AXI + 1;

	bool first_s_query_batch_size = true;

    while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

        for (int query_id = 0; query_id < query_num; query_id++) {

            ap_uint<512> output_header = 0;
            ap_uint<32> topK_header = ef;
            output_header.range(31, 0) = topK_header;
            s_kernel_network_out.write(output_header);

            // send vec IDs first
			for (int s = 0; s < AXI_num_results_vec_ID; s++) {
				ap_uint<512> reg_out = 0;
				for (int k = 0; k < INT_PER_AXI && s * INT_PER_AXI + k < ef; k++) {
					int raw_output = s_out_ids.read();
					ap_uint<32> output = *((ap_uint<32>*) (&raw_output));
					reg_out.range(32 * k + 31, 32 * k) = output;
				}
				s_kernel_network_out.write(reg_out);
			}

            // then send dist
            for (int j = 0; j < AXI_num_results_dist; j++) {
				ap_uint<512> reg_out = 0;
				for (int k = 0; k < FLOAT_PER_AXI && j * FLOAT_PER_AXI + k < ef; k++) {
					float raw_output = s_out_dists.read();
					ap_uint<32> output = *((ap_uint<32>*) (&raw_output));
					reg_out.range(32 * k + 31, 32 * k) = output;
				}
				s_kernel_network_out.write(reg_out);
			}
        }
    } 
}


extern "C" {
void FPGA_intra_query_v1_5(
     // Internal Stream
     hls::stream<pkt512>& s_axis_udp_rx, 
     hls::stream<pkt512>& m_axis_udp_tx, 
     hls::stream<pkt256>& s_axis_udp_rx_meta, 
     hls::stream<pkt256>& m_axis_udp_tx_meta, 
     
     hls::stream<pkt16>& m_axis_tcp_listen_port, 
     hls::stream<pkt8>& s_axis_tcp_port_status, 
     hls::stream<pkt64>& m_axis_tcp_open_connection, 
     hls::stream<pkt32>& s_axis_tcp_open_status, 
     hls::stream<pkt16>& m_axis_tcp_close_connection, 
     hls::stream<pkt128>& s_axis_tcp_notification, 
     hls::stream<pkt32>& m_axis_tcp_read_pkg, 
     hls::stream<pkt16>& s_axis_tcp_rx_meta, 
     hls::stream<pkt512>& s_axis_tcp_rx_data, 
     hls::stream<pkt32>& m_axis_tcp_tx_meta, 
     hls::stream<pkt512>& m_axis_tcp_tx_data, 
     hls::stream<pkt64>& s_axis_tcp_tx_status,
     // Rx & Tx
     int useConn,
     // Rx
     int basePortRx, 
     ap_uint<64> expectedRxByteCnt, // for input & output
     // Tx
     int baseIpAddressTx,
     int basePortTx, 
     ap_uint<64> expectedTxPkgCnt,
     int pkgWordCountTx, // number of 64-byte words per packet, e.g, 16 or 22

     //////////     Accelerator kernel     //////////
	// in initialization
	const int query_num, 
	const int query_batch_size,
	const int entry_point_id,
	const int ef, // size of the result priority queue
	const int candidate_queue_runtime_size, 
	const int max_cand_batch_size, 
	const int max_async_stage_num,
	const int runtime_n_bucket_addr_bits,
	const ap_uint<32> hash_seed,
	const int max_bloom_out_burst_size,
	const int max_link_num_base,

    // in runtime (from DRAM)
    ap_uint<512>* db_vectors_chan_0, 
#if N_CHANNEL >= 2
	ap_uint<512>* db_vectors_chan_1,
#endif
#if N_CHANNEL >= 4
	ap_uint<512>* db_vectors_chan_2,
	ap_uint<512>* db_vectors_chan_3,
#endif
#if N_CHANNEL >= 8
	ap_uint<512>* db_vectors_chan_4,
	ap_uint<512>* db_vectors_chan_5,
	ap_uint<512>* db_vectors_chan_6,
	ap_uint<512>* db_vectors_chan_7,
#endif
#if N_CHANNEL >= 16
	ap_uint<512>* db_vectors_chan_8,
	ap_uint<512>* db_vectors_chan_9,
	ap_uint<512>* db_vectors_chan_10,
	ap_uint<512>* db_vectors_chan_11,
	ap_uint<512>* db_vectors_chan_12,
	ap_uint<512>* db_vectors_chan_13,
	ap_uint<512>* db_vectors_chan_14,
	ap_uint<512>* db_vectors_chan_15,
#endif

    const ap_uint<512>* links_base_chan_0
#if N_CHANNEL >= 2
	, const ap_uint<512>* links_base_chan_1
#endif
#if N_CHANNEL >= 4
	, const ap_uint<512>* links_base_chan_2,
	const ap_uint<512>* links_base_chan_3
#endif
#if N_CHANNEL >= 8
	, const ap_uint<512>* links_base_chan_4
	const ap_uint<512>* links_base_chan_5,
	const ap_uint<512>* links_base_chan_6,
	const ap_uint<512>* links_base_chan_7
#endif
#if N_CHANNEL >= 16
	, const ap_uint<512>* links_base_chan_8,
	const ap_uint<512>* links_base_chan_9,
	const ap_uint<512>* links_base_chan_10,
	const ap_uint<512>* links_base_chan_11,
	const ap_uint<512>* links_base_chan_12,
	const ap_uint<512>* links_base_chan_13,
	const ap_uint<512>* links_base_chan_14,
	const ap_uint<512>* links_base_chan_15
#endif
                      ) {

// network 
#pragma HLS INTERFACE axis port = s_axis_udp_rx
#pragma HLS INTERFACE axis port = m_axis_udp_tx
#pragma HLS INTERFACE axis port = s_axis_udp_rx_meta
#pragma HLS INTERFACE axis port = m_axis_udp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_listen_port
#pragma HLS INTERFACE axis port = s_axis_tcp_port_status
#pragma HLS INTERFACE axis port = m_axis_tcp_open_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_open_status
#pragma HLS INTERFACE axis port = m_axis_tcp_close_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_notification
#pragma HLS INTERFACE axis port = m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_data
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_data
#pragma HLS INTERFACE axis port = s_axis_tcp_tx_status


#pragma HLS INTERFACE m_axi port=db_vectors_chan_0 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors0
#if N_CHANNEL >= 2
#pragma HLS INTERFACE m_axi port=db_vectors_chan_1 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors1
#endif
#if N_CHANNEL >= 4
#pragma HLS INTERFACE m_axi port=db_vectors_chan_2 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors2
#pragma HLS INTERFACE m_axi port=db_vectors_chan_3 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors3
#endif
#if N_CHANNEL >= 8
#pragma HLS INTERFACE m_axi port=db_vectors_chan_4 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors4
#pragma HLS INTERFACE m_axi port=db_vectors_chan_5 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors5
#pragma HLS INTERFACE m_axi port=db_vectors_chan_6 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors6
#pragma HLS INTERFACE m_axi port=db_vectors_chan_7 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors7
#endif
#if N_CHANNEL >= 16
#pragma HLS INTERFACE m_axi port=db_vectors_chan_8 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors8
#pragma HLS INTERFACE m_axi port=db_vectors_chan_9 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors9
#pragma HLS INTERFACE m_axi port=db_vectors_chan_10 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors10
#pragma HLS INTERFACE m_axi port=db_vectors_chan_11 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors11
#pragma HLS INTERFACE m_axi port=db_vectors_chan_12 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors12
#pragma HLS INTERFACE m_axi port=db_vectors_chan_13 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors13
#pragma HLS INTERFACE m_axi port=db_vectors_chan_14 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors14
#pragma HLS INTERFACE m_axi port=db_vectors_chan_15 latency=64 num_read_outstanding=64  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemdb_vectors15
#endif

#pragma HLS INTERFACE m_axi port=links_base_chan_0 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base0
#if N_CHANNEL >= 2
#pragma HLS INTERFACE m_axi port=links_base_chan_1 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base1
#endif
#if N_CHANNEL >= 4
#pragma HLS INTERFACE m_axi port=links_base_chan_2 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base2
#pragma HLS INTERFACE m_axi port=links_base_chan_3 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base3
#endif
#if N_CHANNEL >= 8
#pragma HLS INTERFACE m_axi port=links_base_chan_4 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base4
#pragma HLS INTERFACE m_axi port=links_base_chan_5 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base5
#pragma HLS INTERFACE m_axi port=links_base_chan_6 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base6
#pragma HLS INTERFACE m_axi port=links_base_chan_7 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base7
#endif
#if N_CHANNEL >= 16
#pragma HLS INTERFACE m_axi port=links_base_chan_8 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base8
#pragma HLS INTERFACE m_axi port=links_base_chan_9 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base9
#pragma HLS INTERFACE m_axi port=links_base_chan_10 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base10
#pragma HLS INTERFACE m_axi port=links_base_chan_11 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base11
#pragma HLS INTERFACE m_axi port=links_base_chan_12 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base12
#pragma HLS INTERFACE m_axi port=links_base_chan_13 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base13
#pragma HLS INTERFACE m_axi port=links_base_chan_14 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base14
#pragma HLS INTERFACE m_axi port=links_base_chan_15 latency=32 num_read_outstanding=16  num_write_outstanding=1 max_write_burst_length=2 offset=slave bundle=gmemlinks_base15
#endif


#pragma HLS dataflow

////////////////////     Recv     ////////////////////
          
    listenPorts(
        basePortRx, 
        useConn, 
        m_axis_tcp_listen_port, 
        s_axis_tcp_port_status);

    hls::stream<ap_uint<512>> s_kernel_network_in;
#pragma HLS STREAM variable=s_kernel_network_in depth=depth_network_in

    // Wenqi-customized recv function, resolve deadlock in the case that
    //   input data rate >> FPGA query processing rate
    // recvDataSafe(expectedRxByteCnt, 
    //     s_kernel_network_in,
    //     s_axis_tcp_notification, 
    //     m_axis_tcp_read_pkg, 
    //     s_axis_tcp_rx_meta, 
    //     s_axis_tcp_rx_data);
    recvData(expectedRxByteCnt, 
        s_kernel_network_in,
        s_axis_tcp_notification, 
        m_axis_tcp_read_pkg, 
        s_axis_tcp_rx_meta, 
        s_axis_tcp_rx_data);


////////////////////     Network Input     ////////////////////


    hls::stream<int> s_query_batch_size;
#pragma HLS stream variable=s_query_batch_size depth=depth_control
    
    hls::stream<ap_uint<512> > s_query_vectors_in;
#pragma HLS stream variable=s_query_vectors_in depth=depth_query_vectors
    
    hls::stream<int> s_entry_point_ids;
#pragma HLS stream variable=s_entry_point_ids depth=depth_control

	hls::stream<int> s_finish_batch;
#pragma HLS stream variable=s_finish_batch depth=depth_control

	network_input_processing(
		// in init
		entry_point_id, // hnsw and nsg always start from a single point

		// in runtime
		s_kernel_network_in,
		s_finish_batch,

		// out streams
		s_query_batch_size,
		s_query_vectors_in,
		s_entry_point_ids
	);

////////////////////     Accelerator Simulation     ////////////////////


    hls::stream<int> s_finish_query_task_scheduler; // finish the current query
#pragma HLS stream variable=s_finish_query_task_scheduler depth=depth_control

    hls::stream<int> s_finish_query_results_collection; // finish all queries
#pragma HLS stream variable=s_finish_query_results_collection depth=depth_control
	
	hls::stream<ap_uint<512>> s_query_vectors_replicated[N_CHANNEL];
#pragma HLS stream variable=s_query_vectors_replicated depth=depth_query_vectors

// 	hls::stream<result_t> s_entry_point_base_level;
// #pragma HLS stream variable=s_entry_point_base_level depth=depth_control

	hls::stream<int> s_cand_batch_size;
#pragma HLS stream variable=s_cand_batch_size depth=depth_control

    hls::stream<cand_t> s_top_candidates; // current top candidates
#pragma HLS stream variable=s_top_candidates depth=depth_data

	hls::stream<int> s_num_inserted_candidates;
#pragma HLS stream variable=s_num_inserted_candidates depth=depth_control

	hls::stream<result_t> s_inserted_candidates;
#pragma HLS stream variable=s_inserted_candidates depth=depth_data

	hls::stream<int> s_num_valid_candidates_base_level_total;
#pragma HLS stream variable=s_num_valid_candidates_base_level_total depth=depth_control

	hls::stream<result_t> s_distances_base_level;
#pragma HLS stream variable=s_distances_base_level depth=depth_data

	hls::stream<float> s_largest_result_queue_elements;
#pragma HLS stream variable=s_largest_result_queue_elements depth=depth_control	

	hls::stream<int> s_debug_signals;
#pragma HLS stream variable=s_debug_signals depth=depth_control

	const int rep_factor_s_largest_result_queue_elements = 1 + N_CHANNEL;
	hls::stream<float> s_largest_result_queue_elements_replicated[rep_factor_s_largest_result_queue_elements];
#pragma HLS stream variable=s_largest_result_queue_elements_replicated depth=depth_control		

// 	hls::stream<int> s_debug_num_vec_base_layer;
// #pragma HLS stream variable=s_debug_num_vec_base_layer depth=depth_control

	hls::stream<int> s_finish_query_replicate_s_largest_result_queue_elements;
#pragma HLS stream variable=s_finish_query_replicate_s_largest_result_queue_elements depth=depth_control

	// replicate s_query_batch_size to multiple streams
	const int replicate_factor_s_query_batch_size = 2 * N_CHANNEL + 10;
	hls::stream<int> s_query_batch_size_replicated[replicate_factor_s_query_batch_size];
#pragma HLS stream variable=s_query_batch_size_replicated depth=depth_control

	replicate_s_query_batch_size<replicate_factor_s_query_batch_size>(
		s_query_batch_size,
		s_query_batch_size_replicated
	);

	// controls the traversal and maintains the candidate queue
	task_scheduler(
		candidate_queue_runtime_size,
		max_cand_batch_size,
		max_async_stage_num,

		// in streams
		s_query_batch_size_replicated[0],
		s_query_vectors_in,	
		s_entry_point_ids, 
		s_num_inserted_candidates,
		s_inserted_candidates,
		s_largest_result_queue_elements_replicated[0],
		// s_debug_num_vec_base_layer,
		s_finish_query_replicate_s_largest_result_queue_elements,
		
		// out streams
		s_query_vectors_replicated,
		// s_entry_point_base_level,
		s_cand_batch_size,
		s_top_candidates,
		s_debug_signals,
		s_finish_query_task_scheduler
	);


	hls::stream<int> s_fetch_batch_size;
#pragma HLS stream variable=s_fetch_batch_size depth=depth_control

    hls::stream<ap_uint<512>> s_neighbor_ids_raw;
#pragma HLS stream variable=s_neighbor_ids_raw depth=depth_data

    hls::stream<int> s_finish_query_fetch_neighbor_ids; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_neighbor_ids depth=depth_control

	fetch_neighbor_ids(
		max_link_num_base,
		// in runtime (should from DRAM)
    	links_base_chan_0,
#if N_CHANNEL >= 2
		links_base_chan_1,
#endif
#if N_CHANNEL >= 4
		links_base_chan_2,
		links_base_chan_3,
#endif
#if N_CHANNEL >= 8
		links_base_chan_4,
		links_base_chan_5,
		links_base_chan_6,
		links_base_chan_7,
#endif
#if N_CHANNEL >= 16
		links_base_chan_8,
		links_base_chan_9,
		links_base_chan_10,
		links_base_chan_11,
		links_base_chan_12,
		links_base_chan_13,
		links_base_chan_14,
		links_base_chan_15,
#endif
		// in runtime (stream)
		s_query_batch_size_replicated[1],
		s_top_candidates,
		s_finish_query_task_scheduler,

		// out (stream)
		s_neighbor_ids_raw,
		s_finish_query_fetch_neighbor_ids
	);

    hls::stream<int> s_num_neighbors_base_level_per_channel[N_CHANNEL]; // number of neighbors of the current candidate
#pragma HLS stream variable=s_num_neighbors_base_level_per_channel depth=depth_control

	hls::stream<cand_t> s_fetched_neighbor_ids_per_channel[N_CHANNEL]; 
#pragma HLS stream variable=s_fetched_neighbor_ids_per_channel depth=depth_data

    hls::stream<int> s_finish_query_split_tasks_to_channels; // finish all queries
#pragma HLS stream variable=s_finish_query_split_tasks_to_channels depth=depth_control

	split_tasks_to_channels(
		max_link_num_base,

		// in streams
		s_query_batch_size_replicated[2],
		s_cand_batch_size,
		s_neighbor_ids_raw,
		s_finish_query_fetch_neighbor_ids,

		// out streams
		s_num_neighbors_base_level_per_channel,
		s_fetched_neighbor_ids_per_channel,
		s_finish_query_split_tasks_to_channels
	);

    hls::stream<int> s_finish_query_split_tasks_to_channels_replicated[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_query_split_tasks_to_channels_replicated depth=depth_control

	replicate_s_finish<N_CHANNEL>(
		s_query_batch_size_replicated[3],
		s_finish_query_split_tasks_to_channels,
		s_finish_query_split_tasks_to_channels_replicated
	);

	hls::stream<int> s_num_valid_candidates_base_level_total_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_num_valid_candidates_base_level_total_per_channel depth=depth_control

	hls::stream<result_t> s_distances_base_level_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_distances_base_level_per_channel depth=depth_data

    hls::stream<int> s_finish_query_bloom_fetch_compute_per_channel[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_query_bloom_fetch_compute_per_channel depth=depth_control


	// using loop unrolling for bloom_fetch_compute can lead to compilation error
	//   due to failed dataflow checking in HLS, when using inline pragma for bloom_fetch_compute

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_0,

		// in streams
		s_query_batch_size_replicated[0 + 4],
		s_query_vectors_replicated[0],
		s_num_neighbors_base_level_per_channel[0],
		s_fetched_neighbor_ids_per_channel[0],
		s_finish_query_split_tasks_to_channels_replicated[0],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[0],
		s_distances_base_level_per_channel[0],
		s_finish_query_bloom_fetch_compute_per_channel[0]
	);
	#if N_CHANNEL >= 2
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_1,

		// in streams
		s_query_batch_size_replicated[1 + 4],
		s_query_vectors_replicated[1],
		s_num_neighbors_base_level_per_channel[1],
		s_fetched_neighbor_ids_per_channel[1],
		s_finish_query_split_tasks_to_channels_replicated[1],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[1],
		s_distances_base_level_per_channel[1],
		s_finish_query_bloom_fetch_compute_per_channel[1]
	);
	#endif
	#if N_CHANNEL >= 4
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_2,

		// in streams
		s_query_batch_size_replicated[2 + 4],
		s_query_vectors_replicated[2],
		s_num_neighbors_base_level_per_channel[2],
		s_fetched_neighbor_ids_per_channel[2],
		s_finish_query_split_tasks_to_channels_replicated[2],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[2],
		s_distances_base_level_per_channel[2],
		s_finish_query_bloom_fetch_compute_per_channel[2]
	);
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_3,

		// in streams
		s_query_batch_size_replicated[3 + 4],
		s_query_vectors_replicated[3],
		s_num_neighbors_base_level_per_channel[3],
		s_fetched_neighbor_ids_per_channel[3],
		s_finish_query_split_tasks_to_channels_replicated[3],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[3],
		s_distances_base_level_per_channel[3],
		s_finish_query_bloom_fetch_compute_per_channel[3]
	);
	#endif
	#if N_CHANNEL >= 8
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		
		// in runtime (from DRAM)
		db_vectors_chan_4,

		// in streams
		s_query_batch_size_replicated[4 + 4],
		s_query_vectors_replicated[4],
		s_num_neighbors_base_level_per_channel[4],
		s_fetched_neighbor_ids_per_channel[4],
		s_finish_query_split_tasks_to_channels_replicated[4],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[4],
		s_distances_base_level_per_channel[4],
		s_finish_query_bloom_fetch_compute_per_channel[4]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_5,

		// in streams
		s_query_batch_size_replicated[5 + 4],
		s_query_vectors_replicated[5],
		s_num_neighbors_base_level_per_channel[5],
		s_fetched_neighbor_ids_per_channel[5],
		s_finish_query_split_tasks_to_channels_replicated[5],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[5],
		s_distances_base_level_per_channel[5],
		s_finish_query_bloom_fetch_compute_per_channel[5]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_6,

		// in streams
		s_query_batch_size_replicated[6 + 4],
		s_query_vectors_replicated[6],
		s_num_neighbors_base_level_per_channel[6],
		s_fetched_neighbor_ids_per_channel[6],
		s_finish_query_split_tasks_to_channels_replicated[6],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[6],
		s_distances_base_level_per_channel[6],
		s_finish_query_bloom_fetch_compute_per_channel[6]
	);
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_7,

		// in streams
		s_query_batch_size_replicated[7 + 4],
		s_query_vectors_replicated[7],
		s_num_neighbors_base_level_per_channel[7],
		s_fetched_neighbor_ids_per_channel[7],
		s_finish_query_split_tasks_to_channels_replicated[7],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[7],
		s_distances_base_level_per_channel[7],
		s_finish_query_bloom_fetch_compute_per_channel[7]
	);
	#endif
	#if N_CHANNEL >= 16
	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_8,

		// in streams
		s_query_batch_size_replicated[8 + 4],
		s_query_vectors_replicated[8],
		s_num_neighbors_base_level_per_channel[8],
		s_fetched_neighbor_ids_per_channel[8],
		s_finish_query_split_tasks_to_channels_replicated[8],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[8],
		s_distances_base_level_per_channel[8],
		s_finish_query_bloom_fetch_compute_per_channel[8]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_9,

		// in streams
		s_query_batch_size_replicated[9 + 4],
		s_query_vectors_replicated[9],
		s_num_neighbors_base_level_per_channel[9],
		s_fetched_neighbor_ids_per_channel[9],
		s_finish_query_split_tasks_to_channels_replicated[9],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[9],
		s_distances_base_level_per_channel[9],
		s_finish_query_bloom_fetch_compute_per_channel[9]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,
		
		// in runtime (from DRAM)
		db_vectors_chan_10,

		// in streams
		s_query_batch_size_replicated[10 + 4],
		s_query_vectors_replicated[10],
		s_num_neighbors_base_level_per_channel[10],
		s_fetched_neighbor_ids_per_channel[10],
		s_finish_query_split_tasks_to_channels_replicated[10],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[10],
		s_distances_base_level_per_channel[10],
		s_finish_query_bloom_fetch_compute_per_channel[10]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_11,

		// in streams
		s_query_batch_size_replicated[11 + 4],
		s_query_vectors_replicated[11],
		s_num_neighbors_base_level_per_channel[11],
		s_fetched_neighbor_ids_per_channel[11],
		s_finish_query_split_tasks_to_channels_replicated[11],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[11],
		s_distances_base_level_per_channel[11],
		s_finish_query_bloom_fetch_compute_per_channel[11]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_12,

		// in streams
		s_query_batch_size_replicated[12 + 4],
		s_query_vectors_replicated[12],
		s_num_neighbors_base_level_per_channel[12],
		s_fetched_neighbor_ids_per_channel[12],
		s_finish_query_split_tasks_to_channels_replicated[12],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[12],
		s_distances_base_level_per_channel[12],
		s_finish_query_bloom_fetch_compute_per_channel[12]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_13,

		// in streams
		s_query_batch_size_replicated[13 + 4],
		s_query_vectors_replicated[13],
		s_num_neighbors_base_level_per_channel[13],
		s_fetched_neighbor_ids_per_channel[13],
		s_finish_query_split_tasks_to_channels_replicated[13],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[13],
		s_distances_base_level_per_channel[13],
		s_finish_query_bloom_fetch_compute_per_channel[13]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_14,

		// in streams
		s_query_batch_size_replicated[14 + 4],
		s_query_vectors_replicated[14],
		s_num_neighbors_base_level_per_channel[14],
		s_fetched_neighbor_ids_per_channel[14],
		s_finish_query_split_tasks_to_channels_replicated[14],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[14],
		s_distances_base_level_per_channel[14],
		s_finish_query_bloom_fetch_compute_per_channel[14]
	);

	bloom_fetch_compute(
		// in initialization
		runtime_n_bucket_addr_bits,
		hash_seed,
		max_bloom_out_burst_size,

		// in runtime (from DRAM)
		db_vectors_chan_15,

		// in streams
		s_query_batch_size_replicated[15 + 4],
		s_query_vectors_replicated[15],
		s_num_neighbors_base_level_per_channel[15],
		s_fetched_neighbor_ids_per_channel[15],
		s_finish_query_split_tasks_to_channels_replicated[15],

		// out streams
		s_num_valid_candidates_base_level_total_per_channel[15],
		s_distances_base_level_per_channel[15],
		s_finish_query_bloom_fetch_compute_per_channel[15]
	);
	#endif

	hls::stream<int> s_num_valid_candidates_base_level_filtered_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_num_valid_candidates_base_level_filtered_per_channel depth=depth_control

	hls::stream<result_t> s_distances_base_filtered_per_channel[N_CHANNEL];
#pragma HLS stream variable=s_distances_base_filtered_per_channel depth=depth_data

    hls::stream<int> s_finish_filter_computed_distances_per_channel[N_CHANNEL]; // finish all queries
#pragma HLS stream variable=s_finish_filter_computed_distances_per_channel depth=depth_control	

filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[0 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[0],
    s_largest_result_queue_elements_replicated[0 + 1],
    s_distances_base_level_per_channel[0],
    s_finish_query_bloom_fetch_compute_per_channel[0],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[0],
    s_distances_base_filtered_per_channel[0],
    s_finish_filter_computed_distances_per_channel[0]
);
#if N_CHANNEL >= 2
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[1 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[1],
    s_largest_result_queue_elements_replicated[1 + 1],
    s_distances_base_level_per_channel[1],
    s_finish_query_bloom_fetch_compute_per_channel[1],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[1],
    s_distances_base_filtered_per_channel[1],
    s_finish_filter_computed_distances_per_channel[1]
);
#endif
#if N_CHANNEL >= 4
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[2 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[2],
    s_largest_result_queue_elements_replicated[2 + 1],
    s_distances_base_level_per_channel[2],
    s_finish_query_bloom_fetch_compute_per_channel[2],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[2],
    s_distances_base_filtered_per_channel[2],
    s_finish_filter_computed_distances_per_channel[2]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[3 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[3],
    s_largest_result_queue_elements_replicated[3 + 1],
    s_distances_base_level_per_channel[3],
    s_finish_query_bloom_fetch_compute_per_channel[3],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[3],
    s_distances_base_filtered_per_channel[3],
    s_finish_filter_computed_distances_per_channel[3]
);
#endif
#if N_CHANNEL >= 8
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[4 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[4],
    s_largest_result_queue_elements_replicated[4 + 1],
    s_distances_base_level_per_channel[4],
    s_finish_query_bloom_fetch_compute_per_channel[4],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[4],
    s_distances_base_filtered_per_channel[4],
    s_finish_filter_computed_distances_per_channel[4]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[5 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[5],
    s_largest_result_queue_elements_replicated[5 + 1],
    s_distances_base_level_per_channel[5],
    s_finish_query_bloom_fetch_compute_per_channel[5],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[5],
    s_distances_base_filtered_per_channel[5],
    s_finish_filter_computed_distances_per_channel[5]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[6 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[6],
    s_largest_result_queue_elements_replicated[6 + 1],
    s_distances_base_level_per_channel[6],
    s_finish_query_bloom_fetch_compute_per_channel[6],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[6],
    s_distances_base_filtered_per_channel[6],
    s_finish_filter_computed_distances_per_channel[6]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[7 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[7],
    s_largest_result_queue_elements_replicated[7 + 1],
    s_distances_base_level_per_channel[7],
    s_finish_query_bloom_fetch_compute_per_channel[7],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[7],
    s_distances_base_filtered_per_channel[7],
    s_finish_filter_computed_distances_per_channel[7]
);
#endif
#if N_CHANNEL >= 16
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[8 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[8],
    s_largest_result_queue_elements_replicated[8 + 1],
    s_distances_base_level_per_channel[8],
    s_finish_query_bloom_fetch_compute_per_channel[8],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[8],
    s_distances_base_filtered_per_channel[8],
    s_finish_filter_computed_distances_per_channel[8]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[9 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[9],
    s_largest_result_queue_elements_replicated[9 + 1],
    s_distances_base_level_per_channel[9],
    s_finish_query_bloom_fetch_compute_per_channel[9],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[9],
    s_distances_base_filtered_per_channel[9],
    s_finish_filter_computed_distances_per_channel[9]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[10 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[10],
    s_largest_result_queue_elements_replicated[10 + 1],
    s_distances_base_level_per_channel[10],
    s_finish_query_bloom_fetch_compute_per_channel[10],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[10],
    s_distances_base_filtered_per_channel[10],
    s_finish_filter_computed_distances_per_channel[10]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[11 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[11],
    s_largest_result_queue_elements_replicated[11 + 1],
    s_distances_base_level_per_channel[11],
    s_finish_query_bloom_fetch_compute_per_channel[11],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[11],
    s_distances_base_filtered_per_channel[11],
    s_finish_filter_computed_distances_per_channel[11]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[12 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[12],
    s_largest_result_queue_elements_replicated[12 + 1],
    s_distances_base_level_per_channel[12],
    s_finish_query_bloom_fetch_compute_per_channel[12],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[12],
    s_distances_base_filtered_per_channel[12],
    s_finish_filter_computed_distances_per_channel[12]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[13 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[13],
    s_largest_result_queue_elements_replicated[13 + 1],
    s_distances_base_level_per_channel[13],
    s_finish_query_bloom_fetch_compute_per_channel[13],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[13],
    s_distances_base_filtered_per_channel[13],
    s_finish_filter_computed_distances_per_channel[13]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[14 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[14],
    s_largest_result_queue_elements_replicated[14 + 1],
    s_distances_base_level_per_channel[14],
    s_finish_query_bloom_fetch_compute_per_channel[14],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[14],
    s_distances_base_filtered_per_channel[14],
    s_finish_filter_computed_distances_per_channel[14]
);
filter_computed_distances(
    // in stream
    s_query_batch_size_replicated[15 + N_CHANNEL + 4],
    s_num_valid_candidates_base_level_total_per_channel[15],
    s_largest_result_queue_elements_replicated[15 + 1],
    s_distances_base_level_per_channel[15],
    s_finish_query_bloom_fetch_compute_per_channel[15],

    // out stream
    s_num_valid_candidates_base_level_filtered_per_channel[15],
    s_distances_base_filtered_per_channel[15],
    s_finish_filter_computed_distances_per_channel[15]
);
#endif

    hls::stream<int> s_finish_query_filter_computed_distances; // finish all queries
#pragma HLS stream variable=s_finish_query_filter_computed_distances depth=depth_control

	gather_s_finish<N_CHANNEL>(
    	s_query_batch_size_replicated[2 * N_CHANNEL + 4],
		s_finish_filter_computed_distances_per_channel,
		s_finish_query_filter_computed_distances
	);

    hls::stream<int> s_finish_gather_distances_from_channels; // finish all queries
#pragma HLS stream variable=s_finish_gather_distances_from_channels depth=depth_control

	gather_distances_from_channels(
    	s_query_batch_size_replicated[2 * N_CHANNEL + 5],
		s_num_valid_candidates_base_level_filtered_per_channel,
		s_distances_base_filtered_per_channel,
		s_finish_query_filter_computed_distances,

		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_gather_distances_from_channels
	);

	hls::stream<int> s_out_ids;
#pragma HLS stream variable=s_out_ids depth=depth_data

	hls::stream<float> s_out_dists;
#pragma HLS stream variable=s_out_dists depth=depth_data

	results_collection(
		// in (initialization)
		ef,
		// in runtime (stream)
    	s_query_batch_size_replicated[2 * N_CHANNEL + 6],
		// s_entry_point_base_level,
		s_num_valid_candidates_base_level_total,
		s_distances_base_level,
		s_finish_gather_distances_from_channels,

		// out (stream)
		s_inserted_candidates,
		s_num_inserted_candidates,
		s_largest_result_queue_elements,
		// s_debug_num_vec_base_layer,
		s_finish_query_results_collection,
		s_out_ids,
		s_out_dists
	);

	replicate_s_control<rep_factor_s_largest_result_queue_elements, float>(
		// in (stream)
    	s_query_batch_size_replicated[2 * N_CHANNEL + 7],
		s_largest_result_queue_elements,
		s_finish_query_results_collection,
		
		// out (stream)
		s_largest_result_queue_elements_replicated,
		s_finish_query_replicate_s_largest_result_queue_elements
	);

    hls::stream<int> s_out_ids_forward;
#pragma HLS stream variable=s_out_ids_forward depth=depth_data

    hls::stream<float> s_out_dists_forward;
#pragma HLS stream variable=s_out_dists_forward depth=depth_data

	forward_result_streams(
		ef,

		// in streams
		s_query_batch_size_replicated[2 * N_CHANNEL + 8], // -1: stop
		s_out_ids,
		s_out_dists,
		s_debug_signals,

		// out streams
		s_out_ids_forward,
		s_out_dists_forward,
		s_finish_batch
	);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=depth_data

	network_output_processing(
		// input init
		ef,

		// input streams
		s_query_batch_size_replicated[2 * N_CHANNEL + 9],
		s_out_ids_forward,
		s_out_dists_forward,

		// output
		s_kernel_network_out);

////////////////////     Send     ////////////////////

    ap_uint<16> sessionID [8];

    openConnections(
        useConn, 
        baseIpAddressTx, 
        basePortTx, 
        m_axis_tcp_open_connection, 
        s_axis_tcp_open_status, 
        sessionID);

    ap_uint<64> expectedTxByteCnt = expectedTxPkgCnt * pkgWordCountTx * 64;
    
    // Wenqi: for all the iterations, only send out tx_meta when input data is available
    sendDataProtected(
    // sendData(
        m_axis_tcp_tx_meta, 
        m_axis_tcp_tx_data, 
        s_axis_tcp_tx_status, 
        s_kernel_network_out, 
        sessionID,
        useConn, 
        expectedTxByteCnt, 
        pkgWordCountTx);


////////////////////     Tie off     ////////////////////

    tie_off_udp(s_axis_udp_rx, 
        m_axis_udp_tx, 
        s_axis_udp_rx_meta, 
        m_axis_udp_tx_meta);

    tie_off_tcp_close_con(m_axis_tcp_close_connection);

}

}