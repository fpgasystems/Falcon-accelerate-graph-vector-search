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

#include "constants.hpp"
#include "utils.hpp"
#include "types.hpp"

/*
Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/


void network_input_processing(
	// in init
	const int entry_point_id, // hnsw and nsg always start from a single point

    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
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
    }
}

void simulated_accelerator_kernel(
	// input init
	const int ef,

    // input streams
	hls::stream<int>& s_query_batch_size,
	hls::stream<ap_uint<512>>& s_query_vectors_in,
	hls::stream<int>& s_entry_point_ids,

	// input DRAM
	float* in_DDR,

    // output streams
	hls::stream<int>& s_out_ids,
	hls::stream<float>& s_out_dists) {

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
	const int AXI_num_vec = D % FLOAT_PER_AXI == 0? D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; 

	float bias = in_DDR[0];

	bool first_s_query_batch_size = true;

    while (true) {

		wait_data_fifo_first_iter<int>(
			1, s_query_batch_size, first_s_query_batch_size);
		int query_num = s_query_batch_size.read();
		if (query_num == -1) {
			break;
		}

        for (int query_id = 0; query_id < query_num; query_id++) {

			// consume input
			int entry_point_id = s_entry_point_ids.read();
			ap_uint<512> reg_in;
            for (int i = 0; i < AXI_num_vec; i++) {
			#pragma HLS pipeline II=1
                reg_in = s_query_vectors_in.read();
            }
			
			int out_id = entry_point_id;
			ap_uint<32> uint_out_dist = reg_in.range(31, 0);
			float out_dist = *((float*) (&uint_out_dist)) + bias;

			// generate output
			for (int i = 0; i < ef; i++) {
				s_out_ids.write(entry_point_id);
				s_out_dists.write(out_dist);
			}
        }
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
void network_sim_graph(
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
    // in init
    const int ef,
    const int entry_point_id, 
    float* in_DDR
    // data bank
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

#pragma HLS INTERFACE m_axi port=in_DDR offset=slave bundle=gmem0

#pragma HLS dataflow

////////////////////     Recv     ////////////////////
          
    listenPorts(
        basePortRx, 
        useConn, 
        m_axis_tcp_listen_port, 
        s_axis_tcp_port_status);

    hls::stream<ap_uint<512>> s_kernel_network_in;
#pragma HLS STREAM variable=s_kernel_network_in depth=2048

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
#pragma HLS stream variable=s_query_batch_size depth=512
    
    hls::stream<ap_uint<512> > s_query_vectors_in;
#pragma HLS stream variable=s_query_vectors_in depth=512
    
    hls::stream<int> s_entry_point_ids;
#pragma HLS stream variable=s_entry_point_ids depth=512

	network_input_processing(
		// in init
		entry_point_id, // hnsw and nsg always start from a single point

		// in runtime
		s_kernel_network_in,
		// out streams
		s_query_batch_size,
		s_query_vectors_in,
		s_entry_point_ids
	);

////////////////////     Accelerator Simulation     ////////////////////

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

	simulated_accelerator_kernel(
		// input init
		ef,

		// input streams
		s_query_batch_size_replicated[0],
		s_query_vectors_in,
		s_entry_point_ids,

		// input DRAM
		in_DDR,

		// output streams
		s_out_ids,
		s_out_dists);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=512

	network_output_processing(
		// input init
		ef,

		// input streams
		s_query_batch_size_replicated[1],
		s_out_ids,
		s_out_dists,

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