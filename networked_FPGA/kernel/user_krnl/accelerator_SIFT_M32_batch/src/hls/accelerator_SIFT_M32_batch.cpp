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

#include "ADC.hpp"
#include "constants.hpp"
#include "DRAM_utils.hpp"
#include "helpers.hpp"
#include "hierarchical_priority_queue.hpp"
#include "LUT_construction.hpp"
#include "types.hpp"

/*
Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/


void network_input_processing(
    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
    // out
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<int>& s_cell_ID,
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors
    ) {

    // Format: foe each query
    // packet 0: header (batch_size, nprobe, terminate)
    //   for the following packets, for each query
    // 		packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
    // 		packet k~n: query_vectors
    // 		packet n~m: center_vectors

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    while (true) {

        // decode batch search header
        ap_uint<512> input_header = s_kernel_network_in.read();
        ap_uint<32> batch_size_uint = input_header.range(1 * 32 - 1, 0 * 32);
        ap_uint<32> nprobe_uint = input_header.range(2 * 32 - 1, 1 * 32);
        ap_uint<32> terminate_uint = input_header.range(3 * 32 - 1, 2 * 32);

        int batch_size = batch_size_uint;
        int nprobe = nprobe_uint;
        int terminate = terminate_uint;
        
        batch_header_t batch_header;
        batch_header.batch_size = batch_size;
        batch_header.nprobe = nprobe;
        batch_header.terminate = terminate;
        s_batch_header.write(batch_header);

        // termination detection
        if (terminate) {
            break;
        }

        // num packets for cell IDs
        int size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;

        // decode the content of the batch search
        for (int query_id = 0; query_id < batch_size; query_id++) {

            // cell_IDs
            for (int i = 0; i < size_cell_IDs; i++) {

                ap_uint<512> pkt = s_kernel_network_in.read();

                for (int j = 0; j < 16; j++) {

                    ap_uint<32> cell_ID_uint = pkt.range(32 * j + 31, 32 * j);
                    int cell_ID = *((int*) (&cell_ID_uint));

                    int cell_count = i * 16 + j;
                    if (cell_count < nprobe) {
                        s_cell_ID.write(cell_ID);
                    }
                }
            }

            // query vec
            for (int i = 0; i < size_query_vector; i++) {
                s_query_vectors.write(s_kernel_network_in.read());
            }

            // center vec
            for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {
                for (int i = 0; i < size_center_vector; i++) {
                    s_center_vectors.write(s_kernel_network_in.read());
                }
            }
        }
    }
}


void network_output_processing(
    // input
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<result_t> &s_output, 
    // output
    hls::stream<ap_uint<512>>& s_kernel_network_out) {

    // Format: for each query
    // packet 0: header (topK)
    // packet 1~k: topK results, including vec_ID (8-byte) array and dist_array (4-byte)
	//    -> size = ceil(topK * 8 / 64) + ceil(topK * 4 / 64)

    // in 512-bit packets
    const int size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    const int size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;

    ap_uint<64> vec_ID_buffer [size_results_vec_ID * (512 / 64)] = { 0 };
    float dist_buffer[size_results_dist * (512 / 32)] = { 0 };

    while (true) {

        batch_header_t batch_header = s_batch_header.read();
        int batch_size = batch_header.batch_size;
        int nprobe = batch_header.nprobe;
        int terminate = batch_header.terminate;

        // termination detection
        if (terminate) {
            break;
        }

        for (int query_id = 0; query_id < batch_size; query_id++) {

            ap_uint<512> output_header = 0;
            ap_uint<32> topK_header = PRIORITY_QUEUE_LEN_L2;
            output_header.range(31, 0) = topK_header;
            s_kernel_network_out.write(output_header);

            for (int k = 0; k < TOPK; k++) {
                result_t raw_output = s_output.read();
                vec_ID_buffer[k] = raw_output.vec_ID;
                dist_buffer[k] = raw_output.dist;
            }

            // send vec IDs first
            for (int j = 0; j < size_results_vec_ID; j++) {
                ap_uint<512> pkt = 0;

                for (int k = 0; k < 512 / 64; k++) {
                    
                        ap_uint<64> vec_ID = vec_ID_buffer[j * 512 / 64 + k];
                        pkt.range(64 * k + 63, 64 * k) = vec_ID;
                        
                }
                s_kernel_network_out.write(pkt);
            }

            // then send dist
            for (int j = 0; j < size_results_dist; j++) {
                ap_uint<512> pkt = 0;

                for (int k = 0; k < 512 / 32; k++) {
                    
                        float dist = dist_buffer[j * 512 / 32 + k];
                        ap_uint<32> dist_uint = *((ap_uint<32>*) (&dist));
                        pkt.range(32 * k + 31, 32 * k) = dist_uint;
                }
                s_kernel_network_out.write(pkt);
            }
        }
    } 
}


extern "C" {
void accelerator_SIFT_M32_batch(
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
    int nlist,
    int* meta_data_init, // which consists of the following three stuffs:
                    // int* nlist_PQ_codes_start_addr,
                    // int* nlist_vec_ID_start_addr,
                    // int* nlist_num_vecs,
                    // float* product_quantizer

    // in runtime (should from DRAM)
    const ap_uint<512>* PQ_codes_DRAM_0,
    const ap_uint<512>* PQ_codes_DRAM_1,
    const ap_uint<512>* PQ_codes_DRAM_2,
    const ap_uint<512>* PQ_codes_DRAM_3,
    ap_uint<64>* vec_ID_DRAM_0,
    ap_uint<64>* vec_ID_DRAM_1,
    ap_uint<64>* vec_ID_DRAM_2,
    ap_uint<64>* vec_ID_DRAM_3
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

#pragma HLS INTERFACE s_axilite port=useConn // bundle = control
#pragma HLS INTERFACE s_axilite port=basePortRx // bundle = control
#pragma HLS INTERFACE s_axilite port=expectedRxByteCnt // bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddressTx // bundle=control
#pragma HLS INTERFACE s_axilite port=basePortTx // bundle = control
#pragma HLS INTERFACE s_axilite port=expectedTxPkgCnt // bundle = control
#pragma HLS INTERFACE s_axilite port=pkgWordCountTx // bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = // control

// accelerator kernel 
#pragma HLS INTERFACE s_axilite port=nlist // bundle = control

#pragma HLS INTERFACE m_axi port=meta_data_init offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_0 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_1 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_2 offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=PQ_codes_DRAM_3 offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_0  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_1  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_2  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=vec_ID_DRAM_3  offset=slave bundle=gmem12


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

    hls::stream<batch_header_t> s_batch_header;
#pragma HLS stream variable=s_batch_header depth=512

    const int n_batch_header_streams = 8 + ADC_PE_NUM;
    hls::stream<batch_header_t> s_batch_header_replicated[n_batch_header_streams];
#pragma HLS stream variable=s_batch_header_replicated depth=8
    hls::stream<int> s_cell_ID;
#pragma HLS stream variable=s_cell_ID depth=512
    
    hls::stream<ap_uint<512> > s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=512
    
    hls::stream<ap_uint<512> > s_center_vectors;
#pragma HLS stream variable=s_center_vectors depth=512

    network_input_processing(
        // in runtime
        s_kernel_network_in,
        // out
        s_batch_header,
        s_cell_ID,
        s_query_vectors, 
        s_center_vectors);


    replicate_s_batch_header<n_batch_header_streams>(
        s_batch_header,
        s_batch_header_replicated);

////////////////////     0. Initialization     ////////////////////

    hls::stream<int> s_nlist_PQ_codes_start_addr;
#pragma HLS stream variable=s_nlist_PQ_codes_start_addr depth=512

    hls::stream<int> s_nlist_vec_ID_start_addr; // the top 10 numbers
#pragma HLS stream variable=s_nlist_vec_ID_start_addr depth=512
    
    hls::stream<int> s_nlist_num_vecs;
#pragma HLS stream variable=s_nlist_num_vecs depth=512

    hls::stream<float> s_product_quantizer_init;
#pragma HLS stream variable=s_product_quantizer_init depth=512

    load_meta_data(
        nlist,
        meta_data_init,
        s_nlist_PQ_codes_start_addr,
        s_nlist_vec_ID_start_addr,
        s_nlist_num_vecs,
        s_product_quantizer_init);

////////////////////     1. Construct LUT     ////////////////////


    hls::stream<distance_LUT_parallel_t> s_distance_LUT[ADC_PE_NUM + 1];
#pragma HLS stream variable=s_distance_LUT depth=8
#pragma HLS array_partition variable=s_distance_LUT complete
// #pragma HLS resource variable=s_distance_LUT core=FIFO_SRL

    LUT_construction_wrapper(
		// init
		s_product_quantizer_init,
		// runtime input from network
		s_batch_header_replicated[0], 
		s_query_vectors,
		s_center_vectors,
		// output
		s_distance_LUT[0]);

////////////////////     2. ADC     ////////////////////

    hls::stream<int> s_cell_ID_get_cell_addr_and_size;
#pragma HLS stream variable=s_cell_ID_get_cell_addr_and_size depth=512
    
    hls::stream<int> s_cell_ID_load_PQ_codes;
#pragma HLS stream variable=s_cell_ID_load_PQ_codes depth=512

    replicate_s_cell_ID(
        s_batch_header_replicated[1], 
        s_cell_ID,
        s_cell_ID_get_cell_addr_and_size,
        s_cell_ID_load_PQ_codes);

    hls::stream<int> s_scanned_entries_every_cell;
#pragma HLS stream variable=s_scanned_entries_every_cell depth=512
// #pragma HLS resource variable=s_scanned_entries_every_cell core=FIFO_SRL
    
    hls::stream<int> s_last_valid_PE_ID;
#pragma HLS stream variable=s_last_valid_PE_ID depth=512
// #pragma HLS resource variable=s_last_valid_PE_ID core=FIFO_SRL
    
    hls::stream<int> s_start_addr_every_cell;
#pragma HLS stream variable=s_start_addr_every_cell depth=512
// #pragma HLS resource variable=s_start_addr_every_cell core=FIFO_SRL
    
    hls::stream<int> s_control_iter_num_per_query;
#pragma HLS stream variable=s_control_iter_num_per_query depth=512
// #pragma HLS resource variable=s_control_iter_num_per_query core=FIFO_SRL
    
    get_cell_addr_and_size(
        // in init
	    nlist,
        s_nlist_PQ_codes_start_addr,
        s_nlist_num_vecs,
        // in runtime
		s_batch_header_replicated[2], 
        s_cell_ID_get_cell_addr_and_size,
        // out
        s_scanned_entries_every_cell,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        s_control_iter_num_per_query);

    hls::stream<int> s_scanned_entries_every_cell_ADC[ADC_PE_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_ADC depth=512
#pragma HLS array_partition variable=s_scanned_entries_every_cell_ADC complete
// #pragma HLS resource variable=s_scanned_entries_every_cell_ADC core=FIFO_SRL

    hls::stream<int> s_scanned_entries_every_cell_load_PQ_codes;
#pragma HLS stream variable=s_scanned_entries_every_cell_load_PQ_codes depth=512
// #pragma HLS resource variable=s_scanned_entries_every_cell_load_PQ_codes core=FIFO_SRL

    replicate_s_scanned_entries_every_cell(
        // in
        s_batch_header_replicated[3], 
        s_scanned_entries_every_cell,
        // out
        s_scanned_entries_every_cell_ADC,
        s_scanned_entries_every_cell_load_PQ_codes);

    hls::stream<PQ_in_t> s_PQ_codes[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_codes depth=8
#pragma HLS array_partition variable=s_PQ_codes complete
// #pragma HLS resource variable=s_PQ_codes core=FIFO_SRL

    load_PQ_codes(
        // in runtime
		s_batch_header_replicated[4], 
        s_cell_ID_load_PQ_codes,
        s_scanned_entries_every_cell_load_PQ_codes,
        s_last_valid_PE_ID,
        s_start_addr_every_cell,
        PQ_codes_DRAM_0,
        PQ_codes_DRAM_1,
        PQ_codes_DRAM_2,
        PQ_codes_DRAM_3,
        // out
        s_PQ_codes);

    hls::stream<PQ_out_t> s_PQ_result[ADC_PE_NUM];
#pragma HLS stream variable=s_PQ_result depth=8
#pragma HLS array_partition variable=s_PQ_result complete
// #pragma HLS resource variable=s_PQ_result core=FIFO_SRL


    for (int s = 0; s < ADC_PE_NUM; s++) {
#pragma HLS unroll

#if ADC_DOUBLE_BUF_ENABLE == 0
        PQ_lookup_computation(
            // input streams
			s_batch_header_replicated[5 + s], 
            s_distance_LUT[s],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s + 1],
            s_PQ_result[s]);
#elif ADC_DOUBLE_BUF_ENABLE == 1
        PQ_lookup_computation_double_buffer(
            // input streams
			s_batch_header_replicated[5 + s], 
            s_distance_LUT[s],
            s_PQ_codes[s],
            s_scanned_entries_every_cell_ADC[s],
            // output streams
            s_distance_LUT[s + 1],
            s_PQ_result[s]);
#endif
    }

    dummy_distance_LUT_consumer(
		s_batch_header_replicated[5 + ADC_PE_NUM], 
        s_distance_LUT[ADC_PE_NUM]);

////////////////////     3. K-Selection     ////////////////////

    hls::stream<result_t> s_output; // the topK numbers
#pragma HLS stream variable=s_output depth=512

    hierarchical_priority_queue( 
        nlist,
		s_batch_header_replicated[6 + ADC_PE_NUM],
        s_nlist_vec_ID_start_addr, 
        s_control_iter_num_per_query, 
        s_PQ_result,
        vec_ID_DRAM_0,
        vec_ID_DRAM_1,
        vec_ID_DRAM_2,
        vec_ID_DRAM_3,
        s_output);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=512

    network_output_processing(
        s_batch_header_replicated[7 + ADC_PE_NUM], 
        s_output, 
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