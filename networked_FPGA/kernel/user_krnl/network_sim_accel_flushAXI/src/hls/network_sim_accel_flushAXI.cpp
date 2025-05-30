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
#include "types.hpp"

/*
Receive data from connection A, then forward the data to connection B
Support 1 connection per direction only.
*/


void network_input_processing(
    int query_num,
    int nprobe,
    // in runtime
    hls::stream<ap_uint<512>>& s_kernel_network_in,
    // out
    hls::stream<int>& s_cell_ID,
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors
    ) {

    // Format: foe each query
    // packet 0: header (query_id, nprobe), in the future, nprobe is streamed from network
    // packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
    // packet k~n: query_vectors
    // packet n~m: center_vectors

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        // header meta
        ap_uint<512> header = s_kernel_network_in.read();
        ap_uint<32> query_id_unused_uint = header.range(31, 0);
        ap_uint<32> nprobe_unused_uint = header.range(63, 32);
        int query_id_unused = *((int*) (&query_id_unused_uint));
        int nprobe_unused = *((int*) (&nprobe_unused_uint));

        int size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;

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

void simulated_accelerator_kernel(
    int query_num,
    int nprobe,
    int delay_cycle_per_cell,
    // runtime input
    hls::stream<int>& s_cell_ID,
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors,
    // runtime output
    hls::stream<result_t> &s_output) {

    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    for (int query_id = 0; query_id < query_num; query_id++) {

        ap_uint<512> query;
        for (int i = 0; i < size_query_vector; i++) {
            query = s_query_vectors.read();
        }

        ap_uint<512> center;
        int cell_ID;
        for (int cell_id = 0; cell_id < nprobe; cell_id++) {

            cell_ID = s_cell_ID.read();
            // read
            for (int i = 0; i < size_center_vector; i++) {
                center = s_center_vectors.read();
            }

            // delay: simulate accelerator compute
            volatile int delay_cnt = 0;
            while (delay_cnt < delay_cycle_per_cell) {
                delay_cnt++;
            }
        }

        // send out all results per query
        result_t out_reg;
        out_reg.vec_ID.range(31, 0) = query.range(31, 0);
        out_reg.vec_ID.range(63, 32) = center.range(31, 0);
        out_reg.dist = cell_ID;

        for (int k = 0; k < TOPK; k++) {
            s_output.write(out_reg);
        }
    }
}

// attach to AXI, the default free-runing pipeline (frp) would not work here
//   thus we can try either (a) using flushable pipeline (flp) or disable pipeline
void AXI_data_processing(
    int query_num,
    float* in_DDR,
    hls::stream<result_t> &s_input,
    hls::stream<result_t> &s_output) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int k = 0; k < TOPK; k++) {
// use one of the following
#pragma HLS pipeline style=flp
// #pragma HLS pipeline off
            result_t reg = s_input.read();
            reg.dist += in_DDR[k];
            s_output.write(reg);
        }
    }
}

void network_output_processing(
    int query_num,
    hls::stream<result_t> &s_output, 
    hls::stream<ap_uint<512>>& s_kernel_network_out) {

    // Format: foe each query
    // packet 0: header (query_id, topK)
    // packet 1~k: topK results_pair (vec_ID, dist) -> size = ceil(topK * 8 / 64) 

    // in 512-bit packets
    const int size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    const int size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;

    ap_uint<64> vec_ID_buffer [size_results_vec_ID * (512 / 64)] = { 0 };
    float dist_buffer[size_results_dist * (512 / 32)] = { 0 };

    // only write the last iteration
    for (int i = 0; i < query_num; i++) {
#pragma HLS pipeline II=1

        ap_uint<512> header = 0;
        ap_uint<32> query_id_header = i;
        ap_uint<32> topK_header = PRIORITY_QUEUE_LEN_L2;
        header.range(31, 0) = query_id_header;
        header.range(63, 32) = topK_header;
        s_kernel_network_out.write(header);

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

        // then send disk
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


extern "C" {
void network_sim_accel_flushAXI(
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
    int query_num, 
    int nlist,
    int nprobe,
    int delay_cycle_per_cell,
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

#pragma HLS INTERFACE s_axilite port=useConn // bundle = control
#pragma HLS INTERFACE s_axilite port=basePortRx // bundle = control
#pragma HLS INTERFACE s_axilite port=expectedRxByteCnt // bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddressTx // bundle=control
#pragma HLS INTERFACE s_axilite port=basePortTx // bundle = control
#pragma HLS INTERFACE s_axilite port=expectedTxPkgCnt // bundle = control
#pragma HLS INTERFACE s_axilite port=pkgWordCountTx // bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = // control

// accelerator kernel 
#pragma HLS INTERFACE s_axilite port=query_num // bundle = control
#pragma HLS INTERFACE s_axilite port=nlist // bundle = control
#pragma HLS INTERFACE s_axilite port=nprobe // bundle = control
#pragma HLS INTERFACE s_axilite port=delay_cycle_per_cell // bundle = control

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

    hls::stream<int> s_cell_ID;
#pragma HLS stream variable=s_cell_ID depth=512
    
    hls::stream<ap_uint<512> > s_query_vectors;
#pragma HLS stream variable=s_query_vectors depth=512
    
    hls::stream<ap_uint<512> > s_center_vectors;
#pragma HLS stream variable=s_center_vectors depth=512

    network_input_processing(
        query_num,
        nprobe,
        // in runtime
        s_kernel_network_in,
        // out
        s_cell_ID,
        s_query_vectors, 
        s_center_vectors);


////////////////////     Accelerator Simulation     ////////////////////

    hls::stream<result_t> s_output_A; // the topK numbers
#pragma HLS stream variable=s_output_A depth=512

    simulated_accelerator_kernel(
        query_num,
        nprobe,
        delay_cycle_per_cell,
        // runtime input
        s_cell_ID,
        s_query_vectors,
        s_center_vectors,
        // runtime output
        s_output_A);

    hls::stream<result_t> s_output_B; // the topK numbers
#pragma HLS stream variable=s_output_B depth=512

    AXI_data_processing(
        query_num,
        in_DDR,
        s_output_A,
        s_output_B);

////////////////////     Network Output     ////////////////////

    hls::stream<ap_uint<512>> s_kernel_network_out; 
#pragma HLS stream variable=s_kernel_network_out depth=512

    network_output_processing(
        query_num,
        s_output_B, 
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