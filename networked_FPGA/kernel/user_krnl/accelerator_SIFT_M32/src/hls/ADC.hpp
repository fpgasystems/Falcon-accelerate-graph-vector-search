#pragma once

#include "constants.hpp"
#include "types.hpp"

void PQ_lookup_computation(
    int query_num, 
    int nprobe,
    // input streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_in,
    hls::stream<PQ_in_t>& s_PQ_codes,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    // output streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_out,
    hls::stream<PQ_out_t>& s_PQ_result);

void PQ_lookup_computation_double_buffer(
    int query_num, 
    int nprobe,
    // input streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_in,
    hls::stream<PQ_in_t>& s_single_PQ,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    // output streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_out,
    hls::stream<PQ_out_t>& s_single_PQ_result);
    
void dummy_distance_LUT_consumer(
    int query_num, 
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT);

void PQ_lookup_computation(
    int query_num, 
    int nprobe,
    // input streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_in,
    hls::stream<PQ_in_t>& s_PQ_codes,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    // output streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_out,
    hls::stream<PQ_out_t>& s_PQ_result) {

    float distance_LUT[M][LUT_ENTRY_NUM];
#pragma HLS array_partition variable=distance_LUT dim=1
#pragma HLS resource variable=distance_LUT core=RAM_1P_BRAM

    QUERY_LOOP: 
    for (int query_id = 0; query_id < query_num; query_id++) {

        NPROBE_LOOP:
        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            // TODO: for real test, should use tmp_scanned_entries_every_cell instead of template
            //
            //
            int scanned_entries_every_cell = 
                s_scanned_entries_every_cell_PQ_lookup_computation.read();

            CP_LUT_LOOP:
            // Stage A: init distance LUT
            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
// #pragma HLS unroll factor=2

                // without duplication, HLS cannot achieve II=1
                distance_LUT_parallel_t dist_row = s_distance_LUT_in.read();
                s_distance_LUT_out.write(dist_row);
                
                for (int col_id = 0; col_id < M; col_id++) {
                    distance_LUT[col_id][row_id] = dist_row.dist[col_id]; 
                }
            }

            ADC_LOOP:
            // Stage B: compute estimated distance
            for (int entry_id = 0; entry_id < scanned_entries_every_cell; entry_id++) {
#pragma HLS pipeline II=1

                PQ_in_t PQ_local = s_PQ_codes.read();

                PQ_out_t out; 
                out.cell_ID = PQ_local.cell_ID;
                out.offset = PQ_local.offset;
    
                out.dist = 0;
                for (int b = 0; b < M; b++) {
#pragma HLS unroll
                    out.dist += distance_LUT[b][PQ_local.PQ_code[b]];
                }                
                
                // for padded element, replace its distance by large number
                if (PQ_local.valid == 0) {
                    out.cell_ID = -1;
                    out.offset = -1;
                    out.dist = LARGE_NUM;
                }
                s_PQ_result.write(out);
            }
        }
    }
}


void PQ_lookup_computation_double_buffer(
    int query_num, 
    int nprobe,
    // input streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_in,
    hls::stream<PQ_in_t>& s_single_PQ,
    hls::stream<int>& s_scanned_entries_every_cell_PQ_lookup_computation,
    // output streams
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT_out,
    hls::stream<PQ_out_t>& s_single_PQ_result) {

    // Manual control on the double buffer
    float distance_LUT_A[M][LUT_ENTRY_NUM];
#pragma HLS array_partition variable=distance_LUT_A dim=1
#pragma HLS resource variable=distance_LUT_A core=RAM_1P_BRAM
    float distance_LUT_B[M][LUT_ENTRY_NUM];
#pragma HLS array_partition variable=distance_LUT_B dim=1
#pragma HLS resource variable=distance_LUT_B core=RAM_1P_BRAM

    QUERY_LOOP:
    for (int query_id = 0; query_id < query_num; query_id++) {
        
        NPROBE_LOOP:
        for (int nprobe_id = 0; nprobe_id < nprobe + 1; nprobe_id++) {

            // iter 0: read data only
            // middle iter: read + compute
            // final iter: compute only
            int scanned_entries_every_cell;
            if (nprobe_id > 0) {
                scanned_entries_every_cell = 
                    s_scanned_entries_every_cell_PQ_lookup_computation.read();
            }
            else {
                // for the first iteration, there's no computation, just randomly
                //  init these variables
                scanned_entries_every_cell = 0;
            }

            int max_iter = scanned_entries_every_cell > LUT_ENTRY_NUM?
                scanned_entries_every_cell : LUT_ENTRY_NUM;

            int nprobe_mod = nprobe_id % 2;

            for (int common_iter = 0; common_iter < max_iter; common_iter++) {
#pragma HLS pipeline II=1

                // Note! Use the if else clause such that A and B certainly
                //   won't have bank conflict in a single CC

                // even: load to buffer A, compute by buffer B
                if (nprobe_mod == 0) { 

                    // load part
                    // last iter not read 
                    if (nprobe_id < nprobe) { 

                        // load or not
                        if (common_iter < LUT_ENTRY_NUM) { 

                            distance_LUT_parallel_t dist_row = s_distance_LUT_in.read();
                            s_distance_LUT_out.write(dist_row);

                            // even: load to buffer A
                            for (int col_id = 0; col_id < M; col_id++) {
#pragma HLS UNROLL
                                distance_LUT_A[col_id][common_iter] = dist_row.dist[col_id]; 
                            }
                        }
                    }

                    // compute part
                    // first iter not compute
                    if (nprobe_id > 0) { 

                        // compute or not
                        if (common_iter < scanned_entries_every_cell) { 

                            PQ_in_t PQ_local = s_single_PQ.read();

                            PQ_out_t out; 
                            out.cell_ID = PQ_local.cell_ID;
                            out.offset = PQ_local.offset;
                            
                            out.dist = 0;
                            for (int b = 0; b < M; b++) {
#pragma HLS unroll
                                out.dist += distance_LUT_B[b][PQ_local.PQ_code[b]];
                            }       
                                
                            // for padded element, replace its distance by large number
                            if (PQ_local.valid == 0) {
                                out.cell_ID = -1;
                                out.offset = -1;
                                out.dist = LARGE_NUM;
                            }
                            s_single_PQ_result.write(out);
                        }
                    }
                }
                // odd: load from buffer B, compute by buffer A
                else {

                    // load part
                    // last iter not read 
                    if (nprobe_id < nprobe) { 

                        // load or not
                        if (common_iter < LUT_ENTRY_NUM) { 

                            distance_LUT_parallel_t dist_row = s_distance_LUT_in.read();
                            s_distance_LUT_out.write(dist_row);

                            // odd: load to buffer B
                            for (int col_id = 0; col_id < M; col_id++) {
#pragma HLS UNROLL
                                distance_LUT_B[col_id][common_iter] = dist_row.dist[col_id]; 
                            }
                        }
                    }

                    // compute part
                    // first iter not compute
                    if (nprobe_id > 0) { 

                        // compute or not
                        if (common_iter < scanned_entries_every_cell) { 

                            PQ_in_t PQ_local = s_single_PQ.read();

                            PQ_out_t out; 
                            out.cell_ID = PQ_local.cell_ID;
                            out.offset = PQ_local.offset;
                            
                            out.dist = 0;
                            for (int b = 0; b < M; b++) {
#pragma HLS unroll
                                out.dist += distance_LUT_A[b][PQ_local.PQ_code[b]];
                            }       
                            // for padded element, replace its distance by large number
                            if (PQ_local.valid == 0) {
                                out.cell_ID = -1;
                                out.offset = -1;
                                out.dist = LARGE_NUM;
                            }
                            s_single_PQ_result.write(out);
                        }
                    }
                }
            }
        }
    }
}

void dummy_distance_LUT_consumer(
    int query_num, 
    int nprobe,
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    distance_LUT_parallel_t dist_row;

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int row_id = 0; row_id < LUT_ENTRY_NUM; row_id++) {
#pragma HLS pipeline II=1
                dist_row = s_distance_LUT.read();
            }

        }
    }
}