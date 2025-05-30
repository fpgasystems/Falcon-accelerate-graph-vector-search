#pragma once

#include "types.hpp"
#include "utils.hpp"

// Given an input 64-byte representation, parse it as node meta data type
node_meta_t parse_meta_data(ap_uint<512> in_uint512) {
#pragma HLS inline

    node_meta_t meta_data;

    ap_uint<32> reg_is_leaf_uint32 = in_uint512.range(32 * 0 + 31, 32 * 0);
    meta_data.is_leaf = *((int*) (&reg_is_leaf_uint32));

    ap_uint<32> reg_count_uint32 = in_uint512.range(32 * 1 + 31, 32 * 1);
    meta_data.count = *((int*) (&reg_count_uint32));

    ap_uint<32> reg_obj_id_uint32 = in_uint512.range(32 * 2 + 31, 32 * 2);
    meta_data.obj.id = *((int*) (&reg_obj_id_uint32));

    ap_uint<32> reg_obj_low0_uint32 = in_uint512.range(32 * 3 + 31, 32 * 3);
    meta_data.obj.low0 = *((float*) (&reg_obj_low0_uint32));

    ap_uint<32> reg_obj_high0_uint32 = in_uint512.range(32 * 4 + 31, 32 * 4);
    meta_data.obj.high0 = *((float*) (&reg_obj_high0_uint32));

    ap_uint<32> reg_obj_low1_uint32 = in_uint512.range(32 * 5 + 31, 32 * 5);
    meta_data.obj.low1 = *((float*) (&reg_obj_low1_uint32));

    ap_uint<32> reg_obj_high1_uint32 = in_uint512.range(32 * 6 + 31, 32 * 6);
    meta_data.obj.high1 = *((float*) (&reg_obj_high1_uint32));

    return meta_data;
}

// Input: the read request (pair of page IDs) from the scheduler
// Output: the meta data and entries of the nodes of the pair of nodes
// Note: the current design underutilizes the bandwidth for convenience
void read_nodes(
    // input
    const ap_uint<512>* in_pages_A,
    const ap_uint<512>* in_pages_B,
    hls::stream<pair_t>& s_page_ID_pair_read_nodes,
    hls::stream<int>& s_join_finish_replicated,
    // output
    hls::stream<node_meta_t>& s_meta_A,
    hls::stream<node_meta_t>& s_meta_B,
    hls::stream<obj_t>& s_page_A,
    hls::stream<obj_t>& s_page_B
    ) {
    
#ifdef DEBUG_read_nodes
    ////// debug starts
    pair_t reg_p = s_page_ID_pair_read_nodes.read();
    node_meta_t reg_n;
    obj_t reg_o; 
    reg_n.is_leaf = reg_p.id_A;
    reg_o.id = reg_p.id_A;

    s_meta_A.write(reg_n);
    s_meta_B.write(reg_n);

    s_page_A.write(reg_o);
    s_page_B.write(reg_o);

    int reg_i = s_join_finish_replicated.read();

    ////// debug ends
#else

    while (true) {

        if (!s_join_finish_replicated.empty()) {
            int end = s_join_finish_replicated.read();
            break;
        } else if (!s_page_ID_pair_read_nodes.empty()) {

            pair_t page_ID_pair = s_page_ID_pair_read_nodes.read();
            int page_ID_A = page_ID_pair.id_A;
            int page_ID_B = page_ID_pair.id_B;

            int start_addr_A = PAGE_SIZE_PER_AXI * page_ID_A;
            int start_addr_B = PAGE_SIZE_PER_AXI * page_ID_B;

            // read meta data to get the page
            node_meta_t meta_A = parse_meta_data(in_pages_A[start_addr_A]);
            node_meta_t meta_B = parse_meta_data(in_pages_B[start_addr_B]);
            start_addr_A++;
            start_addr_B++;

            s_meta_A.write(meta_A);
            s_meta_B.write(meta_B);
            
            int max_page_entries = meta_A.count >= meta_B.count? meta_A.count : meta_B.count;
            // number of 512-bit entries that a page contains 
            int addr_per_page = max_page_entries % N_OBJ_PER_AXI == 0?
                max_page_entries / N_OBJ_PER_AXI : max_page_entries / N_OBJ_PER_AXI + 1;

            // the 64-byte header is already counted by ++
            for (int i = 0; i < addr_per_page; i++) {
    #pragma HLS pipeline // II=3 // needs N_OBJ_PER_AXI cycles
                // parse the input to three outputs
                ap_uint<512> reg_A = in_pages_A[start_addr_A + i];
                ap_uint<512> reg_B = in_pages_B[start_addr_B + i];

                for (int j = 0; j < N_OBJ_PER_AXI; j++) {

                    // page A
                    ap_uint<32> id_A_ap_uint_32 = reg_A.range(
                        j * OBJ_BITS + 32 * 1 - 1, j * OBJ_BITS + 32 * 0);
                    ap_uint<32> low0_A_ap_uint_32 = reg_A.range(
                        j * OBJ_BITS + 32 * 2 - 1, j * OBJ_BITS + 32 * 1);
                    ap_uint<32> high0_A_ap_uint_32 = reg_A.range(
                        j * OBJ_BITS + 32 * 3 - 1, j * OBJ_BITS + 32 * 2);
                    ap_uint<32> low1_A_ap_uint_32 = reg_A.range(
                        j * OBJ_BITS + 32 * 4 - 1, j * OBJ_BITS + 32 * 3);
                    ap_uint<32> high1_A_ap_uint_32 = reg_A.range(
                        j * OBJ_BITS + 32 * 5 - 1, j * OBJ_BITS + 32 * 4);

                    obj_t obj_A;
                    obj_A.id = *((int*) (&id_A_ap_uint_32));
                    obj_A.low0 = *((float*) (&low0_A_ap_uint_32)); 
                    obj_A.high0 = *((float*) (&high0_A_ap_uint_32)); 
                    obj_A.low1 = *((float*) (&low1_A_ap_uint_32)); 
                    obj_A.high1 = *((float*) (&high1_A_ap_uint_32)); 
                    if (i * N_OBJ_PER_AXI + j < meta_A.count) {
                        s_page_A.write(obj_A);
                    }

                    // Page B
                    ap_uint<32> id_B_ap_uint_32 = reg_B.range(
                        j * OBJ_BITS + 32 * 1 - 1, j * OBJ_BITS + 32 * 0);
                    ap_uint<32> low0_B_ap_uint_32 = reg_B.range(
                        j * OBJ_BITS + 32 * 2 - 1, j * OBJ_BITS + 32 * 1);
                    ap_uint<32> high0_B_ap_uint_32 = reg_B.range(
                        j * OBJ_BITS + 32 * 3 - 1, j * OBJ_BITS + 32 * 2);
                    ap_uint<32> low1_B_ap_uint_32 = reg_B.range(
                        j * OBJ_BITS + 32 * 4 - 1, j * OBJ_BITS + 32 * 3);
                    ap_uint<32> high1_B_ap_uint_32 = reg_B.range(
                        j * OBJ_BITS + 32 * 5 - 1, j * OBJ_BITS + 32 * 4);

                    obj_t obj_B;
                    obj_B.id = *((int*) (&id_B_ap_uint_32));
                    obj_B.low0 = *((float*) (&low0_B_ap_uint_32)); 
                    obj_B.high0 = *((float*) (&high0_B_ap_uint_32)); 
                    obj_B.low1 = *((float*) (&low1_B_ap_uint_32)); 
                    obj_B.high1 = *((float*) (&high1_B_ap_uint_32)); 
                    if (i * N_OBJ_PER_AXI + j < meta_B.count) {
                        s_page_B.write(obj_B);
                    }
                }
            }
        }
    }

#endif
}


void layer_cache_memory_controller(
    // input
    //   argument
    int root_id_A,
    int root_id_B,
    //   memory 
    ap_uint<64>* layer_cache,
    //   from join PE
    hls::stream<int>& s_intersect_count_directory, 
    hls::stream<pair_t>& s_result_pair_directory,
    //   from scheduler
    hls::stream<int>& s_read_write_control, // 0 -> read from memory; 1 -> write to memory 
    hls::stream<int>& s_read_layer_id,      // layer l 
    hls::stream<int>& s_read_layer_pointer, // pair p in layer l
    hls::stream<int>& s_write_layer_id, 
    hls::stream<int>& s_join_finish_replicated,
    // output
    //   to scheduler
    hls::stream<pair_t>& s_page_pair_scheduler,      // for read request, return pair
    hls::stream<int>& s_intersect_count_directory_scheduler // for write request, return count
) {

#ifdef DEBUG_layer_cache_memory_controller
    ////// debug starts

    // get read request, and return pair
    int read = s_read_write_control.read();
    int layer_id = s_read_layer_id.read();
    int layer_pointer = s_read_layer_pointer.read();
    int sum = read + layer_id + layer_pointer;
    pair_t pair;
    pair.id_A = sum;
    s_page_pair_scheduler.write(pair);

    // get write requests, and write results
    int write = s_read_write_control.read() + s_write_layer_id.read();
    int count = s_intersect_count_directory.read() + write;
    s_intersect_count_directory_scheduler.write(count);
    pair_t result = s_result_pair_directory.read();
    layer_cache[0] = result.id_A;

    // end 
    int end = s_join_finish_replicated.read();


    ////// debug ends
#else

    // Initialization: write the pair (rootA, rootB) in layer cache 0
    pair_t root_pair;
    root_pair.id_A = root_id_A;
    root_pair.id_B = root_id_B;
    ap_uint<64> root_ap_uint_64 = pack_pair(root_pair);
    layer_cache[0] = root_ap_uint_64; 

    // Order:
    //   1. receive a signal from the scheduler: indicating whether it wants to write to / read from memory
    //   If it is a write:
    //      a. receive cache layer id from the scheduler
    //      b. receive the layer cache results from the join PE, write to memory
    //   If it is a read:
    //      a. receive the read layer id and the pointer, read it from memory
    //      b. write the pair back to the scheduler

    while (true) {

        if (!s_join_finish_replicated.empty()) {
            int end = s_join_finish_replicated.read();
            break;
        } else if (!s_read_write_control.empty()) {

            // 0 -> read from memory; 1 -> write to memory 
            int write = s_read_write_control.read();

            if (write) {
                int write_layer_id = s_write_layer_id.read();
                int start_addr = write_layer_id * (LAYER_CACHE_SIZE / 8); 

                bool reach_end = true; 
                for (int i = 0; i < MAX_PAGE_ENTRIES * MAX_PAGE_ENTRIES; i++) {
    #pragma HLS pipeline II=1
                    if (!s_intersect_count_directory.empty() && s_result_pair_directory.empty()) {
                        int count = s_intersect_count_directory.read(); 
                        s_intersect_count_directory_scheduler.write(count);
                        reach_end = false;
                        break;
                    }
                    pair_t result = s_result_pair_directory.read();
                    ap_uint<64> result_ap_uint_64 =pack_pair(result);
                    layer_cache[start_addr + i] = result_ap_uint_64;
                }
                if (reach_end) { // maximum results, haven't read the count yet 
                    int count = s_intersect_count_directory.read(); 
                    s_intersect_count_directory_scheduler.write(count);
                }
            } else { // read 
                int read_layer_id = s_read_layer_id.read();
                int read_layer_pointer = s_read_layer_pointer.read();
                int addr = read_layer_id * (LAYER_CACHE_SIZE / 8) + read_layer_pointer; 

                pair_t next_page_pair = unpack_pair(layer_cache[addr]);
                s_page_pair_scheduler.write(next_page_pair);
            }
        }
    }
#endif
}


// TODO: rewrite this, then write the scheduler
// slow in writing, no burst inferred
void write_results(
    // input
    //   from join PE
    hls::stream<int>& s_intersect_count_leaf, // per page pair
    hls::stream<pair_t>& s_result_pair_leaf,
    //   from the scheduler
    hls::stream<int>& s_join_finish_replicated,  // final end signal 
    // out
    //    out format: the first number writes total intersection count, 
    //                while the rest are intersect ID pairs
    ap_uint<64>* out_intersect) {

#ifdef DEBUG_write_results
    ////// debug starts
    int reg0 = s_intersect_count_leaf.read();
    pair_t reg1 = s_result_pair_leaf.read();

    out_intersect[0] = reg0;
    out_intersect[1] = reg1.id_A;
    
    int end = s_join_finish_replicated.read();

    ////// debug ends
#else 

    ap_uint<64> total_intersect_count = 0;
    const int bias = 1; // the first number writes total intersection count, 
    
    while (true) {

        // the entire join is finished
        if (!s_join_finish_replicated.empty()) {
            int end = s_join_finish_replicated.read();
            break;
        }
        // if data is available, finish writing results of a pair of node join
        else if (!s_result_pair_leaf.empty()) {

            bool reach_end = true; 
            for (int i = 0; i < MAX_PAGE_ENTRIES * MAX_PAGE_ENTRIES; i++) {
#pragma HLS pipeline II=1
                if (!s_intersect_count_leaf.empty() && s_result_pair_leaf.empty()) {
                    total_intersect_count += s_intersect_count_leaf.read(); 
                    reach_end = false;
                    break;
                }
                pair_t result = s_result_pair_leaf.read();
                ap_uint<64> result_ap_uint_64 = pack_pair(result);
                out_intersect[total_intersect_count + i + bias] = result_ap_uint_64;
            }
            if (reach_end) { // maximum results, haven't read the count yet 
                total_intersect_count += s_intersect_count_leaf.read();
            }
        } 
        
    }

    // write the number of intersection in the first address
    out_intersect[0] = total_intersect_count;
#endif
}