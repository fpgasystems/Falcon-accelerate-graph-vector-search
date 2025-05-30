#pragma once

#include "types.hpp"

// join between node pairs
// if both the nodes are leaf, write to the results
// if at least on of the nodes is directory, write to layer cache 
void join_page(
    // input
    hls::stream<node_meta_t>& s_meta_A,
    hls::stream<node_meta_t>& s_meta_B,
    hls::stream<obj_t>& s_page_A,
    hls::stream<obj_t>& s_page_B,
    hls::stream<int>& s_join_finish_replicated,
    // output
    //   for directory nodes: 
    hls::stream<int>& s_intersect_count_directory, // per page pair
    hls::stream<pair_t>& s_result_pair_directory,
    //   for leaf nodes: 
    hls::stream<int>& s_intersect_count_leaf, // per page pair
    hls::stream<pair_t>& s_result_pair_leaf
    ) {

#ifdef DEBUG_join_page
    ////// debug starts

    node_meta_t reg0 = s_meta_A.read();
    node_meta_t reg1 = s_meta_B.read();
    obj_t reg2 = s_page_A.read();
    obj_t reg3 = s_page_B.read();

    int reg_a = reg0.is_leaf + reg1.is_leaf;
    pair_t reg_b;
    reg_b.id_A = reg2.id;
    reg_b.id_B = reg3.id;

    s_intersect_count_directory.write(reg_a);
    s_result_pair_directory.write(reg_b);

    s_intersect_count_leaf.write(reg_a);
    s_result_pair_leaf.write(reg_b);

    int reg4 = s_join_finish_replicated.read();

    ////// debug ends
#else 

    obj_t page_A[MAX_PAGE_ENTRY_NUM];
    obj_t page_B[MAX_PAGE_ENTRY_NUM];

    // WENQI: as an HLS defect, somehow this loop cannot be compiled successfully using while
    //   so I have to use a counter that counts to (almost infinite) to solve the problem
    // for (long infinite_counter = 0; infinite_counter < 1000 * 1000 * 1000 * 1000; infinite_counter++) {
    while (true) {

        if (!s_join_finish_replicated.empty()) {
            int end = s_join_finish_replicated.read();
            break;
        } else if (!s_meta_A.empty() || !s_meta_B.empty()) {

            node_meta_t meta_A = s_meta_A.read();
            node_meta_t meta_B = s_meta_B.read();
            int max_page_entries = meta_A.count >= meta_B.count? meta_A.count : meta_B.count;
    
            // load the two pages 
            LOAD_PAGE:
            for (int j = 0; j < max_page_entries; j++) {
    #pragma HLS pipeline II=1
                if (j < meta_A.count) {
                    page_A[j] = s_page_A.read();
                }
                if (j < meta_B.count) {
                    page_B[j] = s_page_B.read();
                }
            }

            // perform the join 
            // for the case where both trees are data nodes or directory nodes, join directly
            if ((meta_A.is_leaf && meta_B.is_leaf) || (!meta_A.is_leaf && !meta_B.is_leaf)) {

                int intersect_count = 0;
                LOOP_A:
                for (int m = 0; m < meta_A.count; m++) {

                    obj_t obj_A = page_A[m];

                    LOOP_B:
                    for (int n = 0; n < meta_B.count; n++) {
    #pragma HLS pipeline II=1

                        obj_t obj_B = page_B[n];

    #if POINT_INTERSECT_COUNTS
                        // point overlap is regarded as overlap
                        bool overlap = 
                            // horizontal overlap 
                            (obj_A.high0 >= obj_B.low0) && (obj_B.high0 >= obj_A.low0) && 
                            // vertical no overlap 
                            (obj_A.high1 >= obj_B.low1) && (obj_B.high1 >= obj_A.low1);
    #else
                        // point overlap is NOT regarded as overlap, only region overlap counts
                        bool overlap = 
                            // horizontal overlap 
                            (obj_A.high0 > obj_B.low0) && (obj_B.high0 > obj_A.low0) && 
                            // vertical no overlap 
                            (obj_A.high1 > obj_B.low1) && (obj_B.high1 > obj_A.low1);
    #endif

                        if (overlap) {
                            intersect_count++;
                            pair_t result;
                            result.id_A = obj_A.id;
                            result.id_B = obj_B.id;
                            if (meta_A.is_leaf && meta_B.is_leaf) {
                                s_result_pair_leaf.write(result);
                            } else {
                                s_result_pair_directory.write(result);
                            }
                        }
                    }
                }
                if (meta_A.is_leaf && meta_B.is_leaf) {
                    s_intersect_count_leaf.write(intersect_count);
                } else {
                    s_intersect_count_directory.write(intersect_count);
                }
            } 
            // for the case where one is data node while the other is directory nodes, 
            //    write pairs of data nodes's pointer with diretory nodes' children
            else {
                obj_t leaf_obj;
                obj_t* dir_ptr;
                int count;

                if (meta_A.is_leaf) {
                    leaf_obj = meta_A.obj;
                    dir_ptr = page_B;
                    count = meta_B.count;
                } else {
                    leaf_obj = meta_B.obj;
                    dir_ptr = page_A;
                    count = meta_A.count;
                }

                int intersect_count = 0;
                LOOP_C:
                for (int n = 0; n < count; n++) {
    #pragma HLS pipeline II=1

                    obj_t dir_obj = dir_ptr[n];

    #if POINT_INTERSECT_COUNTS
                    // point overlap is regarded as overlap
                    bool overlap = 
                        // horizontal overlap 
                        (leaf_obj.high0 >= dir_obj.low0) && (dir_obj.high0 >= leaf_obj.low0) && 
                        // vertical no overlap 
                        (leaf_obj.high1 >= dir_obj.low1) && (dir_obj.high1 >= leaf_obj.low1);
    #else
                    // point overlap is NOT regarded as overlap, only region overlap counts
                    bool overlap = 
                        // horizontal overlap 
                        (leaf_obj.high0 > dir_obj.low0) && (dir_obj.high0 > leaf_obj.low0) && 
                        // vertical no overlap 
                        (leaf_obj.high1 > dir_obj.low1) && (dir_obj.high1 > leaf_obj.low1);
    #endif

                    if (overlap) {
                        intersect_count++;
                        pair_t result;
                        result.id_A = leaf_obj.id;
                        result.id_B = dir_obj.id;
                        s_result_pair_directory.write(result);
                    }
                }
                s_intersect_count_directory.write(intersect_count);
            }
        }
    }
#endif
}