#pragma once

#include "types.hpp"

typedef struct {
	float float_data[FLOAT_PER_AXI];
} float_pack_t;

float aggregation_sum_float_pack(
    float_pack_t in) {
// #pragma HLS inline // inline and latency are conflicting
#pragma HLS latency max=1

    // manual specification of adder tree is better than HLS inference (all 16 elements in a single line of code)

    float sum_l0_0 = in.float_data[0] + in.float_data[1];
    float sum_l0_1 = in.float_data[2] + in.float_data[3];
    float sum_l0_2 = in.float_data[4] + in.float_data[5];
    float sum_l0_3 = in.float_data[6] + in.float_data[7];
    float sum_l0_4 = in.float_data[8] + in.float_data[9];
    float sum_l0_5 = in.float_data[10] + in.float_data[11];
    float sum_l0_6 = in.float_data[12] + in.float_data[13];
    float sum_l0_7 = in.float_data[14] + in.float_data[15];

    float sum_l1_0 = sum_l0_0 + sum_l0_1;
    float sum_l1_1 = sum_l0_2 + sum_l0_3;
    float sum_l1_2 = sum_l0_4 + sum_l0_5;
    float sum_l1_3 = sum_l0_6 + sum_l0_7;

    float sum_l2_0 = sum_l1_0 + sum_l1_1;
    float sum_l2_1 = sum_l1_2 + sum_l1_3;

    float sum = sum_l2_0 + sum_l2_1;

    return sum;
}

// float aggregation_sum_float_pack(
//     float_pack_t in) {
// // #pragma HLS inline // inline and latency are conflicting
// #pragma HLS latency max=10
//     float sum = in.float_data[0] + in.float_data[1] + in.float_data[2] + in.float_data[3] +
//                 in.float_data[4] + in.float_data[5] + in.float_data[6] + in.float_data[7] + 
//                 in.float_data[8] + in.float_data[9] + in.float_data[10] + in.float_data[11] +
//                 in.float_data[12] + in.float_data[13] + in.float_data[14] + in.float_data[15];
//     return sum;
// }

void compute_distances_sub_PE_A(
    // in initialization
    const int query_num,
    const int d,
    // in runtime (stream)
    hls::stream<ap_uint<512>>& s_query_vectors, 
    hls::stream<ap_uint<512>>& s_fetched_vectors,
    hls::stream<int>& s_finish_query_in,
    
    // out (stream)
    hls::stream<float>& s_partial_distances,
    hls::stream<int>& s_finish_query_out
) {

    const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 

    float query_vector[D_MAX];
#pragma HLS unroll variable=query_vector factor=float_per_axi

    bool first_iter_s_query_vectors = true;

    for (int qid = 0; qid < query_num; qid++) {

        // read query vector
        if (first_iter_s_query_vectors) {
            while (s_query_vectors.empty()) {}
            first_iter_s_query_vectors = false;
        }
        for (int i = 0; i < vec_AXI_num; i++) {
        #pragma HLS pipeline II=1
            ap_uint<512> query_reg = s_query_vectors.read();
            for (int j = 0; j < FLOAT_PER_AXI; j++) {
            #pragma HLS unroll
                ap_uint<32> query_reg_uint32 = query_reg.range(32 * (j + 1) - 1, 32 * j);
                float query_reg_float = *((float*) (&query_reg_uint32));
                query_vector[i * FLOAT_PER_AXI + j] = query_reg_float;
            }
        }

        while (true) {
            
            if (!s_finish_query_in.empty() && s_fetched_vectors.empty()) {
                s_finish_query_out.write(s_finish_query_in.read());
                break;
            } else if (!s_fetched_vectors.empty()) {

                for (int i = 0; i < vec_AXI_num; i++) {
                #pragma HLS pipeline II=1

                    // read dist vector
                    ap_uint<512> db_vec_reg = s_fetched_vectors.read();
                    
                    float_pack_t reg_part_dist_packed;
                    for (int s = 0; s < FLOAT_PER_AXI; s++) {
                    #pragma HLS unroll
                        ap_uint<32> db_vec_reg_uint32 = db_vec_reg.range(32 * (s + 1) - 1, 32 * s);
                        float database_vector_partial = *((float*) (&db_vec_reg_uint32));
                        float diff = query_vector[i * FLOAT_PER_AXI + s] - database_vector_partial;
                        reg_part_dist_packed.float_data[s] = diff * diff;
                    }

                    float distance_partial = aggregation_sum_float_pack(reg_part_dist_packed);
                    s_partial_distances.write(distance_partial);
                }
            }
        }
    }
}


void compute_distances_sub_PE_pack_partial_distances(
    // in initialization
    const int query_num,
    const int d,
    // in runtime (stream)
    hls::stream<float>& s_partial_distances,
    hls::stream<int>& s_finish_query_in,
    
    // out (stream)
    hls::stream<float_pack_t> &s_partial_distances_packed,
    hls::stream<int>& s_finish_query_out
) {

    const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 
    const int packed_partial_dist_num = vec_AXI_num % FLOAT_PER_AXI == 0? vec_AXI_num / FLOAT_PER_AXI : vec_AXI_num / FLOAT_PER_AXI + 1;

    for (int qid = 0; qid < query_num; qid++) {

        while (true) {
            
            if (!s_finish_query_in.empty() && s_partial_distances.empty()) {
                s_finish_query_out.write(s_finish_query_in.read());
                break;
            } else if (!s_partial_distances.empty()) {

                for (int j = 0; j < packed_partial_dist_num; j++) {
                    float_pack_t reg_packed_dist;
                    for (int i = 0; i < FLOAT_PER_AXI && j * FLOAT_PER_AXI + i < vec_AXI_num; i++) {
                    #pragma HLS pipeline II=1

                        // read dist vector
                        float distance_partial = s_partial_distances.read();
                        reg_packed_dist.float_data[i] = distance_partial;
                    }
                    s_partial_distances_packed.write(reg_packed_dist);
                }
            }
        }
    }
}

void compute_distances_sub_PE_B(
    // in initialization
    const int query_num,
    const int d,
    // in runtime (stream)
    hls::stream<float_pack_t> &s_partial_distances_packed,
    hls::stream<bool>& s_fetch_valid,
    hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
    hls::stream<int>& s_finish_query_in,
    
    // out (stream)
    hls::stream<result_t>& s_distances,
    hls::stream<int>& s_finish_query_out
) {
    const int vec_AXI_num = d % FLOAT_PER_AXI == 0? d / FLOAT_PER_AXI : d / FLOAT_PER_AXI + 1; 
    const int packed_partial_dist_num = vec_AXI_num % FLOAT_PER_AXI == 0? vec_AXI_num / FLOAT_PER_AXI : vec_AXI_num / FLOAT_PER_AXI + 1;
    float aggreg_buffer[D_MAX / FLOAT_PER_AXI / FLOAT_PER_AXI] = {0};

    for (int qid = 0; qid < query_num; qid++) {

        while (true) {
            
            if (!s_finish_query_in.empty() && s_partial_distances_packed.empty() 
                && s_fetch_valid.empty() && s_fetched_neighbor_ids_replicated.empty()) {
                s_finish_query_out.write(s_finish_query_in.read());
                break;
            } else if (!s_partial_distances_packed.empty() && !s_fetch_valid.empty() &&!s_fetched_neighbor_ids_replicated.empty()) {

                cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
                bool valid = s_fetch_valid.read();

                for (int i = 0; i < packed_partial_dist_num; i++) {
                #pragma HLS pipeline II=1 

                    // read dist vector
                    float_pack_t reg_part_dist_packed = s_partial_distances_packed.read();
                    // aggregation sum
                    aggreg_buffer[i] = aggregation_sum_float_pack(reg_part_dist_packed);
                }

                float final_dist = aggreg_buffer[0];
                if (packed_partial_dist_num > 1) {
                    for (int i = 1; i < packed_partial_dist_num; i++) {
                        final_dist += aggreg_buffer[i];
                    }
                }

                // write results
                result_t out;
                out.node_id = reg_cand.node_id;
                out.level_id = reg_cand.level_id;
                out.dist = valid? final_dist : large_float;

                s_distances.write(out);
            }
        }
    }
}

void compute_distances(
    // in initialization
    const int query_num,
    const int d,
    // in runtime (stream)
    hls::stream<ap_uint<512>>& s_query_vectors, 
    hls::stream<ap_uint<512>>& s_fetched_vectors,
    hls::stream<bool>& s_fetch_valid,
    hls::stream<cand_t>& s_fetched_neighbor_ids_replicated,
    hls::stream<int>& s_finish_query_in,
    
    // out (stream)
    hls::stream<result_t>& s_distances,
    hls::stream<int>& s_finish_query_out
) {
#pragma HLS dataflow


    hls::stream<int> s_finish_query_fetch_sub_PE_A; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_sub_PE_A depth=16
	
    hls::stream<int> s_finish_query_fetch_sub_PE_pack_partial_distances; // finish all queries
#pragma HLS stream variable=s_finish_query_fetch_sub_PE_pack_partial_distances depth=16
	
    hls::stream<float> s_partial_distances; 
#pragma HLS stream variable=s_partial_distances depth=512

    hls::stream<float_pack_t> s_partial_distances_packed; 
#pragma HLS stream variable=s_partial_distances_packed depth=16

    compute_distances_sub_PE_A(
        // in initialization
        query_num,
        d,
        // in runtime (stream)
        s_query_vectors, 
        s_fetched_vectors,
        s_finish_query_in,
        
        // out (stream)
        s_partial_distances,
        s_finish_query_fetch_sub_PE_A
    );

    compute_distances_sub_PE_pack_partial_distances(
        // in initialization
        query_num,
        d,
        // in runtime (stream)
        s_partial_distances,
        s_finish_query_fetch_sub_PE_A,
        
        // out (stream)
        s_partial_distances_packed,
        s_finish_query_fetch_sub_PE_pack_partial_distances
    );

    compute_distances_sub_PE_B(
        // in initialization
        query_num,
        d,
        // in runtime (stream)
        s_partial_distances_packed,
        s_fetch_valid,
        s_fetched_neighbor_ids_replicated,
        s_finish_query_fetch_sub_PE_pack_partial_distances,
        
        // out (stream)
        s_distances,
        s_finish_query_out
    );
}
