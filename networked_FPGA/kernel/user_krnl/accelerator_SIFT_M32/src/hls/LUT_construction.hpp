#pragma once

#include "constants.hpp"
#include "types.hpp"

///// Instantiate in Top-Level ///// 

void LUT_construction_wrapper(
    // init
    int query_num,
    int nprobe, 
    hls::stream<float>& s_product_quantizer_init,
    // runtime input from network
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors,
    // output
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT);

///// Instantiate in Top-Level end ///// 


template<const int array_len>
float square_sum(float array[array_len]) {
#pragma HLS inline

    float result = 0;
    for (int i = 0; i < array_len; i++) {
#pragma HLS UNROLL
        result += array[i] * array[i];
    }

    return result;
}

void product_quantizer_dispatcher(
    // init
    hls::stream<float>& s_product_quantizer_init,
    // output
    hls::stream<float> (&s_product_quantizer_init_sub_PE)[M]) {

    // Faiss data alignment: M sub quantizers x 256 row x (D / M) sub-vectors
    
    // dispatch to M PEs
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < LUT_ENTRY_NUM; k++) {
            for (int j = 0; j < D / M; j++) {
#pragma HLS pipeline II=1
                s_product_quantizer_init_sub_PE[m].write(s_product_quantizer_init.read());
            }
        }
    }
}

void query_vector_dispatcher(
    // init
    int query_num,
    // runtime input
    hls::stream<ap_uint<512> >& s_query_vectors,
    // output
    hls::stream<float> (&s_sub_query_vectors)[M]) {

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    
    // no need to partition as loading query vector is not the throughput bottleneck
    float query_vector_buffer[size_query_vector * 64 / 4]; 
// #pragma HLS array_partition variable=query_vector_buffer complete

    for (int query_id = 0; query_id < query_num; query_id++) {

        // load query
        for (int i = 0; i < size_query_vector; i++) {

            ap_uint<512> reg = s_query_vectors.read();

            for (int j = 0; j < 64 / 4; j++) {
#pragma HLS pipeline II=1
                ap_uint<32> single_val_uint = reg.range(32 * j + 31, 32 * j);
                float single_val_float = *((float*) (&single_val_uint));
                query_vector_buffer[i * 64 / 4 + j] = single_val_float;
            }
        }

        // dispatch query
        for (int i = 0; i < D / M; i++) {
            for (int m = 0; m < M; m++) {
#pragma HLS pipeline II=1
                // 0 ~ D/M -> PE0; D/M ~ 2 * D/M -> PE1; ...
                s_sub_query_vectors[m].write(query_vector_buffer[m * D / M + i]);
            }
        }
    }
}

void center_vector_dispatcher(
    // init
    int query_num,
    int nprobe, 
    // runtime input
    hls::stream<ap_uint<512> >& s_center_vectors,
    // output
    hls::stream<float> (&s_sub_center_vectors)[M]) {

    // center vector format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    
    // no need to partition as loading center vector is not the throughput bottleneck
    float center_vector_buffer[size_center_vector * 64 / 4]; 
// #pragma HLS array_partition variable=center_vector_buffer complete

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            // load center vector
            for (int i = 0; i < size_center_vector; i++) {

                ap_uint<512> reg = s_center_vectors.read();

                for (int j = 0; j < 64 / 4; j++) {
    #pragma HLS pipeline II=1
                    ap_uint<32> single_val_uint = reg.range(32 * j + 31, 32 * j);
                    float single_val_float = *((float*) (&single_val_uint));
                    center_vector_buffer[i * 64 / 4 + j] = single_val_float;
                }
            }

            // dispatch center vector
            for (int i = 0; i < D / M; i++) {
                for (int m = 0; m < M; m++) {
    #pragma HLS pipeline II=1
                    // 0 ~ D/M -> PE0; D/M ~ 2 * D/M -> PE1; ...
                    s_sub_center_vectors[m].write(center_vector_buffer[m * D / M + i]);
                }
            }
        }
    }
}

void LUT_construction_sub_PE(
    // init
    int query_num,
    int nprobe, 
    hls::stream<float>& s_product_quantizer_init_sub_PE,
    // runtime input
    hls::stream<float>& s_sub_query_vectors,
    hls::stream<float>& s_sub_center_vectors,
    // output
    hls::stream<float>& s_partial_distance_LUT
) {
    // return a column of the distance LUT, i.e., 256 number per Voronoi cell    
    //   throughput: ~= 1 value per cycle, 256 CC per cell, except init time

    float sub_product_quantizer[LUT_ENTRY_NUM][D / M];
#pragma HLS array_partition variable=sub_product_quantizer dim=2
#pragma HLS resource variable=sub_product_quantizer core=RAM_2P_BRAM

    for (int k = 0; k < LUT_ENTRY_NUM; k++) {
        for (int j = 0; j < D / M; j++) {
#pragma HLS pipeline II=1
            sub_product_quantizer[k][j] = s_product_quantizer_init_sub_PE.read();
        }
    }

    // store query and center vector into the format of M sub-vectors
    float sub_query_vector_local[D / M];
#pragma HLS array_partition variable=sub_query_vector_local complete

    float sub_residual_vector[D / M]; // query_vector - center_vector
#pragma HLS array_partition variable=sub_residual_vector complete

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int i = 0; i < D / M; i++) {
#pragma HLS pipeline II=1
            sub_query_vector_local[i] = s_sub_query_vectors.read();
        }

        // for each nprobe, construct LUT
        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int i = 0; i < D / M; i++) {
#pragma HLS pipeline II=1
                // partial center vector
                float reg = s_sub_center_vectors.read();
                sub_residual_vector[i] = sub_query_vector_local[i] - reg;
            }

            for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#if D <= 128
    #pragma HLS pipeline II=1 // For small dimensions we can afford to process 128D per cycle
#elif D <= 256
    #pragma HLS pipeline II=2 // Save resources
#else
    #pragma HLS pipeline II=4 // Save resources
#endif 
                // the L1 diff between sub_residual_vector annd sub-quantizers
                float L1_dist[D/M];
#pragma HLS array_partition variable=L1_dist complete

                    for (int j = 0; j < D / M; j++) {
                        L1_dist[j] = sub_residual_vector[j] - sub_product_quantizer[k][j];
                    }

                // square distance
                float L2_dist = square_sum<D / M>(L1_dist);
                
                s_partial_distance_LUT.write(L2_dist);
            }
        }
    }
}

void gather_LUT_results(
    // init
    int query_num,
    int nprobe, 
    // runtime input
    hls::stream<float> (&s_partial_distance_LUT)[M],
    // output
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    for (int query_id = 0; query_id < query_num; query_id++) {

        for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#pragma HLS pipeline II=1

                distance_LUT_parallel_t row;
                for (int m = 0; m < M; m++) {
#pragma HLS UNROLL
                    row.dist[m] = s_partial_distance_LUT[m].read();
                }
                s_distance_LUT.write(row);
            }
        }
    }
}

void LUT_construction_wrapper(
    // init
    int query_num,
    int nprobe, 
    hls::stream<float>& s_product_quantizer_init,
    // runtime input from network
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors,
    // output
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    // given input query vectors & Voronoi center vectors, return distance LUT
    //   throughput: ~= 1 row (M) of LUT per cycle, 256 CC per cell, except init time
    // output format: sending first row of M values, then second row, and so on...

#pragma HLS inline // inline to a dataflow region

    hls::stream<float> s_product_quantizer_init_sub_PE[M];
#pragma HLS stream variable=s_product_quantizer_init_sub_PE depth=2
#pragma HLS array_partition variable=s_product_quantizer_init_sub_PE complete
    
    product_quantizer_dispatcher(
        s_product_quantizer_init,
        s_product_quantizer_init_sub_PE);

    hls::stream<float> s_sub_query_vectors[M];
#pragma HLS stream variable=s_sub_query_vectors depth=2
#pragma HLS array_partition variable=s_sub_query_vectors complete

    hls::stream<float> s_sub_center_vectors[M];
#pragma HLS stream variable=s_sub_center_vectors depth=2
#pragma HLS array_partition variable=s_sub_center_vectors complete

    query_vector_dispatcher(
        query_num,
        s_query_vectors,
        s_sub_query_vectors);

    center_vector_dispatcher(
        query_num,
        nprobe,
        s_center_vectors,
        s_sub_center_vectors);

    hls::stream<float> s_partial_distance_LUT[M];
#pragma HLS stream variable=s_partial_distance_LUT depth=2
#pragma HLS array_partition variable=s_partial_distance_LUT complete

    for (int m = 0; m < M; m++) {
#pragma HLS UNROLL
        LUT_construction_sub_PE(
            query_num,
            nprobe, 
            s_product_quantizer_init_sub_PE[m],
            s_sub_query_vectors[m],
            s_sub_center_vectors[m],
            s_partial_distance_LUT[m]); 
    }

    gather_LUT_results(
        query_num,
        nprobe, 
        s_partial_distance_LUT,
        s_distance_LUT);
}