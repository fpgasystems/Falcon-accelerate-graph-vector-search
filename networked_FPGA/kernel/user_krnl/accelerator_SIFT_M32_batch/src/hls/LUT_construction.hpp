#pragma once

#include "constants.hpp"
#include "types.hpp"

///// Instantiate in Top-Level ///// 

void LUT_construction_wrapper(
    // init
    hls::stream<float>& s_product_quantizer_init,
    // runtime input from network
    hls::stream<batch_header_t>& s_batch_header,
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
    hls::stream<float> (&s_product_quantizer_init_sub_PE)[LUT_CONSTR_SUB_PE_NUM]) {

    // Faiss data alignment: M sub quantizers x 256 row x (D / M) sub-vectors
    // PE storage format, e.g., LUT_CONSTR_SUB_PE_NUM=16, M=64
    // m 0 -> PE 0 ... m 15 -> PE 15; m 16 -> PE 0, ... m 31 -> PE 15; ...
    //  PE id = m % LUT_CONSTR_SUB_PE_NUM
    
    // dispatch to M PEs
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < LUT_ENTRY_NUM; k++) {
            for (int j = 0; j < D / M; j++) {
#pragma HLS pipeline II=1
                s_product_quantizer_init_sub_PE[m % LUT_CONSTR_SUB_PE_NUM].write(
                    s_product_quantizer_init.read());
            }
        }
    }
}

void query_vector_dispatcher(
    // runtime input
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<ap_uint<512> >& s_query_vectors,
    // output
    hls::stream<float> (&s_sub_query_vectors)[LUT_CONSTR_SUB_PE_NUM]) {

    // query format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    
    // no need to partition as loading query vector is not the throughput bottleneck
    float query_vector_buffer[size_query_vector * 64 / 4]; 
// #pragma HLS array_partition variable=query_vector_buffer complete

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
			// each mini block = D / M elements
			// m 0 -> PE 0 ... m 15 -> PE 15; m 16 -> PE 0, ... m 31 -> PE 15; ...
			//  PE id = m % LUT_CONSTR_SUB_PE_NUM
			for (int i = 0; i < D / M; i++) {
				for (int m = 0; m < M; m++) { // balance the transmission rate per PE
#pragma HLS pipeline II=1
					s_sub_query_vectors[m % LUT_CONSTR_SUB_PE_NUM].write(
						query_vector_buffer[m * D / M + i]);
				}
			}
		}
	}
}

void center_vector_dispatcher(
    // runtime input
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<ap_uint<512> >& s_center_vectors,
    // output
    hls::stream<float> (&s_sub_center_vectors)[LUT_CONSTR_SUB_PE_NUM]) {

    // center vector format: store in 512-bit packets, pad 0 for the last packet if needed
    const int size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    
    // no need to partition as loading center vector is not the throughput bottleneck
    float center_vector_buffer[size_center_vector * 64 / 4]; 
// #pragma HLS array_partition variable=center_vector_buffer complete

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
				// each mini block = D / M elements
				// m 0 -> PE 0 ... m 15 -> PE 15; m 16 -> PE 0, ... m 31 -> PE 15; ...
				//  PE id = m % LUT_CONSTR_SUB_PE_NUM
				for (int i = 0; i < D / M; i++) {
					for (int m = 0; m < M; m++) { // balance the transmission rate per PE
#pragma HLS pipeline II=1
						s_sub_center_vectors[m % LUT_CONSTR_SUB_PE_NUM].write(
							center_vector_buffer[m * D / M + i]);
					}
				}
			}
		}
	}
}

void LUT_construction_sub_PE(
    // init
    hls::stream<float>& s_product_quantizer_init_sub_PE,
    // runtime input
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<float>& s_sub_query_vectors,
    hls::stream<float>& s_sub_center_vectors,
    // output
    hls::stream<float>& s_partial_distance_LUT
) {
    // return one or multiple columns (M / LUT_CONSTR_SUB_PE_NUM) of the distance LUT, 
    //   i.e., 256 number per Voronoi cell    
    //   throughput: ~= 1 value per cycle, 256 CC per cell, except init time

    // m 0 -> PE 0 ... m 15 -> PE 15; m 16 -> PE 0, ... m 31 -> PE 15; ...
    //  PE id = m % LUT_CONSTR_SUB_PE_NUM

    const int array_unroll_factor = D / M;

    float sub_product_quantizer[LUT_ENTRY_NUM][(M / LUT_CONSTR_SUB_PE_NUM) * (D / M)];
#pragma HLS array_partition variable=sub_product_quantizer dim=2 factor=array_unroll_factor 
#pragma HLS resource variable=sub_product_quantizer core=RAM_2P_BRAM

    for (int i = 0; i < M / LUT_CONSTR_SUB_PE_NUM; i++) {
        for (int k = 0; k < LUT_ENTRY_NUM; k++) {
            for (int j = 0; j < D / M; j++) {
#pragma HLS pipeline II=1
                sub_product_quantizer[k][i * (D / M) + j] = s_product_quantizer_init_sub_PE.read();
            }
        }
    }

    // store query and center vector into the format of M sub-vectors
    float sub_query_vector_local[(M / LUT_CONSTR_SUB_PE_NUM) * (D / M)];
#pragma HLS array_partition variable=sub_query_vector_local factor=array_unroll_factor 

    float sub_residual_vector[(M / LUT_CONSTR_SUB_PE_NUM) * (D / M)]; // query_vector - center_vector
#pragma HLS array_partition variable=sub_residual_vector factor=array_unroll_factor 

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

			// order see query_vector_dispatcher
			for (int i = 0; i < D / M; i++) {
				for (int j = 0; j < M / LUT_CONSTR_SUB_PE_NUM; j++) {
#pragma HLS pipeline II=1
					sub_query_vector_local[j * (D / M) + i] = s_sub_query_vectors.read();
				}
			}

			// for each nprobe, construct LUT
			for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {


				// order see center_vector_dispatcher
				for (int i = 0; i < D / M; i++) {
					for (int j = 0; j < M / LUT_CONSTR_SUB_PE_NUM; j++) {
#pragma HLS pipeline II=1
						// partial center vector
						float reg = s_sub_center_vectors.read();
						sub_residual_vector[j * (D / M) + i] = sub_query_vector_local[j * (D / M) + i] - reg;
					}
				}

				for (int k = 0; k < LUT_ENTRY_NUM; k++) {

				for (int m_per_PE = 0; m_per_PE < (M / LUT_CONSTR_SUB_PE_NUM); m_per_PE++) {
#if D/M <= 8
	#pragma HLS pipeline II=1 // For small dimensions we can afford to process 128D per cycle
#elif D/M <= 16
	#pragma HLS pipeline II=2 // Save resources
#elif D/M <= 32
	#pragma HLS pipeline II=4 // Save resources
#else
	#pragma HLS pipeline II=8 // Save resources
#endif 

						// the L1 diff between sub_residual_vector annd sub-quantizers
						float L1_dist[D/M];
#pragma HLS array_partition variable=L1_dist complete

							for (int j = m_per_PE * (D / M); j < (m_per_PE + 1) * (D / M); j++) {
#pragma HLS UNROLL
								L1_dist[j] = sub_residual_vector[j] - sub_product_quantizer[k][j];
							}

						// square distance
						float L2_dist = square_sum<D / M>(L1_dist);
						
						s_partial_distance_LUT.write(L2_dist);
					}
				}
			}
		}
	}
}

void gather_LUT_results(
    // runtime input
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<float> (&s_partial_distance_LUT)[LUT_CONSTR_SUB_PE_NUM],
    // output
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

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

			for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

				for (int k = 0; k < LUT_ENTRY_NUM; k++) {
#pragma HLS pipeline II=1

					// m 0 -> PE 0 ... m 15 -> PE 15; m 16 -> PE 0, ... m 31 -> PE 15; ...
					//  PE id = m % LUT_CONSTR_SUB_PE_NUM
					distance_LUT_parallel_t row;
					for (int m_per_PE = 0; m_per_PE < (M / LUT_CONSTR_SUB_PE_NUM); m_per_PE++) {
#pragma HLS pipeline II=1
						for (int pe_id = 0; pe_id < LUT_CONSTR_SUB_PE_NUM; pe_id++) {
#pragma HLS UNROLL
							row.dist[m_per_PE * LUT_CONSTR_SUB_PE_NUM + pe_id] = 
								s_partial_distance_LUT[pe_id].read();
						}
					}
					s_distance_LUT.write(row);
				}
			}
		}
	}
}

void LUT_construction_wrapper(
    hls::stream<float>& s_product_quantizer_init,
    // runtime input from network
    hls::stream<batch_header_t>& s_batch_header,
    hls::stream<ap_uint<512> >& s_query_vectors,
    hls::stream<ap_uint<512> >& s_center_vectors,
    // output
    hls::stream<distance_LUT_parallel_t>& s_distance_LUT) {

    // given input query vectors & Voronoi center vectors, return distance LUT
    //   throughput: ~= 1 row (M) of LUT per cycle, 256 CC per cell, except init time
    // output format: sending first row of M values, then second row, and so on...

#pragma HLS inline // inline to a dataflow region

    hls::stream<float> s_product_quantizer_init_sub_PE[LUT_CONSTR_SUB_PE_NUM];
#pragma HLS stream variable=s_product_quantizer_init_sub_PE depth=2
#pragma HLS array_partition variable=s_product_quantizer_init_sub_PE complete
    
    product_quantizer_dispatcher(
        s_product_quantizer_init,
        s_product_quantizer_init_sub_PE);

    hls::stream<float> s_sub_query_vectors[LUT_CONSTR_SUB_PE_NUM];
#pragma HLS stream variable=s_sub_query_vectors depth=2
#pragma HLS array_partition variable=s_sub_query_vectors complete

    hls::stream<float> s_sub_center_vectors[LUT_CONSTR_SUB_PE_NUM];
#pragma HLS stream variable=s_sub_center_vectors depth=2
#pragma HLS array_partition variable=s_sub_center_vectors complete

    const int n_batch_header_streams = 3 + LUT_CONSTR_SUB_PE_NUM;
    hls::stream<batch_header_t> s_batch_header_replicated[n_batch_header_streams];
#pragma HLS stream variable=s_batch_header_replicated depth=8
#pragma HLS array_partition variable=s_batch_header_replicated complete

	replicate_s_batch_header<n_batch_header_streams>(
		s_batch_header, 
		s_batch_header_replicated);
    query_vector_dispatcher(
        s_batch_header_replicated[0],
        s_query_vectors,
        s_sub_query_vectors);

    center_vector_dispatcher(
        s_batch_header_replicated[1],
        s_center_vectors,
        s_sub_center_vectors);

    hls::stream<float> s_partial_distance_LUT[LUT_CONSTR_SUB_PE_NUM];
#pragma HLS stream variable=s_partial_distance_LUT depth=2
#pragma HLS array_partition variable=s_partial_distance_LUT complete

    for (int pe_id = 0; pe_id < LUT_CONSTR_SUB_PE_NUM; pe_id++) {
#pragma HLS UNROLL
        LUT_construction_sub_PE(
            s_product_quantizer_init_sub_PE[pe_id],
            s_batch_header_replicated[2 + pe_id],
            s_sub_query_vectors[pe_id],
            s_sub_center_vectors[pe_id],
            s_partial_distance_LUT[pe_id]); 
    }

    gather_LUT_results(
        s_batch_header_replicated[2 + LUT_CONSTR_SUB_PE_NUM], 
        s_partial_distance_LUT,
        s_distance_LUT);
}