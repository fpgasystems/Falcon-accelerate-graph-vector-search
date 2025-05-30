#pragma once

#include "types.hpp"

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

    // Wenqi comments: future upgrade -> less check on query finish, this will delay the progress
    //  approach: add a signal: everytime we check finish, check how many more computations should the unit expect

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
            
            if (!s_finish_query_in.empty() && s_fetched_vectors.empty() 
				&& s_fetch_valid.empty() && s_fetched_neighbor_ids_replicated.empty()) {
				s_finish_query_out.write(s_finish_query_in.read());
				break;
            } else if (!s_fetched_vectors.empty() && !s_fetch_valid.empty() &&!s_fetched_neighbor_ids_replicated.empty()) {

				cand_t reg_cand = s_fetched_neighbor_ids_replicated.read();
				bool valid = s_fetch_valid.read();

				float distance = 0;
				if (!valid) { // skip this vector
					distance = large_float;
					for (int i = 0; i < vec_AXI_num; i++) {
						volatile ap_uint<512> db_vec_reg = s_fetched_vectors.read();
					}
				} else { // compute

					for (int i = 0; i < vec_AXI_num; i++) {
					#pragma HLS pipeline II=1

						// read dist vector
						ap_uint<512> db_vec_reg = s_fetched_vectors.read();
						float distance_partial = 0;
									
						float database_vector_partial[FLOAT_PER_AXI];
						#pragma HLS array_partition variable=database_vector_partial complete

						for (int s = 0; s < FLOAT_PER_AXI; s++) {
						#pragma HLS unroll
							ap_uint<32> db_vec_reg_uint32 = db_vec_reg.range(32 * (s + 1) - 1, 32 * s);
							float db_vec_reg_float = *((float*) (&db_vec_reg_uint32));
							database_vector_partial[s] = *((float*) (&db_vec_reg_float));
							float diff = query_vector[i * FLOAT_PER_AXI + s] - database_vector_partial[s];
							distance_partial += diff * diff;
						}

						// accumulate distance
						distance += distance_partial;
					}
				}

				// write results
				result_t out;
				out.node_id = reg_cand.node_id;
				out.level_id = reg_cand.level_id;
				out.dist = distance;

				s_distances.write(out);
            }
        }
    }
}

