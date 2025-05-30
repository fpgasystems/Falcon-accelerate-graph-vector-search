#pragma once

#include "types.hpp"
#include "utils.hpp"

#define BITS_512_ADDR 9

template<int num_hash_funs, int num_bucket_addr_bits> // ap_uint cannot be used as template type
class BloomFilter {
public:

	ap_uint<512>* buckets;
	ap_uint<32> num_buckets;
	ap_uint<32> num_512b_buckets;
	ap_uint<32> runtime_num_buckets;
	ap_uint<32> runtime_num_512b_buckets; // 2^5-1=31, make sure later on the range selection would not overflow
	ap_uint<5> runtime_num_512b_bucket_addr_bits;

	BloomFilter(const int runtime_n_bucket_addr_bits) {
#pragma HLS inline
		this->num_buckets = 1 << num_bucket_addr_bits;
		const int num_512b_buckets_int = 1 << (num_bucket_addr_bits - BITS_512_ADDR) > 1? 1 << (num_bucket_addr_bits - BITS_512_ADDR) : 1;
		num_512b_buckets = num_512b_buckets_int;

		ap_uint<512> hash_buckets[num_512b_buckets_int];
#pragma HLS bind_storage variable=hash_buckets type=RAM_2P impl=BRAM
		this->buckets = hash_buckets;

		this->runtime_num_buckets = 1 << runtime_n_bucket_addr_bits;
		int runtime_num_512b_buckets_int = 1 << (runtime_n_bucket_addr_bits - BITS_512_ADDR) > 1? 1 << (runtime_n_bucket_addr_bits - BITS_512_ADDR) : 1;
		this->runtime_num_512b_buckets = runtime_num_512b_buckets_int;
		this->runtime_num_512b_bucket_addr_bits = runtime_n_bucket_addr_bits - BITS_512_ADDR;
		// this->reset(); // cannot reset here, in dataflow, bucket can only have a single reader/writer in one PE
	}

	void reset() {
		for (int i = 0; i < this->runtime_num_512b_buckets; i++) {
#pragma HLS pipeline II=1
			this->buckets[i] = false;
		}
	}

	ap_uint<32> MurmurHash2_KeyLen4 (ap_uint<32> key, ap_uint<32> hash_seed) {
#pragma HLS inline

		/* 'm' and 'r' are mixing constants generated offline.
		They're not really 'magic', they just happen to work well.  */

		const ap_uint<32> m = 0x5bd1e995;

		ap_uint<32> k = key;
		k *= m;
		k ^= k >> 24;
		k *= m;

		const int len = 4;
		ap_uint<32> h = hash_seed ^ len;
		h *= m;
		h ^= k;

		h ^= h >> 13;
		h *= m;
		h ^= h >> 15;

		return h;
	} 

	void stream_hash(
		const int query_num, 
		const ap_uint<32> hash_seed,
		// in streams
		hls::stream<int>& s_num_candidates,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<int>& s_finish_in,

		// out streams
		hls::stream<ap_uint<32>>& s_hash_values,
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_all_candidates = true;

		for (int qid = 0; qid < query_num; qid++) {

			while (true) {

				if (!s_finish_in.empty() && s_num_candidates.empty() && s_all_candidates.empty()) {
					s_finish_out.write(s_finish_in.read());
					break;
				} else if (!s_num_candidates.empty()) {
					int num_candidates = s_num_candidates.read();
					wait_data_fifo_first_iter<cand_t>(
						num_candidates, s_all_candidates, first_iter_s_all_candidates);
					for (int i = 0; i < num_candidates; i++) {
					#pragma HLS pipeline II=1
					 	cand_t cand = s_all_candidates.read();
						ap_uint<32> key = cand.node_id;
						ap_uint<32> hash = MurmurHash2_KeyLen4(key, hash_seed);
						s_hash_values.write(hash);
					}
				}
			}
		}
	}

	// the bloom filter's RAM part, check whether the input candidate is visited & update RAM
	void check_update(
		const int query_num, 
		hls::stream<int>& s_num_candidates,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<ap_uint<32>> (&s_hash_values_per_pe)[num_hash_funs],
		hls::stream<int>& s_finish_in,

		hls::stream<int>& s_num_processed_candidates_burst, // number of processed input, no matter whether valid
		hls::stream<int>& s_num_valid_candidates_burst, // does not exist in bloom filter
		hls::stream<cand_t>& s_valid_candidates,
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_all_candidates = true;
		bool first_iter_s_hash_values_per_pe = true;
		this->reset(); // reset before the first query

		// break num valid to smaller units, otherwise the pipeline can be deadlocked
		const int max_burst_size = 64; 

		for (int qid = 0; qid < query_num; qid++) {

			while (true) {

				if (!s_finish_in.empty() && s_num_candidates.empty() && all_streams_empty<num_hash_funs, ap_uint<32>>(s_hash_values_per_pe)) {
					s_finish_out.write(s_finish_in.read());
					// reset the hash buckets
					reset();
					break;
				} else if (!s_num_candidates.empty()) {
					int num_candidates = s_num_candidates.read();
					wait_data_fifo_first_iter<cand_t>(
						num_candidates, s_all_candidates, first_iter_s_all_candidates);
					wait_data_fifo_group_first_iter<num_hash_funs, ap_uint<32>>(
						num_candidates, s_hash_values_per_pe, first_iter_s_hash_values_per_pe);

					int num_valid = 0;
					int num_processed = 0;
					bool sent_out_s_num_valid_candidates_burst = false; // needs to be at least sent once even with 0 results
					bool sent_out_s_num_processed_candidates_burst = false; // needs to be at least sent once even with 0 results
					for (int i = 0; i < num_candidates; i++) {
						// check each bucket, if false, write true
						int bit_match_cnt = 0;
						cand_t cand = s_all_candidates.read();
						for (int j = 0; j < num_hash_funs; j++) {
							ap_uint<32> hash = s_hash_values_per_pe[j].read();
							// layout of bits in the hash code: [...useless bits...] [outer_bucket_id] [inner_bucket_id]
							// outer_bucket_id is the bucket ID to the 512-bit bucket array, while inner_bucket_id is the bit ID in the 512-bit bucket
							ap_uint<32> outer_bucket_id = hash.range(BITS_512_ADDR + this->runtime_num_512b_bucket_addr_bits - 1, BITS_512_ADDR);
							ap_uint<32> inner_bucket_id = hash.range(BITS_512_ADDR - 1, 0);
							if (!this->buckets[outer_bucket_id].range(inner_bucket_id, inner_bucket_id)) {
								this->buckets[outer_bucket_id].range(inner_bucket_id, inner_bucket_id) = 1;
							} else {
								bit_match_cnt++;
							}
						}
						if (bit_match_cnt < num_hash_funs) { // does not contain
							s_valid_candidates.write(cand);
							num_valid++;
						}
						num_processed++;
						// if already some data in data fifo, write num acount
						if (num_valid == max_burst_size) {
							s_num_valid_candidates_burst.write(num_valid);
							s_num_processed_candidates_burst.write(num_processed);
							num_valid = 0;
							num_processed = 0;
							sent_out_s_num_valid_candidates_burst = true;
							sent_out_s_num_processed_candidates_burst = true;
						}
					}
					if (num_valid > 0 || !sent_out_s_num_valid_candidates_burst) {
						s_num_valid_candidates_burst.write(num_valid);
					}
					if (num_processed > 0 || !sent_out_s_num_processed_candidates_burst) {
						s_num_processed_candidates_burst.write(num_processed);
					}
				}
			}
		}
	}

	// the main function that runs the bloom filter
	void run_bloom_filter(
		const int query_num, 
		const ap_uint<32> hash_seed,
		hls::stream<int>& s_num_candidates,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<int>& s_finish_in,

		hls::stream<int>& s_num_processed_candidates_burst, // number of processed input, no matter whether valid
		hls::stream<int>& s_num_valid_candidates_burst, // one round (s_num_candidates) can contain multiple bursts
		hls::stream<cand_t>& s_valid_candidates,
		hls::stream<int>& s_finish_out) {

#pragma HLS inline

		hls::stream<int> s_num_candidates_replicated[num_hash_funs + 1];
#pragma HLS stream variable=s_num_candidates_replicated depth=16

		hls::stream<cand_t> s_all_candidates_replicated[num_hash_funs + 1];
#pragma HLS stream variable=s_all_candidates_replicated depth=512

		hls::stream<int> s_finish_replicate_candidates;
#pragma HLS stream variable=s_finish_replicate_candidates depth=16

		replicate_s_read_iter_and_s_data<num_hash_funs + 1, cand_t>(
			// in (initialization)
			query_num,

			// in (stream)
			s_num_candidates,
			s_all_candidates,
			s_finish_in,
			
			// out (stream)
			s_num_candidates_replicated,
			s_all_candidates_replicated,
			s_finish_replicate_candidates
		);

		hls::stream<int> s_finish_replicate_candidates_replicated[num_hash_funs];
#pragma HLS stream variable=s_finish_replicate_candidates_replicated depth=16

		replicate_s_finish<num_hash_funs>(
			// in (initialization)
			query_num,
			// in (stream)
			s_finish_replicate_candidates,
			// out (stream)
			s_finish_replicate_candidates_replicated
		);

		hls::stream<ap_uint<32>> s_hash_values_per_pe[num_hash_funs];
#pragma HLS stream variable=s_hash_values_per_pe depth=512

		hls::stream<int> s_finish_hash_per_pe[num_hash_funs];
#pragma HLS stream variable=s_finish_hash_per_pe depth=16

		for (int pe_id = 0; pe_id < num_hash_funs; pe_id++) {
#pragma HLS UNROLL
			stream_hash(
				query_num, 
				hash_seed + pe_id, // hash_seed
				// in streams
				s_num_candidates_replicated[pe_id],
				s_all_candidates_replicated[pe_id],
				s_finish_replicate_candidates_replicated[pe_id],

				// out streams
				s_hash_values_per_pe[pe_id],
				s_finish_hash_per_pe[pe_id]
			);
		}

		hls::stream<int> s_finish_hash;
#pragma HLS stream variable=s_finish_hash depth=16

		gather_s_finish<num_hash_funs>(
			// in (initialization)
			query_num,
			// in (stream)
			s_finish_hash_per_pe,
			// out (stream)
			s_finish_hash
		);

		check_update(
			query_num, 
			s_num_candidates_replicated[num_hash_funs],
			s_all_candidates_replicated[num_hash_funs],
			s_hash_values_per_pe,
			s_finish_hash,

			s_num_processed_candidates_burst, // number of processed input, no matter whether valid
			s_num_valid_candidates_burst, // does not exist in bloom filter
			s_valid_candidates,
			s_finish_out
		);
	}
	

	void split_upper_base_workloads(
		const int query_num, 
		const ap_uint<32> hash_seed,

		// in streams: from the fetch neighbor ID PE
		hls::stream<int>& s_num_neighbors_upper_levels,
		hls::stream<int>& s_num_neighbors_base_level,
		hls::stream<cand_t>& s_all_candidates,
		// in streams: from bloom filter
		hls::stream<int>& s_num_processed_candidates_burst_in, // one round (s_num_neighbors) can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_burst_in, // one round (s_num_neighbors) can contain multiple bursts
		hls::stream<cand_t>& s_valid_candidates_burst_in,
		hls::stream<int>& s_finish_in,

		// out streams: to bloom 
		hls::stream<int>& s_num_neighbors_base_level_to_bloom, 
		hls::stream<cand_t>& s_base_candidates_to_bloom,
		// out stream: to output
		hls::stream<int>& s_num_valid_candidates_burst_out, // one round (s_num_neighbors) can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_upper_levels_total_out, // one round can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_base_level_total_out, // one round can contain multiple bursts
		hls::stream<cand_t>& s_valid_candidates_out, // contains both burst base layer / or complete upper layer
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_all_candidates = true;
		bool first_iter_s_valid_candidates_burst_in = true;

		// if input number of candidates > FIFO length, with bursting there would be a deadlock
		const int max_burst_to_bloom_filter = 128; 

		for (int qid = 0; qid < query_num; qid++) {

			int current_base_nodes_before_filtering = 0; // before bloom filter
			int current_base_nodes_sent_to_filter = 0; // already sent to bloom filter (burst by burst)
			int current_base_nodes_filtered = 0; // already processed by bloom filter (burst by burst)
			int current_base_nodes_valid = 0; // already processed & valid by bloom filter (burst by burst)
			int burst_to_bloom_to_be_consumed = 0; // sent to bloom, but not yet consumed

			while (true) {
				if (!s_finish_in.empty() && s_num_neighbors_upper_levels.empty() 
					&& s_num_neighbors_base_level.empty() && s_all_candidates.empty()
					&& s_num_processed_candidates_burst_in.empty() && s_num_valid_candidates_burst_in.empty()
					&& s_valid_candidates_burst_in.empty()) {

					s_finish_out.write(s_finish_in.read());
					break;
				} 
				// forward the upper level neighbors to output, when no other rounds are being processed
				else if (!s_num_neighbors_upper_levels.empty() && current_base_nodes_before_filtering == 0) {

					int num_neighbors = s_num_neighbors_upper_levels.read();
					wait_data_fifo_first_iter<cand_t>(
						num_neighbors, s_all_candidates, first_iter_s_all_candidates);
					s_num_valid_candidates_burst_out.write(num_neighbors);
					s_num_valid_candidates_upper_levels_total_out.write(num_neighbors);
					for (int i = 0; i < num_neighbors; i++) {
					#pragma HLS pipeline II=1
						cand_t cand = s_all_candidates.read();
						s_valid_candidates_out.write(cand);
					}
				} 
				// forward the base level neighbors to bloom filter, when no other rounds are being processed
				// first iteration of forwarding upper levels
				else if (!s_num_neighbors_base_level.empty() && current_base_nodes_before_filtering == 0 && burst_to_bloom_to_be_consumed == 0) {

					int num_neighbors = s_num_neighbors_base_level.read();
					current_base_nodes_before_filtering = num_neighbors;
					wait_data_fifo_first_iter<cand_t>(
						num_neighbors, s_all_candidates, first_iter_s_all_candidates);
					int num_cand_to_bloom_this_burst = num_neighbors > max_burst_to_bloom_filter? max_burst_to_bloom_filter : num_neighbors;
					current_base_nodes_sent_to_filter = num_cand_to_bloom_this_burst;
					burst_to_bloom_to_be_consumed = num_cand_to_bloom_this_burst;
					s_num_neighbors_base_level_to_bloom.write(num_cand_to_bloom_this_burst);
					for (int i = 0; i < num_cand_to_bloom_this_burst; i++) {
					#pragma HLS pipeline II=1
						cand_t cand = s_all_candidates.read();
						s_base_candidates_to_bloom.write(cand);
					}
				}
				// rest rounds of forwarding upper levels
				else if (current_base_nodes_before_filtering > 0 && current_base_nodes_sent_to_filter < current_base_nodes_before_filtering
					&& burst_to_bloom_to_be_consumed < max_burst_to_bloom_filter) {

					int num_cand_to_bloom_this_burst = current_base_nodes_before_filtering - current_base_nodes_sent_to_filter;
					num_cand_to_bloom_this_burst = num_cand_to_bloom_this_burst > max_burst_to_bloom_filter - burst_to_bloom_to_be_consumed? 
						max_burst_to_bloom_filter - burst_to_bloom_to_be_consumed : num_cand_to_bloom_this_burst;
					current_base_nodes_sent_to_filter += num_cand_to_bloom_this_burst;
					burst_to_bloom_to_be_consumed += num_cand_to_bloom_this_burst;
					s_num_neighbors_base_level_to_bloom.write(num_cand_to_bloom_this_burst);
					for (int i = 0; i < num_cand_to_bloom_this_burst; i++) {
					#pragma HLS pipeline II=1
						cand_t cand = s_all_candidates.read();
						s_base_candidates_to_bloom.write(cand);
					}
				}
				// forward the valid candidates & number from bloom to output
				else if (!s_num_processed_candidates_burst_in.empty() && !s_num_valid_candidates_burst_in.empty()) {

					int num_valid = s_num_valid_candidates_burst_in.read();
					current_base_nodes_valid += num_valid;
					wait_data_fifo_first_iter<cand_t>(
						num_valid, s_valid_candidates_burst_in, first_iter_s_valid_candidates_burst_in);
					s_num_valid_candidates_burst_out.write(num_valid);
					for (int i = 0; i < num_valid; i++) {
					#pragma HLS pipeline II=1
						cand_t valid_cand = s_valid_candidates_burst_in.read();
						s_valid_candidates_out.write(valid_cand);
					}
					int num_processed = s_num_processed_candidates_burst_in.read();
					current_base_nodes_filtered += num_processed;
					burst_to_bloom_to_be_consumed -= num_processed;
					// whether finished processing all batches in the current round
					if (current_base_nodes_filtered == current_base_nodes_before_filtering) {
						s_num_valid_candidates_base_level_total_out.write(current_base_nodes_valid);
						current_base_nodes_valid = 0;
						current_base_nodes_filtered = 0;
						current_base_nodes_before_filtering = 0;
						current_base_nodes_sent_to_filter = 0;
						burst_to_bloom_to_be_consumed = 0;
					}
				}
			}
		}
	}

	// the top-level function that runs the bloom filter & bypassing the upper level
	void bloom_filter_top_level(
		const int query_num, 
		const ap_uint<32> hash_seed,
		// in streams
		hls::stream<int>& s_num_neighbors_upper_levels,
		hls::stream<int>& s_num_neighbors_base_level,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<int>& s_finish_in,

		// out streams
		hls::stream<int>& s_num_valid_candidates_burst, // one round (s_num_neighbors) can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_upper_levels_total_out, // one round can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_base_level_total_out, // one round can contain multiple bursts
		hls::stream<cand_t>& s_valid_candidates,
		hls::stream<int>& s_finish_out) {

#pragma HLS dataflow

		hls::stream<int> s_num_neighbors_base_level_to_bloom;
#pragma HLS stream variable=s_num_neighbors_base_level_to_bloom depth=16

		hls::stream<cand_t> s_base_candidates_to_bloom;
#pragma HLS stream variable=s_base_candidates_to_bloom depth=512

		hls::stream<int> s_num_processed_candidates_burst_from_bloom; // one round (s_num_neighbors) can contain multiple bursts
#pragma HLS stream variable=s_num_processed_candidates_burst_from_bloom depth=16

		hls::stream<int> s_num_valid_candidates_burst_from_bloom; // one round (s_num_neighbors) can contain multiple bursts
#pragma HLS stream variable=s_num_valid_candidates_burst_from_bloom depth=16

		hls::stream<cand_t> s_valid_candidates_burst_from_bloom;
#pragma HLS stream variable=s_valid_candidates_burst_from_bloom depth=512

		hls::stream<int> s_finish_split_upper_base_workloads;
#pragma HLS stream variable=s_finish_split_upper_base_workloads depth=16

		split_upper_base_workloads(
			query_num, 
			hash_seed,

			// in streams: from the fetch neighbor ID PE
			s_num_neighbors_upper_levels,
			s_num_neighbors_base_level,
			s_all_candidates,
			// in streams: from bloom filter
			s_num_processed_candidates_burst_from_bloom, // one round (s_num_neighbors) can contain multiple bursts
			s_num_valid_candidates_burst_from_bloom, // one round (s_num_neighbors) can contain multiple bursts
			s_valid_candidates_burst_from_bloom,
			s_finish_in,

			// out streams: to bloom 
			s_num_neighbors_base_level_to_bloom, 
			s_base_candidates_to_bloom,
			// out stream: to output
			s_num_valid_candidates_burst, // one round (s_num_neighbors) can contain multiple bursts
			s_num_valid_candidates_upper_levels_total_out, // one round can contain multiple bursts
			s_num_valid_candidates_base_level_total_out, // one round can contain multiple bursts
			s_valid_candidates, // contains both burst base layer / or complete upper layer
			s_finish_split_upper_base_workloads
		);

		run_bloom_filter(
			query_num, 
			hash_seed,
			s_num_neighbors_base_level_to_bloom,
			s_base_candidates_to_bloom,
			s_finish_split_upper_base_workloads,

			s_num_processed_candidates_burst_from_bloom, // number of processed input, no matter whether valid
			s_num_valid_candidates_burst_from_bloom, // one round (s_num_candidates) can contain multiple bursts
			s_valid_candidates_burst_from_bloom,
			s_finish_out
		);
	}
};