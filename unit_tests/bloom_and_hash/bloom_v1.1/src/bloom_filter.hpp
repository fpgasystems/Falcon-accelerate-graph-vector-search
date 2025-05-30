#pragma once

#include "types.hpp"
#include "utils.hpp"


template<int num_hash_funs, int num_bucket_addr_bits> // ap_uint cannot be used as template type
class BloomFilter {
public:

	bool* buckets;
	const ap_uint<32> num_buckets;
	ap_uint<32> runtime_num_buckets;
	ap_uint<5> runtime_num_bucket_addr_bits; // 2^5-1=31, make sure later on the range selection would not overflow

	BloomFilter(const int runtime_n_bucket_addr_bits) : 
		num_buckets(1 << num_bucket_addr_bits), runtime_num_buckets(1 << runtime_n_bucket_addr_bits), runtime_num_bucket_addr_bits(runtime_n_bucket_addr_bits) {
#pragma HLS inline
		bool hash_buckets[1 << num_bucket_addr_bits]; // cannot use num_buckets as initialization as it is ap_uint type
		this->buckets = hash_buckets;
		// this->reset(); // cannot reset here, in dataflow, bucket can only have a single reader/writer in one PE
	}

	void reset() {
		for (int i = 0; i < this->runtime_num_buckets; i++) {
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
		hls::stream<int>& s_num_keys,
		hls::stream<ap_uint<32>>& s_keys,
		hls::stream<int>& s_finish_in,

		// out streams
		hls::stream<ap_uint<32>>& s_hash_values,
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_keys = true;

		for (int qid = 0; qid < query_num; qid++) {

			while (true) {

				if (!s_finish_in.empty() && s_num_keys.empty() && s_keys.empty()) {
					s_finish_out.write(s_finish_in.read());
					break;
				} else if (!s_num_keys.empty()) {
					int num_keys = s_num_keys.read();
					wait_data_fifo_first_iter<ap_uint<32>>(
						num_keys, s_keys, first_iter_s_keys);
					for (int i = 0; i < num_keys; i++) {
					#pragma HLS pipeline II=1
						ap_uint<32> key = s_keys.read();
						ap_uint<32> hash = MurmurHash2_KeyLen4(key, hash_seed);
						s_hash_values.write(hash);
					}
				}
			}
		}
	}

	void check_update(
		const int query_num, 
		hls::stream<int>& s_num_keys,
		hls::stream<ap_uint<32>>& s_keys,
		hls::stream<ap_uint<32>> (&s_hash_values_per_pe)[num_hash_funs],
		hls::stream<int>& s_finish_in,

		hls::stream<int>& s_num_valid_keys, // does not exist in bloom filter
		hls::stream<ap_uint<32>>& s_valid_keys,
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_keys = true;
		bool first_iter_s_hash_values_per_pe = true;
		this->reset(); // reset before the first query

		// break num valid to smaller units, otherwise the pipeline can be deadlocked
		const int match_burst_size = 64; 

		for (int qid = 0; qid < query_num; qid++) {

			while (true) {

				if (!s_finish_in.empty() && s_num_keys.empty() && s_keys.empty() && all_streams_empty<num_hash_funs, ap_uint<32>>(s_hash_values_per_pe)) {
					s_finish_out.write(s_finish_in.read());
					// reset the hash buckets
					reset();
					break;
				} else if (!s_num_keys.empty()) {
					int num_keys = s_num_keys.read();
					wait_data_fifo_first_iter<ap_uint<32>>(
						num_keys, s_keys, first_iter_s_keys);
					wait_data_fifo_group_first_iter<num_hash_funs, ap_uint<32>>(
						num_keys, s_hash_values_per_pe, first_iter_s_hash_values_per_pe);

					int num_valid = 0;
					bool sent_out_s_num_valid_candidates_burst = false; // needs to be at least sent once even with 0 results
					for (int i = 0; i < num_keys; i++) {
						// check each bucket, if false, write true
						int bit_match_cnt = 0;
						ap_uint<32> key = s_keys.read();
						for (int j = 0; j < num_hash_funs; j++) {
							ap_uint<32> hash = s_hash_values_per_pe[j].read();
							ap_uint<32> bucket_id = hash.range(this->runtime_num_bucket_addr_bits - 1, 0);
							if (!this->buckets[bucket_id]) {
								this->buckets[bucket_id] = true;
							} else {
								bit_match_cnt++;
							}
						}
						if (bit_match_cnt < num_hash_funs) { // does not contain
							s_valid_keys.write(key);
							num_valid++;
						}
						// if already some data in data fifo, write num acount
						if (num_valid == match_burst_size) {
							s_num_valid_keys.write(num_valid);
							sent_out_s_num_valid_candidates_burst = true;
							num_valid = 0;
						}
					}
					if (num_valid > 0 || !sent_out_s_num_valid_candidates_burst) {
						s_num_valid_keys.write(num_valid);
					}
				}
			}
		}
	}

	// the main function that runs the bloom filter
	void run_continuous_insert_check(
		const int query_num, 
		const ap_uint<32> hash_seed,
		hls::stream<int>& s_num_keys,
		hls::stream<ap_uint<32>>& s_keys,
		hls::stream<int>& s_finish_in,

		hls::stream<int>& s_num_valid_keys, // does not exist in bloom filter
		hls::stream<ap_uint<32>>& s_valid_keys,
		hls::stream<int>& s_finish_out) {

#pragma HLS dataflow

		hls::stream<int> s_num_keys_replicated[num_hash_funs + 1];
#pragma HLS stream variable=s_num_keys_replicated depth=16

		hls::stream<ap_uint<32>> s_keys_replicated[num_hash_funs + 1];
#pragma HLS stream variable=s_keys_replicated depth=512

		hls::stream<int> s_finish_replicate_keys;
#pragma HLS stream variable=s_finish_replicate_keys depth=16

		replicate_s_read_iter_and_s_data<num_hash_funs + 1, ap_uint<32>>(
			// in (initialization)
			query_num,

			// in (stream)
			s_num_keys,
			s_keys,
			s_finish_in,
			
			// out (stream)
			s_num_keys_replicated,
			s_keys_replicated,
			s_finish_replicate_keys
		);

		hls::stream<int> s_finish_replicate_keys_replicated[num_hash_funs];
#pragma HLS stream variable=s_finish_replicate_keys_replicated depth=16

		replicate_s_finish<num_hash_funs>(
			// in (initialization)
			query_num,
			// in (stream)
			s_finish_replicate_keys,
			// out (stream)
			s_finish_replicate_keys_replicated
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
				s_num_keys_replicated[pe_id],
				s_keys_replicated[pe_id],
				s_finish_replicate_keys_replicated[pe_id],

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
			s_num_keys_replicated[num_hash_funs],
			s_keys_replicated[num_hash_funs],
			s_hash_values_per_pe,
			s_finish_hash,

			s_num_valid_keys, // does not exist in bloom filter
			s_valid_keys,
			s_finish_out
		);
	}
};