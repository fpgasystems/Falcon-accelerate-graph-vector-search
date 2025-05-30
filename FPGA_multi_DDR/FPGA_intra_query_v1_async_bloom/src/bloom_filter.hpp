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
		// input streams
		hls::stream<int>& s_num_candidates,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<ap_uint<32>> (&s_hash_values_per_pe)[num_hash_funs],
		hls::stream<int>& s_finish_in,

		// output streams
		hls::stream<int>& s_num_valid_candidates_burst, // does not exist in bloom filter
		hls::stream<int>& s_num_valid_candidates_total, // one round can contain multiple bursts
		hls::stream<cand_t>& s_valid_candidates,
		hls::stream<int>& s_finish_out) {

		bool first_iter_s_all_candidates = true;
		bool first_iter_s_hash_values_per_pe = true;
		this->reset(); // reset before the first query

		// break num valid to smaller units, otherwise the pipeline can be deadlocked
		const int max_burst_size = 64; 

		for (int qid = 0; qid < query_num; qid++) {

			while (true) {

				if (!s_finish_in.empty() && s_num_candidates.empty() && s_all_candidates.empty()
					&& all_streams_empty<num_hash_funs, ap_uint<32>>(s_hash_values_per_pe)) {
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

					int num_valid_burst = 0;
					int num_valid_total = 0;
					bool sent_out_s_num_valid_candidates_burst = false; // needs to be at least sent once even with 0 results
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
							num_valid_burst++;
							num_valid_total++;
						}
						// if already some data in data fifo, write num acount
						if (num_valid_burst == max_burst_size) {
							s_num_valid_candidates_burst.write(num_valid_burst);
							num_valid_burst = 0;
							sent_out_s_num_valid_candidates_burst = true;
						}
					}
					if (num_valid_burst > 0 || !sent_out_s_num_valid_candidates_burst) {
						s_num_valid_candidates_burst.write(num_valid_burst);
					}
					s_num_valid_candidates_total.write(num_valid_total);
				}
			}
		}
	}

	// the main function that runs the bloom filter
	void run_bloom_filter(
		const int query_num, 
		const ap_uint<32> hash_seed,

		// in streams
		hls::stream<int>& s_num_candidates,
		hls::stream<cand_t>& s_all_candidates,
		hls::stream<int>& s_finish_in,

		// out streams
		hls::stream<int>& s_num_valid_candidates_burst, // one round (s_num_candidates) can contain multiple bursts
		hls::stream<int>& s_num_valid_candidates_total, // one round can contain multiple bursts
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

			s_num_valid_candidates_burst, // does not exist in bloom filter
			s_num_valid_candidates_total,
			s_valid_candidates,
			s_finish_out
		);
	}
};