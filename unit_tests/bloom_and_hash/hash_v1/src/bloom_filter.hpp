#pragma once

#include "types.hpp"

ap_uint<32> MurmurHash2_KeyLen4 (ap_uint<32> key, ap_uint<32> seed) {
#pragma HLS inline

    /* 'm' and 'r' are mixing constants generated offline.
     They're not really 'magic', they just happen to work well.  */

    const ap_uint<32> m = 0x5bd1e995;

    ap_uint<32> k = key;
    k *= m;
    k ^= k >> 24;
    k *= m;

    const int len = 4;
    ap_uint<32> h = seed ^ len;
    h *= m;
    h ^= k;

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
} 

void stream_hash(
	const int query_num, 
	const ap_uint<32> seed,
    // in streams
    hls::stream<int>& s_num_keys,
    hls::stream<ap_uint<32>>& s_keys,
    hls::stream<int>& s_finish_in,

    // out streams
    hls::stream<ap_uint<32>>& s_hash_values,
    hls::stream<int>& s_finish_out) {

    for (int qid = 0; qid < query_num; qid++) {

        while (true) {

            if (!s_finish_in.empty() && s_num_keys.empty() && s_keys.empty()) {
                s_finish_out.write(s_finish_in.read());
                break;
            } else if (!s_num_keys.empty() && !s_keys.empty()) {
                int num_keys = s_num_keys.read();
                for (int i = 0; i < num_keys; i++) {
#pragma HLS pipeline II=1
                    ap_uint<32> key = s_keys.read();
                    ap_uint<32> hash = MurmurHash2_KeyLen4(key, seed);
                    s_hash_values.write(hash);
                }
            }
        }
    }
}