// #include "constants.hpp"
// #include "DRAM_utils.hpp"
// #include "join_PE.hpp"
// #include "scheduler.hpp"
// #include "types.hpp"

#include <hls_stream.h>

#define EMPTY_CHECK 1

void PE_A(
    // in
    hls::stream<int>& s_C_to_A,
    // out
    hls::stream<int>& s_A_to_B,
    int* out_intersect) {

    // PE A starts the loop
    s_A_to_B.write(1);

    // End by 
//#if EMPTY_CHECK
//    while (s_C_to_A.empty()) {}
//#endif
//    int out = s_C_to_A.read();
//    out_intersect[0] = out;
}

void PE_B(
    // in
    int* layer_cache,
    hls::stream<int>& s_A_to_B,
    // out
    hls::stream<int>& s_B_to_C) {

#if EMPTY_CHECK
    while (s_A_to_B.empty()) {}
#endif
    int in = s_A_to_B.read();
    int out = layer_cache[in];

    s_B_to_C.write(out);
}

void PE_C(
    // in
    hls::stream<int>& s_B_to_C,
    // out
    hls::stream<int>& s_C_to_A) {

#if EMPTY_CHECK
    while (s_B_to_C.empty()) {}
#endif
    int out = s_B_to_C.read();
//    s_C_to_A.write(out);
}


extern "C" {

void vadd(  
    // int max_level_A,
    // int max_level_B,
    // int root_id_A,
    // int root_id_B,
    // // in runtime (should from DRAM)
    // const ap_uint<512>* in_pages_A,
    // const ap_uint<512>* in_pages_B,
    // out (intermediate)
    int* layer_cache,
    // out (result) format: the first number writes total intersection count, 
    //   while the rest are intersect ID pairs
    int* out_intersect 
    )
{
// Share the same AXI interface with several control signals (but they are not allowed in same dataflow)
//    https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Controlling-AXI4-Burst-Behavior

// in runtime (should from DRAM)
// #pragma HLS INTERFACE m_axi port=in_pages_A offset=slave bundle=gmem0
// #pragma HLS INTERFACE m_axi port=in_pages_B offset=slave bundle=gmem1

// out
#pragma HLS INTERFACE m_axi port=layer_cache  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=out_intersect  offset=slave bundle=gmem3

#pragma HLS dataflow
    
    hls::stream<int> s_A_to_B("A to B"); 
#pragma HLS stream variable=s_A_to_B depth=512
    hls::stream<int> s_B_to_C("B to C"); 
#pragma HLS stream variable=s_B_to_C depth=512
    hls::stream<int> s_C_to_A("C to A"); 
#pragma HLS stream variable=s_C_to_A depth=512

    PE_A(
        // in
        s_C_to_A,
        // out
        s_A_to_B,
        out_intersect);

    PE_B(
        // in
        layer_cache,
        s_A_to_B,
        // out
        s_B_to_C);

    PE_C(
        // in
        s_B_to_C,
        // out
        s_C_to_A);
}

}
