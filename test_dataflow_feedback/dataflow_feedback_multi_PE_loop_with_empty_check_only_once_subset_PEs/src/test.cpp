// https://docs.xilinx.com/r/en-US/ug1399-vitis-hls/Writing-a-Test-Bench
// #include "types.hpp"

#include <iostream>

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
    int*  layer_cache,
    // out (result) format: the first number writes total intersection count, 
    //   while the rest are intersect ID pairs
    int*  out_intersect 
    );
}

int main () { 

    int* layer_cache = new int[100];
    int* out_intersect = new int[100];

    // Call the top-level function, passing input stimuli as needed.
	vadd(  
        layer_cache,
        out_intersect);

    std::cout << "Test finished (no result evaluation)" << std::endl;
    
    return 0;
}