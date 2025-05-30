#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "constants.hpp"

// priority queue 
enum Order { Collect_smallest, Collect_largest };

// The input to an ADC computation PE
typedef struct {
    bool valid;  // if invalid, the vector is dummy data, the compute PE
		 //  should handle it respectively
    int cell_ID; // Voronoi cell ID: [0, nlist - 1]
    int offset;  // the i-th vector in the cell in the memory channel
                 // different memory channels can have the same offsets
		 // the offset will be used to lookup the real vecID in the channel
    ap_uint<8> PQ_code[M];
} PQ_in_t;

// The output of an ADC computation PE & the input of L1 priority queues
 typedef struct {
    int cell_ID; // Voronoi cell ID: [0, nlist - 1]
    int offset;  // the i-th vector in the cell in the memory channel
                 // different memory channels can have the same offsets
		 // the offset will be used to lookup the real vecID in the channel
    float dist;  // invalid elements are marked as dist=LARGE_NUM
} PQ_out_t; 


// Convert cell_ID + offset into the read address in memory 
 typedef struct {
    int vec_ID_addr; 
    float dist;  // invalid elements are marked as dist=LARGE_NUM
} PQ_lookup_t; 

// the input & output of L2 priority queue
typedef struct {
    ap_uint<64> vec_ID;
    float dist;
} result_t; 

// the distance LUT input type
typedef struct {
    // each distance LUT has K=256 such row
    // each distance_LUT_PQ16_t is the content of a single row (16 floats)
    float dist[M];
} distance_LUT_parallel_t;
