#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "constants.hpp"


typedef struct {
    // minimum bounding rectangle 
    float low0; 
    float high0; 
    float low1; 
    float high1; 
} mbr_t; 


typedef struct {
    // obj id for data nodes; pointer to children for directory nodes
    int id; 
    // minimum bounding rectangle 
    float low0; 
    float high0; 
    float low1; 
    float high1; 
} obj_t; 

typedef struct {
    // 7 * 4 bytes = 28 bytes
    int is_leaf;  // bool 
    int count;    // valid items
    obj_t obj;    // id/ptr + mbr
} node_meta_t;

typedef struct {
    node_meta_t meta_data; 

    mbr_t mbrs[MAX_PAGE_ENTRIES]; 
    // for directory nodes: page addresses of the children
    // for leaf nodes: object IDs
    int ids[MAX_PAGE_ENTRIES];      
} node_t;

typedef struct {
    // these IDs can either be object IDs (for data nodes)
    //   or pointer to the children (directory nodes)
    int id_A;
    int id_B;
} pair_t;
