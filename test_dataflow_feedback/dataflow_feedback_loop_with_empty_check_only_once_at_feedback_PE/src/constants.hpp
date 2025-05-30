#pragma once

// #define DEBUG_read_nodes
// #define DEBUG_layer_cache_memory_controller
// #define DEBUG_write_results
// #define DEBUG_join_page
// #define DEBUG_scheduler

// 1 -> point/line intersection counts as region intersection
// 0 -> does not count
#define POINT_INTERSECT_COUNTS 1 

#define PAGE_SIZE 4096
#define PAGE_SIZE_PER_AXI (PAGE_SIZE / 64)
// mbr 16 bytes + children/obj_ID 8 bytes = 24 bytes; 128 * 20 = 2560 < 4096 
#define MAX_PAGE_ENTRIES 128 

#define LAYER_CACHE_SIZE (MAX_PAGE_ENTRIES * MAX_PAGE_ENTRIES * 8) // 8 = 2 x int

#define MAX_TREE_LEVEL 32

#define OBJ_BYTES 20 // 20 bytes (1 * id + 4 * boundary)
#define OBJ_BITS (OBJ_BYTES * 8) // 20 bytes (1 * id + 4 * boundary)
#define N_OBJ_PER_AXI 3 // 64 bytes can accommodate 3 objects 

#define MAX_PAGE_ENTRY_NUM 10000 // maximum number of entries per page