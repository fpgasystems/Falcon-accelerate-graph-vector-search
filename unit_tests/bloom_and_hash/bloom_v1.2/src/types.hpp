#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#include "constants.hpp"

// candidate type
typedef struct {
    int node_id;
    int level_id;
} cand_t;

typedef struct {
	int node_id;
    int level_id;
	float dist;
} result_t;

enum Order { Collect_smallest, Collect_largest };