#pragma once

#define FLOAT_PER_AXI 16 // 512 bit / 32 bit = 16
#define INT_PER_AXI 16
#define BYTE_PER_AXI 64
const int float_per_axi = FLOAT_PER_AXI;

#define D_MAX 1024
#define D 128

// largest 32-bit float: 3.4028237 × 10^38
const float large_float = 1E+20f; // 1E+20f is large enough for cosine similarity

// max queue sizes
const int hardware_result_queue_size = 128;
// const int hardware_candidate_queue_size = 128;


const int AXI_num_per_vector_and_padding = D % FLOAT_PER_AXI == 0? 
	D / FLOAT_PER_AXI + 1 : D / FLOAT_PER_AXI + 2; // 16 for D = 512 + visited padding

const int AXI_num_per_vector_only = D % FLOAT_PER_AXI == 0? 
	D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; // 16 for D = 512