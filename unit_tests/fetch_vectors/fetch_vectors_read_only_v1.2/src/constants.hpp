#pragma once

#define FLOAT_PER_AXI 16 // 512 bit / 32 bit = 16
#define INT_PER_AXI 16
#define BYTE_PER_AXI 64
const int float_per_axi = FLOAT_PER_AXI;

#define D_MAX 1024

// largest 32-bit float: 3.4028237 × 10^38
const float large_float = 1E+20f; // 1E+20f is large enough for cosine similarity

// max queue sizes
const int hardware_result_queue_size = 128;
// const int hardware_candidate_queue_size = 128;
