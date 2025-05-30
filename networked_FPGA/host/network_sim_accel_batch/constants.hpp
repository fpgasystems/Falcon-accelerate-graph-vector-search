#pragma once

// Variables that can be configured as a parameters
// #define NLIST 8192
// #define NPROBE 32
// #define QUERY_NUM 10000
// #define ITER_NUM_PER_QUERY 10000

// Variables that can be changed & should be set at compile time
#define D 128
#define M 32
#define LUT_ENTRY_NUM 256

// Derived & Fixed numbers
#define NLIST_MAX 262144 // 256K centroids at max
#define TOPK 100
#define DDR_BANK_NUM 4
#define ADC_PE_PER_CHANNEL (512 / 8 / M) // number of vectors per 512-bit AXI interface
#define ADC_PE_NUM (DDR_BANK_NUM * ADC_PE_PER_CHANNEL)
#define PRIORITY_QUEUE_NUM_L1 (2 * ADC_PE_NUM)
#define PRIORITY_QUEUE_PER_BANK (PRIORITY_QUEUE_NUM_L1 / 4)
#if M == 8 // probablistic approximate priority queue group
    #define PRIORITY_QUEUE_LEN_L1 10
#elif M == 16
    #define PRIORITY_QUEUE_LEN_L1 15
#elif M == 32
    #define PRIORITY_QUEUE_LEN_L1 23
#elif M == 16
    #define PRIORITY_QUEUE_LEN_L1 38
#endif
#define PRIORITY_QUEUE_LEN_L2 TOPK
#define LARGE_NUM 9999999999