// host_single_FPGA: the complete implementation for single FPGA
//   2 thread, 1 for sending query, 1 for receiving results
//   the cells to scan can either be selected using HNSW or brute-force scan
//   the cells to scan is computed at query time

// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c
// Usage (e.g.): ./host_single_FPGA 10.253.74.24 8881 5001 1 32
//  "Usage: " << argv[0] << " <Tx (FPGA) IP_addr> <Tx send_port> <Rx recv_port> <WINDOW_SIZE (1~N, similar to batch size)> <nprobe>

// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdint.h>
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>

#include "ANN_socket.hpp"

// #define DEBUG

int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Init     //////////
    
    std::cout << "Usage: " << argv[0] << " <Tx (FPGA) IP_addr> <Tx send_port> <Rx recv_port> <WINDOW_SIZE (1~N, similar to batch size)> <nprobe>" << std::endl;

    const char* IP_addr;
    if (argc >= 2)
    {
        IP_addr = argv[1];
    } else {
        // IP_addr = "10.253.74.5"; // alveo-build-01
        IP_addr = "10.253.74.12"; // alveo-u250-01
        // IP_addr = "10.253.74.16"; // alveo-u250-02
        // IP_addr = "10.253.74.20"; // alveo-u250-03
        // IP_addr = "10.253.74.24"; // alveo-u250-04
    }

    unsigned int send_port = 8888;
    if (argc >= 3)
    {
        send_port = strtol(argv[2], NULL, 10);
    } 

    unsigned int recv_port = 5001;
    if (argc >= 4)
    {
        recv_port = strtol(argv[3], NULL, 10);
    } 
    
    // how many queries are allow to send ahead of receiving results
    // e.g., when window_size = 2, query 2 can be sent without receiving result 1
    // e.g., when window_size = 1, result 1 must be received before query 2 can be sent, 
    //          but might lead to a deadlock
    int window_size = 1;
    if (argc >= 5)
    {
        window_size = strtol(argv[4], NULL, 10);
    } 

    size_t nprobe = 1;
    if (argc >= 6)
    {
        nprobe = strtol(argv[5], NULL, 10);
    } 

    // Deep100M or Deep1000M or SIFT100M or SIFT1000M or SBERT1000M or SBERT3000M
    std::string db_name = "SBERT3000M"; 
    std::cout << "DB name: " << db_name << std::endl;
    
    std::string index_scan = "hnsw"; // hnsw or brute-force
    // std::string index_scan = "brute_force"; // hnsw or brute-force
    std::cout << "Index scan: " << index_scan << std::endl;

    //////////     Data loading / computing Part     //////////

    size_t D;
    size_t query_num;
    size_t nlist;
    std::string data_dir_prefix;
    std::string raw_gt_vec_ID_suffix_dir;
    std::string raw_gt_dist_suffix_dir;
    std::string gnd_dir;
    std::string product_quantizer_dir_suffix;
    std::string query_vectors_dir_suffix;
    std::string vector_quantizer_dir_suffix;
    size_t raw_gt_vec_ID_size;
    size_t raw_gt_dist_size;
    size_t len_per_result; 
    size_t result_start_bias;
    if (strncmp(db_name.c_str(), "SIFT", 4) == 0) {
        if (db_name == "SIFT100M") {
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF32768,PQ32";
            raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
            raw_gt_dist_suffix_dir = "dis_100M.fvecs";
        }
        else if (db_name == "SIFT1000M") {
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT1000M_IVF32768,PQ32";
            raw_gt_vec_ID_suffix_dir = "idx_1000M.ivecs";
            raw_gt_dist_suffix_dir = "dis_1000M.fvecs";
        }
        D = 128;
        query_num = 10000;
        nlist = 32768;
        gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/";
        product_quantizer_dir_suffix = "product_quantizer_float32_32_256_4_raw";
        query_vectors_dir_suffix = "query_vectors_float32_10000_128_raw";
        vector_quantizer_dir_suffix = "vector_quantizer_float32_32768_128_raw";
        len_per_result = 1001;
        result_start_bias = 1;
        raw_gt_vec_ID_size = 10000 * 1001 * sizeof(int);
        raw_gt_dist_size = 10000 * 1001 * sizeof(float);
    } else if (strncmp(db_name.c_str(), "Deep", 4) == 0) {
        if (db_name == "Deep100M") {
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/Deep100M_IVF32768,PQ32";
            raw_gt_vec_ID_suffix_dir = "gt_idx_100M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_100M.fbin";
        }
        else if (db_name == "Deep1000M") {
            data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/Deep1000M_IVF32768,PQ32";
            raw_gt_vec_ID_suffix_dir = "gt_idx_1000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_1000M.fbin";
        }
        D = 96;
        query_num = 10000;
        nlist = 32768;
        gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/deep1b/";
        product_quantizer_dir_suffix = "product_quantizer_float32_32_256_3_raw";
        query_vectors_dir_suffix = "query_vectors_float32_10000_96_raw";
        vector_quantizer_dir_suffix = "vector_quantizer_float32_32768_96_raw";
        len_per_result = 100;
        result_start_bias = 2;
        raw_gt_vec_ID_size = (2 + 10000 * 100) * sizeof(int);
        raw_gt_dist_size = (2 + 10000 * 100) * sizeof(float);
    }  else if (strncmp(db_name.c_str(), "SBERT", 5) == 0) {
        if (db_name == "SBERT1000M") {
            // if (shard_ID == 0) {
                data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF32768,PQ64_2shards/shard_0";
            // } else if (shard_ID == 1) {
            //     data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF32768,PQ64_2shards/shard_1";
            // }
            nlist = 32768;
            raw_gt_vec_ID_suffix_dir = "gt_idx_1000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_1000M.fbin";
            vector_quantizer_dir_suffix = "vector_quantizer_float32_32768_384_raw";
        }
        else if (db_name == "SBERT3000M") {
            // if (shard_ID == 0) {
                data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT3000M_IVF65536,PQ64_4shards/shard_0";
            // } else if (shard_ID == 1) {
            //     data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF65536,PQ64_4shards/shard_1";
            // } else if (shard_ID == 2) {
            //     data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF65536,PQ64_4shards/shard_2";
            // } else if (shard_ID == 3) {
            //     data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SBERT1000M_IVF65536,PQ64_4shards/shard_3";
            // }
            nlist = 65536;
            raw_gt_vec_ID_suffix_dir = "gt_idx_3000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_3000M.fbin";
            vector_quantizer_dir_suffix = "vector_quantizer_float32_65536_384_raw";
        }
        D = 384;
        query_num = 10000;
        gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/sbert/";
        product_quantizer_dir_suffix = "product_quantizer_float32_64_256_6_raw";
        query_vectors_dir_suffix = "query_vectors_float32_10000_384_raw";
        raw_gt_vec_ID_size = (10000 * 1000 + 2) * sizeof(int);
        raw_gt_dist_size = (10000 * 1000 + 2) * sizeof(float);
        len_per_result = 1000;
        result_start_bias = 2;
    }  else if (strncmp(db_name.c_str(), "GNN", 3) == 0) {
        if (db_name == "GNN1400M") {
            // if (shard_ID == 0) {
                data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/GNN1400M_IVF32768,PQ64_2shards/shard_0";
            // } else if (shard_ID == 1) {
            //     data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/GNN1400M_IVF32768,PQ64_2shards/shard_1";
            // }
            nlist = 32768; 
            raw_gt_vec_ID_suffix_dir = "gt_idx_1000M.ibin";
            raw_gt_dist_suffix_dir = "gt_dis_1000M.fbin";
            vector_quantizer_dir_suffix = "vector_quantizer_float32_32768_256_raw";
        }
        D = 256;
        query_num = 10000;
        gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/Marius_GNN/";
        product_quantizer_dir_suffix = "product_quantizer_float32_64_256_4_raw";
        query_vectors_dir_suffix = "query_vectors_float32_10000_256_raw";
        raw_gt_vec_ID_size = (10000 * 1000 + 2) * sizeof(int);
        raw_gt_dist_size = (10000 * 1000 + 2) * sizeof(float);
        len_per_result = 1000;
        result_start_bias = 2;
    }


    assert (nprobe <= nlist);

    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = TOPK * 64 % 512 == 0?
        TOPK * 64 / 512 : TOPK * 64 / 512 + 1;
    size_t size_results_dist = TOPK * 32 % 512 == 0?
        TOPK * 32 / 512 : TOPK * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet
    size_t out_bytes = query_num * 64 * size_results;
    int recv_bytes_per_query = 64 * size_results;

    std::cout << "recv_bytes_per_query: " << recv_bytes_per_query << std::endl;
    ///////////     get data size from disk     //////////
    
    // info used to Select Cells to Scan
    std::string query_vectors_dir = dir_concat(data_dir_prefix, query_vectors_dir_suffix);
    std::ifstream query_vectors_fstream(
        query_vectors_dir, 
        std::ios::in | std::ios::binary);
    query_vectors_fstream.seekg(0, query_vectors_fstream.end);
    size_t query_vectors_size =  query_vectors_fstream.tellg();
    if (!query_vectors_size) std::cout << "query_vectors_size is 0!";
    query_vectors_fstream.seekg(0, query_vectors_fstream.beg);
    
    std::string vector_quantizer_dir = dir_concat(data_dir_prefix, vector_quantizer_dir_suffix);
    std::ifstream vector_quantizer_fstream(
        vector_quantizer_dir, 
        std::ios::in | std::ios::binary);
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.end);
    size_t vector_quantizer_size =  vector_quantizer_fstream.tellg();
    if (!vector_quantizer_size) std::cout << "vector_quantizer_size is 0!";
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.beg);

    // ground truth 
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);

    std::string raw_gt_dist_dir = dir_concat(gnd_dir, raw_gt_dist_suffix_dir);
    std::ifstream raw_gt_dist_fstream(
        raw_gt_dist_dir,
        std::ios::in | std::ios::binary);

    //////////     Allocate Memory     //////////
    
    // on host side, used to Select Cells to Scan 
    std::vector<float, aligned_allocator<float>> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float, aligned_allocator<float>> vector_quantizer(vector_quantizer_size / sizeof(float));

    // output data
    char* out_buf = new char[out_bytes];
    memset(out_buf, 0, out_bytes);
    
    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    // recall counts the very first nearest neighbor only
    size_t gt_vec_ID_size = 10000 * sizeof(uint32_t);
    std::vector<uint32_t, aligned_allocator<uint32_t>> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(uint32_t), 0);
    std::vector<uint32_t, aligned_allocator<uint32_t>> gt_vec_ID(gt_vec_ID_size / sizeof(uint32_t), 0);
    
    size_t gt_dist_size = 10000 * sizeof(float);
    std::vector<float, aligned_allocator<float>> raw_gt_dist(raw_gt_dist_size / sizeof(float), 0);
    std::vector<float, aligned_allocator<float>> gt_dist(gt_dist_size / sizeof(float), 0);

    //////////     load data from disk     //////////
    std::cout << "Loading data from disk...\n";

    // on host, used to Select Cells to Scan
    char* query_vectors_char = (char*) malloc(query_vectors_size);
    query_vectors_fstream.read(query_vectors_char, query_vectors_size);
    if (!query_vectors_fstream) {
            std::cout << "error: only " << query_vectors_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&query_vectors[0], query_vectors_char, query_vectors_size);
    free(query_vectors_char);
    
    char* vector_quantizer_char = (char*) malloc(vector_quantizer_size);
    vector_quantizer_fstream.read(vector_quantizer_char, vector_quantizer_size);
    if (!vector_quantizer_fstream) {
            std::cout << "error: only " << vector_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vector_quantizer[0], vector_quantizer_char, vector_quantizer_size);
    free(vector_quantizer_char);

    assert(D * nlist * sizeof(float) == vector_quantizer_size);

    // ground truth
    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);
    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    free(raw_gt_vec_ID_char);

    for (int i = 0; i < 10000; i++) {
        gt_vec_ID[i] = raw_gt_vec_ID[i * len_per_result + result_start_bias];
    }

    char* raw_gt_dist_char = (char*) malloc(raw_gt_dist_size);
    raw_gt_dist_fstream.read(raw_gt_dist_char, raw_gt_dist_size);
    if (!raw_gt_dist_fstream) {
        std::cout << "error: only " << raw_gt_dist_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_dist[0], raw_gt_dist_char, raw_gt_dist_size);
    free(raw_gt_dist_char);

    for (int i = 0; i < 10000; i++) {
        gt_dist[i] = raw_gt_dist[i * len_per_result + result_start_bias];
    }

    // HNSWlib index
    hnswlib::AlgorithmInterface<float>* alg_hnswlib;
    hnswlib::L2Space space(D);
    if (index_scan == "brute_force") {
        std::string brute_force_index_dir = dir_concat(data_dir_prefix, "hnswlib_brute_force_index.bin");
        std::ifstream f_hnswlib(brute_force_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "brute_force Index exists, loading index..." << std::endl;
            alg_hnswlib = new hnswlib::BruteforceSearch<float>(&space, brute_force_index_dir);
        }
        else {
            std::cout << "brute_force Index does not exist, creating new index..." << std::endl;
            alg_hnswlib = new hnswlib::BruteforceSearch<float>(&space, nlist);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib->saveIndex(brute_force_index_dir);
        }
    } else if (index_scan == "hnsw") {
        std::string hnsw_index_dir = dir_concat(data_dir_prefix, "hnswlib_hnsw_index.bin");
        std::ifstream f_hnswlib(hnsw_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "HNSW Index exists, loading index..." << std::endl;
            alg_hnswlib = new hnswlib::HierarchicalNSW<float>(&space, hnsw_index_dir);
        }
        else {
            std::cout << "HNSW Index does not exist, creating new index..." << std::endl;
            size_t M_hnswlib = 64;
            size_t ef_construction = 800;
            alg_hnswlib = new hnswlib::HierarchicalNSW<float>(&space, nlist,  M_hnswlib = M_hnswlib, ef_construction = ef_construction);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib->saveIndex(hnsw_index_dir);
        }
        // ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib)->setEf(64);
        std::cout << "ef: " << ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib)->ef_ << std::endl;
    } else {
        std::cout << "index option does not exists, either brute_force or hnsw" << std::endl;
        exit(1);
    }

    //////////     Networking Part     //////////

    // inter-thread communication by shared memory
    int start_recv = 0; 
    int finish_recv_query_id = -1;

    // profiler
    std::vector<std::chrono::system_clock::time_point> query_start_time(query_num);
    std::vector<std::chrono::system_clock::time_point> query_finish_time(query_num);

    std::thread t_send(
        thread_send_packets, IP_addr, send_port, query_num, nprobe,
        query_vectors.data(), vector_quantizer.data(), D, alg_hnswlib,
        window_size, &start_recv, &finish_recv_query_id,
        query_start_time.data());

    std::thread t_recv(
        thread_recv_packets, recv_port, query_num, recv_bytes_per_query, out_buf,
        &start_recv, &finish_recv_query_id, query_finish_time.data());

    t_send.join();
    t_recv.join();

    //////////     Performance evaluation    //////////

    // TCP has a slow start, thus the performance of the first 1000 queries are not counted
    int start_query_id = int(query_num * 0.1);
    std::cout << "Calculating performance skipping the first 10% of query due to the slow "
        "start of TCP/IP..." << std::endl; 

    // calculate QPS
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
            query_finish_time[query_num - 1] - query_start_time[start_query_id]).count());
    std::cout << "Overall QPS = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;


    std::vector<double> durationUs_per_query(query_num - start_query_id);
    for (int i = start_query_id; i < query_num; i++) {
        double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
            query_finish_time[i] - query_start_time[i]).count());
        durationUs_per_query[i - start_query_id] = durationUs;
#ifdef DEBUG
        std::cout << "query " << i << " duration (us) = " << durationUs << std::endl;
#endif
    }
    std::sort(durationUs_per_query.begin(), durationUs_per_query.end());
    std::cout << "Median latency (us) = " << durationUs_per_query[int(durationUs_per_query.size() * 0.5)] << std::endl;
    std::cout << "95% tail latency (us) = " << durationUs_per_query[int(durationUs_per_query.size() * 0.95)] << std::endl;
    std::cout << "Worst latency (us) = " << durationUs_per_query[durationUs_per_query.size() - 1] << std::endl;

    //////////     Recall evaluation    //////////
    std::cout << "Comparing Results..." << std::endl;
    
    int match_count_R1_at_1 = 0;
    int match_count_R1_at_10 = 0;
    int match_count_R1_at_100 = 0;
    int match_count_R_at_1 = 0;
    int match_count_R_at_10 = 0;
    int match_count_R_at_100 = 0;

    for (int query_id = 0; query_id < query_num; query_id++) {

#ifdef DEBUG
        std::cout << "query ID: " << query_id << std::endl;
#endif


        std::vector<long> hw_result_vec_ID_partial(TOPK, 0);
        std::vector<float> hw_result_dist_partial(TOPK, 0);
        std::vector<std::pair<float, long>> hw_result_pair(TOPK);

        int start_result_vec_ID_addr = (query_id * size_results + 1) * 64;
        int start_result_dist_addr = (query_id * size_results + 1 + size_results_vec_ID) * 64;

        // Load data
        memcpy(&hw_result_vec_ID_partial[0], &out_buf[start_result_vec_ID_addr], 8 * TOPK);
        memcpy(&hw_result_dist_partial[0], &out_buf[start_result_dist_addr], 4 * TOPK);
        for (int k = 0; k < TOPK; k++) {
            hw_result_pair[k] = std::make_pair(hw_result_dist_partial[k], hw_result_vec_ID_partial[k]);
        }
        std::sort(hw_result_pair.begin(), hw_result_pair.end());
        for (int k = 0; k < TOPK; k++) {
            hw_result_dist_partial[k] = hw_result_pair[k].first;
            hw_result_vec_ID_partial[k] = hw_result_pair[k].second;
        }
        

        int start_addr_gt = query_id * len_per_result + result_start_bias;

        // R1@K
        for (int k = 0; k < 1; k++) {
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_1++;
                break;
            }
        } 
        for (int k = 0; k < 10; k++) {
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_10++;
                break;
            }
        } 
        for (int k = 0; k < TOPK; k++) {
#ifdef DEBUG
            // std::cout << "hw: " << hw_result_vec_ID_partial[k] << " gt: " << gt_vec_ID[query_id] << 
            //     "hw dist: " << hw_result_dist_partial[k] << " gt dist: " << gt_dist[query_id] << std::endl;
#endif 
            if (hw_result_vec_ID_partial[k] == raw_gt_vec_ID[start_addr_gt]) {
                match_count_R1_at_100++;
                break;
            }
        } 

        // R@K
        std::unordered_set<size_t> gt_set;
        for (int k = 0; k < 1; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 1; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_1++;
            }
        }
        for (int k = 0; k < 10; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 10; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_10++;
            }
        }
        for (int k = 0; k < 100; k++) {
            gt_set.insert(raw_gt_vec_ID[start_addr_gt + k]);
        }
        for (int k = 0; k < 100; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hw_result_vec_ID_partial[k])) { 
                match_count_R_at_100++;
            }
        }
    }

    std::cout << "R1@1: " << float(match_count_R1_at_1) / query_num << std::endl;
    std::cout << "R1@10: " << float(match_count_R1_at_10) / query_num << std::endl;
    std::cout << "R1@100: " << float(match_count_R1_at_100) / query_num << std::endl;
    std::cout << "R@1: " << float(match_count_R_at_1) / (query_num * 1) << std::endl;
    std::cout << "R@10: " << float(match_count_R_at_10) / (query_num * 10) << std::endl;
    std::cout << "R@100: " << float(match_count_R_at_100) / (query_num * 100) << std::endl;

    return 0; 
} 
