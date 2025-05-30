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

#include "./hnswlib/hnswlib.h"
#include "utils.hpp"
// #define DEBUG

int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Init     //////////
    
    // Deep100M or Deep1000M or SIFT100M or SIFT1000M or SBERT1000M or SBERT3000M
    std::string db_name = "SBERT3000M"; 
    std::cout << "DB name: " << db_name << std::endl;
    
    // std::string index_scan = "hnsw"; // hnsw or brute-force
    // std::string index_scan = "brute_force"; // hnsw or brute-force
    // std::cout << "Index scan: " << index_scan << std::endl;

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
    }

    int nprobe_max = 128;
    

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

    //////////     Allocate Memory     //////////
    
    // on host side, used to Select Cells to Scan 
    std::vector<float, aligned_allocator<float>> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float, aligned_allocator<float>> vector_quantizer(vector_quantizer_size / sizeof(float));
    
    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    // recall counts the very first nearest neighbor only
    size_t gt_vec_ID_num = 10000 * nprobe_max;
    std::vector<uint32_t, aligned_allocator<uint32_t>> gt_vec_ID(gt_vec_ID_num, 0);
    std::vector<uint32_t, aligned_allocator<uint32_t>> hnsw_vec_ID(gt_vec_ID_num, 0);
    
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


    // HNSWlib index
    hnswlib::AlgorithmInterface<float>* alg_hnswlib_brute;
    hnswlib::AlgorithmInterface<float>* alg_hnswlib_hnsw;
    hnswlib::L2Space space(D);
    
    {
        std::string brute_force_index_dir = dir_concat(data_dir_prefix, "hnswlib_brute_force_index.bin");
        std::ifstream f_hnswlib(brute_force_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "brute_force Index exists, loading index..." << std::endl;
            alg_hnswlib_brute = new hnswlib::BruteforceSearch<float>(&space, brute_force_index_dir);
        }
        else {
            std::cout << "brute_force Index does not exist, creating new index..." << std::endl;
            alg_hnswlib_brute = new hnswlib::BruteforceSearch<float>(&space, nlist);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib_brute->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib_brute->saveIndex(brute_force_index_dir);
        }
    }

    {
        std::string hnsw_index_dir = dir_concat(data_dir_prefix, "hnswlib_hnsw_index.bin");
        std::ifstream f_hnswlib(hnsw_index_dir);
        bool hnswlib_index_exists = f_hnswlib.good();
        if (hnswlib_index_exists) {
            std::cout << "HNSW Index exists, loading index..." << std::endl;
            alg_hnswlib_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_index_dir);
        }
        else {
            std::cout << "HNSW Index does not exist, creating new index..." << std::endl;
            size_t M_hnswlib = 128;
            size_t ef_construction = 800;
            alg_hnswlib_hnsw = new hnswlib::HierarchicalNSW<float>(&space, nlist,  M_hnswlib = M_hnswlib, ef_construction = ef_construction);
            std::cout << "Adding data..." << std::endl;
            for (size_t i = 0; i < nlist; ++i) {
                alg_hnswlib_hnsw->addPoint(vector_quantizer.data() + D * i, i);
            }
            alg_hnswlib_hnsw->saveIndex(hnsw_index_dir);
        }
        ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib_hnsw)->setEf(10);
        std::cout << "ef: " << ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib_hnsw)->ef_ << std::endl;
    }   

    int search_query_num = 1000;
    std::vector<std::pair<float, uint32_t>> hw_result_pair(nprobe_max);

    for (int query_id = 0; query_id < search_query_num; query_id++) {

        std::cout << "query ID: " << query_id << std::endl;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>

        void* p = &query_vectors[0] + query_id * D;

        {
            auto gd = alg_hnswlib_brute->searchKnn(p, nprobe_max);
            assert(gd.size() == nprobe_max);
            int cnt = 0;
            while (!gd.empty()) {
                hw_result_pair[cnt] = std::make_pair(gd.top().first, gd.top().second);
                // gt_vec_ID[query_id * nprobe_max + cnt] = gd.top().second;
                gd.pop();
                cnt++;
            }
            std::sort(hw_result_pair.begin(), hw_result_pair.end());
            for (int i = 0; i < nprobe_max; i++) {
                gt_vec_ID[query_id * nprobe_max + i] = hw_result_pair[i].second;
            }
        }

    }

    //////////     Recall evaluation    //////////
    std::cout << "Comparing Results..." << std::endl;
    

    std::vector<int> nprobe_list = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int nprobe : nprobe_list) {
    
        int match_count_R_at_nprobe = 0;

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        for (int query_id = 0; query_id < search_query_num; query_id++) {


            // std::cout << "ef: " << ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib_hnsw)->ef_ << std::endl;
            // search
            {
                void* p = &query_vectors[0] + query_id * D;
                auto gd = alg_hnswlib_hnsw->searchKnn(p, nprobe);
                assert(gd.size() == nprobe);
                int cnt = 0;
                while (!gd.empty()) {
                    hnsw_vec_ID[query_id * nprobe_max + cnt] = gd.top().second;
                    gd.pop();
                    cnt++;
                }
            }
        }
        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        // calculate QPS
        double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count());
        float QPS = search_query_num / (durationUs / 1000.0 / 1000.0);

        for (int query_id = 0; query_id < search_query_num; query_id++) {

            int start_addr_gt = query_id * nprobe_max;

            // R@K
            std::unordered_set<size_t> gt_set;
            for (int k = 0; k < nprobe; k++) {
                gt_set.insert(gt_vec_ID[start_addr_gt + k]);
            }
            for (int k = 0; k < nprobe; k++) {
                // count actually means contain here...
                // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
                if (gt_set.count(hnsw_vec_ID[start_addr_gt + k])) { 
                    match_count_R_at_nprobe++;
                }
            }
        }

        std::cout << "R@" << nprobe << ": " << float(match_count_R_at_nprobe) / (search_query_num * nprobe) <<
            "   QPS: " << QPS << std::endl;
    }

    return 0; 
} 
