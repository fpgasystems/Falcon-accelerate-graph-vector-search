// See wenqi_scripts folder in hnswlib_eval repo for more basic scripts (adjusted from cpp_svae_load_index.cpp)

#include "./hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <string>
#include <fstream>
#include <chrono>

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

void test() {

    std::string db_name = "SIFT100M";
    

    int D = 128;
    size_t nlist = 32768;
    size_t nq = 10000;
    size_t topK = 32;
   
   //////

    
    std::string data_dir_prefix;
    if (db_name == "SIFT100M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF32768,PQ32";
    } else if (db_name == "SIFT1000M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT1000M_IVF32768,PQ32";
    }

    std::string query_vectors_dir_suffix("query_vectors_float32_10000_128_raw");
    std::string query_vectors_dir = dir_concat(data_dir_prefix, query_vectors_dir_suffix);
    std::ifstream query_vectors_fstream(
        query_vectors_dir, 
        std::ios::in | std::ios::binary);
    query_vectors_fstream.seekg(0, query_vectors_fstream.end);
    size_t query_vectors_size =  query_vectors_fstream.tellg();
    if (!query_vectors_size) std::cout << "query_vectors_size is 0!";
    query_vectors_fstream.seekg(0, query_vectors_fstream.beg);

    std::string vector_quantizer_dir_suffix("vector_quantizer_float32_32768_128_raw");
    std::string vector_quantizer_dir = dir_concat(data_dir_prefix, vector_quantizer_dir_suffix);
    std::ifstream vector_quantizer_fstream(
        vector_quantizer_dir, 
        std::ios::in | std::ios::binary);
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.end);
    size_t vector_quantizer_size =  vector_quantizer_fstream.tellg();
    if (!vector_quantizer_size) std::cout << "vector_quantizer_size is 0!";
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.beg);

    assert (nlist * D * sizeof(float) == vector_quantizer_size);
    assert (nq * D * sizeof(float) == query_vectors_size);

    std::vector<float> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float> vector_quantizer(vector_quantizer_size / sizeof(float));

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


    hnswlib::L2Space space(D);

    // interface:  HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
    size_t ef_construction = 128;

    // hnsw index
    std::string hnsw_index_dir = dir_concat(data_dir_prefix, "hnswlib_hnsw_index.bin");
    hnswlib::AlgorithmInterface<float>* alg_hnsw;
    if (file_exists(hnsw_index_dir)) {
        std::cout << "HNSW Index exists, loading index..." << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_index_dir);
    }
    else {
        std::cout << "HNSW Index does not exist, creating new index..." << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, nlist, ef_construction);
        std::cout << "Adding data..." << std::endl;
        for (size_t i = 0; i < nlist; ++i) {
            alg_hnsw->addPoint(vector_quantizer.data() + D * i, i);
        }
        alg_hnsw->saveIndex(hnsw_index_dir);
    }

    // brute-force index
    std::string brute_force_index_dir = dir_concat(data_dir_prefix, "hnswlib_brute_force_index.bin");
    hnswlib::AlgorithmInterface<float>* alg_brute_force;
    if (file_exists(brute_force_index_dir)) {
        std::cout << "brute_force Index exists, loading index..." << std::endl;
        alg_brute_force = new hnswlib::BruteforceSearch<float>(&space, brute_force_index_dir);
    }
    else {
        std::cout << "brute_force Index does not exist, creating new index..." << std::endl;
        alg_brute_force = new hnswlib::BruteforceSearch<float>(&space, nlist);
        std::cout << "Adding data..." << std::endl;
        for (size_t i = 0; i < nlist; ++i) {
            alg_brute_force->addPoint(vector_quantizer.data() + D * i, i);
        }
        alg_brute_force->saveIndex(brute_force_index_dir);
    }

    std::cout << "Searching for results..." << std::endl;
    std::vector<size_t> gt_ID(nq * topK);
    std::vector<size_t> hnsw_ID(nq * topK);

    std::cout << "Searching with HNSW..." << std::endl;
    auto start_hnsw = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query_vectors.data() + j * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto res = alg_hnsw->searchKnn(p, topK);
        assert(res.size() == topK);
        int cnt = 0;
        while (!res.empty()) {
            hnsw_ID[j * topK + cnt] = res.top().second;
            res.pop();
            cnt++;
        }
    }
    auto end_hnsw = std::chrono::high_resolution_clock::now();
    double duration_hnsw = (std::chrono::duration_cast<std::chrono::milliseconds>(end_hnsw - start_hnsw).count());
    std::cout << "Duration HNSW search: " << duration_hnsw << " ms" << "\tQPS:" << nq / (duration_hnsw / 1000.0) << std::endl; 

    std::cout << "Searching with brute_force..." << std::endl;
    auto start_brute_force = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query_vectors.data() + j * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto gd = alg_brute_force->searchKnn(p, topK);
        assert(gd.size() == topK);
        int cnt = 0;
        while (!gd.empty()) {
            gt_ID[j * topK + cnt] = gd.top().second;
            gd.pop();
            cnt++;
        }
    }
    auto end_brute_force = std::chrono::high_resolution_clock::now();
    double duration_brute_force = (std::chrono::duration_cast<std::chrono::milliseconds>(end_brute_force - start_brute_force).count());
    std::cout << "Duration brute_force search: " << duration_brute_force << " ms" << "\tQPS:" << nq / (duration_brute_force / 1000.0) << std::endl; 

    
    int match_cnt = 0;
    for (size_t i = 0; i < nq; i++) {
        int start_addr = i * topK;


        // https://en.cppreference.com/w/cpp/container/unordered_set
        std::unordered_set<size_t> gt_set;

        for (int k = 0; k < topK; k++) {
            gt_set.insert(gt_ID[start_addr + k]);
        }
        for (int k = 0; k < topK; k++) {
            // count actually means contain here...
            // https://stackoverflow.com/questions/42532550/why-does-stdset-not-have-a-contains-member-function
            if (gt_set.count(hnsw_ID[start_addr + k])) { 
                match_cnt++;
            }
        }
        // std::cout << "query ID: " << i << " match cnt: " << match_cnt << std::endl;
    }
    std::cout << "Recall@" << topK << ": " << float(match_cnt) / (nq * topK) << std::endl;
    
    delete alg_hnsw;
    delete alg_brute_force;
}

int main() {
    
    test();

    return 0;
}
