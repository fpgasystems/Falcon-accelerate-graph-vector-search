#include <stdint.h>
#include <chrono>
#include <cassert>

#include "host.hpp"

#include "constants.hpp"
// #include "types.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"

#include <sys/stat.h>

#define DEBUG

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

// concat dir
std::string concat_dir(std::string dir, std::string filename) {
    if (dir.back() == '/') {
        return dir + filename;
    } else {
        return dir + "/" + filename;
    }
}

int main(int argc, char** argv)
{
    std::cout << "Usage: ./host <xclbin> <max_cand_per_group (mc)> <max_group_num_in_pipe (mg)> <ef> <graph_type> <dataset> <Max degree (MD)> <batch_size>" << std::endl;
    std::cout << "   Example: ./host xclbin/vadd.hw.xclbin 1 4 64 HNSW SIFT1M 64 10000" << std::endl;

    // in init
    int d = D;
    int query_num = 10000;
    int query_offset = 0; // starting from query x
    int query_num_after_offset = query_num + query_offset > 10000? 10000 - query_offset : query_num;
    int candidate_queue_runtime_size = hardware_candidate_queue_size;
       
    int arg_cnt = 2;
    int max_cand_per_group = 1;
    if (argc > 2) { max_cand_per_group = atoi(argv[arg_cnt++]); } 
    std::cout << "max_cand_per_group=" << max_cand_per_group << std::endl;

    int max_group_num_in_pipe = 4;
    if (argc > 3) { max_group_num_in_pipe = atoi(argv[arg_cnt++]); } 
    std::cout << "max_group_num_in_pipe=" << max_group_num_in_pipe << std::endl;

    int ef = 64;
    if (argc > 4) { ef = atoi(argv[arg_cnt++]); }
    std::cout << "ef=" << ef << std::endl;
    assert(ef <= hardware_result_queue_size);

    std::string graph_type = "HNSW"; // "NSG" or "HNSW"
    if (argc > 5) { graph_type = argv[arg_cnt++]; }
    std::cout << "graph_type=" << graph_type << std::endl;
    assert (graph_type == "NSG" || graph_type == "HNSW");

    std::string dataset = "SIFT1M"; // "SIFT1M" or "SIFT10M" or "Deep1M" or "Deep10M" or "GLOVE" or "SBERT1M"
    if (argc > 6) { dataset = argv[arg_cnt++]; }
    std::cout << "dataset=" << dataset << std::endl;
    if (dataset == "SIFT1M" || dataset == "SIFT10M") {assert (d == 128);}
    else if (dataset == "Deep1M" || dataset == "Deep10M") {assert (d == 96);}
    else if (dataset == "GLOVE") {assert (d == 300);}
    else if (dataset == "SBERT1M") {assert (d == 384);}
    else if (dataset == "SPACEV1M" || dataset == "SPACEV10M") {assert (d == 100);}
    else {std::cout << "Unknown dataset\n"; return -1;}

    int MD = 64;
    if (argc > 7) { MD = atoi(argv[arg_cnt++]); }
    std::cout << "MD (max degree)=" << MD << std::endl;

    int query_batch_size = 10000;
    if (argc > 8) { query_batch_size = atoi(argv[arg_cnt++]); }
    std::cout << "query_batch_size=" << query_batch_size << std::endl;
    assert (query_batch_size <= query_num);

    int max_bloom_out_burst_size = 16; // according to mem & compute speed test
#if N_CHANNEL == 1
    int runtime_n_bucket_addr_bits = 8 + 10; // 256K buckets
#endif
#if N_CHANNEL == 2
    int runtime_n_bucket_addr_bits = 7 + 10; 
#endif
#if N_CHANNEL == 3
    int runtime_n_bucket_addr_bits = 7 + 10; 
#endif
#if N_CHANNEL == 4
    int runtime_n_bucket_addr_bits = 6 + 10;
#endif
    int runtime_n_buckets = 1 << runtime_n_bucket_addr_bits;
    uint32_t hash_seed = 1;
    assert (ef <= hardware_result_queue_size);

    std::string index_dir;

    if (graph_type == "HNSW") {
        if (dataset == "SIFT1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/SIFT1M_MD" + std::to_string(MD);
        } else if (dataset == "SIFT10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/SIFT10M_MD" + std::to_string(MD);
        } else if (dataset == "Deep1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/Deep1M_MD" + std::to_string(MD);
        } else if (dataset == "Deep10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/Deep10M_MD" + std::to_string(MD);
        } else if (dataset == "GLOVE") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/GLOVE_MD" + std::to_string(MD);
        } else if (dataset == "SBERT1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/SBERT1M_MD" + std::to_string(MD);
        } else if (dataset == "SPACEV1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/SPACEV1M_MD" + std::to_string(MD);
        } else if (dataset == "SPACEV10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_hnsw/SPACEV10M_MD" + std::to_string(MD);
        } else {
            std::cout << "Unknown dataset\n";
            return -1;
        }
    } else if (graph_type == "NSG") {
        if (dataset == "SIFT1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SIFT1M_MD" + std::to_string(MD);
        } else if (dataset == "SIFT10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SIFT10M_MD" + std::to_string(MD);
        } else if (dataset == "Deep1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/Deep1M_MD" + std::to_string(MD);
        } else if (dataset == "Deep10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/Deep10M_MD" + std::to_string(MD);
        } else if (dataset == "GLOVE") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/GLOVE_MD" + std::to_string(MD);
        } else if (dataset == "SBERT1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SBERT1M_MD" + std::to_string(MD);
        } else if (dataset == "SPACEV1M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SPACEV1M_MD" + std::to_string(MD);
        } else if (dataset == "SPACEV10M") {
            index_dir = "/mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG/SPACEV10M_MD" + std::to_string(MD);
        }  else {
            std::cout << "Unknown dataset\n";
            return -1;
        }
    } else {
        std::cout << "Unknown graph type\n";
        return -1; 
    }

    std::string dataset_dir;
    std::string fname_query_vectors;
    std::string fname_gt_vec_ID;
    std::string fname_gt_dist;
    if (dataset == "SIFT1M" || dataset == "SIFT10M") {
        dataset_dir = "/mnt/scratch/wenqi/Faiss_experiments/bigann";
        fname_query_vectors = concat_dir(dataset_dir, "bigann_query.bvecs");
        if (dataset == "SIFT1M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gnd/idx_1M.ivecs");
            fname_gt_dist = concat_dir(dataset_dir, "gnd/dis_1M.fvecs");
        } else if (dataset == "SIFT10M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gnd/idx_10M.ivecs");
            fname_gt_dist = concat_dir(dataset_dir, "gnd/dis_10M.fvecs");
        }
    } else if (dataset == "Deep1M" || dataset == "Deep10M") {
        dataset_dir = "/mnt/scratch/wenqi/Faiss_experiments/deep1b";
        fname_query_vectors = concat_dir(dataset_dir, "query.public.10K.fbin");
        if (dataset == "Deep1M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_1M.ibin");
            fname_gt_dist = concat_dir(dataset_dir, "gt_dis_1M.fbin");
        } else if (dataset == "Deep10M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_10M.ibin");
            fname_gt_dist = concat_dir(dataset_dir, "gt_dis_10M.fbin");
        }
    } else if (dataset == "GLOVE") {
        dataset_dir = "/mnt/scratch/wenqi/Faiss_experiments/GLOVE_840B_300d";
        fname_query_vectors = concat_dir(dataset_dir, "query_10K.fbin");
        fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_1M.ibin");
        fname_gt_dist = concat_dir(dataset_dir, "gt_dis_1M.fbin");
    } else if (dataset == "SBERT1M") {
        dataset_dir = "/mnt/scratch/wenqi/Faiss_experiments/sbert";
        fname_query_vectors = concat_dir(dataset_dir, "query_10K.fvecs");
        fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_1M.ibin");
        fname_gt_dist = concat_dir(dataset_dir, "gt_dis_1M.fbin");
    } else if (dataset == "SPACEV1M" || dataset == "SPACEV10M") {
        dataset_dir = "/mnt/scratch/wenqi/Faiss_experiments/SPACEV";
        fname_query_vectors = concat_dir(dataset_dir, "query_10K.bin");
        if (dataset == "SPACEV1M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_1M.ibin");
            fname_gt_dist = concat_dir(dataset_dir, "gt_dis_1M.fbin");
        } else if (dataset == "SPACEV10M") {
            fname_gt_vec_ID = concat_dir(dataset_dir, "gt_idx_10M.ibin");
            fname_gt_dist = concat_dir(dataset_dir, "gt_dis_10M.fbin");
        }
    }

    // initialization values
    int max_level; // = 16;
    int max_link_num_upper; // = 32;
    int max_link_num_base; // = 32;
    int entry_point_id; // = 0;
    int num_db_vec; // = 1000 * 1000;

    // load metadata from file 
    //  cur_element_count, maxlevel_, enterpoint_node_, maxM_, maxM0_
    FILE* f_metadata = fopen(concat_dir(index_dir, "meta.bin").c_str(), "rb");
    if (graph_type == "HNSW") {
        fread(&num_db_vec, sizeof(int), 1, f_metadata);
        fread(&max_level, sizeof(int), 1, f_metadata);
        fread(&entry_point_id, sizeof(int), 1, f_metadata);
        fread(&max_link_num_upper, sizeof(int), 1, f_metadata);
        fread(&max_link_num_base, sizeof(int), 1, f_metadata);
        fclose(f_metadata);
        std::cout << "num_db_vec=" << num_db_vec << std::endl;
        std::cout << "max_level=" << max_level << std::endl;
        std::cout << "entry_point_id=" << entry_point_id << std::endl;
        std::cout << "max_link_num_upper=" << max_link_num_upper << std::endl;
        std::cout << "max_link_num_base=" << max_link_num_base << std::endl;
    } else if (graph_type == "NSG") {
        fread(&num_db_vec, sizeof(int), 1, f_metadata);
        fread(&entry_point_id, sizeof(int), 1, f_metadata);
        fread(&max_link_num_base, sizeof(int), 1, f_metadata);
        fclose(f_metadata);
        std::cout << "num_db_vec=" << num_db_vec << std::endl;
        std::cout << "entry_point_id=" << entry_point_id << std::endl;
        std::cout << "max_link_num_base=" << max_link_num_base << std::endl;
    }
    
    size_t bytes_per_vec = d * sizeof(float);
    size_t bytes_per_db_vec_plus_padding = d % 16 == 0? d * sizeof(float) : (d + 16 - d % 16) * sizeof(float);
    assert (bytes_per_db_vec_plus_padding % 64 == 0);
    int d_after_padding = bytes_per_db_vec_plus_padding / sizeof(float);
    size_t bytes_entry_vector = bytes_per_db_vec_plus_padding;
    size_t bytes_entry_point_ids = query_num * sizeof(int);
    size_t bytes_query_vectors = query_num * bytes_per_db_vec_plus_padding;
    size_t bytes_out_id = query_num * ef * sizeof(int);
    size_t bytes_out_dist = query_num * ef * sizeof(float);	
    size_t bytes_mem_debug = query_num * 5 * sizeof(int);

// here, for inter-query parallel, replicate rather than using different content
#if N_CHANNEL == 1
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#elif N_CHANNEL == 2
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_1 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#elif N_CHANNEL == 3
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_1 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_2 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#elif N_CHANNEL == 4
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_1 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_2 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_3 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#elif N_CHANNEL == 8
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_1 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_2 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_3 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_4 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_5 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_6 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_7 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#elif N_CHANNEL == 16
    std::string fname_ground_links_chan_0 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_1 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_2 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_3 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_4 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_5 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_6 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_7 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_8 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_9 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_10 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_11 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_12 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_13 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_14 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
    std::string fname_ground_links_chan_15 = concat_dir(index_dir, "ground_links_1_chan_0.bin");
#endif

    std::string fname_ground_labels = concat_dir(index_dir, "ground_labels.bin");

#if N_CHANNEL == 1
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#elif N_CHANNEL == 2
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_1 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#elif N_CHANNEL == 3
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_1 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_2 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#elif N_CHANNEL == 4
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_1 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_2 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_3 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#elif N_CHANNEL == 8
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_1 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_2 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_3 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_4 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_5 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_6 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_7 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#elif N_CHANNEL == 16
    std::string fname_ground_vectors_chan_0 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_1 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_2 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_3 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_4 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_5 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_6 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_7 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_8 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_9 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_10 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_11 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_12 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_13 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_14 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
    std::string fname_ground_vectors_chan_15 = concat_dir(index_dir, "ground_vectors_1_chan_0.bin");
#endif

    FILE* f_ground_links_chan_0 = fopen(fname_ground_links_chan_0.c_str(), "rb");
#if N_CHANNEL >= 2
    FILE* f_ground_links_chan_1 = fopen(fname_ground_links_chan_1.c_str(), "rb");
#endif
#if N_CHANNEL >= 3
    FILE* f_ground_links_chan_2 = fopen(fname_ground_links_chan_2.c_str(), "rb");
#endif
#if N_CHANNEL >= 4
    FILE* f_ground_links_chan_3 = fopen(fname_ground_links_chan_3.c_str(), "rb");
#endif
#if N_CHANNEL >= 8
    FILE* f_ground_links_chan_4 = fopen(fname_ground_links_chan_4.c_str(), "rb");
    FILE* f_ground_links_chan_5 = fopen(fname_ground_links_chan_5.c_str(), "rb");
    FILE* f_ground_links_chan_6 = fopen(fname_ground_links_chan_6.c_str(), "rb");
    FILE* f_ground_links_chan_7 = fopen(fname_ground_links_chan_7.c_str(), "rb");
#endif
#if N_CHANNEL >= 16
    FILE* f_ground_links_chan_8 = fopen(fname_ground_links_chan_8.c_str(), "rb");
    FILE* f_ground_links_chan_9 = fopen(fname_ground_links_chan_9.c_str(), "rb");
    FILE* f_ground_links_chan_10 = fopen(fname_ground_links_chan_10.c_str(), "rb");
    FILE* f_ground_links_chan_11 = fopen(fname_ground_links_chan_11.c_str(), "rb");
    FILE* f_ground_links_chan_12 = fopen(fname_ground_links_chan_12.c_str(), "rb");
    FILE* f_ground_links_chan_13 = fopen(fname_ground_links_chan_13.c_str(), "rb");
    FILE* f_ground_links_chan_14 = fopen(fname_ground_links_chan_14.c_str(), "rb");
    FILE* f_ground_links_chan_15 = fopen(fname_ground_links_chan_15.c_str(), "rb");
#endif

    FILE* f_ground_labels;
    if (graph_type == "HNSW") {
        f_ground_labels = fopen(fname_ground_labels.c_str(), "rb");
    }

    FILE* f_ground_vectors_chan_0 = fopen(fname_ground_vectors_chan_0.c_str(), "rb");
#if N_CHANNEL >= 2
    FILE* f_ground_vectors_chan_1 = fopen(fname_ground_vectors_chan_1.c_str(), "rb");
#endif
#if N_CHANNEL >= 3
    FILE* f_ground_vectors_chan_2 = fopen(fname_ground_vectors_chan_2.c_str(), "rb");
#endif
#if N_CHANNEL >= 4
    FILE* f_ground_vectors_chan_3 = fopen(fname_ground_vectors_chan_3.c_str(), "rb");
#endif
#if N_CHANNEL >= 8
    FILE* f_ground_vectors_chan_4 = fopen(fname_ground_vectors_chan_4.c_str(), "rb");
    FILE* f_ground_vectors_chan_5 = fopen(fname_ground_vectors_chan_5.c_str(), "rb");
    FILE* f_ground_vectors_chan_6 = fopen(fname_ground_vectors_chan_6.c_str(), "rb");
    FILE* f_ground_vectors_chan_7 = fopen(fname_ground_vectors_chan_7.c_str(), "rb");
#endif
#if N_CHANNEL >= 16
    FILE* f_ground_vectors_chan_8 = fopen(fname_ground_vectors_chan_8.c_str(), "rb");
    FILE* f_ground_vectors_chan_9 = fopen(fname_ground_vectors_chan_9.c_str(), "rb");
    FILE* f_ground_vectors_chan_10 = fopen(fname_ground_vectors_chan_10.c_str(), "rb");
    FILE* f_ground_vectors_chan_11 = fopen(fname_ground_vectors_chan_11.c_str(), "rb");
    FILE* f_ground_vectors_chan_12 = fopen(fname_ground_vectors_chan_12.c_str(), "rb");
    FILE* f_ground_vectors_chan_13 = fopen(fname_ground_vectors_chan_13.c_str(), "rb");
    FILE* f_ground_vectors_chan_14 = fopen(fname_ground_vectors_chan_14.c_str(), "rb");
    FILE* f_ground_vectors_chan_15 = fopen(fname_ground_vectors_chan_15.c_str(), "rb");
#endif

FILE* f_query_vectors = fopen(fname_query_vectors.c_str(), "rb");
FILE* f_gt_vec_ID = fopen(fname_gt_vec_ID.c_str(), "rb");
FILE* f_gt_dist = fopen(fname_gt_dist.c_str(), "rb");

// get file size
size_t bytes_db_vectors_chan_0 = GetFileSize(fname_ground_vectors_chan_0);
#if N_CHANNEL >= 2
    size_t bytes_db_vectors_chan_1 = GetFileSize(fname_ground_vectors_chan_1);
#endif
#if N_CHANNEL >= 3
    size_t bytes_db_vectors_chan_2 = GetFileSize(fname_ground_vectors_chan_2);
#endif
#if N_CHANNEL >= 4
    size_t bytes_db_vectors_chan_3 = GetFileSize(fname_ground_vectors_chan_3);
#endif
#if N_CHANNEL >= 8
    size_t bytes_db_vectors_chan_4 = GetFileSize(fname_ground_vectors_chan_4);
    size_t bytes_db_vectors_chan_5 = GetFileSize(fname_ground_vectors_chan_5);
    size_t bytes_db_vectors_chan_6 = GetFileSize(fname_ground_vectors_chan_6);
    size_t bytes_db_vectors_chan_7 = GetFileSize(fname_ground_vectors_chan_7);
#endif
#if N_CHANNEL >= 16
    size_t bytes_db_vectors_chan_8 = GetFileSize(fname_ground_vectors_chan_8);
    size_t bytes_db_vectors_chan_9 = GetFileSize(fname_ground_vectors_chan_9);
    size_t bytes_db_vectors_chan_10 = GetFileSize(fname_ground_vectors_chan_10);
    size_t bytes_db_vectors_chan_11 = GetFileSize(fname_ground_vectors_chan_11);
    size_t bytes_db_vectors_chan_12 = GetFileSize(fname_ground_vectors_chan_12);
    size_t bytes_db_vectors_chan_13 = GetFileSize(fname_ground_vectors_chan_13);
    size_t bytes_db_vectors_chan_14 = GetFileSize(fname_ground_vectors_chan_14);
    size_t bytes_db_vectors_chan_15 = GetFileSize(fname_ground_vectors_chan_15);
#endif
    
    size_t bytes_links_base_chan_0 = GetFileSize(fname_ground_links_chan_0);
#if N_CHANNEL >= 2
    size_t bytes_links_base_chan_1 = GetFileSize(fname_ground_links_chan_1);
#endif
#if N_CHANNEL >= 3
    size_t bytes_links_base_chan_2 = GetFileSize(fname_ground_links_chan_2);
#endif	   
#if N_CHANNEL >= 4
    size_t bytes_links_base_chan_3 = GetFileSize(fname_ground_links_chan_3);
#endif
#if N_CHANNEL >= 8
    size_t bytes_links_base_chan_4 = GetFileSize(fname_ground_links_chan_4);
    size_t bytes_links_base_chan_5 = GetFileSize(fname_ground_links_chan_5);
    size_t bytes_links_base_chan_6 = GetFileSize(fname_ground_links_chan_6);
    size_t bytes_links_base_chan_7 = GetFileSize(fname_ground_links_chan_7);
#endif
#if N_CHANNEL >= 16
    size_t bytes_links_base_chan_8 = GetFileSize(fname_ground_links_chan_8);
    size_t bytes_links_base_chan_9 = GetFileSize(fname_ground_links_chan_9);
    size_t bytes_links_base_chan_10 = GetFileSize(fname_ground_links_chan_10);
    size_t bytes_links_base_chan_11 = GetFileSize(fname_ground_links_chan_11);
    size_t bytes_links_base_chan_12 = GetFileSize(fname_ground_links_chan_12);
    size_t bytes_links_base_chan_13 = GetFileSize(fname_ground_links_chan_13);
    size_t bytes_links_base_chan_14 = GetFileSize(fname_ground_links_chan_14);
    size_t bytes_links_base_chan_15 = GetFileSize(fname_ground_links_chan_15);
#endif
    size_t bytes_labels_base;
    if (graph_type == "HNSW") {
        bytes_labels_base = GetFileSize(fname_ground_labels); // int = 4 bytes
    }
    size_t raw_query_vectors_size = GetFileSize(fname_query_vectors);
    size_t raw_gt_vec_ID_size = GetFileSize(fname_gt_vec_ID);
    size_t raw_gt_dist_size = GetFileSize(fname_gt_dist);
    std::cout << "bytes_db_vectors_chan_0=" << bytes_db_vectors_chan_0 << std::endl;
    std::cout << "bytes_links_base_chan_0=" << bytes_links_base_chan_0 << std::endl;
    std::cout << "raw_query_vectors_size=" << raw_query_vectors_size << std::endl;

    // input vecs
    std::vector<int, aligned_allocator<int>> entry_point_ids(bytes_entry_point_ids / sizeof(int));
    std::vector<float, aligned_allocator<float>> query_vectors(bytes_query_vectors / sizeof(float));

    // db vec
    std::vector<float, aligned_allocator<float>> db_vectors_chan_0(bytes_db_vectors_chan_0 / sizeof(float));
#if N_CHANNEL >= 2
    std::vector<float, aligned_allocator<float>> db_vectors_chan_1(bytes_db_vectors_chan_1 / sizeof(float));
#endif
#if N_CHANNEL >= 3
    std::vector<float, aligned_allocator<float>> db_vectors_chan_2(bytes_db_vectors_chan_2 / sizeof(float));
#endif
#if N_CHANNEL >= 4
    std::vector<float, aligned_allocator<float>> db_vectors_chan_3(bytes_db_vectors_chan_3 / sizeof(float));
#endif
#if N_CHANNEL >= 8
    std::vector<float, aligned_allocator<float>> db_vectors_chan_4(bytes_db_vectors_chan_4 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_5(bytes_db_vectors_chan_5 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_6(bytes_db_vectors_chan_6 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_7(bytes_db_vectors_chan_7 / sizeof(float));
#endif
#if N_CHANNEL >= 16
    std::vector<float, aligned_allocator<float>> db_vectors_chan_8(bytes_db_vectors_chan_8 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_9(bytes_db_vectors_chan_9 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_10(bytes_db_vectors_chan_10 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_11(bytes_db_vectors_chan_11 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_12(bytes_db_vectors_chan_12 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_13(bytes_db_vectors_chan_13 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_14(bytes_db_vectors_chan_14 / sizeof(float));
    std::vector<float, aligned_allocator<float>> db_vectors_chan_15(bytes_db_vectors_chan_15 / sizeof(float));
#endif
    
    // links
    std::vector<int, aligned_allocator<int>> links_base_chan_0(bytes_links_base_chan_0 / sizeof(int));
#if N_CHANNEL >= 2
    std::vector<int, aligned_allocator<int>> links_base_chan_1(bytes_links_base_chan_1 / sizeof(int));
#endif
#if N_CHANNEL >= 3
    std::vector<int, aligned_allocator<int>> links_base_chan_2(bytes_links_base_chan_2 / sizeof(int));
#endif
#if N_CHANNEL >= 4
    std::vector<int, aligned_allocator<int>> links_base_chan_3(bytes_links_base_chan_3 / sizeof(int));
#endif
#if N_CHANNEL >= 8
    std::vector<int, aligned_allocator<int>> links_base_chan_4(bytes_links_base_chan_4 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_5(bytes_links_base_chan_5 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_6(bytes_links_base_chan_6 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_7(bytes_links_base_chan_7 / sizeof(int));
#endif
#if N_CHANNEL >= 16
    std::vector<int, aligned_allocator<int>> links_base_chan_8(bytes_links_base_chan_8 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_9(bytes_links_base_chan_9 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_10(bytes_links_base_chan_10 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_11(bytes_links_base_chan_11 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_12(bytes_links_base_chan_12 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_13(bytes_links_base_chan_13 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_14(bytes_links_base_chan_14 / sizeof(int));
    std::vector<int, aligned_allocator<int>> links_base_chan_15(bytes_links_base_chan_15 / sizeof(int));
#endif
    
    // output
    std::vector<int, aligned_allocator<int>> out_id(bytes_out_id / sizeof(int));
    std::vector<float, aligned_allocator<float>> out_dist(bytes_out_dist / sizeof(float));
    std::vector<int, aligned_allocator<int>> mem_debug(bytes_mem_debug / sizeof(int));

    // intermediate buffer for queries, and ground truth
    std::vector<int> labels_base;
    if (graph_type == "HNSW") {
        labels_base.resize(bytes_labels_base / sizeof(int));
    }
    // init query vectors as zeros (there will be paddings in some cases for unusual d)
    std::vector<char> raw_query_vectors(raw_query_vectors_size / sizeof(char));
    memset(raw_query_vectors.data(), 0, raw_query_vectors_size);
    std::vector<int> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(int));
    std::vector<float> raw_gt_dist(raw_gt_dist_size / sizeof(float));

    int max_topK = 100; // cutting ground truth to with only 100 top queries
    std::vector<int> gt_vec_ID(query_num * max_topK);
    std::vector<float> gt_dist(query_num * max_topK);

    // read data from file
    std::cout << "Reading database vectors from file...\n";
    fread(db_vectors_chan_0.data(), 1, bytes_db_vectors_chan_0, f_ground_vectors_chan_0);
    fclose(f_ground_vectors_chan_0);
#if N_CHANNEL >= 2
    fread(db_vectors_chan_1.data(), 1, bytes_db_vectors_chan_1, f_ground_vectors_chan_1);
    fclose(f_ground_vectors_chan_1);
#endif
#if N_CHANNEL >= 3
    fread(db_vectors_chan_2.data(), 1, bytes_db_vectors_chan_2, f_ground_vectors_chan_2);
    fclose(f_ground_vectors_chan_2);
#endif
#if N_CHANNEL >= 4
    fread(db_vectors_chan_3.data(), 1, bytes_db_vectors_chan_3, f_ground_vectors_chan_3);
    fclose(f_ground_vectors_chan_3);
#endif
#if N_CHANNEL >= 8
    fread(db_vectors_chan_4.data(), 1, bytes_db_vectors_chan_4, f_ground_vectors_chan_4);
    fclose(f_ground_vectors_chan_4);
    fread(db_vectors_chan_5.data(), 1, bytes_db_vectors_chan_5, f_ground_vectors_chan_5);
    fclose(f_ground_vectors_chan_5);
    fread(db_vectors_chan_6.data(), 1, bytes_db_vectors_chan_6, f_ground_vectors_chan_6);
    fclose(f_ground_vectors_chan_6);
    fread(db_vectors_chan_7.data(), 1, bytes_db_vectors_chan_7, f_ground_vectors_chan_7);
    fclose(f_ground_vectors_chan_7);
#endif
#if N_CHANNEL >= 16
    fread(db_vectors_chan_8.data(), 1, bytes_db_vectors_chan_8, f_ground_vectors_chan_8);
    fclose(f_ground_vectors_chan_8);
    fread(db_vectors_chan_9.data(), 1, bytes_db_vectors_chan_9, f_ground_vectors_chan_9);
    fclose(f_ground_vectors_chan_9);
    fread(db_vectors_chan_10.data(), 1, bytes_db_vectors_chan_10, f_ground_vectors_chan_10);
    fclose(f_ground_vectors_chan_10);
    fread(db_vectors_chan_11.data(), 1, bytes_db_vectors_chan_11, f_ground_vectors_chan_11);
    fclose(f_ground_vectors_chan_11);
    fread(db_vectors_chan_12.data(), 1, bytes_db_vectors_chan_12, f_ground_vectors_chan_12);
    fclose(f_ground_vectors_chan_12);
    fread(db_vectors_chan_13.data(), 1, bytes_db_vectors_chan_13, f_ground_vectors_chan_13);
    fclose(f_ground_vectors_chan_13);
    fread(db_vectors_chan_14.data(), 1, bytes_db_vectors_chan_14, f_ground_vectors_chan_14);
    fclose(f_ground_vectors_chan_14);
    fread(db_vectors_chan_15.data(), 1, bytes_db_vectors_chan_15, f_ground_vectors_chan_15);
    fclose(f_ground_vectors_chan_15);
#endif


    std::cout << "Reading base links from file...\n";
    fread(links_base_chan_0.data(), 1, bytes_links_base_chan_0, f_ground_links_chan_0);
    fclose(f_ground_links_chan_0);
#if N_CHANNEL >= 2
    fread(links_base_chan_1.data(), 1, bytes_links_base_chan_1, f_ground_links_chan_1);
    fclose(f_ground_links_chan_1);
#endif
#if N_CHANNEL >= 3
    fread(links_base_chan_2.data(), 1, bytes_links_base_chan_2, f_ground_links_chan_2);
    fclose(f_ground_links_chan_2);
#endif
#if N_CHANNEL >= 4
    fread(links_base_chan_3.data(), 1, bytes_links_base_chan_3, f_ground_links_chan_3);
    fclose(f_ground_links_chan_3);
#endif
#if N_CHANNEL >= 8
    fread(links_base_chan_4.data(), 1, bytes_links_base_chan_4, f_ground_links_chan_4);
    fclose(f_ground_links_chan_4);
    fread(links_base_chan_5.data(), 1, bytes_links_base_chan_5, f_ground_links_chan_5);
    fclose(f_ground_links_chan_5);
    fread(links_base_chan_6.data(), 1, bytes_links_base_chan_6, f_ground_links_chan_6);
    fclose(f_ground_links_chan_6);
    fread(links_base_chan_7.data(), 1, bytes_links_base_chan_7, f_ground_links_chan_7);
    fclose(f_ground_links_chan_7);
#endif
#if N_CHANNEL >= 16
    fread(links_base_chan_8.data(), 1, bytes_links_base_chan_8, f_ground_links_chan_8);
    fclose(f_ground_links_chan_8);
    fread(links_base_chan_9.data(), 1, bytes_links_base_chan_9, f_ground_links_chan_9);
    fclose(f_ground_links_chan_9);
    fread(links_base_chan_10.data(), 1, bytes_links_base_chan_10, f_ground_links_chan_10);
    fclose(f_ground_links_chan_10);
    fread(links_base_chan_11.data(), 1, bytes_links_base_chan_11, f_ground_links_chan_11);
    fclose(f_ground_links_chan_11);
    fread(links_base_chan_12.data(), 1, bytes_links_base_chan_12, f_ground_links_chan_12);
    fclose(f_ground_links_chan_12);
    fread(links_base_chan_13.data(), 1, bytes_links_base_chan_13, f_ground_links_chan_13);
    fclose(f_ground_links_chan_13);
    fread(links_base_chan_14.data(), 1, bytes_links_base_chan_14, f_ground_links_chan_14);
    fclose(f_ground_links_chan_14);
    fread(links_base_chan_15.data(), 1, bytes_links_base_chan_15, f_ground_links_chan_15);
    fclose(f_ground_links_chan_15);
#endif

    std::cout << "Reading queries and ground truths from file...\n";
    if (graph_type == "HNSW") {
        fread(labels_base.data(), 1, bytes_labels_base, f_ground_labels);
        fclose(f_ground_labels);
    }
    fread(raw_query_vectors.data(), 1, raw_query_vectors_size, f_query_vectors);
    fclose(f_query_vectors);
    fread(raw_gt_vec_ID.data(), 1, raw_gt_vec_ID_size, f_gt_vec_ID);
    fclose(f_gt_vec_ID);
    fread(raw_gt_dist.data(), 1, raw_gt_dist_size, f_gt_dist);
    fclose(f_gt_dist);
    
    if (dataset == "SIFT1M" || dataset == "SIFT10M") {
        size_t len_per_query = 4 + d;
        // conversion from uint8 to float
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < d; i++) {
                query_vectors[qid * d + i] = (float) raw_query_vectors[(qid + query_offset) * len_per_query + 4 + i];
            }
        }
        // ground truth = 4-byte ID + 1000 * 4-byte ID + 1000 or 4-byte distances
        size_t len_per_gt = 1001;
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < max_topK; i++) {
                gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[(qid + query_offset) * len_per_gt + 1 + i];
                gt_dist[qid * max_topK + i] = raw_gt_dist[(qid + query_offset) * len_per_gt + 1 + i];
            }
        }
    } else if (dataset == "Deep1M" || dataset == "Deep10M" || dataset == "GLOVE") {
        // queries: fbin, ground truth: ibin, first 8 bytes are num vec & dim
        size_t len_per_query = d * sizeof(float);
        size_t offset_bytes = 8; // first 8 bytes are num vec & dim
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            memcpy(&query_vectors[qid * d], &raw_query_vectors[(qid + query_offset) * len_per_query + offset_bytes], len_per_query);
        }
        size_t len_per_gt = 1000;
        size_t offset = 2; // first 8 bytes are num vec & dim
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < max_topK; i++) {
                gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[(qid + query_offset) * len_per_gt + offset + i];
                gt_dist[qid * max_topK + i] = raw_gt_dist[(qid + query_offset) * len_per_gt + offset + i];
            }
        }
    } else if (dataset == "SPACEV1M" || dataset == "SPACEV10M") {
        // queries: fbin, ground truth: ibin, first 8 bytes are num vec & dim
        size_t len_per_query = d * sizeof(char);
        size_t offset_bytes = 8; // first 8 bytes are num vec & dim

        // raw query vectors are in int8 format, need to convert to flow format for query_vectors
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < d; i++) {
                // first load as unsigned int, thus reducing offset of 128 between int8 and uint8
                float val = (float) raw_query_vectors[(qid + query_offset) * len_per_query + offset_bytes + i];
                query_vectors[qid * d_after_padding + i] = val;
            }
        }
        // print out first query (include padding):
        // std::cout << "First query: ";
        // for (int i = 0; i < d_after_padding; i++) {
        // 	std::cout << query_vectors[i] << " ";
        // }
        size_t len_per_gt = 1000;
        size_t offset = 2; // first 8 bytes are num vec & dim
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < max_topK; i++) {
                gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[(qid + query_offset) * len_per_gt + offset + i];
                gt_dist[qid * max_topK + i] = raw_gt_dist[(qid + query_offset) * len_per_gt + offset + i];
            }
        }
    } else if (dataset == "SBERT1M") {
        // queries: raw bin, ground truth: ibin, first 8 bytes are num vec & dim
        size_t len_per_query = d * sizeof(float);
        size_t offset_bytes = 0; // raw bin
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            memcpy(&query_vectors[qid * d], &raw_query_vectors[(qid + query_offset) * len_per_query + offset_bytes], len_per_query);
        }
        size_t len_per_gt = 1000;
        size_t offset = 2; // first 8 bytes are num vec & dim
        for (int qid = 0; qid < query_num_after_offset; qid++) {
            for (int i = 0; i < max_topK; i++) {
                gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[(qid + query_offset) * len_per_gt + offset + i];
                gt_dist[qid * max_topK + i] = raw_gt_dist[(qid + query_offset) * len_per_gt + offset + i];
            }
        }
    } else {
        std::cout << "Unsupported dataset\n";
        exit(1);
    }

    for (int qid = 0; qid < query_num_after_offset; qid++) {
        entry_point_ids[qid] = entry_point_id;
    }

// OPENCL HOST CODE AREA START

    cl_int err;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

    std::vector<cl::Device> devices = get_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    //Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device);

    // Import XCLBIN
    xclbin_file_name = argv[1];
    cl::Program::Binaries vadd_bins = import_binary_file();

    // Program and Kernel
    devices.resize(1);
    cl::Program program(context, devices, vadd_bins);
    cl::Kernel krnl_vector_add(program, "vadd");

    std::cout << "Finish loading bitstream...\n";
    // in 
    OCL_CHECK(err, cl::Buffer buffer_entry_point_ids (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            bytes_entry_point_ids, entry_point_ids.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_query_vectors (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            bytes_query_vectors, query_vectors.data(), &err));
            
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_0 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_0, links_base_chan_0.data(), &err));

#if N_CHANNEL >= 2
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_1 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_1, links_base_chan_1.data(), &err));
#endif

#if N_CHANNEL >= 3
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_2 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_2, links_base_chan_2.data(), &err));
#endif
#if N_CHANNEL >= 4
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_3 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_3, links_base_chan_3.data(), &err));
#endif

#if N_CHANNEL >= 8
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_4 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_4, links_base_chan_4.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_5 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_5, links_base_chan_5.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_6 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_6, links_base_chan_6.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_7 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_7, links_base_chan_7.data(), &err));
#endif

#if N_CHANNEL >= 16
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_8 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_8, links_base_chan_8.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_9 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_9, links_base_chan_9.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_10 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_10, links_base_chan_10.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_11 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_11, links_base_chan_11.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_12 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_12, links_base_chan_12.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_13 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_13, links_base_chan_13.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_14 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_14, links_base_chan_14.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_links_base_chan_15 (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        bytes_links_base_chan_15, links_base_chan_15.data(), &err));
#endif

    // in & out (db vec is mixed with visited list)
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_0 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_0, db_vectors_chan_0.data(), &err));
#if N_CHANNEL >= 2
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_1 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_1, db_vectors_chan_1.data(), &err));
#endif
#if N_CHANNEL >= 3
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_2 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_2, db_vectors_chan_2.data(), &err));
#endif
#if N_CHANNEL >= 4
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_3 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_3, db_vectors_chan_3.data(), &err));
#endif
#if N_CHANNEL >= 8
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_4 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_4, db_vectors_chan_4.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_5 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_5, db_vectors_chan_5.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_6 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_6, db_vectors_chan_6.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_7 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_7, db_vectors_chan_7.data(), &err));
#endif
#if N_CHANNEL >= 16
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_8 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_8, db_vectors_chan_8.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_9 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_9, db_vectors_chan_9.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_10 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_10, db_vectors_chan_10.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_11 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_11, db_vectors_chan_11.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_12 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_12, db_vectors_chan_12.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_13 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_13, db_vectors_chan_13.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_14 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_14, db_vectors_chan_14.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_db_vectors_chan_15 (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors_chan_15, db_vectors_chan_15.data(), &err));
#endif

    // out
    OCL_CHECK(err, cl::Buffer buffer_out_id (context,CL_MEM_USE_HOST_PTR,// | CL_MEM_WRITE_ONLY,
            bytes_out_id, out_id.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_out_dist (context,CL_MEM_USE_HOST_PTR,// | CL_MEM_WRITE_ONLY,
            bytes_out_dist, out_dist.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mem_debug (context,CL_MEM_USE_HOST_PTR,// | CL_MEM_WRITE_ONLY,
            bytes_mem_debug, mem_debug.data(), &err));

    std::cout << "Finish allocate buffer...\n";

    int arg_counter = 0;    
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_batch_size)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(ef)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(candidate_queue_runtime_size)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_cand_per_group)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_group_num_in_pipe)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(runtime_n_bucket_addr_bits)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(hash_seed)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_bloom_out_burst_size)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(d)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_link_num_base)));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_entry_point_ids));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_query_vectors));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_0));
#if N_CHANNEL >= 2
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_1));
#endif
#if N_CHANNEL >= 3
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_2));
#endif
#if N_CHANNEL >= 4
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_3));
#endif
#if N_CHANNEL >= 8
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_4));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_5));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_6));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_7));
#endif
#if N_CHANNEL >= 16
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_8));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_9));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_10));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_11));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_12));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_13));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_14));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors_chan_15));
#endif


    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_0));
#if N_CHANNEL >= 2
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_1));
#endif
#if N_CHANNEL >= 3
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_2));
#endif
#if N_CHANNEL >= 4
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_3));
#endif
#if N_CHANNEL >= 8
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_4));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_5));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_6));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_7));
#endif
#if N_CHANNEL >= 16
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_8));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_9));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_10));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_11));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_12));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_13));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_14));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base_chan_15));
#endif

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_id));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_dist));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_mem_debug));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
        buffer_entry_point_ids,
        buffer_query_vectors,
        buffer_links_base_chan_0,
#if N_CHANNEL >= 2
        buffer_links_base_chan_1,
#endif
#if N_CHANNEL >= 3
        buffer_links_base_chan_2,
#endif
#if N_CHANNEL >= 4
        buffer_links_base_chan_3,
#endif
#if N_CHANNEL >= 8
        buffer_links_base_chan_4,
        buffer_links_base_chan_5,
        buffer_links_base_chan_6,
        buffer_links_base_chan_7,
#endif
#if N_CHANNEL >= 16
        buffer_links_base_chan_8,
        buffer_links_base_chan_9,
        buffer_links_base_chan_10,
        buffer_links_base_chan_11,
        buffer_links_base_chan_12,
        buffer_links_base_chan_13,
        buffer_links_base_chan_14,
        buffer_links_base_chan_15,
#endif
        // in & out

        buffer_db_vectors_chan_0
#if N_CHANNEL >= 2
        , buffer_db_vectors_chan_1
#endif
#if N_CHANNEL >= 3
        , buffer_db_vectors_chan_2
#endif
#if N_CHANNEL >= 4
        , buffer_db_vectors_chan_3
#endif
#if N_CHANNEL >= 8
        , buffer_db_vectors_chan_4,
        buffer_db_vectors_chan_5,
        buffer_db_vectors_chan_6,
        buffer_db_vectors_chan_7
#endif
#if N_CHANNEL >= 16
        , buffer_db_vectors_chan_8,
        buffer_db_vectors_chan_9,
        buffer_db_vectors_chan_10,
        buffer_db_vectors_chan_11,
        buffer_db_vectors_chan_12,
        buffer_db_vectors_chan_13,
        buffer_db_vectors_chan_14,
        buffer_db_vectors_chan_15
#endif
        },0/* 0 means from host*/));

    std::cout << "Launching kernel...\n";
    // Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        buffer_out_id, buffer_out_dist, buffer_mem_debug}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);

    std::cout << "Duration (including memcpy out): " << duration << " sec" << std::endl; 

    // Translate physical node IDs to real label IDs
    if (graph_type == "HNSW") {
        for (int i = 0; i < query_num * ef; i++) {
            out_id[i] = labels_base[out_id[i]];
        }
    }

#ifdef DEBUG
    // print out the debug signals (each 4 byte):

    // debug signals (each 4 byte): 
    //   0: number of hops in base layer (number of pop operations)
    int print_qnum = 10 < query_num? 10 : query_num;
    for (int i = 0; i < print_qnum; i++) {
        std::cout << "query " << i << "\t#hops (base layer) =" << mem_debug[i * debug_size];
        if (gt_vec_ID[i * max_topK] != out_id[i * ef]) {
                std::cout << "Mismatch ";
        }    
        std::cout << "gt ID: " << gt_vec_ID[i * max_topK] << "\tgt dist: " << gt_dist[i * max_topK] 
            << "\thw ID: " << out_id[i * ef] << "\thw dist: " << out_dist[i * ef] <<   std::endl;
    }
#endif

    // verify top-1 result
    std::cout << "Verifying top-1 and top-10 result...\n";
    int top1_correct_count = 0;
    int top10_correct_count = 0;
    int k = 1;

    int dist_match_id_mismatch_cnt = 0;

    for (int qid = 0; qid < query_num_after_offset; qid++) {

        k = 1;
        for (int i = 0; i < k; i++) {
            int gt = gt_vec_ID[qid * max_topK];
            // float gt_dist_cur = gt_dist[qid * max_topK];
            int hw_id = out_id[qid * ef];
            // float hw_dist = out_dist[qid * ef];
            
            if (hw_id == gt) {
                top1_correct_count++;
            } else if (out_dist[qid * ef] == gt_dist[qid * max_topK]) {
                std::cout << "qid = " << qid << " Distance is the same" << " hw dist: " << out_dist[qid * ef] << " gt dist: " << gt_dist[qid * max_topK] <<
                    "hw id: " << hw_id << " gt id: " << gt << std::endl; 
                dist_match_id_mismatch_cnt++;
            }
        }

        // Check top-10 recall
        k = 10;
        for (int i = 0; i < k; i++) {
            int gt = gt_vec_ID[qid * max_topK + i];
            // check if it matches any top-10 ground truth
            for (int j = 0; j < k; j++) {
                int hw_id = out_id[qid * ef + j];
                if (hw_id == gt) {
                    top10_correct_count++;
                    break;
                }
            }
        }
    }

    // Print recall
    std::cout << "Recall@1=" << (float) top1_correct_count / query_num_after_offset << std::endl;
    std::cout << "Recall@10=" << (float) top10_correct_count / (query_num_after_offset * 10) << std::endl;

    std::cout << "Dist match id mismatch count=" << dist_match_id_mismatch_cnt << std::endl;

    // count avg #hops on base layer
    if (debug_size == 2) {
        int total_hops = 0;
        int total_visited_nodes = 0;
        for (int i = 0; i < query_num_after_offset; i++) {
            total_hops += mem_debug[2 * i];
            total_visited_nodes += mem_debug[2 * i + 1];
        }
        std::cout << "Average #hops on base layer=" << (float) total_hops / query_num_after_offset << std::endl;
        std::cout << "Average #visited nodes=" << (float) total_visited_nodes / query_num_after_offset << std::endl;
    } else if (debug_size == 1) {
        int total_hops = 0;
        for (int i = 0; i < query_num_after_offset; i++) {
            total_hops += mem_debug[i];
        }
        std::cout << "Average #hops on base layer=" << (float) total_hops / query_num_after_offset << std::endl;
    }

    return  0;
}
