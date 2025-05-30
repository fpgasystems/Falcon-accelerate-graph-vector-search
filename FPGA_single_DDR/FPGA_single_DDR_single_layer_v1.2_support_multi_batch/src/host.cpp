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

int main(int argc, char** argv)
{
    cl_int err;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 

    std::cout << "Allocating memory...\n";

    // in init
    int query_num = 10000;
	int query_batch_size = 10000;
	int query_offset = 0; // starting from query x
	int query_num_after_offset = query_num + query_offset > 10000? 10000 - query_offset : query_num;
    int ef = 64;
    int candidate_queue_runtime_size = hardware_candidate_queue_size;
    int max_cand_batch_size;
    int max_async_stage_num;
	if (argc > 2) {
		max_cand_batch_size = atoi(argv[2]);
	} else {
		max_cand_batch_size = 1;
	}
	if (argc > 3) {
		max_async_stage_num = atoi(argv[3]);
	} else {
		max_async_stage_num = 4;
	}
    int d = D;
	int max_bloom_out_burst_size = 16; // according to mem & compute speed test
	int runtime_n_bucket_addr_bits = 8 + 10; // 256K buckets
	int runtime_n_buckets = 1 << runtime_n_bucket_addr_bits;
	uint32_t hash_seed = 1;
    assert (ef <= hardware_result_queue_size);

    // initialization values
    int max_level; // = 16;
    int max_link_num_upper; // = 32;
    int max_link_num_base; // = 32;
    int entry_point_id; // = 0;
    int num_db_vec; // = 1000 * 1000;

    // load metadata from file /mnt/scratch/wenqi/hnswlib-eval/FPGA_indexes/SIFT1M_index_M_32/meta.bin
    //  cur_element_count, maxlevel_, enterpoint_node_, maxM_, maxM0_
    FILE* f_metadata = fopen("/mnt/scratch/wenqi/hnswlib-eval/FPGA_indexes/SIFT1M_index_M_32/meta.bin", "rb");
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
    
    size_t bytes_per_vec = d * sizeof(float);
    size_t bytes_per_db_vec_plus_padding = d % 16 == 0? d * sizeof(float) + 64 : (d + 16 - d % 16) * sizeof(float) + 64;
    size_t bytes_entry_vector = bytes_per_vec;
	size_t bytes_entry_point_ids = query_num * sizeof(int);
    size_t bytes_query_vectors = query_num * bytes_per_vec;
    size_t bytes_out_id = query_num * ef * sizeof(int);
    size_t bytes_out_dist = query_num * ef * sizeof(float);	
    size_t bytes_mem_debug = query_num * 5 * sizeof(int);

    const char* fname_ground_links = "/mnt/scratch/wenqi/hnswlib-eval/FPGA_indexes/SIFT1M_index_M_32/ground_links_1_chan_0.bin";
    const char* fname_ground_labels = "/mnt/scratch/wenqi/hnswlib-eval/FPGA_indexes/SIFT1M_index_M_32/ground_labels.bin";
    const char* fname_ground_vectors = "/mnt/scratch/wenqi/hnswlib-eval/FPGA_indexes/SIFT1M_index_M_32/ground_vectors_1_chan_0.bin";
    const char* fname_query_vectors =  "/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs";
    const char* fname_gt_vec_ID = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_1M.ivecs";
    const char* fname_gt_dist = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/dis_1M.fvecs";
    FILE* f_ground_links = fopen(fname_ground_links, "rb");
	FILE* f_ground_labels = fopen(fname_ground_labels, "rb");
    FILE* f_ground_vectors = fopen(fname_ground_vectors, "rb");
    FILE* f_query_vectors = fopen(fname_query_vectors, "rb");
    FILE* f_gt_vec_ID = fopen(fname_gt_vec_ID, "rb");
    FILE* f_gt_dist = fopen(fname_gt_dist, "rb");


    // get file size
    size_t bytes_db_vectors = GetFileSize(fname_ground_vectors);
    size_t bytes_links_base = GetFileSize(fname_ground_links);
	size_t bytes_labels_base = GetFileSize(fname_ground_labels); // int = 4 bytes
    size_t raw_query_vectors_size = GetFileSize(fname_query_vectors);
    size_t raw_gt_vec_ID_size = GetFileSize(fname_gt_vec_ID);
    size_t raw_gt_dist_size = GetFileSize(fname_gt_dist);
    std::cout << "bytes_db_vectors=" << bytes_db_vectors << std::endl;
    std::cout << "bytes_links_base=" << bytes_links_base << std::endl;
	std::cout << "raw_query_vectors_size=" << raw_query_vectors_size << std::endl;
    assert(bytes_per_db_vec_plus_padding * num_db_vec == bytes_db_vectors);

    // input vecs
	std::vector<int, aligned_allocator<int>> entry_point_ids(bytes_entry_point_ids / sizeof(int));
    std::vector<float, aligned_allocator<float>> query_vectors(bytes_query_vectors / sizeof(float));

    // db vec
    std::vector<float, aligned_allocator<float>> db_vectors(bytes_db_vectors / sizeof(float));
    
    // links
    std::vector<int, aligned_allocator<int>> links_base(bytes_links_base / sizeof(int));
    
    // output
    std::vector<int, aligned_allocator<int>> out_id(bytes_out_id / sizeof(int));
    std::vector<float, aligned_allocator<float>> out_dist(bytes_out_dist / sizeof(float));
    std::vector<int, aligned_allocator<int>> mem_debug(bytes_mem_debug / sizeof(int));

    // intermediate buffer for queries, and ground truth
	std::vector<int> labels_base(bytes_labels_base / sizeof(int));
    std::vector<unsigned char> raw_query_vectors(raw_query_vectors_size / sizeof(unsigned char));
    std::vector<int> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(int));
    std::vector<float> raw_gt_dist(raw_gt_dist_size / sizeof(float));

    int max_topK = 100;
    std::vector<int> gt_vec_ID(query_num * max_topK);
    std::vector<float> gt_dist(query_num * max_topK);

    // read data from file
    std::cout << "Reading database vectors from file...\n";
    fread(db_vectors.data(), 1, bytes_db_vectors, f_ground_vectors);
    fclose(f_ground_vectors);

    std::cout << "Reading base links from file...\n";
    fread(links_base.data(), 1, bytes_links_base, f_ground_links);
    fclose(f_ground_links);

    std::cout << "Reading queries and ground truths from file...\n";
	fread(labels_base.data(), 1, bytes_labels_base, f_ground_labels);
	fclose(f_ground_labels);
    fread(raw_query_vectors.data(), 1, raw_query_vectors_size, f_query_vectors);
    fclose(f_query_vectors);
    fread(raw_gt_vec_ID.data(), 1, raw_gt_vec_ID_size, f_gt_vec_ID);
    fclose(f_gt_vec_ID);
    fread(raw_gt_dist.data(), 1, raw_gt_dist_size, f_gt_dist);
    fclose(f_gt_dist);

    // query vector = 4-byte ID + d * (uint8) vectors
    size_t len_per_query = 4 + d;
    for (int qid = 0; qid < query_num_after_offset; qid++) {
        for (int i = 0; i < d; i++) {
            query_vectors[qid * d + i] = (float) raw_query_vectors[(qid + query_offset) * len_per_query + 4 + i];
        }
    }

	for (int qid = 0; qid < query_num_after_offset; qid++) {
		entry_point_ids[qid] = entry_point_id;
	}

    // ground truth = 4-byte ID + 1000 * 4-byte ID + 1000 or 4-byte distances
    size_t len_per_gt = (4 + 1000 * 4) / 4;
    for (int qid = 0; qid < query_num_after_offset; qid++) {
        for (int i = 0; i < max_topK; i++) {
            gt_vec_ID[qid * max_topK + i] = raw_gt_vec_ID[(qid + query_offset) * len_per_gt + 1 + i];
            gt_dist[qid * max_topK + i] = raw_gt_dist[(qid + query_offset) * len_per_gt + 1 + i];
        }
    }


// OPENCL HOST CODE AREA START

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
    OCL_CHECK(err, cl::Buffer buffer_links_base (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            bytes_links_base, links_base.data(), &err));

    // in & out (db vec is mixed with visited list)
    OCL_CHECK(err, cl::Buffer buffer_db_vectors (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors, db_vectors.data(), &err));

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
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_cand_batch_size)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_async_stage_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(runtime_n_bucket_addr_bits)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(hash_seed)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_bloom_out_burst_size)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(d)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_link_num_base)));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_entry_point_ids));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_query_vectors));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_links_base));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_id));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_dist));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_mem_debug));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
		buffer_entry_point_ids,
        buffer_query_vectors,
        buffer_links_base,
        // in & out
        buffer_db_vectors
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
	for (int i = 0; i < query_num * ef; i++) {
		out_id[i] = labels_base[out_id[i]];
	}

#ifdef DEBUG
    // print out the debug signals (each 4 byte):

	// debug signals (each 4 byte): 
	//   0: number of hops in base layer (number of pop operations)
    int print_qnum = 10 < query_num? 10 : query_num;
	int debug_size = 1;
    for (int i = 0; i < print_qnum; i++) {
        std::cout << "query " << i << "\t#hops (base layer) =" << mem_debug[i * debug_size];
        if (gt_vec_ID[i * max_topK] != out_id[i * ef]) {
                std::cout << "Mismatch ";
        }    
        std::cout << "gt: " << gt_vec_ID[i * max_topK] << " hw: " << out_id[i * ef] << " hw dist: " << out_dist[i * ef] <<   std::endl;
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
	int total_hops = 0;
	int total_visited_nodes = 0;
	for (int i = 0; i < query_num_after_offset; i++) {
		total_hops += mem_debug[2 * i];
		total_visited_nodes += mem_debug[2 * i + 1];
	}
	std::cout << "Average #hops on base layer=" << (float) total_hops / query_num_after_offset << std::endl;
	std::cout << "Average #visited nodes=" << (float) total_visited_nodes / query_num_after_offset << std::endl;

    return  0;
}
