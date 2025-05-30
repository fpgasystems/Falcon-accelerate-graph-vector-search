#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <limits>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "constants.hpp"

#include "xcl2.hpp"

#define DATA_SIZE 62500000

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char **argv) {

    //////////     Part 1. Parse the arguments & Program the FPGA     //////////

    if (argc != 7) {
        // Rx bytes = Tx byte (forwarding the data)
        std::cout << "Usage: " << argv[0] << " <XCLBIN File 1> <local_FPGA_IP 2> <RxPort 3> <TxIP 4> <TxPort 5> <nprobe 6>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    // arg 3
    uint32_t local_IP = 0x0A01D498;
    {
        std::string s = argv[2];
        std::string delimiter = ".";
        int ip [4];
        size_t pos = 0;
        std::string token;
        int i = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            ip [i] = stoi(token);
            s.erase(0, pos + delimiter.length());
            i++;
        }
        ip[i] = stoi(s); 
        local_IP = ip[3] | (ip[2] << 8) | (ip[1] << 16) | (ip[0] << 24);
    }

    // Rx
    int32_t basePortRx = 5001; 
    {
        basePortRx = strtol(argv[3], NULL, 10);
    }


    // Tx
    int32_t TxIPAddr = 0x0A01D46E;//alveo0
    {
        std::string s = argv[4];
        std::string delimiter = ".";
        int ip [4];
        size_t pos = 0;
        std::string token;
        int i = 0;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            ip [i] = stoi(token);
            s.erase(0, pos + delimiter.length());
            i++;
        }
        ip[i] = stoi(s); 
        TxIPAddr = ip[3] | (ip[2] << 8) | (ip[1] << 16) | (ip[0] << 24);
    }

    int32_t basePortTx = 5002; 
    {
        basePortTx = strtol(argv[5], NULL, 10);
    }

    size_t nprobe = 1;
    {
        nprobe = strtol(argv[6], NULL, 10);
    }

    auto size = DATA_SIZE;
    
    //Allocate Memory in Host Memory
    auto vector_size_bytes = sizeof(int) * size;
    std::vector<int, aligned_allocator<int>> network_ptr0(size);
    std::vector<int, aligned_allocator<int>> network_ptr1(size);


    //OPENCL HOST CODE AREA START
    //Create Program and Kernel
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;

    cl::Kernel user_kernel;
    cl::Kernel network_kernel;

    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                  cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err,
                      network_kernel = cl::Kernel(program, "network_krnl", &err));
            OCL_CHECK(err,
                      user_kernel = cl::Kernel(program, "accelerator_SIFT_M32", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    ///////////     Part 2. Load Data     //////////
    
    std::string db_name = "SIFT1000M"; // SIFT100M
    std::cout << "DB name: " << db_name << std::endl;
    
    std::string data_dir_prefix;
    if (db_name == "SIFT100M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF32768,PQ32";
    }
    else if (db_name == "SIFT1000M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT1000M_IVF32768,PQ32";
    }
    // std::string gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/";

    ///////////     get data size from disk     //////////

    // PQ codes
    std::string PQ_codes_DRAM_0_dir_suffix("DDR_bank_0_PQ_raw");
    std::string PQ_codes_DRAM_0_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_0_dir_suffix);
    std::ifstream PQ_codes_DRAM_0_fstream(
        PQ_codes_DRAM_0_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_0_fstream.seekg(0, PQ_codes_DRAM_0_fstream.end);
    size_t PQ_codes_DRAM_0_size =  PQ_codes_DRAM_0_fstream.tellg();
    if (!PQ_codes_DRAM_0_size) std::cout << "PQ_codes_DRAM_0_size is 0!";
    PQ_codes_DRAM_0_fstream.seekg(0, PQ_codes_DRAM_0_fstream.beg);

    std::string PQ_codes_DRAM_1_dir_suffix("DDR_bank_1_PQ_raw");
    std::string PQ_codes_DRAM_1_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_1_dir_suffix);
    std::ifstream PQ_codes_DRAM_1_fstream(
        PQ_codes_DRAM_1_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_1_fstream.seekg(0, PQ_codes_DRAM_1_fstream.end);
    size_t PQ_codes_DRAM_1_size =  PQ_codes_DRAM_1_fstream.tellg();
    if (!PQ_codes_DRAM_1_size) std::cout << "PQ_codes_DRAM_1_size is 0!";
    PQ_codes_DRAM_1_fstream.seekg(0, PQ_codes_DRAM_1_fstream.beg);
    
    std::string PQ_codes_DRAM_2_dir_suffix("DDR_bank_2_PQ_raw");
    std::string PQ_codes_DRAM_2_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_2_dir_suffix);
    std::ifstream PQ_codes_DRAM_2_fstream(
        PQ_codes_DRAM_2_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_2_fstream.seekg(0, PQ_codes_DRAM_2_fstream.end);
    size_t PQ_codes_DRAM_2_size =  PQ_codes_DRAM_2_fstream.tellg();
    if (!PQ_codes_DRAM_2_size) std::cout << "PQ_codes_DRAM_2_size is 0!";
    PQ_codes_DRAM_2_fstream.seekg(0, PQ_codes_DRAM_2_fstream.beg);
    
    std::string PQ_codes_DRAM_3_dir_suffix("DDR_bank_3_PQ_raw");
    std::string PQ_codes_DRAM_3_dir = dir_concat(data_dir_prefix, PQ_codes_DRAM_3_dir_suffix);
    std::ifstream PQ_codes_DRAM_3_fstream(
        PQ_codes_DRAM_3_dir, 
        std::ios::in | std::ios::binary);
    PQ_codes_DRAM_3_fstream.seekg(0, PQ_codes_DRAM_3_fstream.end);
    size_t PQ_codes_DRAM_3_size =  PQ_codes_DRAM_3_fstream.tellg();
    if (!PQ_codes_DRAM_3_size) std::cout << "PQ_codes_DRAM_3_size is 0!";
    PQ_codes_DRAM_3_fstream.seekg(0, PQ_codes_DRAM_3_fstream.beg);

    // vec IDs
    std::string vec_ID_DRAM_0_dir_suffix("DDR_bank_0_vec_ID_raw");
    std::string vec_ID_DRAM_0_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_0_dir_suffix);
    std::ifstream vec_ID_DRAM_0_fstream(
        vec_ID_DRAM_0_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_0_fstream.seekg(0, vec_ID_DRAM_0_fstream.end);
    size_t vec_ID_DRAM_0_size =  vec_ID_DRAM_0_fstream.tellg();
    if (!vec_ID_DRAM_0_size) std::cout << "vec_ID_DRAM_0_size is 0!";
    vec_ID_DRAM_0_fstream.seekg(0, vec_ID_DRAM_0_fstream.beg);

    std::string vec_ID_DRAM_1_dir_suffix("DDR_bank_1_vec_ID_raw");
    std::string vec_ID_DRAM_1_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_1_dir_suffix);
    std::ifstream vec_ID_DRAM_1_fstream(
        vec_ID_DRAM_1_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_1_fstream.seekg(0, vec_ID_DRAM_1_fstream.end);
    size_t vec_ID_DRAM_1_size =  vec_ID_DRAM_1_fstream.tellg();
    if (!vec_ID_DRAM_1_size) std::cout << "vec_ID_DRAM_1_size is 0!";
    vec_ID_DRAM_1_fstream.seekg(0, vec_ID_DRAM_1_fstream.beg);

    std::string vec_ID_DRAM_2_dir_suffix("DDR_bank_2_vec_ID_raw");
    std::string vec_ID_DRAM_2_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_2_dir_suffix);
    std::ifstream vec_ID_DRAM_2_fstream(
        vec_ID_DRAM_2_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_2_fstream.seekg(0, vec_ID_DRAM_2_fstream.end);
    size_t vec_ID_DRAM_2_size =  vec_ID_DRAM_2_fstream.tellg();
    if (!vec_ID_DRAM_2_size) std::cout << "vec_ID_DRAM_2_size is 0!";
    vec_ID_DRAM_2_fstream.seekg(0, vec_ID_DRAM_2_fstream.beg);

    std::string vec_ID_DRAM_3_dir_suffix("DDR_bank_3_vec_ID_raw");
    std::string vec_ID_DRAM_3_dir = dir_concat(data_dir_prefix, vec_ID_DRAM_3_dir_suffix);
    std::ifstream vec_ID_DRAM_3_fstream(
        vec_ID_DRAM_3_dir, 
        std::ios::in | std::ios::binary);
    vec_ID_DRAM_3_fstream.seekg(0, vec_ID_DRAM_3_fstream.end);
    size_t vec_ID_DRAM_3_size =  vec_ID_DRAM_3_fstream.tellg();
    if (!vec_ID_DRAM_3_size) std::cout << "vec_ID_DRAM_3_size is 0!";
    vec_ID_DRAM_3_fstream.seekg(0, vec_ID_DRAM_3_fstream.beg);

    // control signals
    std::string nlist_PQ_codes_start_addr_dir_suffix("nlist_PQ_codes_start_addr");
    std::string nlist_PQ_codes_start_addr_dir = dir_concat(data_dir_prefix, nlist_PQ_codes_start_addr_dir_suffix);
    std::ifstream nlist_PQ_codes_start_addr_fstream(
        nlist_PQ_codes_start_addr_dir, 
        std::ios::in | std::ios::binary);
    nlist_PQ_codes_start_addr_fstream.seekg(0, nlist_PQ_codes_start_addr_fstream.end);
    size_t nlist_PQ_codes_start_addr_size =  nlist_PQ_codes_start_addr_fstream.tellg();
    if (!nlist_PQ_codes_start_addr_size) std::cout << "nlist_PQ_codes_start_addr_size is 0!";
    nlist_PQ_codes_start_addr_fstream.seekg(0, nlist_PQ_codes_start_addr_fstream.beg);

    std::string nlist_vec_ID_start_addr_dir_suffix("nlist_vec_ID_start_addr");
    std::string nlist_vec_ID_start_addr_dir = dir_concat(data_dir_prefix, nlist_vec_ID_start_addr_dir_suffix);
    std::ifstream nlist_vec_ID_start_addr_fstream(
        nlist_vec_ID_start_addr_dir, 
        std::ios::in | std::ios::binary);
    nlist_vec_ID_start_addr_fstream.seekg(0, nlist_vec_ID_start_addr_fstream.end);
    size_t nlist_vec_ID_start_addr_size =  nlist_vec_ID_start_addr_fstream.tellg();
    if (!nlist_vec_ID_start_addr_size) std::cout << "nlist_vec_ID_start_addr_size is 0!";
    nlist_vec_ID_start_addr_fstream.seekg(0, nlist_vec_ID_start_addr_fstream.beg);

    std::string nlist_num_vecs_dir_suffix("nlist_num_vecs");
    std::string nlist_num_vecs_dir = dir_concat(data_dir_prefix, nlist_num_vecs_dir_suffix);
    std::ifstream nlist_num_vecs_fstream(
        nlist_num_vecs_dir, 
        std::ios::in | std::ios::binary);
    nlist_num_vecs_fstream.seekg(0, nlist_num_vecs_fstream.end);
    size_t nlist_num_vecs_size =  nlist_num_vecs_fstream.tellg();
    if (!nlist_num_vecs_size) std::cout << "nlist_num_vecs_size is 0!";
    nlist_num_vecs_fstream.seekg(0, nlist_num_vecs_fstream.beg);

    std::string product_quantizer_dir_suffix("product_quantizer_float32_32_256_4_raw");
    std::string product_quantizer_dir = dir_concat(data_dir_prefix, product_quantizer_dir_suffix);
    std::ifstream product_quantizer_fstream(
        product_quantizer_dir, 
        std::ios::in | std::ios::binary);
    product_quantizer_fstream.seekg(0, product_quantizer_fstream.end);
    size_t product_quantizer_size =  product_quantizer_fstream.tellg();
    if (!product_quantizer_size) std::cout << "product_quantizer_size is 0!";
    product_quantizer_fstream.seekg(0, product_quantizer_fstream.beg);

    //////////     Allocate Memory     //////////

    std::cout << "Allocating memory...\n";
    auto start_load = std::chrono::high_resolution_clock::now();

    // in init
    size_t query_num = 10000;
    size_t nlist = 32768;

    assert (nprobe <= nlist);

    size_t meta_data_init_bytes = 3 * nlist * sizeof(int) + D * LUT_ENTRY_NUM * sizeof(float);

    assert(nlist * 4 ==  nlist_PQ_codes_start_addr_size);
    assert(nlist * 4 ==  nlist_vec_ID_start_addr_size);
    assert(nlist * 4 ==  nlist_num_vecs_size);
    assert(D * LUT_ENTRY_NUM * 4 == product_quantizer_size);

    std::vector<int ,aligned_allocator<int >> meta_data_init(meta_data_init_bytes / sizeof(int));

    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t in_DRAM_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    // std::vector<int ,aligned_allocator<int >> in_DRAM(in_DRAM_bytes / sizeof(int));


    // in runtime (should from DRAM)
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_0(PQ_codes_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_1(PQ_codes_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_2(PQ_codes_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_3(PQ_codes_DRAM_3_size / sizeof(int));

    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_0(vec_ID_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_1(vec_ID_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_2(vec_ID_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_3(vec_ID_DRAM_3_size / sizeof(int));
    
    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    size_t size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet
    size_t out_bytes = query_num * 64 * size_results;
    // std::vector<int ,aligned_allocator<int>> out(out_bytes / sizeof(int));

    //////////     load data from disk     //////////
    std::cout << "Loading data from disk...\n";

    // PQ codes
    char* PQ_codes_DRAM_0_char = (char*) malloc(PQ_codes_DRAM_0_size);
    PQ_codes_DRAM_0_fstream.read(PQ_codes_DRAM_0_char, PQ_codes_DRAM_0_size);
    if (!PQ_codes_DRAM_0_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_0_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_0[0], PQ_codes_DRAM_0_char, PQ_codes_DRAM_0_size);
    free(PQ_codes_DRAM_0_char);

    char* PQ_codes_DRAM_1_char = (char*) malloc(PQ_codes_DRAM_1_size);
    PQ_codes_DRAM_1_fstream.read(PQ_codes_DRAM_1_char, PQ_codes_DRAM_1_size);
    if (!PQ_codes_DRAM_1_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_1_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_1[0], PQ_codes_DRAM_1_char, PQ_codes_DRAM_1_size);
    free(PQ_codes_DRAM_1_char);

    char* PQ_codes_DRAM_2_char = (char*) malloc(PQ_codes_DRAM_2_size);
    PQ_codes_DRAM_2_fstream.read(PQ_codes_DRAM_2_char, PQ_codes_DRAM_2_size);
    if (!PQ_codes_DRAM_2_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_2_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_2[0], PQ_codes_DRAM_2_char, PQ_codes_DRAM_2_size);
    free(PQ_codes_DRAM_2_char);
    
    char* PQ_codes_DRAM_3_char = (char*) malloc(PQ_codes_DRAM_3_size);
    PQ_codes_DRAM_3_fstream.read(PQ_codes_DRAM_3_char, PQ_codes_DRAM_3_size);
    if (!PQ_codes_DRAM_3_fstream) {
            std::cout << "error: only " << PQ_codes_DRAM_3_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&PQ_codes_DRAM_3[0], PQ_codes_DRAM_3_char, PQ_codes_DRAM_3_size);
    free(PQ_codes_DRAM_3_char);

    // vec ID
    char* vec_ID_DRAM_0_char = (char*) malloc(vec_ID_DRAM_0_size);
    vec_ID_DRAM_0_fstream.read(vec_ID_DRAM_0_char, vec_ID_DRAM_0_size);
    if (!vec_ID_DRAM_0_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_0_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_0[0], vec_ID_DRAM_0_char, vec_ID_DRAM_0_size);
    free(vec_ID_DRAM_0_char);

    char* vec_ID_DRAM_1_char = (char*) malloc(vec_ID_DRAM_1_size);
    vec_ID_DRAM_1_fstream.read(vec_ID_DRAM_1_char, vec_ID_DRAM_1_size);
    if (!vec_ID_DRAM_1_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_1_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_1[0], vec_ID_DRAM_1_char, vec_ID_DRAM_1_size);
    free(vec_ID_DRAM_1_char);

    char* vec_ID_DRAM_2_char = (char*) malloc(vec_ID_DRAM_2_size);
    vec_ID_DRAM_2_fstream.read(vec_ID_DRAM_2_char, vec_ID_DRAM_2_size);
    if (!vec_ID_DRAM_2_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_2_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_2[0], vec_ID_DRAM_2_char, vec_ID_DRAM_2_size);
    free(vec_ID_DRAM_2_char);
    
    char* vec_ID_DRAM_3_char = (char*) malloc(vec_ID_DRAM_3_size);
    vec_ID_DRAM_3_fstream.read(vec_ID_DRAM_3_char, vec_ID_DRAM_3_size);
    if (!vec_ID_DRAM_3_fstream) {
            std::cout << "error: only " << vec_ID_DRAM_3_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vec_ID_DRAM_3[0], vec_ID_DRAM_3_char, vec_ID_DRAM_3_size);
    free(vec_ID_DRAM_3_char);

    // control signals
    // meta_data_init = nlist_PQ_codes_start_addr, nlist_vec_ID_start_addr, nlist_num_vecs,
    char* nlist_PQ_codes_start_addr_char = (char*) malloc(nlist_PQ_codes_start_addr_size);
    nlist_PQ_codes_start_addr_fstream.read(nlist_PQ_codes_start_addr_char, nlist_PQ_codes_start_addr_size);
    if (!nlist_PQ_codes_start_addr_fstream) {
            std::cout << "error: only " << nlist_PQ_codes_start_addr_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[0], nlist_PQ_codes_start_addr_char, nlist_PQ_codes_start_addr_size);
    free(nlist_PQ_codes_start_addr_char);

    char* nlist_vec_ID_start_addr_char = (char*) malloc(nlist_vec_ID_start_addr_size);
    nlist_vec_ID_start_addr_fstream.read(nlist_vec_ID_start_addr_char, nlist_vec_ID_start_addr_size);
    if (!nlist_vec_ID_start_addr_fstream) {
            std::cout << "error: only " << nlist_vec_ID_start_addr_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[nlist], nlist_vec_ID_start_addr_char, nlist_vec_ID_start_addr_size);
    free(nlist_vec_ID_start_addr_char);
    
    char* nlist_num_vecs_char = (char*) malloc(nlist_num_vecs_size);
    nlist_num_vecs_fstream.read(nlist_num_vecs_char, nlist_num_vecs_size);
    if (!nlist_num_vecs_fstream) {
            std::cout << "error: only " << nlist_num_vecs_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[2 * nlist], nlist_num_vecs_char, nlist_num_vecs_size);
    free(nlist_num_vecs_char);
#ifdef DEBUG
    for (int i = 0; i < nlist; i++) {
        std::cout << "cell_ID = " << i << " nlist_PQ_codes_start_addr = " << meta_data_init[i] <<
            " nlist_vec_ID_start_addr = " << meta_data_init[i + nlist] << 
            " nlist_num_vecs = " << meta_data_init[i + 2 * nlist] << std::endl;
    }
#endif

    char* product_quantizer_char = (char*) malloc(product_quantizer_size);
    product_quantizer_fstream.read(product_quantizer_char, product_quantizer_size);
    if (!product_quantizer_fstream) {
            std::cout << "error: only " << product_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&meta_data_init[3 * nlist], product_quantizer_char, product_quantizer_size);
    free(product_quantizer_char);

    assert(D * 256 * sizeof(float) == product_quantizer_size);

    auto end_load = std::chrono::high_resolution_clock::now();
    double duration_load = (std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count());

    std::cout << "Duration memory allocation & disk load: " << duration_load << " ms" << std::endl; 

    ///////////     Part 3. Lauch the kernel     //////////

    wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    // fixed or calculated network param
    uint32_t boardNum = 1;
    int32_t useConn = 1;
    uint64_t rxByteCnt = in_DRAM_bytes;
    int32_t pkgWordCountTx = 1; // or 64, 16, etc.
    // int32_t pkgWordCountTx = 16; // or 64, 16, etc.
    uint64_t expectedTxPkgCnt = out_bytes / pkgWordCountTx / 64;
    assert(out_bytes % (pkgWordCountTx * 64) == 0);

    printf("local_IP:%x, boardNum:%d\n", local_IP, boardNum); 

    // Set network kernel arguments
    OCL_CHECK(err, err = network_kernel.setArg(0, local_IP)); // Default IP address
    OCL_CHECK(err, err = network_kernel.setArg(1, boardNum)); // Board number
    OCL_CHECK(err, err = network_kernel.setArg(2, local_IP)); // ARP lookup

    OCL_CHECK(err,
              cl::Buffer buffer_r1(context,
                                   CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   vector_size_bytes,
                                   network_ptr0.data(),
                                   &err));
    OCL_CHECK(err,
            cl::Buffer buffer_r2(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                vector_size_bytes,
                                network_ptr1.data(),
                                &err));

    OCL_CHECK(err, err = network_kernel.setArg(3, buffer_r1));
    OCL_CHECK(err, err = network_kernel.setArg(4, buffer_r2));

    printf("enqueue network kernel...\n");
    OCL_CHECK(err, err = q.enqueueTask(network_kernel));
    OCL_CHECK(err, err = q.finish());

    // accelerator kernel CL buffer
    // in init 
    OCL_CHECK(err, cl::Buffer buffer_meta_data_init   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            meta_data_init_bytes, meta_data_init.data(), &err));

    // in runtime (should from DRAM)
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_0_size, PQ_codes_DRAM_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_1_size, PQ_codes_DRAM_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_2_size, PQ_codes_DRAM_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_PQ_codes_DRAM_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            PQ_codes_DRAM_3_size, PQ_codes_DRAM_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_0   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_0_size, vec_ID_DRAM_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_1   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_1_size, vec_ID_DRAM_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_2_size, vec_ID_DRAM_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_vec_ID_DRAM_3   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vec_ID_DRAM_3_size, vec_ID_DRAM_3.data(), &err));


    //Set user Kernel Arguments
    int start_param_network = 16;

    std::cout << "useConn: " << useConn << std::endl; 
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 0, useConn));

    std::cout << "basePortRx: " << basePortRx << std::endl; 
    std::cout << "rxByteCnt: " << rxByteCnt << std::endl; 

    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 1, basePortRx));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 2, rxByteCnt));

    printf("TxIPAddr:%x \n", TxIPAddr);
    std::cout << "basePortTx: " << basePortTx << std::endl; 
    std::cout << "expectedTxPkgCnt: " << expectedTxPkgCnt << std::endl; 
    std::cout << "pkgWordCountTx: " << pkgWordCountTx << std::endl; 
    std::cout << "(calculated) expected Tx bytes: expectedTxPkgCnt * pkgWordCountTx * 64: " << 
        expectedTxPkgCnt * pkgWordCountTx * 64 << std::endl; 
    
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 3, TxIPAddr));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 4, basePortTx));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 5, expectedTxPkgCnt));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_network + 6, pkgWordCountTx));

    int start_param_accelerator = 16 + 7;

    // in init
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 0, int(query_num)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 1, int(nlist)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 2, int(nprobe)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 3, buffer_meta_data_init));

    // in runtime (should from DRAM)
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 4, buffer_PQ_codes_DRAM_0));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 5, buffer_PQ_codes_DRAM_1));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 6, buffer_PQ_codes_DRAM_2));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 7, buffer_PQ_codes_DRAM_3));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 8, buffer_in_vec_ID_DRAM_0));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 9, buffer_in_vec_ID_DRAM_1));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 10, buffer_in_vec_ID_DRAM_2));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 11, buffer_in_vec_ID_DRAM_3));


    double durationUs = 0.0;

    //Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    printf("enqueue user kernel...\n");
    OCL_CHECK(err, err = q.enqueueTask(user_kernel));
    OCL_CHECK(err, err = q.finish());
    auto end = std::chrono::high_resolution_clock::now();
    durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
    printf("durationUs:%f\n",durationUs);
    //OPENCL HOST CODE AREA END    

    std::cout << "EXIT recorded" << std::endl;
}
