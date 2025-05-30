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
        std::cout << "Usage: " << argv[0] << " <XCLBIN File 1> <local_FPGA_IP 2> <RxPort (C2F) 3> <TxIP (CPU IP) 4> <TxPort (F2C) 5> <FPGA_board_ID 6" << std::endl;
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

    uint32_t boardNum = strtol(argv[6], NULL, 10);

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
                      user_kernel = cl::Kernel(program, "accelerator_SIFT_M32_batch", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    ///////////     Part 2. Load Data     //////////
    
	// 1 B in 1 FPGAs
    int64_t dbsize = 1000 * 1000 * 1000;
    std::cout << "DB size (in THIS SHARD): " << dbsize << "\tD: " << D << std::endl;
    
    //////////     Allocate Memory     //////////

    std::cout << "Allocating memory...\n";
    auto start_load = std::chrono::high_resolution_clock::now();

    // in init
    size_t nlist = 32768;
    size_t meta_data_init_bytes = 3 * nlist * sizeof(int) + D * LUT_ENTRY_NUM * sizeof(float);
	// meta data consists of the following three stuffs:
	// int* nlist_PQ_codes_start_addr,
	// int* nlist_vec_ID_start_addr,
	// int* nlist_num_vecs,
	// float* product_quantizer
    size_t nlist_PQ_codes_start_addr_size = nlist * 4;
    size_t nlist_vec_ID_start_addr_size = nlist * 4;
    size_t nlist_num_vecs_size = nlist * 4;
    size_t product_quantizer_size = D * LUT_ENTRY_NUM * 4;

    std::vector<int ,aligned_allocator<int >> meta_data_init(meta_data_init_bytes / sizeof(int));

	// int* nlist_PQ_codes_start_addr,
	int num_vec_per_channel_per_list = dbsize % (4 * nlist) == 0 ? dbsize / (4 * nlist) : dbsize / (4 * nlist) + 1;
	assert(64 % M == 0);
	int vec_per_AXI = 64 / M; 
	int num_vec_AXI_per_channel_per_list = num_vec_per_channel_per_list % vec_per_AXI == 0 ? num_vec_per_channel_per_list / vec_per_AXI : num_vec_per_channel_per_list / vec_per_AXI + 1;
	for (int i = 0; i < nlist; i++) {
		meta_data_init[i] = i * num_vec_AXI_per_channel_per_list;
	}
	
	// int* nlist_vec_ID_start_addr,
	int ID_per_AXI = 64 / 8;
	int num_ID_AXI_per_channel_per_list = num_vec_per_channel_per_list % ID_per_AXI == 0 ? num_vec_per_channel_per_list / ID_per_AXI : num_vec_per_channel_per_list / ID_per_AXI + 1;
	for (int i = 0; i < nlist; i++) {
		meta_data_init[i + nlist] = i * num_ID_AXI_per_channel_per_list;
	}

	// int* nlist_num_vecs
	for (int i = 0; i < nlist; i++) {
		meta_data_init[i + 2 * nlist] = 4 * num_vec_per_channel_per_list;
	}

	size_t PQ_codes_DRAM_0_size = num_vec_AXI_per_channel_per_list * nlist * 64;
	size_t PQ_codes_DRAM_1_size = num_vec_AXI_per_channel_per_list * nlist * 64;
	size_t PQ_codes_DRAM_2_size = num_vec_AXI_per_channel_per_list * nlist * 64;
	size_t PQ_codes_DRAM_3_size = num_vec_AXI_per_channel_per_list * nlist * 64;

	size_t vec_ID_DRAM_0_size = num_ID_AXI_per_channel_per_list * nlist * 64;
	size_t vec_ID_DRAM_1_size = num_ID_AXI_per_channel_per_list * nlist * 64;
	size_t vec_ID_DRAM_2_size = num_ID_AXI_per_channel_per_list * nlist * 64;
	size_t vec_ID_DRAM_3_size = num_ID_AXI_per_channel_per_list * nlist * 64;

    // in runtime (should from DRAM)
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_0(PQ_codes_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_1(PQ_codes_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_2(PQ_codes_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> PQ_codes_DRAM_3(PQ_codes_DRAM_3_size / sizeof(int));

    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_0(vec_ID_DRAM_0_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_1(vec_ID_DRAM_1_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_2(vec_ID_DRAM_2_size / sizeof(int));
    std::vector<int ,aligned_allocator<int >> vec_ID_DRAM_3(vec_ID_DRAM_3_size / sizeof(int));

    //////////     load data from disk     //////////


    ///////////     Part 3. Lauch the kernel     //////////

    wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

    // fixed or calculated network param
    int32_t useConn = 1;
    uint64_t rxByteCnt = 1UL << (6 * 8); // 1 << 6 = 64 
    int32_t pkgWordCountTx = 1; // or 64, 16, etc.
    // int32_t pkgWordCountTx = 16; // or 64, 16, etc.
    uint64_t expectedTxPkgCnt = 1UL << (6 * 8);
	std::cout << "Setting very large send/recv sizes: rxByteCnt = " << rxByteCnt << " expectedTxPkgCnt = " << 
		expectedTxPkgCnt << std::endl;

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
	int arg_cnt = 0;

    // in init
    // OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, int(query_num)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, int(nlist)));
    // OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, int(nprobe)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_meta_data_init));

    // in runtime (should from DRAM)
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_PQ_codes_DRAM_0));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_PQ_codes_DRAM_1));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_PQ_codes_DRAM_2));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_PQ_codes_DRAM_3));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_in_vec_ID_DRAM_0));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_in_vec_ID_DRAM_1));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_in_vec_ID_DRAM_2));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + arg_cnt++, buffer_in_vec_ID_DRAM_3));


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
