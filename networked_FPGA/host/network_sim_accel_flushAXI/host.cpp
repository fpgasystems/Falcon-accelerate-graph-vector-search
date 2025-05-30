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
                      user_kernel = cl::Kernel(program, "network_sim_accel_flushAXI", &err));
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
    
    int delay_cycle_per_cell;
    if (db_name == "SIFT1000M") {
        delay_cycle_per_cell = float(1000 * 1000 * 1000) / 32768 / (64 * 4 / M);
    }
    else if (db_name == "SIFT100M") {
        delay_cycle_per_cell = float(100 * 1000 * 1000) / 32768 / (64 * 4 / M);
    }
    std::cout << "delay_cycle_per_cell: " << delay_cycle_per_cell << std::endl;


    // in init
    size_t query_num = 10000;
    size_t nlist = 32768;

    assert (nprobe <= nlist);
    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t in_DRAM_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);


    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = PRIORITY_QUEUE_LEN_L2 * 64 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 64 / 512 : PRIORITY_QUEUE_LEN_L2 * 64 / 512 + 1;
    size_t size_results_dist = PRIORITY_QUEUE_LEN_L2 * 32 % 512 == 0?
        PRIORITY_QUEUE_LEN_L2 * 32 / 512 : PRIORITY_QUEUE_LEN_L2 * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet
    size_t out_bytes = query_num * 64 * size_results;


    size_t in_DDR_size = TOPK * sizeof(float);
    std::vector<float ,aligned_allocator<float >> in_DDR(TOPK);

    OCL_CHECK(err, cl::Buffer buffer_in_DDR   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            in_DDR_size, in_DDR.data(), &err));

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
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 3, int(delay_cycle_per_cell)));
    OCL_CHECK(err, err = user_kernel.setArg(start_param_accelerator + 4, buffer_in_DDR));

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
