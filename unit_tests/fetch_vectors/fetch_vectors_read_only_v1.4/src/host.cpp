#include <stdint.h>
#include <chrono>
#include <cassert>

#include "host.hpp"

#include "constants.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"

#include <sys/stat.h>

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
    int query_num = 1000;
	int read_iter_per_query = 1000;
	int read_batch_size = 10;

	int d = D;
	int total_node_num = 1000 * 1000; // 1 M nodes

	const int AXI_num_per_vector = d % FLOAT_PER_AXI == 0? 
		d / FLOAT_PER_AXI + 1 : d / FLOAT_PER_AXI + 2; // 16 for d = 512 + visited padding

	size_t bytes_mem_read_node_id = read_iter_per_query * sizeof(int);
	size_t bytes_db_vectors = AXI_num_per_vector * total_node_num * 64;
	size_t bytes_out_dist = 1024;

	// input vecs
	std::vector<int, aligned_allocator<int>> mem_read_node_id(bytes_mem_read_node_id / sizeof(int));
	std::vector<float, aligned_allocator<float>> db_vectors(bytes_db_vectors / sizeof(float));

	// output
	std::vector<float, aligned_allocator<float>> out_dist(bytes_out_dist / sizeof(float));

	// generate random query and fetched vectors
	for (int i = 0; i < read_iter_per_query; i++) {
		mem_read_node_id[i] = 1000 * i % total_node_num; // make sure every time the memory addresses are very far from each other
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
	OCL_CHECK(err, cl::Buffer buffer_mem_read_node_id (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			bytes_mem_read_node_id, mem_read_node_id.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_db_vectors (context,CL_MEM_USE_HOST_PTR,
			bytes_db_vectors, db_vectors.data(), &err));

	// out
	OCL_CHECK(err, cl::Buffer buffer_out_dist (context,CL_MEM_USE_HOST_PTR,
			bytes_out_dist, out_dist.data(), &err));

	std::cout << "Finish allocate buffer...\n";

	int arg_counter = 0;    
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_num)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(read_iter_per_query)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(read_batch_size)));
	// OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(d)));

	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_mem_read_node_id));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors));
	
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_dist));


    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
		buffer_mem_read_node_id,
		buffer_db_vectors
        },0/* 0 means from host*/));

    std::cout << "Launching kernel...\n";
    // Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out_dist},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);

    std::cout << "Duration (including memcpy out): " << duration << " sec" << std::endl; 

	// compare software and hardware results
	std::cout << "No result checking." << std::endl;

	return  0;
}
