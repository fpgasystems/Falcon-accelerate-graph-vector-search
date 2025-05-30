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
    int query_num = 1;
	int read_iter_per_query = 500;
    int d = 128;


	// turn on / off run upper / base layer
	int run_upper_levels = 1;
	int run_base_level = 0;

	size_t bytes_mem_keys = read_iter_per_query * sizeof(int);
	size_t bytes_out = 2 * read_iter_per_query * sizeof(int); // first read_iter_per_query upper level; second for base level

    // initialization values
    int num_db_vec = 1000 * 1000;
    
    size_t bytes_per_vec = d * sizeof(float);
    size_t bytes_entry_vector = bytes_per_vec;
    size_t bytes_query_vectors = query_num * bytes_per_vec;
	size_t bytes_db_vectors = num_db_vec * bytes_per_vec;

    // input vecs
    std::vector<float, aligned_allocator<float>> query_vectors(bytes_query_vectors / sizeof(float));
	std::vector<uint32_t, aligned_allocator<uint32_t>> mem_keys(bytes_mem_keys / sizeof(uint32_t));
    
	// db vec
    std::vector<float, aligned_allocator<float>> db_vectors(bytes_db_vectors / sizeof(float));
    
	// output
	std::vector<uint32_t, aligned_allocator<uint32_t>> out(bytes_out / sizeof(uint32_t));
	std::vector<uint32_t, aligned_allocator<uint32_t>> sw_results(bytes_out / sizeof(uint32_t));

	// TODO: FILL in mem_keys
	for (int i = 0; i < read_iter_per_query; i++) {
		mem_keys[i] = i * 100 % num_db_vec;
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
    OCL_CHECK(err, cl::Buffer buffer_query_vectors (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            bytes_query_vectors, query_vectors.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_mem_keys (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			bytes_mem_keys, mem_keys.data(), &err));
    // in & out (db vec is mixed with visited list)
    OCL_CHECK(err, cl::Buffer buffer_db_vectors (context,CL_MEM_USE_HOST_PTR,	
            bytes_db_vectors, db_vectors.data(), &err));

    // out
    OCL_CHECK(err, cl::Buffer buffer_out (context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            bytes_out, out.data(), &err));
	std::cout << "Finish allocate buffer...\n";

    int arg_counter = 0;    
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_num)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(read_iter_per_query)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(d)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(run_upper_levels)));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(run_base_level)));
	
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_query_vectors));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_mem_keys));
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_db_vectors));

    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
        buffer_query_vectors,
        buffer_mem_keys,
        // in & out
        buffer_db_vectors
        },0/* 0 means from host*/));

    std::cout << "Launching kernel...\n";
    // Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);

    std::cout << "Duration (including memcpy out): " << duration << " sec" << std::endl; 

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
