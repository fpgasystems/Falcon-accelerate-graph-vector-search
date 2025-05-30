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
    int query_num = 1;
	int num_fetched_vectors_per_query = 64;
	int d = 128;
	bool valid = true;

	size_t bytes_query_vectors = d * sizeof(float);
	size_t bytes_fetched_vectors = d * sizeof(float);
	size_t bytes_out_dist = num_fetched_vectors_per_query * sizeof(float);

	// input vecs
	std::vector<float, aligned_allocator<float>> query_vectors(bytes_query_vectors / sizeof(float));
	std::vector<float, aligned_allocator<float>> fetched_vectors(bytes_fetched_vectors / sizeof(float));

	// output
	std::vector<float, aligned_allocator<float>> out_dist(bytes_out_dist / sizeof(float));
	std::vector<float, aligned_allocator<float>> sw_out_dist(bytes_out_dist / sizeof(float));

	// generate random query and fetched vectors
	for (int i = 0; i < d; i++) {
		query_vectors[i] = (float)rand();
		fetched_vectors[i] = (float)rand();
	}
	// compute software results: L2 distance square
	float dist = 0;
	for (int j = 0; j < d; j++) {
		float diff = query_vectors[j] - fetched_vectors[j];
		dist += diff * diff;
	}
	for (int i = 0; i < num_fetched_vectors_per_query; i++) {
		sw_out_dist[i] = dist;
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
	OCL_CHECK(err, cl::Buffer buffer_fetched_vectors (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			bytes_fetched_vectors, fetched_vectors.data(), &err));

	// out
	OCL_CHECK(err, cl::Buffer buffer_out_dist (context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
			bytes_out_dist, out_dist.data(), &err));

	std::cout << "Finish allocate buffer...\n";

	int arg_counter = 0;    
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_num)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(num_fetched_vectors_per_query)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(d)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, bool(valid)));

	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_query_vectors));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_fetched_vectors));
	
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out_dist));


    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
		buffer_query_vectors,
		buffer_fetched_vectors
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
	std::cout << "Comparing the results of the Device to the CPU...\n";
	bool overall_result = true;
	for (int i = 0; i < num_fetched_vectors_per_query; i++) {
		if (out_dist[i] == sw_out_dist[i]) {
			std::cout << "Match: " << out_dist[i] << " (hw) = " << sw_out_dist[i] << " (sw)" << std::endl;
		} else {
			std::cout << "Mismatch: " << out_dist[i] << " (hw) != " << sw_out_dist[i] << " (sw)" << std::endl;
			overall_result = false;
		}
	}
	if (overall_result) {
		std::cout << "Overall: Match" << std::endl;
	} else {
		std::cout << "Overall: Mismatch" << std::endl;
	}

    return  0;
}
