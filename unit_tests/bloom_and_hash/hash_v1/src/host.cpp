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

uint32_t MurmurHash2 ( const void * key, int len, uint32_t seed )
{
  /* 'm' and 'r' are mixing constants generated offline.
     They're not really 'magic', they just happen to work well.  */

  const uint32_t m = 0x5bd1e995;
  const int r = 24;

  /* Initialize the hash to a 'random' value */

  uint32_t h = seed ^ len;

  /* Mix 4 bytes at a time into the hash */

  const unsigned char * data = (const unsigned char *)key;

  while(len >= 4)
  {
    uint32_t k = *(uint32_t*)data;

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    len -= 4;
  }

  /* Handle the last few bytes of the input array  */

  switch(len)
  {
  case 3: h ^= data[2] << 16;
  case 2: h ^= data[1] << 8;
  case 1: h ^= data[0];
      h *= m;
  };

  /* Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.  */

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
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
    int query_num = 10;
	int read_iter_per_query = 1000;
	int seed = 1;

	size_t bytes_mem_keys = read_iter_per_query * sizeof(int);
	size_t bytes_out = read_iter_per_query * sizeof(int);

	// input vecs
	std::vector<int, aligned_allocator<int>> mem_keys(bytes_mem_keys / sizeof(int));

	// output
	std::vector<int, aligned_allocator<int>> out(bytes_out / sizeof(int));
	std::vector<int, aligned_allocator<int>> sw_results(bytes_out / sizeof(int));

	// generate random query and fetched vectors
	for (int i = 0; i < read_iter_per_query; i++) {
		mem_keys[i] = i;
		sw_results[i] = MurmurHash2(&mem_keys[i], 4, seed);
	}

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
	OCL_CHECK(err, cl::Buffer buffer_mem_keys (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
			bytes_mem_keys, mem_keys.data(), &err));

	// out
	OCL_CHECK(err, cl::Buffer buffer_out (context,CL_MEM_USE_HOST_PTR,
			bytes_out, out.data(), &err));

	std::cout << "Finish allocate buffer...\n";

	int arg_counter = 0;    
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(query_num)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(read_iter_per_query)));
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(seed)));

	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_mem_keys));
	
	OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out));


    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
		buffer_mem_keys
        },0/* 0 means from host*/));

    std::cout << "Launching kernel...\n";
    // Launch the Kernel
    auto start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000.0);

    std::cout << "Duration (including memcpy out): " << duration << " sec" << std::endl; 

	// compare software and hardware results
	std::cout << "Comparing software and hardware results...\n";
	bool match = true;
	for (int i = 0; i < read_iter_per_query; i++) {
		if (sw_results[i] != out[i]) {
			std::cout << "Mismatch at " << i << ": " << sw_results[i] << " vs " << out[i] << std::endl;
			match = false;
		}
	}
	if (match) {
		std::cout << "Software and hardware results match." << std::endl;
	} else {
		std::cout << "Software and hardware results DO NOT match." << std::endl;
	}

	return  0;
}
