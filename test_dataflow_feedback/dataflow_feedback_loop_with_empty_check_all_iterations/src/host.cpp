#include <stdint.h>
#include <chrono>

#include "host.hpp"

#include "constants.hpp"
// Wenqi: seems 2022.1 somehow does not support linking ap_uint.h to host?
// #include "ap_uint.h"


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

    int max_level_A = 2;
    int max_level_B = 2;
    int root_id_A = 0;
    int root_id_B = 0;

    // in init
    int root_children_num = 10;
    size_t page_num_A = 1 + root_children_num; // root and first level children
    size_t page_num_B = 1 + root_children_num; // root and first level children

    size_t bytes_page_A = page_num_A * PAGE_SIZE;
    size_t bytes_page_B = page_num_B * PAGE_SIZE;

    std::cout << "bytes per page: " << PAGE_SIZE << std::endl;
    std::vector<int ,aligned_allocator<int>> in_pages_A(bytes_page_A);
    std::vector<int ,aligned_allocator<int>> in_pages_B(bytes_page_B);

    // size_t out_bytes = 10 * 1024 * 1024;
    size_t layer_cache_bytes = 1 * size_t(1000) * size_t(1000) * size_t(1000); // no more than 16 GB
    std::cout << "layer_cache_bytes: " << layer_cache_bytes << std::endl;
    std::vector<int64_t ,aligned_allocator<int64_t>> layer_cache(layer_cache_bytes / sizeof(int64_t));

    // size_t out_bytes = 10 * 1024 * 1024;
    size_t out_bytes = 4 * size_t(1000) * size_t(1000) * size_t(1000); // no more than 16 GB
    std::cout << "out_bytes: " << out_bytes << std::endl;
    std::vector<int64_t ,aligned_allocator<int64_t>> out(out_bytes / sizeof(int64_t));

    // set page contents
    // The first 64-bytes per page:
    // typedef struct {
    //     // 7 * 4 bytes = 28 bytes
    //     int is_leaf;  // bool 
    //     int count;    // valid items
    //     obj_t obj;    // id/ptr + mbr
    // } node_meta_t;
    
    // root 
    in_pages_A[0] = 0; // is_leaf
    in_pages_A[1] = MAX_PAGE_ENTRIES; // valid items
    in_pages_A[2] = 0; // id items

    
    in_pages_B[0] = 0; // is_leaf
    in_pages_B[1] = MAX_PAGE_ENTRIES; // valid items
    in_pages_B[2] = 0; // id items

    // data nodes
    for (int i = 1; i < page_num_A; i++) {
        int bias = i * PAGE_SIZE / sizeof(int);
        in_pages_A[bias] = 1; // is_leaf
        in_pages_A[bias + 1] = MAX_PAGE_ENTRIES; // valid items
        in_pages_A[bias + 2] = i; // id
    }
    for (int i = 1; i < page_num_B; i++) {
        int bias = i * PAGE_SIZE / sizeof(int);
        in_pages_B[bias] = 1; // is_leaf
        in_pages_B[bias + 1] = MAX_PAGE_ENTRIES; // valid items
        in_pages_B[bias + 2] = i; // id
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
    // in init 
    OCL_CHECK(err, cl::Buffer buffer_in_pages_A   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            bytes_page_A, in_pages_A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in_pages_B   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            bytes_page_B, in_pages_B.data(), &err));
    // in & out
    OCL_CHECK(err, cl::Buffer buffer_layer_cache(context,CL_MEM_USE_HOST_PTR, 
            layer_cache_bytes, layer_cache.data(), &err));
	// out
    OCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            out_bytes, out.data(), &err));

    std::cout << "Finish allocate buffer...\n";

    int arg_counter = 0;    
    // in 
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_level_A)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(max_level_B)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(root_id_A)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, int(root_id_B)));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_in_pages_A));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_in_pages_B));

    // in & out
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_layer_cache));

    // out
    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_out));


    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        // in
        // buffer_in_pages_A,
        // buffer_in_pages_B,
        buffer_layer_cache,
        buffer_out
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

    std::cout << "Intersect pair number: " << out[0] << std::endl;
    std::cout << "Overall page per second = " << (page_num_A * page_num_B) / duration << std::endl;

    std::cout << "TEST FINISHED (NO RESULT CHECK)" << std::endl; 

    return  0;
}
