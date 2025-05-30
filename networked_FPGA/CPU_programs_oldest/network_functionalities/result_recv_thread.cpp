// Reference: https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_recv.c
// Usage (e.g.): ./result_recv_thread 5001

// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>


#define RECV_PKG_SIZE 4096 // 1024

#define TOPK 100

#define DEBUG

void thread_recv_packets(
    unsigned int recv_port, int query_num, int recv_bytes_per_query, char* out_buf);

template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};


// boost::filesystem does not compile well, so implement this myself
std::string dir_concat(std::string dir1, std::string dir2) {
    if (dir1.back() != '/') {
        dir1 += '/';
    }
    return dir1 + dir2;
}

int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Initialization     //////////

    
    std::cout << "Usage: " << argv[0] << "<Rx recv_port>" << std::endl;

    unsigned int recv_port = 5001;
    if (argc >= 2)
    {
        recv_port = strtol(argv[1], NULL, 10);
    } 

    int query_num = 100;

    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = TOPK * 64 % 512 == 0?
        TOPK * 64 / 512 : TOPK * 64 / 512 + 1;
    size_t size_results_dist = TOPK * 32 % 512 == 0?
        TOPK * 32 / 512 : TOPK * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet
    size_t out_bytes = query_num * 64 * size_results;
    int recv_bytes_per_query = 64 * size_results;

    std::cout << "recv_bytes_per_query: " << recv_bytes_per_query << std::endl;

    //////////     Data loading / computing Part     //////////

    std::string gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/";

    ///////////     get data size from disk     //////////
    
    // ground truth 
    std::string raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);

    std::string raw_gt_dist_suffix_dir = "dis_100M.fvecs";
    std::string raw_gt_dist_dir = dir_concat(gnd_dir, raw_gt_dist_suffix_dir);
    std::ifstream raw_gt_dist_fstream(
        raw_gt_dist_dir,
        std::ios::in | std::ios::binary);

    char* out_buf = new char[out_bytes];
    
    // the raw ground truth size is the same for idx_1M.ivecs, idx_10M.ivecs, idx_100M.ivecs
    // recall counts the very first nearest neighbor only
    size_t raw_gt_vec_ID_size = 10000 * 1001 * sizeof(int);
    size_t gt_vec_ID_size = 10000 * sizeof(int);
    std::vector<int, aligned_allocator<int>> raw_gt_vec_ID(raw_gt_vec_ID_size / sizeof(int), 0);
    std::vector<int, aligned_allocator<int>> gt_vec_ID(gt_vec_ID_size / sizeof(int), 0);
    
    size_t raw_gt_dist_size = 10000 * 1001 * sizeof(float);
    size_t gt_dist_size = 10000 * sizeof(float);
    std::vector<float, aligned_allocator<float>> raw_gt_dist(raw_gt_dist_size / sizeof(float), 0);
    std::vector<float, aligned_allocator<float>> gt_dist(gt_dist_size / sizeof(float), 0);

    //////////     load data from disk     //////////

    // ground truth
    char* raw_gt_vec_ID_char = (char*) malloc(raw_gt_vec_ID_size);
    raw_gt_vec_ID_fstream.read(raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    if (!raw_gt_vec_ID_fstream) {
        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_vec_ID[0], raw_gt_vec_ID_char, raw_gt_vec_ID_size);
    free(raw_gt_vec_ID_char);

    for (int i = 0; i < 10000; i++) {
        gt_vec_ID[i] = raw_gt_vec_ID[i * 1001 + 1];
    }

    char* raw_gt_dist_char = (char*) malloc(raw_gt_dist_size);
    raw_gt_dist_fstream.read(raw_gt_dist_char, raw_gt_dist_size);
    if (!raw_gt_dist_fstream) {
        std::cout << "error: only " << raw_gt_dist_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&raw_gt_dist[0], raw_gt_dist_char, raw_gt_dist_size);
    free(raw_gt_dist_char);

    for (int i = 0; i < 10000; i++) {
        gt_dist[i] = raw_gt_dist[i * 1001 + 1];
    }

    //////////     Networking Part     //////////
    std::thread th0(thread_recv_packets, recv_port, query_num, recv_bytes_per_query, out_buf);
    
    th0.join();

    //////////     Recall evaluation    //////////
    std::cout << "Comparing Results..." << std::endl;
    int count = 0;
    int match_count = 0;

    for (int query_id = 0; query_id < query_num; query_id++) {

#ifdef DEBUG
        std::cout << "query ID: " << query_id << std::endl;
#endif

        std::vector<long> hw_result_vec_ID_partial(TOPK, 0);
        std::vector<float> hw_result_dist_partial(TOPK, 0);

        int start_result_vec_ID_addr = (query_id * size_results + 1) * 64;
        int start_result_dist_addr = (query_id * size_results + 1 + size_results_vec_ID) * 64;

        // Load data
        memcpy(&hw_result_vec_ID_partial[0], &out_buf[start_result_vec_ID_addr], 8 * TOPK);
        memcpy(&hw_result_dist_partial[0], &out_buf[start_result_dist_addr], 4 * TOPK);
        
        // Check correctness
        count++;
        // std::cout << "query id" << query_id << std::endl;
        for (int k = 0; k < TOPK; k++) {
#ifdef DEBUG
            std::cout << "hw: " << hw_result_vec_ID_partial[k] << " gt: " << gt_vec_ID[query_id] << 
                "hw dist: " << hw_result_dist_partial[k] << " gt dist: " << gt_dist[query_id] << std::endl;
#endif 
            if (hw_result_vec_ID_partial[k] == gt_vec_ID[query_id]) {
                match_count++;
                break;
            }
        } 
    }
    float recall = ((float) match_count / (float) count);
    printf("\n=====  Recall: %.8f  =====\n", recall);


    return 0; 
} 




void thread_recv_packets(
    unsigned int recv_port, int query_num, int recv_bytes_per_query, char* out_buf) { 

    printf("Printing recv_port from Thread %d\n", recv_port); 

    int server_fd, sock;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR , &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(recv_port);

    // Forcefully attaching socket to the recv_port 8080 
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((sock = accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t*)&addrlen))<0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    printf("Successfully built connection.\n"); 


    printf("Start receiving data.\n");
    ////////////////   Data transfer   ////////////////

    auto start = std::chrono::high_resolution_clock::now();

    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.

    for (int query_id = 0; query_id < query_num; query_id++) {

        int total_recv_bytes = 0;
        while (total_recv_bytes < recv_bytes_per_query) {
            int recv_bytes_this_iter = (recv_bytes_per_query - total_recv_bytes) < RECV_PKG_SIZE? (recv_bytes_per_query - total_recv_bytes) : RECV_PKG_SIZE;
            int recv_bytes = read(sock, out_buf + query_id * recv_bytes_per_query + total_recv_bytes, recv_bytes_this_iter);
            total_recv_bytes += recv_bytes;
            if (recv_bytes == -1) {
                printf("Receiving data UNSUCCESSFUL!\n");
                return;
            }
#ifdef DEBUG
        else {
            std::cout << "query_id: " << query_id << " recv_bytes" << total_recv_bytes << std::endl;
        }
#endif
        }

        if (total_recv_bytes != recv_bytes_per_query) {
            printf("Receiving error, receiving more bytes than a block\n");
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
#ifdef DEBUG
    std::cout << "Duration (us) = " << durationUs << std::endl;
#endif

    return; 
} 
