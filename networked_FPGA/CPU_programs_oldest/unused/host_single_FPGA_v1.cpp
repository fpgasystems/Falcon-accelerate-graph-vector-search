// host_single_FPGA_v1: the basic implementation
//   2 thread, 1 for sending query, 1 for receiving results
//   compute all cell to scan before the query sending thread (by manual implementation of brute-force scan)


// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c
// Usage (e.g.): ./host_single_FPGA 10.253.74.24 8881 5001

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
#include <algorithm>

#define D 128
#define TOPK 100

#define SEND_PKG_SIZE 17088 // 1024 
// #define SEND_PKG_SIZE 4096 // 1024 
#define RECV_PKG_SIZE 4096 // 1024

// #define DEBUG


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

void thread_send_packets(
    const char* IP_addr, unsigned int send_port, int query_num, int send_bytes_per_query, char* in_buf,
    int send_recv_query_gap, int* start_recv, int* finish_recv_query_id); 

void thread_recv_packets(
    unsigned int send_port, int query_num, int recv_bytes_per_query, char* out_buf,
    int* start_recv, int* finish_recv_query_id); 
    

int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Init     //////////
    
    std::cout << "Usage: " << argv[0] << " <Tx IP_addr> <Tx send_port> <Rx recv_port>" << std::endl;

    const char* IP_addr;
    if (argc >= 2)
    {
        IP_addr = argv[1];
    } else {
        // IP_addr = "10.253.74.5"; // alveo-build-01
        // IP_addr = "10.253.74.16"; // alveo-u250-02
        // IP_addr = "10.253.74.20"; // alveo-u250-03
        IP_addr = "10.253.74.24"; // alveo-u250-04
    }

    unsigned int send_port = 8888;
    if (argc >= 3)
    {
        send_port = strtol(argv[2], NULL, 10);
    } 

    unsigned int recv_port = 5001;
    if (argc >= 4)
    {
        recv_port = strtol(argv[3], NULL, 10);
    } 
    
    std::string db_name = "SIFT1000M"; // SIFT100M

    size_t query_num = 10000;
    size_t nlist = 32768;
    size_t nprobe = 32;

    // how many queries are allow to send ahead of receiving results
    // e.g., when send_recv_query_gap = 1, query 2 can be sent without receiving result 1
    // e.g., when send_recv_query_gap = 1, result 1 must be received before query 2 can be sent
    int send_recv_query_gap = 5; 

    assert (nprobe <= nlist);

    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t FPGA_input_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    int send_bytes_per_query = 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    std::cout << "send_bytes_per_query: " << send_bytes_per_query << std::endl;

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

    std::string data_dir_prefix;
    if (db_name == "SIFT100M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF32768,PQ32";
    } else if (db_name == "SIFT1000M") {
        data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT1000M_IVF32768,PQ32";
    }
    std::string gnd_dir = "/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/";

    ///////////     get data size from disk     //////////
    
    // info used to Select Cells to Scan
    std::string query_vectors_dir_suffix("query_vectors_float32_10000_128_raw");
    std::string query_vectors_dir = dir_concat(data_dir_prefix, query_vectors_dir_suffix);
    std::ifstream query_vectors_fstream(
        query_vectors_dir, 
        std::ios::in | std::ios::binary);
    query_vectors_fstream.seekg(0, query_vectors_fstream.end);
    size_t query_vectors_size =  query_vectors_fstream.tellg();
    if (!query_vectors_size) std::cout << "query_vectors_size is 0!";
    query_vectors_fstream.seekg(0, query_vectors_fstream.beg);
    
    std::string vector_quantizer_dir_suffix("vector_quantizer_float32_32768_128_raw");
    std::string vector_quantizer_dir = dir_concat(data_dir_prefix, vector_quantizer_dir_suffix);
    std::ifstream vector_quantizer_fstream(
        vector_quantizer_dir, 
        std::ios::in | std::ios::binary);
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.end);
    size_t vector_quantizer_size =  vector_quantizer_fstream.tellg();
    if (!vector_quantizer_size) std::cout << "vector_quantizer_size is 0!";
    vector_quantizer_fstream.seekg(0, vector_quantizer_fstream.beg);

    // ground truth 
    std::string raw_gt_vec_ID_suffix_dir;
    if (db_name == "SIFT100M") {
        raw_gt_vec_ID_suffix_dir = "idx_100M.ivecs";
    } else if (db_name == "SIFT1000M") {
        raw_gt_vec_ID_suffix_dir = "idx_1000M.ivecs";
    }
    std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
    std::ifstream raw_gt_vec_ID_fstream(
        raw_gt_vec_ID_dir,
        std::ios::in | std::ios::binary);

    std::string raw_gt_dist_suffix_dir;
    if (db_name == "SIFT100M") {
        raw_gt_dist_suffix_dir = "dis_100M.fvecs";
    } else if (db_name == "SIFT1000M") {
        raw_gt_dist_suffix_dir = "dis_1000M.fvecs";
    }
    std::string raw_gt_dist_dir = dir_concat(gnd_dir, raw_gt_dist_suffix_dir);
    std::ifstream raw_gt_dist_fstream(
        raw_gt_dist_dir,
        std::ios::in | std::ios::binary);

    //////////     Allocate Memory     //////////

    // input data
    char* in_buf = new char[FPGA_input_bytes];
    memset(in_buf, 0, FPGA_input_bytes);

    // on host side, used to Select Cells to Scan 
    std::vector<float, aligned_allocator<float>> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float, aligned_allocator<float>> vector_quantizer(vector_quantizer_size / sizeof(float));

    // output data
    char* out_buf = new char[out_bytes];
    memset(out_buf, 0, out_bytes);
    
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
    std::cout << "Loading data from disk...\n";

    // on host, used to Select Cells to Scan
    char* query_vectors_char = (char*) malloc(query_vectors_size);
    query_vectors_fstream.read(query_vectors_char, query_vectors_size);
    if (!query_vectors_fstream) {
            std::cout << "error: only " << query_vectors_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&query_vectors[0], query_vectors_char, query_vectors_size);
    free(query_vectors_char);
    
    char* vector_quantizer_char = (char*) malloc(vector_quantizer_size);
    vector_quantizer_fstream.read(vector_quantizer_char, vector_quantizer_size);
    if (!vector_quantizer_fstream) {
            std::cout << "error: only " << vector_quantizer_fstream.gcount() << " could be read";
        exit(1);
    }
    memcpy(&vector_quantizer[0], vector_quantizer_char, vector_quantizer_size);
    free(vector_quantizer_char);

    assert(D * nlist * sizeof(float) == vector_quantizer_size);

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

    //////////     Select Cells to Scan     //////////
    std::cout << "selecting cells to scan..." << std::endl;

    auto start_LUT = std::chrono::high_resolution_clock::now();

    std::vector<std::pair <float, int>> dist_array(nlist); // (dist, cell_ID)

    // select cells to scan
    for (size_t query_id = 0; query_id < query_num; query_id++) {

        for (size_t nlist_id = 0; nlist_id < nlist; nlist_id ++) {

            float dist = 0;
            for (size_t d = 0; d < D; d++) {
                dist += 
                    (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]) *
                    (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]);
            }
            dist_array[nlist_id] = std::make_pair(dist, nlist_id);
            // std::cout << "dist: " << dist << " cell ID: " << nlist_id << "\n";
        }
        
        size_t start_addr_FPGA_input_per_query = query_id * send_bytes_per_query;
        size_t start_addr_FPGA_input_cell_ID = start_addr_FPGA_input_per_query + size_header * 64;
        size_t start_addr_FPGA_input_query_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs) * 64;
        size_t start_addr_FPGA_input_center_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs + size_query_vector) * 64;

        // select cells
        std::nth_element(dist_array.begin(), dist_array.begin() + nprobe, dist_array.end()); // get nprobe smallest
        std::sort(dist_array.begin(), dist_array.begin() + nprobe); // sort first nprobe
            
        // write cell ID
        for (size_t n = 0; n < nprobe; n++) {
            int cell_ID = dist_array[n].second;
            memcpy(&in_buf[start_addr_FPGA_input_cell_ID + n * sizeof(int)], &cell_ID, sizeof(int));
#ifdef DEBUG
            std::cout << "dist: " << dist_array[n].first << " cell ID: " << dist_array[n].second << "\n";
#endif
        } 

        // write query vector
	    memcpy(&in_buf[start_addr_FPGA_input_query_vector], &query_vectors[query_id * D], D * sizeof(float));

        // write center vectors      
        for (size_t nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            int cell_ID = dist_array[nprobe_id].second;
	        memcpy(&in_buf[start_addr_FPGA_input_center_vector + nprobe_id * D * sizeof(float)], 
                   &vector_quantizer[cell_ID * D], D * sizeof(float));
        }        
    }
    auto end_LUT = std::chrono::high_resolution_clock::now();
    double duration_LUT = (std::chrono::duration_cast<std::chrono::milliseconds>(end_LUT - start_LUT).count());

    std::cout << "Duration Select Cells to Scan: " << duration_LUT << " ms" << std::endl; 

    //////////     Networking Part     //////////

    // inter-thread communication by shared memory
    int start_recv = 0; 
    int finish_recv_query_id = -1;

    std::thread t_recv(
        thread_recv_packets, recv_port, query_num, recv_bytes_per_query, out_buf,
        &start_recv, &finish_recv_query_id);

    std::thread t_send(
        thread_send_packets, IP_addr, send_port, query_num, send_bytes_per_query, in_buf,
        send_recv_query_gap, &start_recv, &finish_recv_query_id);

    t_recv.join();
    t_send.join();

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
            // std::cout << "hw: " << hw_result_vec_ID_partial[k] << " gt: " << gt_vec_ID[query_id] << 
            //     "hw dist: " << hw_result_dist_partial[k] << " gt dist: " << gt_dist[query_id] << std::endl;
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
    unsigned int recv_port, int query_num, int recv_bytes_per_query, char* out_buf,
    int* start_recv, int* finish_recv_query_id) { 

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
    *start_recv = 1; // set shared register


    printf("Start receiving data.\n");
    ////////////////   Data transfer   ////////////////


    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.

    auto start = std::chrono::high_resolution_clock::now(); // reset after recving the first query

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
        *finish_recv_query_id = query_id; // set shared register
        if (query_id == 0) {
            // start counting when the first recv finished
            start = std::chrono::high_resolution_clock::now();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

    std::cout << "Recv side Duration (us) = " << durationUs << std::endl;
    if (query_num > 1) {
        std::cout << "Recv side QPS () = " << (query_num - 1) / (durationUs / 1000.0 / 1000.0) << std::endl;
    } 

    return; 
} 

void thread_send_packets(
    const char* IP_addr, unsigned int send_port, int query_num, int send_bytes_per_query, char* in_buf,
    int send_recv_query_gap, int* start_recv, int* finish_recv_query_id) { 
       
    volatile int dummy_count = 0;
    while(!(*start_recv)) { 
        dummy_count++;
    }
 
    printf("Printing send_port from Thread %d\n", send_port); 
    
    int sock = 0; 
    struct sockaddr_in serv_addr; 

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(send_port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    //if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    //if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
    if(inet_pton(AF_INET, IP_addr, &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return; 
    } 

    printf("Start sending data.\n");
    ////////////////   Data transfer   ////////////////

    auto start = std::chrono::high_resolution_clock::now();

    for (int query_id = 0; query_id < query_num; query_id++) {

        std::cout << "query_id " << query_id << std::endl;
        int total_sent_bytes = 0;

        while (total_sent_bytes < send_bytes_per_query) {
            int send_bytes_this_iter = (send_bytes_per_query - total_sent_bytes) < SEND_PKG_SIZE? (send_bytes_per_query - total_sent_bytes) : SEND_PKG_SIZE;
            int sent_bytes = send(sock, &in_buf[query_id * send_bytes_per_query + total_sent_bytes], send_bytes_this_iter, 0);
            total_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return;
            }
#ifdef DEBUG
        else {
            printf("total sent bytes = %d\n", total_sent_bytes);
        }
#endif
        }

        if (total_sent_bytes != send_bytes_per_query) {
            printf("Sending error, sending more bytes than a block\n");
        }

        volatile int dummy_count = 0;
        while((*finish_recv_query_id) < query_id - send_recv_query_gap) {  ///// Wenqi: send 5 queries in advance
            dummy_count++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    
    std::cout << "Send side Duration (us) = " << durationUs << std::endl;
    std::cout << "Send side QPS () = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;

    return; 
} 
