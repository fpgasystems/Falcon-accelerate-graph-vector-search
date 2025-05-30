// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c
// Usage (e.g.): ./query_send_thread 10.253.74.24 8888

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

#define SEND_PKG_SIZE 4096 // 1024 

#define DEBUG


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
    const char* IP_addr, unsigned int send_port, int query_num, int send_bytes_per_query, char* in_buf); 

int main(int argc, char const *argv[]) 
{ 


    //////////     Parameter Init     //////////

    
    std::cout << "Usage: " << argv[0] << " <Tx IP_addr> <Tx send_port>" << std::endl;

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
    

    size_t query_num = 100;
    size_t nlist = 32768;
    size_t nprobe = 32;

    assert (nprobe <= nlist);

    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t FPGA_input_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    int send_bytes_per_query = 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    std::cout << "send_bytes_per_query: " << send_bytes_per_query << std::endl;

    //////////     Data loading / computing Part     //////////

    std::string data_dir_prefix = "/mnt/scratch/wenqi/Faiss_Enzian_U250_index/SIFT100M_IVF32768,PQ32";

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

    //////////     Allocate Memory     //////////


    char* in_buf = new char[FPGA_input_bytes];
    memset(in_buf, 0, FPGA_input_bytes);

    // on host side, used to Select Cells to Scan 
    std::vector<float, aligned_allocator<float>> query_vectors(query_vectors_size / sizeof(float));
    std::vector<float, aligned_allocator<float>> vector_quantizer(vector_quantizer_size / sizeof(float));

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

    //////////     Select Cells to Scan     //////////
    std::cout << "selecting cells to scan..." << std::endl;

    auto start_LUT = std::chrono::high_resolution_clock::now();

//     std::vector<std::pair <float, int>> dist_array(nlist); // (dist, cell_ID)

//     // select cells to scan
//     for (size_t query_id = 0; query_id < query_num; query_id++) {

//         for (size_t nlist_id = 0; nlist_id < nlist; nlist_id ++) {

//             float dist = 0;
//             for (size_t d = 0; d < D; d++) {
//                 dist += 
//                     (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]) *
//                     (query_vectors[query_id * D + d] - vector_quantizer[nlist_id * D + d]);
//             }
//             dist_array[nlist_id] = std::make_pair(dist, nlist_id);
//             // std::cout << "dist: " << dist << " cell ID: " << nlist_id << "\n";
//         }
        
//         size_t start_addr_FPGA_input_per_query = query_id * send_bytes_per_query;
//         size_t start_addr_FPGA_input_cell_ID = start_addr_FPGA_input_per_query + size_header * 64;
//         size_t start_addr_FPGA_input_query_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs) * 64;
//         size_t start_addr_FPGA_input_center_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs + size_query_vector) * 64;

//         // select cells
//         std::nth_element(dist_array.begin(), dist_array.begin() + nprobe, dist_array.end()); // get nprobe smallest
//         std::sort(dist_array.begin(), dist_array.begin() + nprobe); // sort first nprobe
            
//         // write cell ID
//         for (size_t n = 0; n < nprobe; n++) {
//             int cell_ID = dist_array[n].second;
//             memcpy(&in_buf[start_addr_FPGA_input_cell_ID + n * sizeof(int)], &cell_ID, sizeof(int));
// #ifdef DEBUG
//             std::cout << "dist: " << dist_array[n].first << " cell ID: " << dist_array[n].second << "\n";
// #endif
//         } 

//         // write query vector
// 	    memcpy(&in_buf[start_addr_FPGA_input_query_vector], &query_vectors[query_id * D], D * sizeof(float));

//         // write center vectors      
//         for (size_t nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

//             int cell_ID = dist_array[nprobe_id].second;
// 	        memcpy(&in_buf[start_addr_FPGA_input_center_vector + nprobe_id * D * sizeof(float)], 
//                    &vector_quantizer[cell_ID * D], D * sizeof(float));
//         }        
//     }
    auto end_LUT = std::chrono::high_resolution_clock::now();
    double duration_LUT = (std::chrono::duration_cast<std::chrono::milliseconds>(end_LUT - start_LUT).count());

    std::cout << "Duration Select Cells to Scan: " << duration_LUT << " ms" << std::endl; 

#ifdef DEBUG
    int count_zero = 0;
    for (int i = 0; i < send_bytes_per_query * query_num; i++) {
        if (in_buf[i] == 0) { count_zero++; }
    }
    std::cout << "there are " << count_zero << " 0s in in_buf" << std::endl;
#endif

    //////////     Network Part     //////////
    
    std::thread th0(thread_send_packets, IP_addr, send_port, query_num, send_bytes_per_query, in_buf);

    th0.join();

    return 0; 
} 

// A normal C function that is executed as a thread  
void thread_send_packets(
    const char* IP_addr, unsigned int send_port, int query_num, int send_bytes_per_query, char* in_buf) { 
        
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

// #ifdef DEBUG
//     char* buf = new char[send_bytes_per_query];
// #endif 

    for (int query_id = 0; query_id < query_num; query_id++) {

        std::cout << "query_id " << query_id << std::endl;
// #ifdef DEBUG
//         memcpy(buf, &in_buf[query_id * send_bytes_per_query], send_bytes_per_query);
// #endif 
        // if (query_id >= 1) { 
        //     std::cout << "Set send buffer to 0" << std::endl; 
        //     memset(buf, 0, send_bytes_per_query); 
        // }
        int total_sent_bytes = 0;

        while (total_sent_bytes < send_bytes_per_query) {
            int send_bytes_this_iter = (send_bytes_per_query - total_sent_bytes) < SEND_PKG_SIZE? (send_bytes_per_query - total_sent_bytes) : SEND_PKG_SIZE;
            // int sent_bytes = send(sock, buf + total_sent_bytes, send_bytes_this_iter, 0);
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

        int sleep_ms = 100;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        std::cout << "sleep " << sleep_ms << " ms before next query" << std::endl;
    }

    return; 
} 
