// FPGA_simulator: simulate the behavior for a single FPGA
//   2 thread, 1 for sending results, 1 for receiving queries

// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c
// Usage (e.g.): ./FPGA_simulator 10.253.74.5 5001 8881 32
//  "Usage: " << argv[0] << " <Tx (CPU) IP_addr> <Tx send_port> <Rx recv_port> <nprobe>"

// Network order:
//   Open host_single_thread (CPU) first
//   then open FPGA simulator
//   FPGA open connection -> CPU recv -> CPU send query -> FPGA recv

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

#define SEND_PKG_SIZE 1024 // 1024 
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

void thread_send_results(
    const char* IP_addr, unsigned int send_port, int query_num, 
    int* start_send, int* finish_recv_query_id); 
    
void thread_recv_query(
    unsigned int recv_port, int query_num, int nprobe,
    int* start_send, int* finish_recv_query_id);

int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Init     //////////
    
    std::cout << "Usage: " << argv[0] << " <Tx (CPU) IP_addr> <Tx send_port> <Rx recv_port> <nprobe>" << std::endl;

    const char* IP_addr;
    if (argc >= 2)
    {
        IP_addr = argv[1];
    } else {
        IP_addr = "10.253.74.5"; // alveo-build-01
    }

    unsigned int send_port = 5008; // send out results
    if (argc >= 3)
    {
        send_port = strtol(argv[2], NULL, 10);
    } 

    unsigned int recv_port = 8888; // receive query
    if (argc >= 4)
    {
        recv_port = strtol(argv[3], NULL, 10);
    } 

    size_t nprobe = 1;
    if (argc >= 5)
    {
        nprobe = strtol(argv[4], NULL, 10);
    } 

    size_t query_num = 10000;
    size_t nlist = 32768;

    assert (nprobe <= nlist);

    
    //////////     Networking Part     //////////

    // inter-thread communication by shared memory
    int start_send = 0; 
    int finish_recv_query_id = -1;

    // profiler
    std::vector<std::chrono::system_clock::time_point> query_start_time(query_num);
    std::vector<std::chrono::system_clock::time_point> query_finish_time(query_num);

    std::thread t_send(
        thread_send_results, IP_addr, send_port, query_num,
        &start_send, &finish_recv_query_id);

    std::thread t_recv(
        thread_recv_query, recv_port, query_num, nprobe,
        &start_send, &finish_recv_query_id);


    t_send.join();
    t_recv.join();

    return 0; 
} 

void thread_send_results(
    const char* IP_addr, unsigned int send_port, int query_num, 
    int* start_send, int* finish_recv_query_id) { 
      
    // out
    // 128 is a random padding for network headers
    // header = 1 pkt
    size_t size_results_vec_ID = TOPK * 64 % 512 == 0?
        TOPK * 64 / 512 : TOPK * 64 / 512 + 1;
    size_t size_results_dist = TOPK * 32 % 512 == 0?
        TOPK * 32 / 512 : TOPK * 32 / 512 + 1;
    size_t size_results = 1 + size_results_vec_ID + size_results_dist; // in 512-bit packet


    size_t out_bytes = query_num * 64 * size_results;
    int result_bytes_per_query = 64 * size_results;

    std::cout << "result_bytes_per_query: " << result_bytes_per_query << std::endl;
    
    // output data
    char* out_buf = new char[out_bytes];
    memset(out_buf, 0, out_bytes); 

    printf("Printing send_port from Thread %d\n", send_port); 
    
    int sock = 0; 
    struct sockaddr_in serv_addr; 

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return; 
    } 
    std::cout << "send sock " << sock << std::endl; 
   
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
    *start_send = 1;

    ////////////////   Data transfer + Select Cells   ////////////////


    for (int query_id = 0; query_id < query_num; query_id++) {

        volatile int cnt = 0;
        while (query_id > *finish_recv_query_id) {
            cnt++;
            sleep(0.001);
        }

        std::cout << "send query_id " << query_id << std::endl;

        // send data
        int total_sent_bytes = 0;

        while (total_sent_bytes < result_bytes_per_query) {
            int send_bytes_this_iter = (result_bytes_per_query - total_sent_bytes) < SEND_PKG_SIZE? (result_bytes_per_query - total_sent_bytes) : SEND_PKG_SIZE;
            int sent_bytes = send(sock, &out_buf[query_id * result_bytes_per_query + total_sent_bytes], send_bytes_this_iter, 0);
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

        if (total_sent_bytes != result_bytes_per_query) {
            printf("Sending error, sending more bytes than a block\n");
        }
    }

    return; 
} 

void thread_recv_query(
    unsigned int recv_port, int query_num, int nprobe,
    int* start_send, int* finish_recv_query_id) { 

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

    std::cout << "recv sock " << sock << std::endl; 
    // wait for ready
    // volatile int dummy_count = 0;
    // while(!(*start_send)) { 
    //     dummy_count++;
    // }


    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t FPGA_input_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    int query_bytes = 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    std::cout << "query_bytes: " << query_bytes << std::endl;
    std::vector<char ,aligned_allocator<char >> FPGA_input(FPGA_input_bytes);

    printf("Start receiving data.\n");
    ////////////////   Data transfer   ////////////////


    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.


    for (int query_id = 0; query_id < query_num; query_id++) {

        std::cout << "recv query_id " << query_id << std::endl;
        int total_recv_bytes = 0;
        while (total_recv_bytes < query_bytes) {
            int recv_bytes_this_iter = (query_bytes - total_recv_bytes) < RECV_PKG_SIZE? (query_bytes - total_recv_bytes) : RECV_PKG_SIZE;
            int recv_bytes = read(sock, FPGA_input.data() + query_id * query_bytes + total_recv_bytes, recv_bytes_this_iter);
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
        // set shared register as soon as the first packet of the results is received
        *finish_recv_query_id = query_id; 

        if (total_recv_bytes != query_bytes) {
            printf("Receiving error, receiving more bytes than a block\n");
        }
    }

    return; 
} 
