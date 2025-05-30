// FPGA_simulator: simulate the behavior for a single FPGA
//   2 thread, 1 for sending results, 1 for receiving queries

// Usage (e.g.): ./FPGA_simulator 10.253.74.5 5001 8881 128 100
//  "Usage: " << argv[0] << " <1 Tx (CPU) CPU_IP_addr> <2 Tx F2C_port> <3 Rx C2F_port> <4 D>"

// Network order:
//   Open host_single_thread (CPU) first
//   then open FPGA simulator -> the FPGA will establish socket con for both F2C and C2F threads
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
#include <semaphore>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <netinet/tcp.h>

#include "constants.hpp"
// #include "my_semaphore.hpp"
#include "utils.hpp"

#define MAX_BATCH_NUM (1000 * 1000)
#define MAX_C2F_BUF_SIZE (10 * 1000 * 1000)

class FPGARetriever {

/* 
    FPGA input (C2F) format:
    // Format: foe each query
    // packet 0: header (batch_size, nprobe, terminate)
    //   for the following packets, for each query
    // 		packet 1~k: cell_IDs to scan -> size = ceil(nprobe * 4 / 64) 
    // 		packet k~n: query_vectors
    // 		packet n~m: center_vectors

    FPGA output (F2C) format:
    // Format: for each query
    // packet 0: header (FPGA_TOPK)
    // packet 1~k: FPGA_TOPK results, including vec_ID (8-byte) array and dist_array (4-byte)
    //    -> size = ceil(FPGA_TOPK * 8 / 64) + ceil(FPGA_TOPK * 4 / 64)
*/

public:

    const size_t D;  
    const size_t max_batch_num; // need to allocate a buffer to store the header states

    const char* CPU_IP_addr; 
    const unsigned int F2C_port; // FPGA send, CPU receive
    const unsigned int C2F_port; // FPGA recv, CPU send
    
    // states during data transfer
    int finish_F2C_query_id;
    int finish_C2F_query_id;
    bool start_F2C;
    bool start_C2F;
    int batch_size; 
    int nprobe;		
    int terminate;	
    int C2F_batch_id;
    int F2C_batch_id;

    // available results to send
      sem_t sem_available_results; // once receiving a query, ++, once send out a result, --
    sem_t sem_available_batches; // once start to receiving a batch, ++, once start to sending out a batch, --
    // MySemaphore sem_available_results;
    // MySemaphore sem_available_batches;

    // bit & byte const
    const size_t bit_int = 32; 
    const size_t bit_float = 32; 
    const size_t bit_long_int = 64; 
    const size_t bit_AXI = 512; 
    
    const size_t byte_AXI = 64; 

    // size in number of 512-bit AXI data packet
    //   C2F
    size_t AXI_size_C2F_header;
    size_t AXI_size_C2F_cell_IDs;
    size_t AXI_size_C2F_query_vector; 
    size_t AXI_size_C2F_center_vector; 
    //   F2C
    size_t AXI_size_F2C_header;
    size_t AXI_size_F2C_vec_ID;
    size_t AXI_size_F2C_dist;
    size_t AXI_size_F2C;

    // size in bytes
    size_t bytes_C2F_header;
    size_t bytes_F2C_per_query;
    size_t bytes_C2F_per_query;

    // header states
    char* buf_header;

    // C2F & F2C buffers, length = single query
    char* buf_F2C; 
    char* buf_C2F;

    // constructor
    FPGARetriever(
        const size_t in_D,
        const size_t in_max_batch_num,
        const char* in_CPU_IP_addr,
        const unsigned int in_F2C_port,
        const unsigned int in_C2F_port) :
        D(in_D), max_batch_num(in_max_batch_num), CPU_IP_addr(in_CPU_IP_addr), 
        F2C_port(in_F2C_port), C2F_port(in_C2F_port) {

        finish_F2C_query_id = -1;
        finish_C2F_query_id = -1;
        start_F2C = false;
        start_C2F = false;
        batch_size = -1;
        nprobe = -1;
        terminate = 0;	
        C2F_batch_id = -1;
        F2C_batch_id = -1;

        sem_init(&sem_available_results, 0, 0); // 0 = share between threads of a process
        sem_init(&sem_available_batches, 0, 0); // 0 = share between threads of a process
        // sem_available_results = MySemaphore(0);
        // sem_available_batches = MySemaphore(0);

        // C2F sizes
        AXI_size_C2F_header = 1;
        AXI_size_C2F_query_vector = D * bit_float % bit_AXI == 0? D * bit_float / bit_AXI : D * bit_float / bit_AXI + 1; 
        AXI_size_C2F_center_vector = D * bit_float % bit_AXI == 0? D * bit_float / bit_AXI : D * bit_float / bit_AXI + 1; 

        // F2C sizes
        AXI_size_F2C_header = 1;
        AXI_size_F2C_vec_ID = FPGA_TOPK * bit_long_int % bit_AXI == 0?
            FPGA_TOPK * bit_long_int / bit_AXI : FPGA_TOPK * bit_long_int / bit_AXI + 1;
        AXI_size_F2C_dist = FPGA_TOPK * bit_float % bit_AXI == 0?
            FPGA_TOPK * bit_float / bit_AXI : FPGA_TOPK * bit_float / bit_AXI + 1;
        AXI_size_F2C = AXI_size_F2C_header + AXI_size_F2C_vec_ID + AXI_size_F2C_dist; 

        bytes_C2F_header = AXI_size_F2C_header * byte_AXI;
        bytes_F2C_per_query = byte_AXI * AXI_size_F2C;
        std::cout << "bytes_F2C_per_query: " << bytes_F2C_per_query << std::endl;

        buf_header = (char*) malloc(MAX_BATCH_NUM * byte_AXI);

        buf_F2C = (char*) malloc(bytes_F2C_per_query);
        buf_C2F = (char*) malloc(MAX_C2F_BUF_SIZE);
    }

    void thread_C2F() { 

        printf("Printing C2F_port from Thread %d\n", C2F_port); 

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
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0) {
            perror("setsockopt(SO_REUSEADDR) failed");
        }
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0) {
            perror("setsockopt(SO_REUSEPORT) failed");
        }
        // send sock, set immediately send out small msg: https://stackoverflow.com/questions/32274907/why-does-tcp-socket-slow-down-if-done-in-multiple-system-calls
        int yes = 1;
        if (setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int))) {
            perror("setsockopt");
            exit(EXIT_FAILURE);
        }

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(C2F_port);

        // Forcefully attaching socket to the C2F_port 8080 
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

        std::cout << "C2F sock " << sock << std::endl; 
        start_C2F = true;
        // wait for ready
        while(!start_F2C) {}
        printf("Start C2F.\n");

        ////////////////   Data transfer   ////////////////


        // Should wait until the server said all the data was sent correctly,
        // otherwise the sender may send packets yet the server did not receive.

        while (true) {
                     
            // recv batch header 
            char header_buf[bytes_C2F_header];
            size_t header_C2F_bytes = 0;
            while (header_C2F_bytes < bytes_C2F_header) {
                int C2F_bytes_this_iter = bytes_C2F_header - header_C2F_bytes;
                int C2F_bytes = read(sock, header_buf + header_C2F_bytes, C2F_bytes_this_iter);
                header_C2F_bytes += C2F_bytes;
                if (C2F_bytes == -1) {
                    printf("Receiving data UNSUCCESSFUL!\n");
                    return;
                }
            }
            int batch_size = decode_int(header_buf);
            int nprobe = decode_int(header_buf + 4);
            int terminate = decode_int(header_buf + 8);
            memcpy(buf_header + (C2F_batch_id + 1) * byte_AXI, header_buf, bytes_C2F_header);
            C2F_batch_id++; // mark it as valid only when header buffer copy finished

            sem_post(&sem_available_batches);
            // sem_available_batches.produce();

            if (terminate) {
                break;
            }

            AXI_size_C2F_cell_IDs = nprobe * bit_int % bit_AXI == 0? nprobe * bit_int / bit_AXI: nprobe * bit_int / bit_AXI + 1;
            bytes_C2F_per_query = byte_AXI * (AXI_size_C2F_cell_IDs + AXI_size_C2F_query_vector + nprobe * AXI_size_C2F_center_vector); // not consider header 
            std::cout << "bytes_C2F_per_query (exclude 64-byte header) " << bytes_C2F_per_query << std::endl;
            assert(bytes_C2F_per_query <= MAX_C2F_BUF_SIZE);

            // recv the batch 
            for (int query_id = 0; query_id < batch_size; query_id++) {

                size_t total_C2F_bytes = 0;
                while (total_C2F_bytes < bytes_C2F_per_query) {
                    int C2F_bytes_this_iter = (bytes_C2F_per_query - total_C2F_bytes) < C2F_PKG_SIZE? (bytes_C2F_per_query - total_C2F_bytes) : C2F_PKG_SIZE;
                    int C2F_bytes = read(sock, buf_C2F + total_C2F_bytes, C2F_bytes_this_iter);
                    total_C2F_bytes += C2F_bytes;
                    
                    if (C2F_bytes == -1) {
                        printf("Receiving data UNSUCCESSFUL!\n");
                        return;
                    }
#ifdef DEBUG
                    else {
                        std::cout << "query_id: " << query_id << " C2F_bytes" << total_C2F_bytes << std::endl;
                    }
#endif
                }
                // set shared register as soon as the first packet of the results is received
                finish_C2F_query_id++; 
                std::cout << "finish_C2F_query_id: " << finish_C2F_query_id << std::endl;
                
                sem_post(&sem_available_results); // notify the F2C thread that there is a result to send
                // sem_available_results.produce();

                if (total_C2F_bytes != bytes_C2F_per_query) {
                    printf("Receiving error, receiving more bytes than a block\n");
                }
            }

            // reset state
            batch_size = -1;
            nprobe = -1;
            terminate = 0;	
        }

        return; // end C2F thread
    } 


    void thread_F2C() { 
        
        printf("Printing F2C_port from Thread %d\n", F2C_port); 

        int sock = 0; 
        struct sockaddr_in serv_addr; 

        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
        { 
            printf("\n Socket creation error \n"); 
            return; 
        } 
        int opt = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0) {
            perror("setsockopt(SO_REUSEADDR) failed");
        }
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0) {
            perror("setsockopt(SO_REUSEPORT) failed");
        }
        // send sock, set immediately send out small msg: https://stackoverflow.com/questions/32274907/why-does-tcp-socket-slow-down-if-done-in-multiple-system-calls
        int yes = 1;
        if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int))) {
            perror("setsockopt");
            exit(EXIT_FAILURE);
        }

        std::cout << "F2C sock " << sock << std::endl; 
    
        serv_addr.sin_family = AF_INET; 
        serv_addr.sin_port = htons(F2C_port); 
        
        // Convert IPv4 and IPv6 addresses from text to binary form 
        //if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
        //if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
        if(inet_pton(AF_INET, CPU_IP_addr, &serv_addr.sin_addr)<=0)  
        { 
            printf("\nInvalid address/ Address not supported \n"); 
            return; 
        } 
    
        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
        { 
            printf("\nConnection Failed \n"); 
            return; 
        } 

        start_F2C = true;
        while(!start_C2F) {}
        printf("Start F2C.\n");

        ////////////////   Data transfer + Select Cells   ////////////////


        while (true) {

            F2C_batch_id++;
            
            sem_wait(&sem_available_batches);
            // sem_available_batches.consume();
            char header_buf[byte_AXI];

            memcpy(header_buf, buf_header + F2C_batch_id * byte_AXI, bytes_C2F_header);
            int batch_size = decode_int(header_buf);
            int nprobe = decode_int(header_buf + 4);
            int terminate = decode_int(header_buf + 8);
            
            if (terminate) {
                break;
            }

            for (int query_id = 0; query_id < batch_size; query_id++) {

                sem_wait(&sem_available_results); // wait for available results
                // sem_available_results.consume();

                // send data
                size_t total_sent_bytes = 0;

                while (total_sent_bytes < bytes_F2C_per_query) {
                    int F2C_bytes_this_iter = (bytes_F2C_per_query - total_sent_bytes) < F2C_PKG_SIZE? (bytes_F2C_per_query - total_sent_bytes) : F2C_PKG_SIZE;
                    int sent_bytes = send(sock, buf_F2C + total_sent_bytes, F2C_bytes_this_iter, 0);
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
                if (total_sent_bytes != bytes_F2C_per_query) {
                    printf("Sending error, sending more bytes than a block\n");
                }
                finish_F2C_query_id++;
                std::cout << "finish_F2C_query_id: " << finish_F2C_query_id << std::endl;
            }
        }
        return; // end F2C thread
    } 

    void start_C2F_F2C_threads() {

        // start thread with member function: https://stackoverflow.com/questions/10673585/start-thread-with-member-function
        std::thread t_F2C(&FPGARetriever::thread_F2C, this);
        std::thread t_C2F(&FPGARetriever::thread_C2F, this);

        t_F2C.join();
        t_C2F.join();
    }
};



int main(int argc, char const *argv[]) 
{ 
    //////////     Parameter Init     //////////
    
    std::cout << "Usage: " << argv[0] << " <1 CPU_IP_addr> <2 F2C_port> <3 C2F_port> <4 D>" << std::endl;

    std::cout << "The maximum number of batches the FPGA simulator can handle: " << MAX_BATCH_NUM << std::endl;

    const char* CPU_IP_addr;
    if (argc >= 2)
    {
        CPU_IP_addr = argv[1];
    } else {
        CPU_IP_addr = "10.253.74.5"; // alveo-build-01
    }

    unsigned int F2C_port = 5008; 
    if (argc >= 3)
    {
        F2C_port = strtol(argv[2], NULL, 10);
    } 

    unsigned int C2F_port = 8888; 
    if (argc >= 4)
    {
        C2F_port = strtol(argv[3], NULL, 10);
    } 

    size_t D = 128;
    if (argc >= 5)
    {
        D = strtol(argv[4], NULL, 10);
    } 
    std::cout << "D: " << D << std::endl;


    FPGARetriever retriever(
        D,
        MAX_BATCH_NUM, 
        CPU_IP_addr,
        F2C_port,
        C2F_port);

    retriever.start_C2F_F2C_threads();

    return 0; 
} 

