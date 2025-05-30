#pragma once

#include <sys/socket.h> 
#include "./hnswlib/hnswlib.h"
#include "utils.hpp"

#define TOPK 100

#define SEND_PKG_SIZE 1024 // 1024 
#define RECV_PKG_SIZE 4096 // 1024

void thread_HNSW(
    int query_num, int nprobe, 
    float* query_vectors_ptr, float* vector_quantizer_ptr, size_t D,
    hnswlib::AlgorithmInterface<float>* alg_hnswlib,
    int num_FPGA, int window_size, 
    int* start_send, int* finish_recv_query_id, int* finish_hnsw_id,
    char* FPGA_input,
    std::chrono::system_clock::time_point* query_start_time_ptr) { 
       
    printf("Printing from HNSW Thread\n"); 

    // ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib)->setEf(64);
    std::cout << "ef: " << ((hnswlib::HierarchicalNSW<float>*) alg_hnswlib)->ef_ << std::endl;

    // in runtime (should from network) in 512-bit packet
    size_t size_header = 1;
    size_t size_cell_IDs = nprobe * 4 % 64 == 0? nprobe * 4 / 64: nprobe * 4 / 64 + 1;
    size_t size_query_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 
    size_t size_center_vector = D * 4 % 64 == 0? D * 4 / 64: D * 4 / 64 + 1; 

    size_t FPGA_input_bytes = query_num * 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);
    int send_bytes_per_query = 64 * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector);

    std::vector<std::pair <float, int>> dist_array(nprobe); // (dist, cell_ID)

    // wait for send thread to be ready
    volatile int dummy_count = 0;
    while (true) { 
        // wait until all FPGAs are ready before starting
        bool start = true;
        for (int i = 0; i < num_FPGA; i++) {
            if (start_send[i] == 0) {
                start = false;
            }
        }
        if (start) {
            break;
        } else {
            dummy_count++;
        }
    }

    printf("Start running HNSW.\n");
    ////////////////   Data transfer + Select Cells   ////////////////

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (int query_id = 0; query_id < query_num; query_id++) {

        std::cout << "HNSW query_id " << query_id << std::endl;

        *(query_start_time_ptr + query_id) = std::chrono::system_clock::now();

        // select cells to scan
        size_t start_addr_FPGA_input_per_query = query_id * (size_header + size_cell_IDs + size_query_vector + nprobe * size_center_vector) * 64;
        size_t start_addr_FPGA_input_cell_ID = start_addr_FPGA_input_per_query + size_header * 64;
        size_t start_addr_FPGA_input_query_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs) * 64;
        size_t start_addr_FPGA_input_center_vector = start_addr_FPGA_input_per_query + (size_header + size_cell_IDs + size_query_vector) * 64;

        void* p = query_vectors_ptr + query_id * D;
        // searchKNN return type: std::priority_queue<std::pair<dist_t, labeltype >>
        auto gd = alg_hnswlib->searchKnn(p, nprobe);
        assert(gd.size() == nprobe);
        int cnt = 0;
        while (!gd.empty()) {
            dist_array[cnt] = std::make_pair(gd.top().first, int(gd.top().second));
            gd.pop();
            cnt++;
        }

        // write cell ID
        for (size_t nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {
            memcpy(&FPGA_input[start_addr_FPGA_input_cell_ID + nprobe_id * sizeof(int)], &dist_array[nprobe_id].second, sizeof(int));
#ifdef DEBUG
            // std::cout << "dist: " << dist_array[nprobe_id].first << " cell ID: " << dist_array[nprobe_id].second << "\n";
#endif
        } 

        // write query vector
	    memcpy(&FPGA_input[start_addr_FPGA_input_query_vector], query_vectors_ptr + query_id * D, D * sizeof(float));

        // write center vectors      
        for (size_t nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {

            size_t cell_ID = dist_array[nprobe_id].second;
	        memcpy(&FPGA_input[start_addr_FPGA_input_center_vector + nprobe_id * D * sizeof(float)], 
                   vector_quantizer_ptr + cell_ID * D, D * sizeof(float));
        }        

        *finish_hnsw_id = query_id;

        ///// Wenqi: send several queries in advance
        volatile int dummy_count = 0;
        while (true) { 
            // wait until all FPGAs are ready before starting
            bool start = true;
            for (int i = 0; i < num_FPGA; i++) {
                if (finish_recv_query_id[i] < query_id - (window_size - 1)) {
                    start = false;
                }
            }
            if (start) {
                break;
            } else {
                dummy_count++;
            }
        }
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    
    std::cout << "HNSW side Duration (us) = " << durationUs << std::endl;
    std::cout << "HNSW side QPS = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;

    std::cout << "HNSW side finished." << std::endl;
    return; 
} 


void thread_send_packets(
    const char* IP_addr, unsigned int send_port, 
    int num_FPGA, int thread_id, int query_num, int nprobe, int send_bytes_per_query,
    char* FPGA_input, int* finish_hnsw_id, int* finish_send_query_id, int* start_recv, int* start_send) { 

    printf("Printing send_port %d from Thread %d\n", send_port, thread_id); 

    // wait for recv thread to be ready
    volatile int dummy_count = 0;
    while (true) { 
        // wait until all FPGAs are ready before starting
        bool start = true;
        for (int i = 0; i < num_FPGA; i++) {
            if (start_recv[i] == 0) {
                start = false;
            }
        }
        if (start) {
            break;
        } else {
            dummy_count++;
        }
    } 
    
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

    printf("Start sending data send_port %d from Thread %d\n", send_port, thread_id); 
    start_send[thread_id] = 1;
    ////////////////   Data transfer   ////////////////

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (int query_id = 0; query_id < query_num; query_id++) {

        volatile int dummy_count = 0;
        // wait for HNSW thread
        while(*finish_hnsw_id < query_id) {  
            dummy_count++;
        }

        std::cout << "thread " << thread_id << " send query_id " << query_id << std::endl;

        // send data
        int total_sent_bytes = 0;

        while (total_sent_bytes < send_bytes_per_query) {
            int send_bytes_this_iter = (send_bytes_per_query - total_sent_bytes) < SEND_PKG_SIZE? (send_bytes_per_query - total_sent_bytes) : SEND_PKG_SIZE;
            int sent_bytes = send(sock, &FPGA_input[query_id * send_bytes_per_query + total_sent_bytes], send_bytes_this_iter, 0);
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
        finish_send_query_id[thread_id] = query_id;
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    
    std::cout << "Send side thread " << thread_id << " Duration (us) = " << durationUs << std::endl;
    std::cout << "Send side " << thread_id << " QPS = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;

    std::cout << "Send side " << thread_id << " finished." << std::endl;
    return; 
} 


void thread_recv_packets(
    unsigned int recv_port, int num_FPGA, int thread_id, int query_num, int recv_bytes_per_query, char* out_buf,
    int* start_recv, int* finish_send_query_id, int* finish_recv_query_id) { 

    printf("Printing recv_port %d from Thread %d\n", recv_port, thread_id); 

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
    printf("Successfully built connectionrecv_port %d from Thread %d\n", recv_port, thread_id); 
    start_recv[thread_id] = 1; // set shared register


    printf("Start receiving data recv_port %d from Thread %d\n", recv_port, thread_id); 
    ////////////////   Data transfer   ////////////////


    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now(); // reset after recving the first query

    for (int query_id = 0; query_id < query_num; query_id++) {
 
        // wait for send thread, if sock begins early then send, it will occupy socket resource,
        //   and other threads recv might not easy to get in (very slow)
        volatile int dummy_count = 0;
        while(finish_send_query_id[thread_id] < query_id) {  
            dummy_count++;
        }

        std::cout << "thread " << thread_id << " recv query_id " << query_id << std::endl;
        int total_recv_bytes = 0;
        while (total_recv_bytes < recv_bytes_per_query) {
            int recv_bytes_this_iter = (recv_bytes_per_query - total_recv_bytes) < RECV_PKG_SIZE? (recv_bytes_per_query - total_recv_bytes) : RECV_PKG_SIZE;
            int recv_bytes = read(sock, out_buf + (num_FPGA * query_id + thread_id) * recv_bytes_per_query + total_recv_bytes, recv_bytes_this_iter);
            total_recv_bytes += recv_bytes;
            
            if (recv_bytes == -1) {
                printf("Receiving data UNSUCCESSFUL!\n");
                return;
            }
#ifdef DEBUG
            else {
                std::cout << "thread " << thread_id << " query_id: " << query_id << " recv_bytes" << total_recv_bytes << std::endl;
            }
#endif
//             if (recv_bytes > 0) {
//                 // set shared register as soon as the first packet of the results is received
//                 *finish_recv_query_id = query_id; 
// #ifdef DEBUG
//                 std::cout << "set finish_recv_query_id: " << query_id  << std::endl;
// #endif
//             }
        }
        // set shared register as soon as the first packet of the results is received
        finish_recv_query_id[thread_id] = query_id; 

        if (total_recv_bytes != recv_bytes_per_query) {
            printf("Receiving error, receiving more bytes than a block\n");
        }
        if (query_id == 0) {
            // start counting when the first recv finished
            start = std::chrono::system_clock::now();
        }
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

    std::cout << "Recv side " << thread_id << " Duration (us) = " << durationUs << std::endl;
    if (query_num > 1) {
        std::cout << "Recv side " << thread_id << " QPS = " << (query_num - 1) / (durationUs / 1000.0 / 1000.0) << std::endl;
    } 
    std::cout << "Recv side " << thread_id << " Finished." << std::endl;

    return; 
} 

void thread_result_gather(
    int num_FPGA, int query_num, int*finish_recv_query_id,
    std::chrono::system_clock::time_point* query_finish_time_ptr) {
    // gather the results per thread
    // Save the result recv time


    for (int query_id = 0; query_id < query_num; query_id++) {

        volatile int dummy_count = 0;
        while (true) { 
            // wait until all FPGAs are ready before starting
            bool start = true;
            for (int i = 0; i < num_FPGA; i++) {
                if (finish_recv_query_id[i] < query_id) {
                    start = false;
                }
            }
            if (start) {
                break;
            } else {
                dummy_count++;
            }
        }

        *(query_finish_time_ptr + query_id) = std::chrono::system_clock::now();
    }
}