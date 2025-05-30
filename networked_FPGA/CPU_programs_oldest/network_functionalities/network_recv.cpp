// Reference: https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_recv.c

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
#include <cassert>
#include <iostream>

#define RECV_PKG_SIZE 4096 
// #define RECV_PKG_SIZE 4096 

#define DEBUG

void thread_recv_packets(unsigned int port, int recv_bytes); 

int main(int argc, char const *argv[]) 
{ 
    int recv_bytes = 10000 * 1344; // 1024 * 128; // the number of bytes to be received
    // unsigned int port = 8880;
    unsigned int port = 8888;

    std::thread th0(thread_recv_packets, port, recv_bytes);
    
    th0.join();

    return 0; 
} 

void thread_recv_packets(unsigned int port, int recv_bytes) { 

    printf("Printing Port from Thread %d\n", port); 

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
    address.sin_port = htons(port);

    // Forcefully attaching socket to the port 8080 
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
    char* recv_buf = (char*) malloc(recv_bytes);

    auto start = std::chrono::high_resolution_clock::now();

    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.

    int total_recv_bytes = 0;
    while (total_recv_bytes < recv_bytes) {
        int recv_bytes_this_iter = (recv_bytes - total_recv_bytes) < RECV_PKG_SIZE? (recv_bytes - total_recv_bytes) : RECV_PKG_SIZE;
    	int recv_bytes = read(sock, recv_buf + total_recv_bytes, recv_bytes_this_iter);
    	//int recv_bytes = read(sock, recv_buf + total_recv_bytes, recv_bytes - total_recv_bytes);
        total_recv_bytes += recv_bytes;
        if (recv_bytes == -1) {
            printf("Receiving data UNSUCCESSFUL!\n");
            return;
        }
#ifdef DEBUG
	else {
	    printf("totol received bytes: %d\n", total_recv_bytes);
	}
#endif
    }

    if (total_recv_bytes != recv_bytes) {
        printf("Receiving error, receiving more bytes than a block\n");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    printf("durationUs:%f\n",durationUs);
    printf("Transfer Throughput: %f GB / sec\n", recv_bytes / (durationUs / 1000.0 / 1000.0) / (1024.0 * 1024.0 * 1024.0)); 

    // for (int i = 0; i < recv_bytes; i++) {
    //     if(recv_buf[i] != i%128) {
    //         std::cout << "i = " << i << " content = " << int(recv_buf[i]) << std::endl;
    //     }
    // }
} 
