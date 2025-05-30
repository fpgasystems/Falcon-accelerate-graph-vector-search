// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c

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

#define SEND_PKG_SIZE 64 
// #define SEND_PKG_SIZE 4096 

#define DEBUG


void thread_send_packets(const char* IP_addr, unsigned int port, int send_bytes); 


int main(int argc, char const *argv[]) 
{ 

    unsigned int port = 5008;
    int send_bytes = 10000 * 17088; // 1024 * 1024;
    const char* IP_addr = "10.253.74.5"; // alveo-build-01
    // const char* IP_addr = "10.253.74.16"; // alveo-u250-02
    // const char* IP_addr = "10.253.74.20"; // alveo-u250-03
    // const char* IP_addr = "10.253.74.24"; // alveo-u250-04


    std::thread th0(thread_send_packets, IP_addr, port, send_bytes);

    th0.join();

    return 0; 
} 

void thread_send_packets(const char* IP_addr, unsigned int port, int send_bytes) { 

    int sock = 0; 
    struct sockaddr_in serv_addr; 

    char* send_buf = (char*) malloc(send_bytes);

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
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


    int total_sent_bytes = 0;

    while (total_sent_bytes < send_bytes) {
	    int send_bytes_this_iter = (send_bytes - total_sent_bytes) < SEND_PKG_SIZE? (send_bytes - total_sent_bytes) : SEND_PKG_SIZE;
        int sent_bytes = send(sock, send_buf + total_sent_bytes, send_bytes_this_iter, 0);
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

    if (total_sent_bytes != send_bytes) {
        printf("Sending error, sending more bytes than a block\n");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    printf("durationUs:%f\n",durationUs);
    printf("Transfer Throughput: %f GB / sec\n", send_bytes / (durationUs / 1000.0 / 1000.0) / (1024.0 * 1024.0 * 1024.0)); 
} 
