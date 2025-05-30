// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c

// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
#include <chrono>
#include <thread>

//#define SEND_BYTES 1
#define QUERY_NUM 100
#define SEND_BYTES 17088 // the number of bytes to be send per query

#define PORT 8888
// #define PORT 5002

#define DEBUG

struct Thread_info {
    int port;
    char* IP_addr;
};

// A normal C function that is executed as a thread  
void *thread_send_packets(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 
    

    int sock = 0, valread; 
    struct sockaddr_in serv_addr; 

    char* send_buf = (char*) malloc(SEND_BYTES);
    //for (int i = 0; i < BLOCK_ENTRY_NUM; i++) {
    //    send_buf[i] = 1;
    //}

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return 0; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    //if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    //if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)  
    if(inet_pton(AF_INET, t_info -> IP_addr, &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return 0; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return 0; 
    } 

    printf("Start sending data.\n");
    ////////////////   Data transfer   ////////////////
    int i = 0;

    clock_t start = clock();


    for (int query_id = 0; query_id < QUERY_NUM; query_id++) {

        int total_sent_bytes = 0;

        while (total_sent_bytes < SEND_BYTES) {
            int send_bytes_this_iter = (SEND_BYTES - total_sent_bytes) < 4096? (SEND_BYTES - total_sent_bytes) : 4096;
            int sent_bytes = send(sock, send_buf + total_sent_bytes, send_bytes_this_iter, 0);
            total_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return 0;
            }
#ifdef DEBUG
        else {
            printf("total sent bytes = %d\n", total_sent_bytes);
        }
#endif
        }

        if (total_sent_bytes != SEND_BYTES) {
            printf("Sending error, sending more bytes than a block\n");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100) );
    }

    clock_t end = clock();

    return NULL; 
} 

int main(int argc, char const *argv[]) 
{ 

    pthread_t thread_id_0; 
    //pthread_t thread_id_1; 
    printf("Before Thread\n"); 

    struct Thread_info t_info_0;
    //struct Thread_info t_info_1;
    t_info_0.port = PORT;
    // t_info_0.IP_addr = "10.253.74.5"; // alveo-build-01
    // t_info_0.IP_addr = "10.253.74.16"; // alveo-u250-02
    // t_info_0.IP_addr = "10.253.74.20"; // alveo-u250-03
    t_info_0.IP_addr = "10.253.74.24"; // alveo-u250-04
    //t_info_1.port = PORT + 1;

    pthread_create(&thread_id_0, NULL, thread_send_packets, (void*) &t_info_0); 
    //pthread_create(&thread_id_1, NULL, thread_send_packets, (void*) &t_info_1); 

    pthread_join(thread_id_0, NULL); 
    //pthread_join(thread_id_1, NULL); 
    printf("After Thread\n"); 

    return 0; 
} 