/* Host CPU communicates with one or multiple FPGAs and one GPU cooridnator,
  it forward the query & centroid vectors and received from the GPU coordinator, and broadcast them to the FPGAs

  3 threads: 
     1 for receiving query (G2C)
    1 for sending query (C2F)
    1 for receiving and forwarding results (F2C & C2G)
  

 Usage (e.g.):

  std::cout << "Usage: " << argv[0] << 
  " <L=1 G2C_port> "
    " <L=1 num_FPGA> "
    " <L=num_FPGA FPGA_IP_addrs> " 
    " <L=num_FPGA C2F_ports> " 
    " <L=num_FPGA F2C_ports> "
    " <L=1 D> <L=1 CPU_TOPK> <L=1 batch_size> "
    " <L=1 total_batch_num> <L=1 nprobe> "
    " <L=1 query_window_size> <L=1 batch_window_size>" << std::endl;

  Single FPGA example:
     ./CPU_cooridnator_for_GPU_FPGA 9091 1 10.253.74.24 8881 5001 128 100 32 100 32 10 4 
  
  Two FPGAs example:
     ./CPU_cooridnator_for_GPU_FPGA 9091 2 10.253.74.24 10.253.74.28 8881 8882 5001 5002 128 100 32 100 32 10 4 
*/

#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <semaphore>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <netinet/tcp.h>

#include "constants.hpp"
#include "ring_buffer.hpp"
// #include "hnswlib_omp/hnswlib.h"

// #define DEBUG // uncomment to activate debug print-statements

#ifdef DEBUG
#define IF_DEBUG_DO(x)                                                                                                                                         \
  do {                                                                                                                                                         \
    x                                                                                                                                                          \
  } while (0)
#else
#define IF_DEBUG_DO(x)                                                                                                                                         \
  do {                                                                                                                                                         \
  } while (0)
#endif

#define G2C_C2F_QUEUE_SIZE 10 // size of producer-consumer-queue buffer between G2C and C2F
#define F2C_C2G_QUEUE_SIZE 10 // size of producer-consumer-queue buffer between F2C and C2G

#define MAX_FPGA_NUM 16

class CPUCoordinator {

public:
  // parameters
  const size_t D;
  const size_t CPU_TOPK;
  const int batch_size;
  const int total_batch_num; // total number of batches to send
  int total_query_num;       // total number of numbers of queries
  const int nprobe;
  const int query_window_size; // gap between query IDs of C2F and F2C
  const int batch_window_size; // whether enable inter-batch pipeline overlap (0 = low latency; 1 = hgih throughput)

  const int num_FPGA; // <= MAX_FPGA_NUM

  // arrays of FPGA IP addresses and ports
  const char** FPGA_IP_addrs;
  const unsigned int* C2F_ports; // FPGA recv, CPU send
  const unsigned int* F2C_ports; // FPGA send, CPU receive
  const unsigned int G2C_port; // GPU send/recv, CPU send/recv

  // states during data transfer
  int start_G2C; // signal that index thread has finished initialization
  int start_F2C; // signal that F2C thread has finished setup connection
  int start_C2F; // signal that C2F thread has finished setup connection

  // semaphores to keep track of how many batches have been sent to FPGA and how many more we can send before the query_window_size is exeeded
  // used by index thread & F2C thread to control index scan rate:
  sem_t sem_batch_window_free_slots; // available slots in the batch window, cnt = batch_window_size - (C2F_batch_id - F2C_batch_id)
  // used by F2C thread & C2F thread to control send rate:
  sem_t sem_query_window_free_slots; // available slots in the query window, cnt = query_window_size - (C2F_query_id - F2C_query_id)
  // used by index thread & C2F thread to control send rate:
  sem_t sem_available_batches_to_send; // index scanned, yet not sent batches

//   unsigned int C2F_send_index;
//   unsigned int F2C_rcv_index;

  int C2F_batch_id;        // signal until what batch of data has been sent to FPGA
  int finish_C2F_query_id; 
  int finish_F2C_query_id;

  // size in bytes
  size_t bytes_C2F_header;
  size_t bytes_F2C_per_query; // expected bytes received per query including header
  size_t bytes_F2C_per_batch;
  size_t bytes_C2F_body_per_query;
  size_t bytes_C2F_per_batch;
  size_t bytes_G2C_per_batch;
  size_t bytes_C2G_per_query;
  size_t bytes_C2G_per_batch;

  /* An illustration of the semaphore logics:
  
  sem_batch_window_free_slots is used for constraint the total amount of queries flowing between CPU and FPGA:
    the query receive from G2C should not happen much earlier before the last batch is sent to FPGA,
      controlled by batch_window_size

  query_window_size is used for constraint communication:
    the F2C thread should not send much earlier before the last query is sent to FPGA,
      controlled by query_window_size

  batch_window_size is used to track how many computed batches are not sent to FPGA yet,
    determining when the C2F thread can fetch data to send
  
  ------------------------------
  |      G2C thread            |      |
  ------------------------------      |
    | sem_available_batches_to_send   |
    v                                 |
  ------------------------------      |
  |      C2F thread            |      |  sem_batch_window_free_slots
  ------------------------------      |
    | sem_query_window_free_slots     |
    v                                 |
  ------------------------------      |
  |      F2C & C2G thread      |      |
  ------------------------------      v
  */

  // variables used for connections
  int* sock_c2f;
  int* sock_f2c;
  int sock_g2c; // G2C & C2G share the same sock

  // C2F & F2C buffers, length = single batch of queries including padding
  char* buf_F2C;
  char* buf_C2F;
  char* buf_G2C;
  char* buf_C2G;

  RingBuffer* ring_buffer_G2C;

  // terminate signal to be sent in the packet header to FPGA
  // it is primarily part of a payload but it is also used as a signal as it affects the control flow
  // TODO: should no longer be used for signaling
  int terminate;

  std::chrono::system_clock::time_point* batch_start_time_array;
  std::chrono::system_clock::time_point* batch_finish_time_array;
  // end-to-end performance
  double* batch_duration_ms_array;
  double QPS;

  // constructor
  CPUCoordinator(
    const size_t in_D,
    const size_t in_CPU_TOPK, 
    const int in_batch_size,
    const int in_total_batch_num,
    const int in_nprobe,
    const int in_query_window_size,
    const int in_batch_window_size,
    const int in_num_FPGA,
    const char** in_FPGA_IP_addrs,
    const unsigned int* in_C2F_ports,
    const unsigned int* in_F2C_ports,
    const unsigned int in_G2C_port) :
    D(in_D), CPU_TOPK(in_CPU_TOPK), batch_size(in_batch_size), total_batch_num(in_total_batch_num),
    nprobe(in_nprobe), query_window_size(in_query_window_size), batch_window_size(in_batch_window_size),
    num_FPGA(in_num_FPGA), FPGA_IP_addrs(in_FPGA_IP_addrs), C2F_ports(in_C2F_ports), F2C_ports(in_F2C_ports), G2C_port(in_G2C_port) {
        
    // Initialize internal variables
    total_query_num = batch_size * total_batch_num;

    start_G2C = 0;
    start_F2C = 0;
    start_C2F = 0;
    finish_C2F_query_id = -1; 
    finish_F2C_query_id = -1;
    // C2F_send_index = 0;
    // F2C_rcv_index = 0;
    terminate = 0;
    C2F_batch_id = -1;
    // F2C_batch_id = -1;
    // F2C_batch_finish = 1; // set to one to allow first C2F iteration run
    sem_init(&sem_query_window_free_slots, 0, query_window_size); // 0 = share between threads of a process
    sem_init(&sem_batch_window_free_slots, 0, batch_window_size); // 0 = share between threads of a process
    sem_init(&sem_available_batches_to_send, 0, 0); // 0 = share between threads of a process

    // C2F sizes
    bytes_C2F_header = num_packages::AXI_size_C2F_header * bit_byte_const::byte_AXI;
    bytes_C2F_body_per_query = bit_byte_const::byte_AXI * (num_packages::AXI_size_C2F_cell_IDs(nprobe) + num_packages::AXI_size_C2F_query_vector(D) +
                                                      nprobe * num_packages::AXI_size_C2F_center_vector(D)); // not consider header
    bytes_C2F_per_batch = bytes_C2F_header + batch_size * bytes_C2F_body_per_query;
    std::cout << "bytes_C2F_per_batch: " << bytes_C2F_per_batch << std::endl;

    std::cout << "bytes_C2F_body_per_query (exclude 64-byte header): " << bytes_C2F_body_per_query << std::endl;

    // F2C sizes
    size_t AXI_size_F2C = num_packages::AXI_size_F2C_header + num_packages::AXI_size_F2C_vec_ID(FPGA_TOPK) + num_packages::AXI_size_F2C_dist(FPGA_TOPK);
    bytes_F2C_per_query = bit_byte_const::byte_AXI * AXI_size_F2C; // expected bytes received per query including header
    bytes_F2C_per_batch = bytes_F2C_per_query * batch_size;
    // std::cout << "bytes_F2C_per_query: " << bytes_F2C_per_query << std::endl;
    std::cout << "bytes_F2C_per_batch: " << bytes_F2C_per_batch << std::endl;

    // G2C sizes
    // header: 16 bytes = batch_size, dim, nprobe, k, all in int32, queries: batch_size * dim * 4 bytes (float32), list_IDs: batch_size * nprobe * 8 bytes (int64)
    bytes_G2C_per_batch = 16 + bit_byte_const::byte_float * batch_size * D + bit_byte_const::byte_long_int * batch_size * nprobe;
    bytes_C2G_per_query = bit_byte_const::byte_long_int * CPU_TOPK + bit_byte_const::byte_float * CPU_TOPK;
    bytes_C2G_per_batch = batch_size * bytes_C2G_per_query;

    sock_f2c = (int*) malloc(num_FPGA * sizeof(int));
    sock_c2f = (int*) malloc(num_FPGA * sizeof(int));

    buf_F2C = (char*) malloc(bytes_F2C_per_query);
    buf_C2F = (char*) malloc(bytes_C2F_body_per_query);
    buf_G2C = (char*) malloc(bytes_C2F_per_batch);
    buf_C2G = (char*) malloc(bytes_C2G_per_batch);
    std::cout << "bytes_C2G_per_batch: " << bytes_C2G_per_batch << std::endl;

	ring_buffer_G2C = new RingBuffer(bytes_G2C_per_batch, G2C_C2F_QUEUE_SIZE);

    batch_start_time_array = (std::chrono::system_clock::time_point*) malloc(in_total_batch_num * sizeof(std::chrono::system_clock::time_point));
    batch_finish_time_array = (std::chrono::system_clock::time_point*) malloc(in_total_batch_num * sizeof(std::chrono::system_clock::time_point));
    batch_duration_ms_array = (double*) malloc(in_total_batch_num * sizeof(double));

    assert (in_num_FPGA < MAX_FPGA_NUM);
  }

  void connect_G2C() {

    std::cout << "Should finish F2C first before G2C" << std::endl;  
    printf("Printing G2C_ports %d\n", G2C_port);

    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
    }
    // send sock, set immediately send out small msg: https://stackoverflow.com/questions/32274907/why-does-tcp-socket-slow-down-if-done-in-multiple-system-calls
    int yes = 1;
    if (setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int))) {
      perror("setsockopt");
      exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0) {
        perror("setsockopt(SO_REUSEADDR) failed");
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0) {
        perror("setsockopt(SO_REUSEPORT) failed");
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(G2C_port);

    // Forcefully attaching socket to the F2C_ports 8080
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
      perror("bind failed");
      exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
      perror("listen");
      exit(EXIT_FAILURE);
    }
    if ((sock_g2c = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
      perror("accept");
      exit(EXIT_FAILURE);
    }
    
    printf("Successfully built G2C connection.\n");
  }

  void receive_G2C_batch_query() {

    size_t total_G2C_bytes = 0;
    while (total_G2C_bytes < bytes_G2C_per_batch) {
      int G2C_bytes_this_iter = (bytes_G2C_per_batch - total_G2C_bytes) < G2C_PKG_SIZE ? (bytes_G2C_per_batch - total_G2C_bytes) : G2C_PKG_SIZE;
      int G2C_bytes = read(sock_g2c, &buf_G2C[total_G2C_bytes], G2C_bytes_this_iter);
      total_G2C_bytes += G2C_bytes;

      if (G2C_bytes == -1) {
        printf("receive_G2C_batch_query Receiving data UNSUCCESSFUL!\n");
        return;
      } else {
        // IF_DEBUG_DO(std::cout << "query_id: " << query_id << " G2C_bytes" << total_G2C_bytes << std::endl;);
      }
    }

    if (total_G2C_bytes != bytes_G2C_per_batch) {
      printf("Receiving error, receiving more bytes than a block\n");
    }      
	// move to ring buffer
	ring_buffer_G2C->write_slot(buf_G2C);
  }

  void thread_G2C() {

    connect_G2C();
    start_G2C = 1;
    while(!start_C2F) {}

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (int g2c_batch_id = 0; g2c_batch_id < total_batch_num; g2c_batch_id++) {

      std::cout << "g2c_batch_id: " << g2c_batch_id << std::endl;
      
      sem_wait(&sem_batch_window_free_slots);
      batch_start_time_array[g2c_batch_id] = std::chrono::system_clock::now();
      receive_G2C_batch_query();
      sem_post(&sem_available_batches_to_send);
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    
    std::cout << "G2C side Duration (us) = " << durationUs << std::endl;
    std::cout << "G2C side QPS () = " << total_query_num / (durationUs / 1000.0 / 1000.0) << std::endl;
    std::cout << "G2C side finished." << std::endl;
    
    return; 
  }


  void connect_C2F() {

    for (int n = 0; n < num_FPGA; n++) {
      
      printf("Printing C2F_ports from C2F thread %d\n", C2F_ports[n]);

      struct sockaddr_in serv_addr;

      if ((sock_c2f[n] = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return;
      }
      // send sock, set immediately send out small msg: https://stackoverflow.com/questions/32274907/why-does-tcp-socket-slow-down-if-done-in-multiple-system-calls
      int yes = 1;
      int opt = 1;
      if (setsockopt(sock_c2f[n], IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
      }
      if (setsockopt(sock_c2f[n], SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int)) < 0) {
          perror("setsockopt(SO_REUSEADDR) failed");
      }
      if (setsockopt(sock_c2f[n], SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int)) < 0) {
          perror("setsockopt(SO_REUSEPORT) failed");
      }

      serv_addr.sin_family = AF_INET;
      serv_addr.sin_port = htons(C2F_ports[n]);

      // Convert IPv4 and IPv6 addresses from text to binary form
      // if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
      // if(inet_pton(AF_INET, "10.1.212.153", &serv_addr.sin_addr)<=0)
      if (inet_pton(AF_INET, FPGA_IP_addrs[n], &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return;
      }

      if (connect(sock_c2f[n], (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return;
      }
      std::cout << "C2F sock: " << sock_c2f[n] << std::endl;
    }

    printf("Start sending data.\n");
  }

  // C2F send batch header
  void send_C2F_header(char *buf_header) {
    for (int n = 0; n < num_FPGA; n++) {
      size_t sent_header_bytes = 0;
      while (sent_header_bytes < bytes_C2F_header) {
        int C2F_bytes_this_iter = (bytes_C2F_header - sent_header_bytes) < C2F_PKG_SIZE ? (bytes_C2F_header - sent_header_bytes) : C2F_PKG_SIZE;
        int C2F_bytes = send(sock_c2f[n], &buf_header[sent_header_bytes], C2F_bytes_this_iter, 0);
        sent_header_bytes += C2F_bytes;
        if (C2F_bytes == -1) {
          printf("Sending data UNSUCCESSFUL!\n");
          return;
        }
      }
    }
  }

  // C2F send a single query
  void send_C2F_query() {
    for (int n = 0; n < num_FPGA; n++) {
      size_t total_C2F_bytes = 0;
      while (total_C2F_bytes < bytes_C2F_body_per_query) {
        int C2F_bytes_this_iter = (bytes_C2F_body_per_query - total_C2F_bytes) < C2F_PKG_SIZE ? (bytes_C2F_body_per_query - total_C2F_bytes) : C2F_PKG_SIZE;
        int C2F_bytes = send(sock_c2f[n], &buf_C2F[total_C2F_bytes], C2F_bytes_this_iter, 0);
        total_C2F_bytes += C2F_bytes;
        if (C2F_bytes == -1) {
          printf("Sending data UNSUCCESSFUL!\n");
          return;
        } else {
          IF_DEBUG_DO(std::cout << "total C2F bytes = " << total_C2F_bytes << std::endl;);
        }
      }
      if (total_C2F_bytes != bytes_C2F_body_per_query) {
        printf("Sending error, sending more bytes than a block\n");
      }
    }
  }

  /* This method is meant to be run in a separate thread.
   * It establishes and maintains a connection to the FPGA.
   * It is resposible for sending the queries to the FPGA.
   */
  void thread_C2F() { 
      
	char* G2C_conv_buf = (char*) malloc(bytes_G2C_per_batch); // load the G2C slot, and later convert to C2F format
    // wait for ready
    while(!start_F2C) {}
    while(!start_G2C) {}

    connect_C2F();

    start_C2F = 1;

    ////////////////   Data transfer + Select Cells   ////////////////

    // used to prepare the header data in the exact layout the FPGA expects
    char buf_header[bytes_C2F_header];

    // used to temporary store the complete batch of queries as received from GPU
    float *query_batch = (float *)malloc(batch_size * D * sizeof(float));

    // used to store the data in the exact format the FPGA expects
    // => the batch of queries is splited up into individual queries and interleaved with cell-ids and corresponsing center-vectors
    // additionally padding (to fill the AXI package) needs to be added after every part of the payload
    // [cell ids] [padding] [query vector] [padding] [center_vector 1] [padding] ... [center_vector nprobe] [padding]
    int n_bytes_cell_ids = num_packages::AXI_size_C2F_cell_IDs(nprobe) * bit_byte_const::byte_AXI;
    int n_bytes_query_vector = num_packages::AXI_size_C2F_query_vector(D) * bit_byte_const::byte_AXI;
    int n_bytes_center_vectors = nprobe * num_packages::AXI_size_C2F_center_vector(D) * bit_byte_const::byte_AXI;

    char *buf_cell_ids = (char *)malloc(n_bytes_cell_ids);
    char *buf_query = (char *)malloc(n_bytes_query_vector);
    char *buf_center_vectors = (char *)malloc(n_bytes_center_vectors);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (int C2F_batch_id = 0; C2F_batch_id < total_batch_num + 1; C2F_batch_id++) {

      std::cout << "C2F_batch_id: " << C2F_batch_id << std::endl;

      terminate = C2F_batch_id == total_batch_num? 1 : 0; 

      memcpy(buf_header, &batch_size, 4);
      memcpy(buf_header + 4, &nprobe, 4);
      memcpy(buf_header + 8, &terminate, 4);

      send_C2F_header(buf_header);

      if (terminate) {
        break;
      }
      
      sem_wait(&sem_available_batches_to_send);

	  ring_buffer_G2C->read_slot(G2C_conv_buf);

      for (int query_id = 0; query_id < batch_size; query_id++) {

        // this semaphore controls that the window size is adhered to
        // If the  semaphore currently has the value zero, then the call blocks
        //   until either it becomes possible to perform the decrement
        sem_wait(&sem_query_window_free_slots);


        float *current_query = query_batch + (query_id * D);
        // TODO: here, the data should be copied from somewhere -> from HNSW
        memcpy(buf_query, current_query, D * bit_byte_const::byte_float);
        // memcpy(buf_center_vectors, center_vectors, n_bytes_center_vectors);

		// header: 16 bytes = batch_size, dim, nprobe, k, all in int32, queries: batch_size * dim * 4 bytes (float32), list_IDs: batch_size * nprobe * 8 bytes (int64)
   		int G2C_per_batch_bias = 16 + bit_byte_const::byte_float * batch_size * D + bit_byte_const::byte_long_int * query_id * nprobe;
		for (int nprobe_id = 0; nprobe_id < nprobe; nprobe_id++) {
			int64_t cell_id = *(int64_t*)(G2C_conv_buf + G2C_per_batch_bias + nprobe_id * 8);
			int32_t cell_id_int32 = cell_id;
			std::cout << cell_id << " "; 
			memcpy(buf_C2F + 4 * nprobe_id, (void*) &cell_id_int32, bit_byte_const::byte_int);
		}

		// DUMMY query & center vectors
        memcpy(buf_C2F + n_bytes_cell_ids, buf_query, n_bytes_query_vector);
        memcpy(buf_C2F + n_bytes_cell_ids + n_bytes_query_vector, buf_center_vectors, n_bytes_center_vectors);
        send_C2F_query();
        finish_C2F_query_id++;
        std::cout << "C2F finish query_id " << finish_C2F_query_id << std::endl;
      }
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    
    std::cout << "C2F side Duration (us) = " << durationUs << std::endl;
    std::cout << "C2F side QPS () = " << total_query_num / (durationUs / 1000.0 / 1000.0) << std::endl;
    std::cout << "C2F side finished." << std::endl;
    
    return; 
  } 

  void connect_F2C() {

    std::cout << "FPGA programs must be started in order (same as the input argument) " <<
      " because the F2C receive side receives connections in order " << std::endl;

    for (int n = 0; n < num_FPGA; n++) {
      
      printf("Printing F2C_ports from Thread %d\n", F2C_ports[n]);

      int server_fd;
      struct sockaddr_in address;
      int opt = 1;
      int addrlen = sizeof(address);

      // Creating socket file descriptor
      if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
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
      address.sin_port = htons(F2C_ports[n]);

      // Forcefully attaching socket to the F2C_ports 8080
      if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
      }
      if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
      }
      if ((sock_f2c[n] = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
      }
      printf("Successfully built connection.\n");
    }
      printf("Successfully built all F2C connection.\n");
  }

  void receive_F2C_batch_result() {

    // TODO: each FPGA's msg should has its own buffer, not a shared buf_F2C
    for (int n = 0; n < num_FPGA; n++) {
      size_t total_F2C_bytes = 0;
      while (total_F2C_bytes < bytes_F2C_per_query) {
        int F2C_bytes_this_iter = (bytes_F2C_per_query - total_F2C_bytes) < F2C_PKG_SIZE ? (bytes_F2C_per_query - total_F2C_bytes) : F2C_PKG_SIZE;
        int F2C_bytes = read(sock_f2c[n], &buf_F2C[total_F2C_bytes], F2C_bytes_this_iter);
        total_F2C_bytes += F2C_bytes;

        if (F2C_bytes == -1) {
          printf("receive_F2C_batch_result Receiving data UNSUCCESSFUL!\n");
          return;
        } else {
          // IF_DEBUG_DO(std::cout << "query_id: " << query_id << " F2C_bytes" << total_F2C_bytes << std::endl;);
        }
      }

      if (total_F2C_bytes != bytes_F2C_per_query) {
        printf("Receiving error, receiving more bytes than a block\n");
      }
    }
  }

  void send_C2G_batch_result() {

    size_t sent_bytes = 0;
    while (sent_bytes < bytes_C2G_per_batch) {
      int C2G_bytes_this_iter = (bytes_C2G_per_batch - sent_bytes) < C2G_PKG_SIZE ? (bytes_C2G_per_batch - sent_bytes) : C2G_PKG_SIZE;
      int C2G_bytes = send(sock_g2c, &buf_C2G[sent_bytes], C2G_bytes_this_iter, 0);
      sent_bytes += C2G_bytes;
      if (C2G_bytes == -1) {
        printf("Sending data UNSUCCESSFUL!\n");
        return;
      }
    }
  }

  void thread_F2C_C2G() { 

    connect_F2C();
    start_F2C = 1;
    while(!start_C2F) {}
    ///////////////////////////////////////
    //// START RECEIVING DATA /////////////
    ///////////////////////////////////////

    printf("Start receiving data.\n");

    ////////////////   Data transfer   ////////////////

    // Should wait until the server said all the data was sent correctly,
    // otherwise the C2Fer may send packets yet the server did not receive.

    size_t n_bytes_top_k = batch_size * bit_byte_const::byte_int;
    size_t n_bytes_vector_ids = batch_size * FPGA_TOPK * (bit_byte_const::byte_long_int);
    size_t n_bytes_distances = batch_size * FPGA_TOPK * (bit_byte_const::byte_float);

    int *top_k_buf = (int *)malloc(n_bytes_top_k);
    long *vector_ids_buf = (long *)malloc(n_bytes_vector_ids);
    float *distances_buf = (float *)malloc(n_bytes_distances);

    size_t byte_offset_vector_ids_buf = num_packages::AXI_size_F2C_header * bit_byte_const::byte_AXI;
    size_t byte_offset_distances_buf = (num_packages::AXI_size_F2C_header + num_packages::AXI_size_F2C_vec_ID(FPGA_TOPK)) * bit_byte_const::byte_AXI;

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now(); // reset after recving the first query

    // TODO: find a good way to signal to this thread that terminate signal was received and when the last batch is received.
    for (int F2C_batch_id = 0; F2C_batch_id < total_batch_num; F2C_batch_id++) {

      std::cout << "F2C & C2G batch_id: " << F2C_batch_id << std::endl;

      for (int query_id = 0; query_id < batch_size; query_id++) {

        receive_F2C_batch_result();

        // Each query received consists of [header] [FPGA_TOPK x ids] [FPGA_TOPK x dists] all three padded to the next AXI packet size
        // We copy all answers into the same array such that the GPU can simply interpret it at as a 2D array
        memcpy(top_k_buf + query_id, buf_F2C, 4);
        memcpy(vector_ids_buf + query_id * FPGA_TOPK, buf_F2C + byte_offset_vector_ids_buf, FPGA_TOPK * (bit_byte_const::bit_long_int / 8));
        memcpy(distances_buf + query_id * FPGA_TOPK, buf_F2C + byte_offset_distances_buf, FPGA_TOPK * (bit_byte_const::bit_float / 8));
    
        // TODO: potentially also cp to the C2G buffer here

        finish_F2C_query_id++;
        std::cout << "F2C finish query_id " << finish_F2C_query_id << std::endl;
        // sem_post() increments (unlocks) the semaphore pointed to by sem
        sem_post(&sem_query_window_free_slots);
      }
      
      send_C2G_batch_result();
      std::cout << "G2C finish batch_id " << F2C_batch_id << std::endl;

      batch_finish_time_array[F2C_batch_id] = std::chrono::system_clock::now();
      sem_post(&sem_batch_window_free_slots);
    }

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

  std::cout << "F2C & C2G side Duration (us) = " << durationUs << std::endl;
  std::cout << "F2C & C2G side QPS = " << total_query_num / (durationUs / 1000.0 / 1000.0) << std::endl;
  std::cout << "F2C & C2G side Finished." << std::endl;

  return;  
  }

  void start_C2F_F2C_threads() {

    // start thread with member function: https://stackoverflow.com/questions/10673585/start-thread-with-member-function
    std::thread t_G2C(&CPUCoordinator::thread_G2C, this);
    std::thread t_F2C_C2G(&CPUCoordinator::thread_F2C_C2G, this);
    std::thread t_C2F(&CPUCoordinator::thread_C2F, this);

    t_G2C.join();
    t_F2C_C2G.join();
    t_C2F.join();
  }

  void calculate_latency() {

    std::vector<double> sorted_duration_ms;
    double total_ms = 0.0;

    for (int b = 0; b < total_batch_num; b++) {
      double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
        batch_finish_time_array[b] - batch_start_time_array[b]).count());
      double durationMs = durationUs / 1000.0;
      batch_duration_ms_array[b] = durationMs;
      sorted_duration_ms.push_back(durationMs);
      total_ms += durationMs;
    }
    double ave_ms = total_ms / total_batch_num;

    std::sort(sorted_duration_ms.begin(), sorted_duration_ms.end());
    std::cout << "Latency from batches: " << std::endl;
    std::cout << "  Min (ms): " << sorted_duration_ms.front() << std::endl;
    std::cout << "  Max (ms): " << sorted_duration_ms.back() << std::endl;
    std::cout << "  Medium (ms): " << sorted_duration_ms.at(total_batch_num / 2) << std::endl;
    std::cout << "  Average (ms): " << ave_ms << std::endl;


    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
      batch_finish_time_array[total_batch_num - 1] - batch_start_time_array[0]).count());
    double durationMs = durationUs / 1000.0;
    QPS = total_query_num / (durationUs / 1000.0 / 1000.0);
    std::cout << "End-to-end Duration (ms) = " << durationMs << std::endl;
    std::cout << "End-to-end QPS = " << QPS << std::endl;

    // write latency and throughput to file in double-precision
    FILE *file_latency = fopen("profile_latency_ms_distribution.double", "w");
    fwrite(batch_duration_ms_array, sizeof(double), total_batch_num, file_latency);
    fclose(file_latency);

    FILE *file_throughput = fopen("profile_QPS.double", "w");
    fwrite(&QPS, sizeof(double), 1, file_throughput);
    fclose(file_throughput);
  }
};


int main(int argc, char const *argv[]) 
{ 
  //////////     Parameter Init     //////////  
  std::cout << "Usage: " << argv[0] << 
  " <L=1 G2C_port> "
    " <L=1 num_FPGA> "
    " <L=num_FPGA FPGA_IP_addrs> " 
    " <L=num_FPGA C2F_ports> " 
    " <L=num_FPGA F2C_ports> "
    " <L=1 D> <L=1 CPU_TOPK> <L=1 batch_size> "
    " <L=1 total_batch_num> <L=1 nprobe> "
    " <L=1 query_window_size> <L=1 batch_window_size>" << std::endl;

  int argv_cnt = 1;
  std::cout << argc;

  unsigned int G2C_port = strtol(argv[argv_cnt++], NULL, 10);
  
  int num_FPGA = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "num_FPGA: " << num_FPGA << std::endl;
  assert(argc == 10 + 3 * num_FPGA);
  assert(num_FPGA <= MAX_FPGA_NUM);


  const char* FPGA_IP_addrs[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      FPGA_IP_addrs[n] = argv[argv_cnt++];
      std::cout << "FPGA " << n << " IP addr: " << FPGA_IP_addrs[n] << std::endl;
  }   
  // FPGA_IP_addrs = "10.253.74.5"; // alveo-build-01
  // FPGA_IP_addrs = "10.253.74.12"; // alveo-u250-01
  // FPGA_IP_addrs = "10.253.74.16"; // alveo-u250-02
  // FPGA_IP_addrs = "10.253.74.20"; // alveo-u250-03
  // FPGA_IP_addrs = "10.253.74.24"; // alveo-u250-04

  unsigned int C2F_ports[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      C2F_ports[n] = strtol(argv[argv_cnt++], NULL, 10);
      std::cout << "C2F_ports " << n << ": " << C2F_ports[n] << std::endl;
  } 

  unsigned int F2C_ports[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      F2C_ports[n] = strtol(argv[argv_cnt++], NULL, 10);
      std::cout << "F2C_ports " << n << ": " << F2C_ports[n] << std::endl;
  } 
    
  size_t D = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "D: " << D << std::endl;

  size_t CPU_TOPK = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "CPU_TOPK: " << CPU_TOPK << std::endl;

  int batch_size = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "batch_size: " << batch_size << std::endl;

  int total_batch_num = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "total_batch_num: " << total_batch_num << std::endl;

  int nprobe = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "nprobe: " << nprobe << std::endl;
    
  // how many queries are allow to send ahead of receiving results
  // e.g., when query_window_size = 1 (which is the min), query 2 cannot be sent without receiving result 1
  //          but might lead to a deadlock
  int query_window_size = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "query_window_size: " << query_window_size << 
    ", query window size controls the network communication pressure between CPU and FPGA (communication control)" << std::endl;
  assert (query_window_size >= 1);

  // 1 = high-throughput mode, allowing inter-batch pipeline
  // 0 = low latency mode, does not send data before the last batch is finished
  int batch_window_size = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "batch_window_size: " << batch_window_size << 
    ", batch window size controls how many batches can be computed for index scan in advance (compute control)" << std::endl;
  assert (batch_window_size >= 1);

    
  CPUCoordinator cpu_coordinator(
    D,
    CPU_TOPK, 
    batch_size,
    total_batch_num,
    nprobe,
    query_window_size,
    batch_window_size,
    num_FPGA,
    FPGA_IP_addrs,
    C2F_ports,
    F2C_ports,
  G2C_port);

  cpu_coordinator.start_C2F_F2C_threads();
  cpu_coordinator.calculate_latency();

  return 0; 
} 
