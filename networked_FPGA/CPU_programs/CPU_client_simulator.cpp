/* 

Host CPU communicates with one or multiple FPGAs, can choose to use index or not.
   2 thread, 1 for sending query, 1 for receiving results

 Usage (e.g.):

  std::cout << "Usage: " << argv[0] << " <1 num_FPGA> "
      "<2 ~ 2 + num_FPGA - 1 FPGA_IP_addr> " 
    "<2 + num_FPGA ~ 2 + 2 * num_FPGA - 1 C2F_port> " 
    "<2 + 2 * num_FPGA ~ 2 + 3 * num_FPGA - 1 F2C_port> "
    "<2 + 3 * num_FPGA D> <3 + 3 * num_FPGA ef> " 
    "<4 + 3 * num_FPGA query_num> " "<5 + 3 * num_FPGA batch_size> "
    "<6 + 3 * num_FPGA query_window_size> <7 + 3 * num_FPGA batch_window_size> " 
*/

#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <semaphore.h>
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
#include "types.hpp"
#include "utils.hpp"

#define DEBUG // uncomment to activate debug print-statements

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

#define MAX_FPGA_NUM 16


class CPU_client_simulator {

public:
  // parameters
  const size_t D;
  const size_t ef;
  const int query_num;
  const int batch_size;
  int total_batch_num; // total number of batches to send
  const int query_window_size; // gap between query IDs of C2F and F2C
  const int batch_window_size; // whether enable inter-batch pipeline overlap (0 = low latency; 1 = hgih throughput)

  const int num_FPGA; // <= MAX_FPGA_NUM

  // arrays of FPGA IP addresses and ports
  const char** FPGA_IP_addr;
  const unsigned int* C2F_port; // FPGA recv, CPU send
  const unsigned int* F2C_port; // FPGA send, CPU receive

  // states during data transfer
  int start_F2C; // signal that F2C thread has finished setup connection
  int start_C2F; // signal that C2F thread has finished setup connection

  // semaphores to keep track of how many batches have been sent to FPGA and how many more we can send before the query_window_size is exeeded
  // used by index thread & F2C thread to control index scan rate:
  sem_t sem_batch_window_free_slots; // available slots in the batch window, cnt = batch_window_size - (C2F_batch_id - F2C_batch_id)
  // used by F2C thread & C2F thread to control send rate:
  sem_t sem_query_window_free_slots; // available slots in the query window, cnt = query_window_size - (C2F_query_id - F2C_query_id)

  unsigned int C2F_send_index;
  unsigned int F2C_rcv_index;

  int C2F_batch_id;        // signal until what batch of data has been sent to FPGA
  int finish_C2F_query_id; 
  int finish_F2C_query_id;

  // size in bytes
  size_t bytes_C2F_header;
  size_t bytes_F2C_header;
  size_t bytes_vec;
  size_t bytes_F2C_per_query; // expected bytes received per query including header
  size_t bytes_C2F_per_query;

  /* An illustration of the semaphore logics:
  
  query_window_size is used for constraint communication:
    the F2C thread should not send much earlier before the last query is sent to FPGA,
      controlled by query_window_size

  batch_window_size is used to track how many batches are not sent to FPGA yet,
    determining when the C2F thread can fetch data to send
  
  ------------------------------      
  |      C2F thread            |        |
  ------------------------------        |
    | sem_query_window_free_slots       | 
    v                                   | sem_batch_window_free_slots
  ------------------------------        |
  |      F2C thread            |        |
  ------------------------------        v
  
  */

  // variables used for connections
  int* sock_c2f;
  int* sock_f2c;

  // C2F & F2C buffers, length = single batch of queries including padding
  char* buf_F2C_per_FPGA[MAX_FPGA_NUM];
  char* buf_C2F;

  std::chrono::system_clock::time_point* batch_start_time_array;
  std::chrono::system_clock::time_point* batch_finish_time_array;
  // end-to-end performance
  double* batch_duration_ms_array;
  double QPS;

  // constructor
  CPU_client_simulator(
    const size_t in_D,
    const size_t in_ef, 
    const int in_query_num,
    const int in_batch_size,
    const int in_query_window_size,
    const int in_batch_window_size,
    const int in_num_FPGA,
    const char** in_FPGA_IP_addr,
    const unsigned int* in_C2F_port,
    const unsigned int* in_F2C_port) :
    D(in_D), ef(in_ef), query_num(in_query_num), batch_size(in_batch_size), 
    query_window_size(in_query_window_size), batch_window_size(in_batch_window_size),
    num_FPGA(in_num_FPGA), FPGA_IP_addr(in_FPGA_IP_addr), C2F_port(in_C2F_port), F2C_port(in_F2C_port) {
        
    // Initialize internal variables
    total_batch_num = query_num % batch_size == 0? query_num / batch_size : query_num / batch_size + 1;

    start_F2C = 0;
    start_C2F = 0;
    finish_C2F_query_id = -1; 
    finish_F2C_query_id = -1;
    C2F_send_index = 0;
    F2C_rcv_index = 0;
    C2F_batch_id = -1;
    
    sem_init(&sem_query_window_free_slots, 0, query_window_size); // 0 = share between threads of a process
    sem_init(&sem_batch_window_free_slots, 0, batch_window_size); // 0 = share between threads of a process

    // C2F sizes
    const int AXI_num_header = 1;
    const int AXI_num_vec = D % FLOAT_PER_AXI == 0? D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; 

    size_t bytes_header = AXI_num_header * BYTES_PER_AXI;
    bytes_C2F_header = bytes_header;
	bytes_F2C_header = bytes_header;
    bytes_vec = AXI_num_vec * BYTES_PER_AXI;
    bytes_C2F_per_query = bytes_C2F_header + bytes_vec; 

    // F2C sizes
    const int AXI_num_results_vec_ID = ef % INT_PER_AXI == 0? ef / INT_PER_AXI : ef / INT_PER_AXI + 1;
    const int AXI_num_results_dist = ef % FLOAT_PER_AXI == 0? ef / FLOAT_PER_AXI : ef / FLOAT_PER_AXI + 1;

    size_t bytes_results_vec_ID = AXI_num_results_vec_ID * BYTES_PER_AXI;
    size_t bytes_results_dist = AXI_num_results_dist * BYTES_PER_AXI;
    bytes_F2C_per_query = bytes_F2C_header + bytes_results_vec_ID + bytes_results_dist;

    std::cout << "bytes_C2F_per_query (include 64-byte header): " << bytes_C2F_per_query << std::endl;
    std::cout << "bytes_F2C_per_query (include 64-byte header):" << bytes_F2C_per_query << std::endl;

    sock_f2c = (int*) malloc(num_FPGA * sizeof(int));
    sock_c2f = (int*) malloc(num_FPGA * sizeof(int));

    for (int i = 0; i < num_FPGA; i++) {
      buf_F2C_per_FPGA[i] = (char*) malloc(bytes_F2C_per_query * query_num);
    }
    buf_C2F = (char*) malloc(bytes_vec * query_num); // only the queries

    batch_start_time_array = (std::chrono::system_clock::time_point*) malloc(total_batch_num * sizeof(std::chrono::system_clock::time_point));
    batch_finish_time_array = (std::chrono::system_clock::time_point*) malloc(total_batch_num * sizeof(std::chrono::system_clock::time_point));
    batch_duration_ms_array = (double*) malloc(total_batch_num * sizeof(double));

    assert (in_num_FPGA < MAX_FPGA_NUM);
  }

  // C2F send batch header
  void send_header(char *buf_header) {
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
  void send_query(char *buf_query_vec) {
    for (int n = 0; n < num_FPGA; n++) {
      size_t total_C2F_bytes = 0;
      while (total_C2F_bytes < bytes_vec) {
        int C2F_bytes_this_iter = (bytes_vec - total_C2F_bytes) < C2F_PKG_SIZE ? (bytes_vec - total_C2F_bytes) : C2F_PKG_SIZE;
        int C2F_bytes = send(sock_c2f[n], buf_query_vec + total_C2F_bytes, C2F_bytes_this_iter, 0);
        total_C2F_bytes += C2F_bytes;
        if (C2F_bytes == -1) {
          printf("Sending data UNSUCCESSFUL!\n");
          return;
        } else {
          IF_DEBUG_DO(std::cout << "total C2F bytes = " << total_C2F_bytes << std::endl;);
        }
      }
      if (total_C2F_bytes != bytes_vec) {
        printf("Sending error, sending more bytes than a block\n");
      }
    }
  }

  /* This method is meant to be run in a separate thread.
   * It establishes and maintains a connection to the FPGA.
   * It is resposible for sending the queries to the FPGA.
   */
  void thread_C2F() { 
      
    // wait for ready
    while(!start_F2C) {}

    for (int i = 0; i < num_FPGA; i++) {
      sock_c2f[i] = send_open_conn(FPGA_IP_addr[i], C2F_port[i]);
    }

    start_C2F = 1;

    ////////////////   Data transfer + Select Cells   ////////////////

    // used to prepare the header data in the exact layout the FPGA expects
    char buf_header[bytes_C2F_header];

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    for (int C2F_batch_id = 0; C2F_batch_id < total_batch_num; C2F_batch_id++) {

      std::cout << "C2F_batch_id: " << C2F_batch_id << std::endl;
      sem_wait(&sem_batch_window_free_slots);

      batch_start_time_array[C2F_batch_id] = std::chrono::system_clock::now();
      int current_batch_size = query_num - C2F_batch_id * batch_size < batch_size? query_num - C2F_batch_id * batch_size : batch_size;
      memcpy(buf_header, &current_batch_size, 4);
      send_header(buf_header);

      for (int query_id = C2F_batch_id * batch_size; query_id < C2F_batch_id * batch_size + current_batch_size; query_id++) {

        // this semaphore controls that the window size is adhered to
        // If the  semaphore currently has the value zero, then the call blocks
        //   until either it becomes possible to perform the decrement
        sem_wait(&sem_query_window_free_slots);

        char* current_query_addr = buf_C2F + bytes_vec * query_id;
        send_query(current_query_addr);

        finish_C2F_query_id++;
        std::cout << "C2F finish query_id " << finish_C2F_query_id << std::endl;
      }
    }
  
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
   
    // send finish: set batch_size as -1
    int finish_batch_size = -1;
    memcpy(buf_header, &finish_batch_size, 4);
    send_header(buf_header); 

    std::cout << "C2F side Duration (us) = " << durationUs << std::endl;
    std::cout << "C2F side QPS () = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;
    std::cout << "C2F side finished." << std::endl;
    
    return; 
  } 

  void receive_answer_to_query(size_t byte_offset) {

    for (int n = 0; n < num_FPGA; n++) {
      size_t total_F2C_bytes = 0;
      while (total_F2C_bytes < bytes_F2C_per_query) {
        int F2C_bytes_this_iter = (bytes_F2C_per_query - total_F2C_bytes) < F2C_PKG_SIZE ? (bytes_F2C_per_query - total_F2C_bytes) : F2C_PKG_SIZE;
        // IF_DEBUG_DO(std::cout << "F2C_bytes_this_iter" << F2C_bytes_this_iter << std::endl;);
        int F2C_bytes = read(sock_f2c[n], &buf_F2C_per_FPGA[n][byte_offset + total_F2C_bytes], F2C_bytes_this_iter);
        total_F2C_bytes += F2C_bytes;

        if (F2C_bytes == -1) {
          printf("Receiving data UNSUCCESSFUL!\n");
          return;
        } else {
          IF_DEBUG_DO(std::cout << " F2C_bytes" << total_F2C_bytes << std::endl;);
        }
      }

      if (total_F2C_bytes != bytes_F2C_per_query) {
        printf("Receiving error, receiving more bytes than a block\n");
      }
    }
  }

  void thread_F2C() { 

    std::cout << "FPGA programs must be started in order (same as the input argument) " <<
      " because the F2C receive side receives connections in order " << std::endl;
    for (int i = 0; i < num_FPGA; i++) {
      sock_f2c[i] = recv_accept_conn(F2C_port[i]);
    }
    start_F2C = 1;
    while(!start_C2F) {}
    ///////////////////////////////////////
    //// START RECEIVING DATA /////////////
    ///////////////////////////////////////

    printf("Start receiving data.\n");

    ////////////////   Data transfer   ////////////////

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now(); // reset after recving the first query

    for (int F2C_batch_id = 0; F2C_batch_id < total_batch_num; F2C_batch_id++) {

      std::cout << "F2C_batch_id: " << F2C_batch_id << std::endl;
      int current_batch_size = query_num - F2C_batch_id * batch_size < batch_size? query_num - F2C_batch_id * batch_size : batch_size;

      for (int query_id = F2C_batch_id * batch_size; query_id < F2C_batch_id * batch_size + current_batch_size; query_id++) {

        IF_DEBUG_DO(std::cout << "F2C query_id " << query_id << std::endl;);
        size_t byte_offset = query_id * bytes_F2C_per_query;
        receive_answer_to_query(byte_offset);

        finish_F2C_query_id++;
        std::cout << "F2C finish query_id " << finish_F2C_query_id << std::endl;
        // sem_post() increments (unlocks) the semaphore pointed to by sem
        sem_post(&sem_query_window_free_slots);
      }
      batch_finish_time_array[F2C_batch_id] = std::chrono::system_clock::now();
      sem_post(&sem_batch_window_free_slots);
    }

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

    std::cout << "F2C side Duration (us) = " << durationUs << std::endl;
    std::cout << "F2C side QPS = " << query_num / (durationUs / 1000.0 / 1000.0) << std::endl;
    std::cout << "F2C side Finished." << std::endl;

    return;  
  }

  void start_C2F_F2C_threads() {

    // start thread with member function: https://stackoverflow.com/questions/10673585/start-thread-with-member-function
    std::thread t_F2C(&CPU_client_simulator::thread_F2C, this);
    std::thread t_C2F(&CPU_client_simulator::thread_C2F, this);

    t_F2C.join();
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
    // for (int b = 0; b < total_batch_num; b++) {
    //   std::cout << "  Batch " << b << " (ms): " << batch_duration_ms_array[b] << std::endl;
    // }


    double durationUs = (std::chrono::duration_cast<std::chrono::microseconds>(
      batch_finish_time_array[total_batch_num - 1] - batch_start_time_array[0]).count());
    double durationMs = durationUs / 1000.0;
    QPS = query_num / (durationUs / 1000.0 / 1000.0);
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
  std::cout << "Usage: " << argv[0] << " <1 num_FPGA> "
      "<2 ~ 2 + num_FPGA - 1 FPGA_IP_addr> " 
    "<2 + num_FPGA ~ 2 + 2 * num_FPGA - 1 C2F_port> " 
    "<2 + 2 * num_FPGA ~ 2 + 3 * num_FPGA - 1 F2C_port> "
    "<2 + 3 * num_FPGA D> <3 + 3 * num_FPGA ef> " 
    "<4 + 3 * num_FPGA query_num> " "<5 + 3 * num_FPGA batch_size> "
    "<6 + 3 * num_FPGA query_window_size> <7 + 3 * num_FPGA batch_window_size> " 
    << std::endl;

  int argv_cnt = 1;
  int num_FPGA = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "num_FPGA: " << num_FPGA << std::endl;
  assert(argc == 8 + 3 * num_FPGA);
  assert(num_FPGA <= MAX_FPGA_NUM);

  const char* FPGA_IP_addr[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      FPGA_IP_addr[n] = argv[argv_cnt++];
      std::cout << "FPGA " << n << " IP addr: " << FPGA_IP_addr[n] << std::endl;
  }   
  // FPGA_IP_addr = "10.253.74.5"; // alveo-build-01
  // FPGA_IP_addr = "10.253.74.12"; // alveo-u250-01
  // FPGA_IP_addr = "10.253.74.16"; // alveo-u250-02
  // FPGA_IP_addr = "10.253.74.20"; // alveo-u250-03
  // FPGA_IP_addr = "10.253.74.24"; // alveo-u250-04

  unsigned int C2F_port[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      C2F_port[n] = strtol(argv[argv_cnt++], NULL, 10);
      std::cout << "C2F_port " << n << ": " << C2F_port[n] << std::endl;
  } 

  unsigned int F2C_port[num_FPGA];
  for (int n = 0; n < num_FPGA; n++) {
      F2C_port[n] = strtol(argv[argv_cnt++], NULL, 10);
      std::cout << "F2C_port " << n << ": " << F2C_port[n] << std::endl;
  } 
    
  size_t D = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "D: " << D << std::endl;

  size_t ef = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "ef: " << ef << std::endl;

  int query_num = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "query_num: " << query_num << std::endl;

  int batch_size = strtol(argv[argv_cnt++], NULL, 10);
  std::cout << "batch_size: " << batch_size << std::endl;

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
    
  CPU_client_simulator cpu_coordinator(
    D,
    ef, 
    query_num,
    batch_size,
    query_window_size,
    batch_window_size,
    num_FPGA,
    FPGA_IP_addr,
    C2F_port,
    F2C_port);

  cpu_coordinator.start_C2F_F2C_threads();
  cpu_coordinator.calculate_latency();

  return 0; 
} 
