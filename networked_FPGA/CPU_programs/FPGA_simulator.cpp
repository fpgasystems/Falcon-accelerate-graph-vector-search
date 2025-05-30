// FPGA_simulator: simulate the behavior for a single FPGA
//   2 thread, 1 for sending results, 1 for receiving queries

// Refer to https://github.com/WenqiJiang/FPGA-ANNS-with_network/blob/master/CPU_scripts/unused/network_send.c
// std::cout << "Usage: 
//  " << argv[0] << " <Tx (CPU) IP_addr> <Tx F2C_port> <Rx C2F_port> " 
// 	"<TOPK/ef> <D> <query_num> " << std::endl;

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

#include "constants.hpp"
#include "types.hpp"
#include "utils.hpp"

#define DEBUG

void thread_F2C(
  const char* IP_addr,
  unsigned int F2C_port,
  int query_num,
  int* start_send,
  int* finish_recv_query_id,
  int* finish_all,
  int D,
  int TOPK
) { 
      
    // const char* IP_addr = send_thread_input.IP_addr;
  // unsigned int F2C_port = send_thread_input.F2C_port;
    // int query_num = send_thread_input.query_num;
    // int* start_send = send_thread_input.start_send;
    // int* finish_recv_query_id = send_thread_input.finish_recv_query_id;
    // int D = send_thread_input.D;
    // int TOPK = send_thread_input.TOPK;
  int ef = TOPK;

    // Format: for each query
    // packet 0: header (topK == ef)
    // packet 1~k: topK results, including vec_ID (4-byte) array and dist_array (4-byte)
  //    -> size = ceil(topK * 4 / 64) + ceil(topK * 4 / 64)

    // in 512-bit packets
  const int AXI_num_header = 1; 
  const int AXI_num_results_vec_ID = ef % INT_PER_AXI == 0? ef / INT_PER_AXI : ef / INT_PER_AXI + 1;
  const int AXI_num_results_dist = ef % FLOAT_PER_AXI == 0? ef / FLOAT_PER_AXI : ef / FLOAT_PER_AXI + 1;
  const int AXI_num_output_per_query = AXI_num_header + AXI_num_results_vec_ID + AXI_num_results_dist;

  size_t bytes_header = AXI_num_header * BYTES_PER_AXI;
  size_t bytes_results_vec_ID = AXI_num_results_vec_ID * BYTES_PER_AXI;
  size_t bytes_results_dist = AXI_num_results_dist * BYTES_PER_AXI;
  size_t bytes_output_per_query = bytes_header + bytes_results_vec_ID + bytes_results_dist;

  size_t bytes_out_total = query_num * bytes_output_per_query;
  char* out_buf = new char[bytes_out_total];
  memset(out_buf, 0, bytes_out_total); 

  std::cout << "bytes_output_per_query: " << bytes_output_per_query << std::endl;
  printf("Printing F2C_port from Thread %d\n", F2C_port); 
    
  int sock = send_open_conn(IP_addr, F2C_port);

  printf("Start sending data.\n");
  *start_send = 1;

  ////////////////   Data transfer + Select Cells   ////////////////


  int query_id = 0;
  while (true) {

    volatile int cnt = 0;
    while (query_id > *finish_recv_query_id && *finish_all == 0) {
        cnt++;
        // sleep(0.001);
    }
    if (query_id > *finish_recv_query_id && *finish_all == 1) {
      std::cout << "thread F2C finished all" << std::endl;
      break;
    }

    std::cout << "send query_id " << query_id << std::endl;

	// send data until finish_recv_query_id
	int send_query_num_this_iter = *finish_recv_query_id - query_id + 1;
	int total_sent_bytes = 0;
	while (total_sent_bytes < send_query_num_this_iter * bytes_output_per_query) {
		int send_bytes_this_iter = (send_query_num_this_iter * bytes_output_per_query - total_sent_bytes);
		int sent_bytes = send(sock, out_buf + query_id * bytes_output_per_query + total_sent_bytes, send_bytes_this_iter, 0);
		total_sent_bytes += sent_bytes;
		if (sent_bytes == -1) {
			printf("Sending data UNSUCCESSFUL!\n");
			return;
		}
	}
	query_id += send_query_num_this_iter;

//     // send data
//     int total_sent_bytes = 0;

//     while (total_sent_bytes < bytes_output_per_query) {
//       int send_bytes_this_iter = (bytes_output_per_query - total_sent_bytes) < F2C_PKG_SIZE? (bytes_output_per_query - total_sent_bytes) : F2C_PKG_SIZE;
//       int sent_bytes = send(sock, &out_buf[query_id * bytes_output_per_query + total_sent_bytes], send_bytes_this_iter, 0);
//       total_sent_bytes += sent_bytes;
//       if (sent_bytes == -1) {
//         printf("Sending data UNSUCCESSFUL!\n");
//         return;
//       }
// #ifdef DEBUG
//       else {
//         printf("total sent bytes = %d\n", total_sent_bytes);
//       }
// #endif
//       }

//       if (total_sent_bytes != bytes_output_per_query) {
//         printf("Sending error, sending more bytes than a block\n");
//       }
//       query_id++;
    }

    return; 
} 

void thread_C2F(
  unsigned int C2F_port,
  int query_num,
  int* start_send,
  int* finish_recv_query_id,
  int* finish_all,
  int D,
  int TOPK
) { 


  // unsigned int C2F_port = recv_thread_input.C2F_port;
  // int query_num = recv_thread_input.query_num;
  // int* start_send = recv_thread_input.start_send; 
  // int* finish_recv_query_id = recv_thread_input.finish_recv_query_id;
  // int D = recv_thread_input.D;
  // int TOPK = recv_thread_input.TOPK;

  printf("Printing C2F_port from Thread %d\n", C2F_port); 

  int sock = recv_accept_conn(C2F_port);
    std::cout << "recv sock " << sock << std::endl; 


  // Format: for each query
  // packet 0: header (batch_size, ) -> batch size as -1 or 0 means terminate
  //   for the following packets, for each query
  // 		packet 1~k: query_vectors

  // query format: store in 512-bit packets, pad 0 for the last packet if needed
  const int AXI_num_header = 1; 
  const int AXI_num_vec = D % FLOAT_PER_AXI == 0? D / FLOAT_PER_AXI : D / FLOAT_PER_AXI + 1; 
  const int AXI_num_input_per_query = AXI_num_header + AXI_num_vec;

  size_t bytes_header = AXI_num_header * BYTES_PER_AXI;
  size_t bytes_vec = AXI_num_vec * BYTES_PER_AXI;
  size_t bytes_input_per_query = bytes_vec;

  std::cout << "bytes_input_per_query: " << bytes_input_per_query << std::endl;
  size_t FPGA_input_bytes = query_num * bytes_input_per_query;
  std::vector<char ,aligned_allocator<char >> FPGA_input(FPGA_input_bytes);

  printf("Start receiving data.\n");

  ////////////////   Data transfer   ////////////////

  // Should wait until the server said all the data was sent correctly,
  // otherwise the sender may send packets yet the server did not receive.

  // recv header -> recv query; etc. until there's a header where the batch size is -1
  while (true) {

    int total_recv_bytes = 0;
    char header_buf[bytes_header];
    while (total_recv_bytes < bytes_header) {
      int recv_bytes_this_iter = (bytes_header - total_recv_bytes);
    //   int recv_bytes_this_iter = (bytes_header - total_recv_bytes) < C2F_PKG_SIZE? (bytes_header - total_recv_bytes) : C2F_PKG_SIZE;
      int recv_bytes = read(sock, header_buf + total_recv_bytes, recv_bytes_this_iter);
      total_recv_bytes += recv_bytes;
      if (recv_bytes == -1) {
        printf("Receiving data UNSUCCESSFUL!\n");
        return;
      }
    }
    int batch_size = *(int*)header_buf;
    if (batch_size == -1) {
      *finish_all = 1;
      printf("thread C2F finished all");
      break;
    }

	// receive queries of batch size at once
	total_recv_bytes = 0;
	while (total_recv_bytes < bytes_input_per_query * batch_size) {
		int recv_bytes_this_iter = (bytes_input_per_query * batch_size - total_recv_bytes);
		int recv_bytes = read(sock, FPGA_input.data() + total_recv_bytes, recv_bytes_this_iter);
		total_recv_bytes += recv_bytes;
		if (recv_bytes == -1) {
			printf("Receiving data UNSUCCESSFUL!\n");
			return;
		}
	}
	*finish_recv_query_id += batch_size;


//     for (int query_id = 0; query_id < batch_size; query_id++) {

//       std::cout << "recv query_id " << *finish_recv_query_id + 1 << std::endl;
//       int total_recv_bytes = 0;
//       while (total_recv_bytes < bytes_input_per_query) {
//         int recv_bytes_this_iter = (bytes_input_per_query - total_recv_bytes) < C2F_PKG_SIZE? (bytes_input_per_query - total_recv_bytes) : C2F_PKG_SIZE;
//         int recv_bytes = read(sock, FPGA_input.data() + query_id * bytes_input_per_query + total_recv_bytes, recv_bytes_this_iter);
//         total_recv_bytes += recv_bytes;
        
//         if (recv_bytes == -1) {
//           printf("Receiving data UNSUCCESSFUL!\n");
//           return;
//         }
// #ifdef DEBUG
//         else {
//           std::cout << "query_id: " << *finish_recv_query_id + 1 << " recv_bytes" << total_recv_bytes << std::endl;
//         }
// #endif
//       }
//       // set shared register as soon as the first packet of the results is received
//       (*finish_recv_query_id)++; 

    //   if (total_recv_bytes != bytes_input_per_query) {
    //     printf("Receiving error, receiving more bytes than a block\n");
    //   }
    // }
  }

  return; 
} 


int main(int argc, char const *argv[]) 
{ 
  //////////     Parameter Init     //////////
  
  std::cout << "Usage: " << argv[0] << " <Tx (CPU) IP_addr> <Tx F2C_port> <Rx C2F_port> " 
  "<TOPK/ef> <D> <query_num> " << std::endl;

  int argv_cnt = 1;

  const char* IP_addr;
  if (argc >= 2) {
      IP_addr = argv[argv_cnt++];
  } else { 
  IP_addr = "10.253.74.5"; // alveo-build-01
  }

  unsigned int F2C_port = 5008; // send out results
  if (argc >= 3) {
      F2C_port = strtol(argv[argv_cnt++], NULL, 10);
  } 

  unsigned int C2F_port = 8888; // receive query
  if (argc >= 4) {
      C2F_port = strtol(argv[argv_cnt++], NULL, 10);
  } 

  int TOPK = 64;
  if (argc >= 5) {
    TOPK = strtol(argv[argv_cnt++], NULL, 10);
  }

  int D = 128;
  if (argc >= 6) {
    D = strtol(argv[argv_cnt++], NULL, 10);
  }

  size_t query_num = 10000;
  if (argc >= 7) {
      query_num = strtol(argv[argv_cnt++], NULL, 10);
  }

  
  //////////     Networking Part     //////////

  // inter-thread communication by shared memory
  int start_send = 0; 
  int finish_recv_query_id = -1;
  int finish_all = 0;

  // profiler
  std::vector<std::chrono::system_clock::time_point> query_start_time(query_num);
  std::vector<std::chrono::system_clock::time_point> query_finish_time(query_num);

  // launch 
  std::thread t_send(thread_F2C,
  IP_addr, F2C_port, query_num, &start_send, &finish_recv_query_id, &finish_all, D, TOPK);
  std::thread t_recv(thread_C2F,
  C2F_port, query_num, &start_send, &finish_recv_query_id, &finish_all, D, TOPK);

  // sync finish
  t_send.join();
  t_recv.join();

  return 0; 
} 