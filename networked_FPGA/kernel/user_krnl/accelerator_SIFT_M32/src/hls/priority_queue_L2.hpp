#pragma once 

// #include <type_traits>
#include "constants.hpp"
// NOTES: 1. use 2020.2 
// 2. Use constructor to partition member array / include pragma 
//    ERROR: [v++ 214-122] '#pragma HLS' is only allowed in function scope: /mnt/scratch/wenqi/FPGA-ANNS/building_blocks/class_priority_queue/src/priority_queue.hpp:23
// enum class Order { Collect_smallest, Collect_largest };

template<typename T, const int queue_size, Order order> 
class Priority_queue_L2;

template<const int queue_size> 
class Priority_queue_L2<result_t, queue_size, Collect_smallest> {

    public: 

        Priority_queue_L2() {
#pragma HLS inline
        }

        void insert_wrapper(
            int query_num,
            hls::stream<int> &s_control_iter_num_per_query,
            hls::stream<result_t> &s_input, 
            hls::stream<result_t> &s_output) {
            
            // smaller value swapped to the right (larger index ID)
            // queue[0] used to store input value
            result_t queue[queue_size + 1];
#pragma HLS array_partition variable=queue complete

            for (int query_id = 0; query_id < query_num; query_id++) {

                int iter_num = s_control_iter_num_per_query.read();

                // init
                for (int i = 0; i < queue_size + 1; i++) {
#pragma HLS UNROLL
                    // queue[i].vec_ID = -1;
                    queue[i].dist = LARGE_NUM;
                }

                // insert: 
                for (int i = 0; i < iter_num; i++) {
#pragma HLS pipeline II=1
                    queue[0] = s_input.read();

                    compare_swap_array_step_A(queue);

                    compare_swap_array_step_B(queue);
                }

                // write
                for (int i = 1; i < queue_size + 1; i++) {
#pragma HLS pipeline II=1
                    s_output.write(queue[i]);
                }
            }
        }


    private:
    
        void compare_swap(result_t* array, int idxA, int idxB) {
            // if smaller -> swap to right
            // note: idxA must < idxB
#pragma HLS inline
            if (array[idxA].dist < array[idxB].dist) {
                result_t regA = array[idxA];
                result_t regB = array[idxB];
                array[idxA] = regB;
                array[idxB] = regA;
            }
        }

        void compare_swap_array_step_A(result_t* array) {
            // start from idx 0, odd-even swap
#pragma HLS inline
            for (int j = 0; j < (queue_size + 1) / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j, 2 * j + 1);
            }
        }
                    
        void compare_swap_array_step_B(result_t* array) {
            // start from idx 1, odd-even swap
#pragma HLS inline
            for (int j = 0; j < queue_size / 2; j++) {
#pragma HLS UNROLL
                compare_swap(array, 2 * j + 1, 2 * j + 2);
            }
        }
};
